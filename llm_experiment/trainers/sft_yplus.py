import os
os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ['WANDB_START_METHOD'] = 'thread'

import accelerate
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trainers.network_utils import AutoModelForCausalLMWithValueHead
from trainers.sft_yplus_trainer import SFTYPlusTrainer
from trainers.sft_yplus_config import SFTYPlusConfig
from alpaca_farm.models.reward_model import RewardModel, RewardConfig
import torch
from absl import flags, app
import os
import gc
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import wandb
import re
from collections import defaultdict
from functools import reduce
from trainers.utils import (
    logprobs_from_logits,
    entropy_from_logits,
)

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'reweighted_bc', 'the wandb project name')
flags.DEFINE_string('run_name', 'reweighted_bc', 'the wandb run name')
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_string('dataset_path', "tatsu-lab/alpaca_farm", 'the path to the dataset')
flags.DEFINE_string('tokenizer_type', "EleutherAI/pythia-1.4b", 'the model name')
flags.DEFINE_string('pretrained_dir', "", 'the path to the pretrained model')
flags.DEFINE_float('learning_rate', 1.0e-6, 'the learning rate')
flags.DEFINE_float('cosine_annealing_lr_eta_min', 1.0e-7, 'the cosine annealing eta min')
flags.DEFINE_integer('num_train_epochs', 50, 'the number of training epochs')
flags.DEFINE_integer('inner_iteration_steps', 1, 'the number of training epochs')
flags.DEFINE_integer('eval_every_steps', 10, 'how often to evaluate')
flags.DEFINE_integer('save_every_steps', 1000, 'how often to save checkpoints')
flags.DEFINE_integer('num_eval_batches', 8, 'the number of evaluation batches of size gold shard size')
flags.DEFINE_float('clip_range', 0.2, 'the clip range')
flags.DEFINE_float('gae_lambda', 0.95, 'the GAE lambda')
flags.DEFINE_integer('batch_size', 64, 'the batch size')
flags.DEFINE_integer('max_gen_batch_size', 8, 'the max generation batch size')
flags.DEFINE_integer('mini_batch_size', 8, 'the chunk size')
flags.DEFINE_integer('seed', 42, 'the random seed')
flags.DEFINE_integer('gradient_accumulation_steps', 1, 'the gradient accumulation steps')
# score manipulation
flags.DEFINE_bool('use_score_scaling', False, 'whether to use score scaling')
flags.DEFINE_bool('use_score_norm', False, 'whether to use score normalization')
# flags for reinforce loss
flags.DEFINE_float('temperature', 1.0, 'the temperature for reweighting')
# flags for preference dataset
flags.DEFINE_string('preference_dataset_path', 'tatsu-lab/alpaca_farm', 'the path to the preference dataset')
flags.DEFINE_string("preference_dataset_subset", "alpaca_human_preference", "Dataset name")
flags.DEFINE_string("preference_dataset_split", "preference", "Dataset name")
flags.DEFINE_integer('preference_num_samples', 19000, 'the number of samples to use from the preference dataset')
flags.DEFINE_bool("batched", True, "Whether to use batched processing")
flags.DEFINE_integer("num_proc", 32, "Number of processes to use")
flags.DEFINE_float('mixing_ratio', 0.5, 'the mixing ratio for preference dataset')
# flags for generation
flags.DEFINE_integer('num_actions_per_prompt', 1, 'the number of actions per prompt for generation')
flags.DEFINE_string('cache_dir', '', 'the cache directory')
# flags for tpu
flags.DEFINE_bool('use_tpu', False, 'whether to use tpus')

def get_dataset(path, num_samples=-1, return_test_data=True, num_samples_test=1000):
    assert os.path.exists(path)
    folders = os.listdir(path)
    regex = r"^\d+-\d+$"
    folders = [x for x in folders if re.search(regex, x)]
    folders.sort(key=lambda x: int(x.split("-")[0]))
    total_samples = int(folders[-1].split("-")[-1])

    assert 0 < num_samples <= total_samples - num_samples_test, f"num_samples {num_samples} must be between 0 and {total_samples} - {num_samples_test}"
    assert 0 < num_samples_test <= total_samples, f"num_samples_test {num_samples_test} must be between 0 and {total_samples}"
    
    num_samples_train = num_samples if num_samples > 0 else total_samples - num_samples_test
    test_folders = [x for x in folders if int(x.split("-")[0]) >= num_samples_train]
    folders = [x for x in folders if int(x.split("-")[0]) < num_samples_train]
    
    datasets = [load_from_disk(os.path.join(path, x)) for x in folders]
    full_data =  concatenate_datasets(datasets)
    
    if num_samples > 0:
        full_data = full_data.select(range(num_samples))
        
    if return_test_data:
        test_datasets = [load_from_disk(os.path.join(path, x)) for x in test_folders]
        test_data = concatenate_datasets(test_datasets)
        if num_samples_test > 0:
            test_data = test_data.select(range(num_samples_test))
        return full_data, test_data
    
    return full_data
    

def construct_dataset(
    path,
    num_samples=-1,
    concatenate_prompt=False,
    num_samples_test=1000,
):
    data, test_data = get_dataset(path, num_samples=num_samples, return_test_data=True, num_samples_test=num_samples_test)

    if concatenate_prompt:
        def map_fn(d):
            for k in ["y_ref", "y_w", "y_l"]:
                d[k] = d["prompt"] + d[k]
            return d
        
        data = data.map(
            map_fn,
            num_proc=FLAGS.num_proc,
        )

    dataset_name = os.path.basename(path).split(".")[0]

    ds = DatasetDict(
        {
            "train": data,
            "test": test_data,
        }
    ) 
    return dataset_name, ds


PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'
EOS_TOKEN = '<|endoftext|>'

def main(_):
    dataset = load_dataset(FLAGS.dataset_path, split="unlabeled")
    eval_dataset = load_dataset(FLAGS.dataset_path, split="val")

    output_dir = f"{FLAGS.output_dir}/{FLAGS.wandb_project}/{FLAGS.run_name}"
    model_name = f"{FLAGS.wandb_project}_{FLAGS.run_name}"
    
    print('Output dir:', output_dir)
    print('Model name:', model_name)

    batch_size_pref_data = FLAGS.batch_size
    batch_size_online_data = 0
    
    if FLAGS.preference_dataset_path in ['tatsu-lab/alpaca_farm', 'Asap7772/alpaca_human_preference_gold', 'Asap7772/alpaca_human_preference_minlength', 'Asap7772/alpaca_human_preference_maxlength']:
        if FLAGS.preference_dataset_path == 'tatsu-lab/alpaca_farm':
            pref_dataset = load_dataset(FLAGS.preference_dataset_path, FLAGS.preference_dataset_subset, split="preference")
        else:
            split='train' if 'length' in FLAGS.preference_dataset_path else FLAGS.preference_dataset_split
            pref_dataset = load_dataset(FLAGS.preference_dataset_path, split=split)
        pref_dataset = pref_dataset.train_test_split(test_size=0.1, seed=FLAGS.seed)
        
        def process_dataset(batch):
            new_batch = defaultdict(list)
            for inst, inp, out1, out2, pref in zip(batch['instruction'], batch['input'], batch['output_1'], batch['output_2'], batch['preference']):
                if pref == 1:
                    selected = out1
                    rejected = out2
                else:
                    selected = out2
                    rejected = out1
                if inp:
                    text = f"{PROMPT_TOKEN}{inst}\n{inp}{EOS_TOKEN}{ASSISTANT_TOKEN}"
                else:
                    text = f"{PROMPT_TOKEN}{inst}{EOS_TOKEN}{ASSISTANT_TOKEN}"
                
                new_batch['prompt'].append(text)
                new_batch['y_w'].append(f"{text}{selected}{EOS_TOKEN}")
                new_batch['y_l'].append(f"{text}{rejected}{EOS_TOKEN}")
            return new_batch
        
        pref_dataset = pref_dataset.map(
            process_dataset,
            batched=FLAGS.batched,
            num_proc=FLAGS.num_proc,
        )
        
        pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
        remove_columns = ['instruction', 'input', 'output_1', 'output_2', 'preference', 'raw_preference', 'prompt', 'y_w', 'y_l']
    else:
        if FLAGS.preference_dataset_path.startswith('Asap7772'):
            pref_dataset_name = os.path.basename(FLAGS.preference_dataset_path)
            pref_dataset = load_dataset(FLAGS.preference_dataset_path)
        else:
            pref_dataset_name, pref_dataset = construct_dataset(
                path=FLAGS.preference_dataset_path,
                num_samples=FLAGS.preference_num_samples,
                concatenate_prompt=False,
            )
        print('Loaded dataset', pref_dataset_name)
        pref_dataset, eval_pref_dataset = pref_dataset['train'], pref_dataset['test']
        remove_columns = ['output', 'text', 'alpaca_text', 'y_ref', 'y_1', 'y_2', 'y_w', 'y_w_alpaca', 'y_l', 'y_l_alpaca', 'y_w_score', 'y_l_score', 'score_diff', 'prompt', 'alpaca_prompt']


    def process_dataset(batch):
        new_batch = {}
        new_batch['query'] = batch['prompt']
        new_batch['text_w'] =  batch['y_w'] 
        new_batch['text_l'] = batch['y_l']
        new_batch['response_w'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w']]
        new_batch['response_l'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_l']]
        
        shapes = {}
        for k, v in new_batch.items():
            shapes[k] = len(v)
        if reduce(lambda x,y: x if x==y else -1, list(shapes.values())) == -1:
            assert False, f"Shapes of all columns must be equal, but got {shapes}, {list(shapes.values())}"
        return new_batch


    pref_dataset = pref_dataset.map(
        process_dataset,
        batched=FLAGS.batched,
        num_proc=FLAGS.num_proc,
        remove_columns=remove_columns,
    )
    
    eval_pref_dataset = eval_pref_dataset.map(
        process_dataset,
        batched=FLAGS.batched,
        num_proc=FLAGS.num_proc,
        remove_columns=remove_columns,
    )

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    wandb_output_dir = tempfile.mkdtemp()
    config = SFTYPlusConfig(
        model_name=FLAGS.pretrained_dir,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        learning_rate=FLAGS.learning_rate,
        lam=FLAGS.gae_lambda,
        cliprange=FLAGS.clip_range,
        cliprange_value=FLAGS.clip_range,
        batch_size=FLAGS.batch_size,
        dataloader_batch_size=max(batch_size_online_data, 1),
        mini_batch_size=FLAGS.mini_batch_size,
        ppo_epochs=FLAGS.inner_iteration_steps,
        tracker_project_name=FLAGS.wandb_project,
        use_score_scaling=FLAGS.use_score_scaling,
        use_score_norm=FLAGS.use_score_norm,
        temperature=FLAGS.temperature,
        use_tpu=FLAGS.use_tpu,
        project_kwargs={
            'project_dir': output_dir,
        },
        tracker_kwargs={
            "wandb": {
                "name": FLAGS.run_name, 
                "id": unique_str,
                "dir": wandb_output_dir,
            }
        },
        log_with='wandb',
        seed=FLAGS.seed,
    )

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.tokenizer_type)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    eos = tokenizer.eos_token
    policy = AutoModelForCausalLM.from_pretrained(
        FLAGS.pretrained_dir,
        cache_dir=FLAGS.cache_dir, 
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map='balanced'
    )
    policy.resize_token_embeddings(len(tokenizer))
    model = AutoModelForCausalLMWithValueHead(policy)

    def formatting_prompts_func(example):
        inst, inp = example['instruction'], example['input']
        if inp:
            query = f"{PROMPT_TOKEN}{inst}\n{inp}{eos}{ASSISTANT_TOKEN}"
        else:
            query = f"{PROMPT_TOKEN}{inst}{eos}{ASSISTANT_TOKEN}"
        example['query'] = query
        return example

    dataset = dataset.map(formatting_prompts_func, batched=False)
    eval_dataset = eval_dataset.map(formatting_prompts_func, batched=False)

    print('Sample Train prompt:', dataset[0]['query'])
    print('Sample Eval prompt:', eval_dataset[0]['query'])

    trainer = SFTYPlusTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        additional_config_kwargs=FLAGS.flag_values_dict(),
    )

    generation_kwargs = {
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 256, # specify how many tokens you want to generate at most
        "temperature": 1.0, # control the temperature of the softmax, 1.0 means no change, lower means more greedy, higher means more diverse
        "use_cache": True, # whether or not the model should use the past key/values attentions (if the model supports it)
    }
    
    def empty_cache():
        gc.collect()
        if FLAGS.use_tpu:
            return
        torch.cuda.empty_cache()
        gc.collect()

    def empty_cache_decorator(func):
        def func_wrapper(*args, **kwargs):
            empty_cache()
            return func(*args, **kwargs)
        return func_wrapper
    
    def save_model(checkpoint_dir, epoch_num, add_prefix=True):
        if add_prefix:
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            print(f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")
 
    pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=max(batch_size_pref_data, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    train_as_eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=FLAGS.mini_batch_size,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        eval_pref_dataset,
        batch_size=FLAGS.mini_batch_size,
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    all_eval_dataloaders = {
        "train_as_eval_pref": train_as_eval_pref_dataset_dataloader,
        "eval_pref": eval_pref_dataset_dataloader,
    }

    
    zipped_dataloaders = pref_dataset_dataloader
    total_len = len(pref_dataset_dataloader)
    
    @empty_cache_decorator
    @torch.no_grad()
    def process_pref_batch(pref_batch):
        ### Process preference dataset
        pref_query = tokenizer(pref_batch["query"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').input_ids
        pref_query_tensors = accelerate.utils.send_to_device(pref_query, trainer.accelerator.device)
        
        # Tokenize together to be the same length
        all_pref = pref_batch["response_w"] + pref_batch["response_l"]
        tokenized = tokenizer(all_pref, padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=64+generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
        
        pref_response_w_tensors = tokenized[:len(pref_batch["response_w"])]
        pref_response_w_tensors = accelerate.utils.send_to_device(pref_response_w_tensors, trainer.accelerator.device)
        
        pref_response_l_tensors = tokenized[len(pref_batch["response_w"]):]
        pref_response_l_tensors = accelerate.utils.send_to_device(pref_response_l_tensors, trainer.accelerator.device)
        assert pref_response_l_tensors.shape[0] == len(pref_batch["response_l"])
        
        return pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors
    
    @empty_cache_decorator
    @torch.no_grad()
    def process_input_ids(input_ids):

        input_data = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        logits, _, _ = trainer.model(**input_data)

        old_logits, _, _ = trainer.ref_model(**input_data)
        old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])

        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        entropy = entropy_from_logits(logits)

        return logprobs, old_logprobs, entropy, logits

    @empty_cache_decorator
    @torch.no_grad()
    def process_batch_sft_yplus(batch):
        if batch is not None:
            #### Construct query tensors
            query_tensors = tokenizer(batch["query"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').to('cuda')

            #### Get generations from SFTModel (including prompt)
            all_generation_tokens = []
            for i in range(FLAGS.num_actions_per_prompt): # generate multiple completions per prompt
                if query_tensors.input_ids.shape[0] > FLAGS.max_gen_batch_size:
                    generation_tokens = []
                    for i in tqdm(range(0, query_tensors.input_ids.shape[0], FLAGS.max_gen_batch_size), desc=f"Generating for epoch {epoch}"):
                        sub_query_tensor = {k: v[i:i+FLAGS.max_gen_batch_size] for k, v in query_tensors.items()}
                        generation_tokens.append(trainer.model.generate(**sub_query_tensor, **generation_kwargs))
                        torch.cuda.empty_cache()
                    generation_tokens = torch.cat(generation_tokens, dim=0)
                else:
                    generation_tokens = trainer.model.generate(**query_tensors, **generation_kwargs)
                all_generation_tokens.append(generation_tokens)
            all_generation_tokens = torch.cat(all_generation_tokens, dim=0)

            logprobs, old_logprobs, entropy, logits = process_input_ids(all_generation_tokens)

            texts = tokenizer.batch_decode(all_generation_tokens, skip_special_tokens=True)

            #### Update batch with response
            batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]
            response_tensors = tokenizer(batch["response"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            response_tensors = accelerate.utils.send_to_device(response_tensors, trainer.accelerator.device)
            
            ### Reprocess query tensors
            query_tensors = query_tensors.input_ids
            # Ensure query and response tensors are same length
            query_tensors = query_tensors.repeat(FLAGS.num_actions_per_prompt, 1)
            assert query_tensors.shape[0] == response_tensors.shape[0], f"query_tensors.shape[0] = {query_tensors.shape[0]} != response_tensors.shape[0] = {response_tensors.shape[0]}"
            batch["query"] = batch["query"] * FLAGS.num_actions_per_prompt
            query_tensors, response_tensors = query_tensors.to(policy.device), response_tensors.to(policy.device)
            logprobs, old_logprobs, entropy, logits = logprobs.to(policy.device), old_logprobs.to(policy.device), entropy.to(policy.device), logits.to(policy.device)
        else:
            query_tensors, response_tensors = None, None
            logprobs, old_logprobs, entropy, logits = None, None, None, None
        return batch, query_tensors, response_tensors, logprobs, old_logprobs, entropy, logits

    print("Starting training")
    total_iterations = 0
    columns_to_log: list[str] = ["query", "response_w", "response_l"]
    columns_to_log_eval: list[str] = ["query", "response"]

    for epoch in tqdm(range(FLAGS.num_train_epochs), desc="Epochs"):
        for sub_iteration, pref_batch in tqdm(enumerate(zipped_dataloaders), desc="Batches", total=total_len):
            empty_cache()

            stats = {}

            if total_iterations % FLAGS.eval_every_steps == 0:
                # TODO: add eval stats
                for eval_name, eval_dataloader in all_eval_dataloaders.items():
                    print(f"Running evaluation on {eval_name}")
                    eval_batch = next(iter(eval_dataloader))
                    eval_batch, query_tensors, response_tensors, logprobs, old_logprobs, entropy, logits = process_batch_sft_yplus(eval_batch)
                    all_to_log = {}
                    for k in columns_to_log_eval:
                        all_to_log[k] = (all_to_log.get(k, []) + [x.cpu().numpy().item() if isinstance(x, torch.Tensor) else x for x in eval_batch[k]])
                    del eval_batch, query_tensors, response_tensors
                    empty_cache()
                    
                    # log lengths
                    char_lengths = np.array([len(x) for x in all_to_log['response']])
                    stats[f"{eval_name}/char_lengths"] = char_lengths
                    stats[f"{eval_name}/char_lengths_mean"] = char_lengths.mean()
                    stats[f"{eval_name}/char_lengths_std"] = char_lengths.std()
                    stats[f"{eval_name}/char_lengths_max"] = char_lengths.max()
                    stats[f"{eval_name}/char_lengths_min"] = char_lengths.min()
                    
                    token_lengths = np.array([len(tokenizer(x).input_ids) for x in all_to_log['response']])
                    stats[f"{eval_name}/token_lengths"] = token_lengths
                    stats[f"{eval_name}/token_lengths_mean"] = token_lengths.mean()
                    stats[f"{eval_name}/token_lengths_std"] = token_lengths.std()
                    stats[f"{eval_name}/token_lengths_max"] = token_lengths.max()
                    stats[f"{eval_name}/token_lengths_min"] = token_lengths.min()
                    
                    word_lengths = np.array([len(re.findall("[a-zA-Z_]+", x)) for x in all_to_log['response']])
                    stats[f"{eval_name}/word_lengths"] = word_lengths
                    stats[f"{eval_name}/word_lengths_mean"] = word_lengths.mean()
                    stats[f"{eval_name}/word_lengths_std"] = word_lengths.std()
                    stats[f"{eval_name}/word_lengths_max"] = word_lengths.max()
                    stats[f"{eval_name}/word_lengths_min"] = word_lengths.min()

                    stats[f"{eval_name}/entropy"] = entropy.mean().item()
                    stats[f"{eval_name}/logprobs"] = logprobs.mean().item()
                    stats[f"{eval_name}/old_logprobs"] = old_logprobs.mean().item()
                    stats[f"{eval_name}/approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2).mean()).item()
                    stats[f"{eval_name}/policykl"] = (logprobs - old_logprobs).mean().item()
                    stats[f"{eval_name}/sequence_approxkl"] = (0.5 * ((logprobs - old_logprobs) ** 2)).sum(-1).mean().item()
                    stats[f"{eval_name}/sequence_policykl"] = (logprobs - old_logprobs).sum(-1).mean().item()

                    # log table of completions
                    table_rows = list(r for r in zip(*[all_to_log[col] for col in columns_to_log_eval]))
                    stats[f"{eval_name}/table"] = wandb.Table(columns=[*columns_to_log_eval], rows=table_rows)

                empty_cache()

            pref_batch, pref_query_tensors, pref_response_w_tensors, pref_response_l_tensors = process_pref_batch(pref_batch) # use pref data completions
            
            output_batch = {k: pref_batch[k] for k in columns_to_log}
            #### Run Trainer step
            train_stats = trainer.step(queries=pref_query_tensors, responses_w=pref_response_w_tensors, responses_l=pref_response_l_tensors)
            for key in train_stats:
                stats[key] = train_stats[key]
                
            rewards = torch.zeros(pref_query_tensors.shape[0], dtype=torch.float32)
            rewards = accelerate.utils.send_to_device(rewards, trainer.accelerator.device)

            stats['epoch'] = epoch + sub_iteration/len(zipped_dataloaders)
            stats['total_iterations'] = total_iterations 
            stats['gradient_steps'] = total_iterations * FLAGS.inner_iteration_steps 

            if stats['total_iterations'] % FLAGS.save_every_steps == 0:
                num_batches = stats['total_iterations']
                save_model(model_name + f"_num_batches_{num_batches}", epoch)

            total_iterations += 1
            trainer.log_stats(
                stats=stats,
                batch=output_batch,
                rewards=rewards,
                columns_to_log=columns_to_log
            )
        save_model(model_name + f"_epoch_{epoch}", epoch)

if __name__ == "__main__":
    app.run(main)