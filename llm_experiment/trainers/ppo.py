import os
os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"
os.environ['WANDB_START_METHOD'] = 'thread'

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore")

from datasets import concatenate_datasets, load_dataset, load_from_disk, DatasetDict
from trainers.ppo_config import PPOConfig
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from trainers.ppo_trainer import PPOTrainer
from trainers.network_utils import AutoModelForCausalLMWithValueHead
from alpaca_farm.models.reward_model import RewardModel, RewardConfig
from transformers import pipeline
import torch
from absl import flags, app
import os
import accelerate
import gc
import datetime
import numpy as np
import tempfile
from tqdm import tqdm
import wandb
import re
from collections import defaultdict
from functools import reduce
import ipdb

FLAGS = flags.FLAGS
flags.DEFINE_string('wandb_project', 'ppo_rew_padfix_rew', 'the wandb project name')
flags.DEFINE_string('run_name', 'ppo_rew_labnoise0.0_rewnorm', 'the wandb run name')
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_string('dataset_path', "tatsu-lab/alpaca_farm", 'the path to the dataset')
flags.DEFINE_string('tokenizer_type', "EleutherAI/pythia-1.4b", 'the model name')
flags.DEFINE_string('pretrained_dir', "", 'the path to the pretrained model')
flags.DEFINE_string('reward_model', "", 'the path to the reward model')
flags.DEFINE_float('learning_rate', 1.0e-6, 'the learning rate')
flags.DEFINE_float('cosine_annealing_lr_eta_min', 1.0e-7, 'the cosine annealing eta min')
flags.DEFINE_integer('num_train_epochs', 10, 'the number of training epochs')
flags.DEFINE_integer('inner_iteration_steps', 4, 'the number of training epochs')
flags.DEFINE_integer('eval_every_steps', 10, 'how often to evaluate')
flags.DEFINE_integer('save_every_steps', 100, 'how often to save checkpoints')
flags.DEFINE_integer('num_eval_batches', 8, 'the number of evaluation batches of size gold shard size')
flags.DEFINE_float('clip_range', 0.2, 'the clip range')
flags.DEFINE_float('gae_lambda', 0.95, 'the GAE lambda')
flags.DEFINE_float('vf_coef', 0.1, 'Scaling factor for value loss')
flags.DEFINE_float('target_kl', 6.0, 'Stop early if we exceed this value by over 50%')
# adaptive KL control
flags.DEFINE_bool('adap_kl_ctrl', True, 'Whether to use adaptive KL control')
flags.DEFINE_float('init_kl_coef', 0.2, 'Initial KL penalty coefficient (used for adaptive and linear control)')
flags.DEFINE_float('target', 6.0, 'Target KL value for adaptive KL control')
flags.DEFINE_integer('horizon', 10000, 'Horizon for adaptive KL control')
flags.DEFINE_string('kl_penalty', 'kl', 'the type of KL penalty-kl, abs, mse or full')
# value/gradient clipping
flags.DEFINE_float('cliprange', 0.2, 'Range for clipping in PPO policy gradient loss')
flags.DEFINE_float('cliprange_value', 0.2, 'Range for clipping values in loss calculation')
# batch sizes
flags.DEFINE_integer('batch_size', 256, 'the batch size')
flags.DEFINE_integer('max_gen_batch_size', 16, 'the max generation batch size')
flags.DEFINE_integer('mini_batch_size', 8, 'the chunk size')
flags.DEFINE_integer('seed', 42, 'the random seed')
flags.DEFINE_integer('gradient_accumulation_steps', 1, 'the gradient accumulation steps')
# score manipulation
flags.DEFINE_bool('use_score_scaling', False, 'whether to use score scaling')
flags.DEFINE_bool('use_score_norm', False, 'whether to use score normalization')
# flags for preference dataset
flags.DEFINE_string('preference_dataset_path', 'tatsu-lab/alpaca_farm', 'the path to the preference dataset')
flags.DEFINE_string("preference_dataset_subset", "alpaca_human_preference", "Dataset name")
flags.DEFINE_string("preference_dataset_split", "preference", "Dataset name")
flags.DEFINE_integer('preference_num_samples', 19000, 'the number of samples to use from the preference dataset')
flags.DEFINE_bool("batched", True, "Whether to use batched processing")
flags.DEFINE_integer("num_proc", 32, "Number of processes to use")
flags.DEFINE_float('mixing_ratio', 0.0, 'the mixing ratio for preference dataset')
# flags to setup length reward model.
flags.DEFINE_bool('use_length_reward', False, 'whether to use length reward')
flags.DEFINE_string('length_type', 'token', 'the length type')
# flags to setup gold reward model.
flags.DEFINE_bool('use_gold_reward_model', False, 'whether to use gold reward model')
flags.DEFINE_string('gold_reward_model_path', '', 'the path to the reward model')
flags.DEFINE_integer('gold_shard_size', 8, 'the number of shards to use for gold reward model')
flags.DEFINE_string('sft10k_path', '', 'the path to the sft10k model')
flags.DEFINE_bool('flash_attn', False, 'whether to use flash attention')
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

def main(_):
    dataset = load_dataset(FLAGS.dataset_path, split="unlabeled")
    eval_dataset = load_dataset(FLAGS.dataset_path, split="val")

    output_dir = f"{FLAGS.output_dir}/{FLAGS.wandb_project}/{FLAGS.run_name}"
    model_name = f"{FLAGS.wandb_project}_{FLAGS.run_name}"
    
    print('Output dir:', output_dir)
    print('Model name:', model_name)

    batch_size_pref_data = int(FLAGS.batch_size * FLAGS.mixing_ratio)
    batch_size_online_data = FLAGS.batch_size - batch_size_pref_data
    
    if FLAGS.preference_dataset_path == "tatsu-lab/alpaca_farm":
        pref_dataset = load_dataset(FLAGS.preference_dataset_path, FLAGS.preference_dataset_subset, split=FLAGS.preference_dataset_split)
        pref_dataset = pref_dataset.train_test_split(test_size=0.1, seed=FLAGS.seed)
        
        PROMPT_TOKEN = '<|prompter|>'
        ASSISTANT_TOKEN = '<|assistant|>'
        EOS_TOKEN = '<|endoftext|>'
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
        new_batch['query'] = batch['prompt'] + batch['prompt']
        new_batch['text'] =  batch['y_w'] + batch['y_l']
        new_batch['response'] = [x.split(ASSISTANT_TOKEN)[-1] for x in batch['y_w'] + batch['y_l']]
        
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
    config = PPOConfig(
        model_name=FLAGS.pretrained_dir,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        learning_rate=FLAGS.learning_rate,
        reward_model=FLAGS.reward_model,
        lam=FLAGS.gae_lambda,
        vf_coef=FLAGS.vf_coef,
        batch_size=FLAGS.batch_size,
        dataloader_batch_size=max(batch_size_online_data, 1),
        adap_kl_ctrl=FLAGS.adap_kl_ctrl,
        init_kl_coef=FLAGS.init_kl_coef,
        target=FLAGS.target,
        horizon=FLAGS.horizon,
        cliprange=FLAGS.cliprange,
        cliprange_value=FLAGS.cliprange_value,
        mini_batch_size=FLAGS.mini_batch_size,
        ppo_epochs=FLAGS.inner_iteration_steps,
        tracker_project_name=FLAGS.wandb_project,
        use_score_scaling=FLAGS.use_score_scaling,
        use_score_norm=FLAGS.use_score_norm,
        score_clip=None,
        optimize_cuda_cache=True,
        early_stopping=False,
        target_kl=FLAGS.target_kl,
        kl_penalty=FLAGS.kl_penalty,
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
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map='auto'
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

    trainer = PPOTrainer(
        model=model,
        config=config,
        dataset=dataset,
        tokenizer=tokenizer,
        additional_config_kwargs=FLAGS.flag_values_dict(),
    )

    generation_kwargs = {
        #"min_length": -1,
        "top_k": 0.0, # no top-k sampling
        "top_p": 1.0, # no nucleus sampling
        "do_sample": True, # yes, we want to sample
        "pad_token_id": tokenizer.eos_token_id, # most decoder models don't have a padding token - use EOS token instead
        "max_new_tokens": 256, # specify how many tokens you want to generate at most
        "temperature": 1.0, # control the temperature of the softmax, 1.0 means no change, lower means more greedy, higher means more diverse
        "use_cache": True, # whether or not the model should use the past key/values attentions (if the model supports it)
    }
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        FLAGS.reward_model,
        cache_dir=FLAGS.cache_dir, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map='auto'
    )
    reward_tokenizer = AutoTokenizer.from_pretrained(FLAGS.reward_model)
    if not FLAGS.use_gold_reward_model:
        reward_model = trainer.accelerator.prepare_model(reward_model)
    reward_model.eval()

    print("Loaded reward model")
    
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

    @empty_cache_decorator
    @torch.no_grad()
    def get_pred_reward(text, max_len=512):
        encoded_input = reward_tokenizer(text, padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=max_len, return_tensors='pt')
        encoded_input = accelerate.utils.send_to_device(encoded_input, trainer.accelerator.device)
        output = reward_model(**encoded_input)
        logits = output.logits.squeeze()
        return logits

    if FLAGS.use_gold_reward_model:
        gold_tokenizer = AutoTokenizer.from_pretrained(FLAGS.gold_reward_model_path)
        gold_model = RewardModel.from_pretrained(
            FLAGS.gold_reward_model_path,
            flash_attn=FLAGS.flash_attn,
            mixed_precision=True,
            torch_dtype=torch.bfloat16 if FLAGS.use_tpu else torch.float32,
            cache_dir=FLAGS.cache_dir,
            low_cpu_mem_usage=True,
            config=RewardConfig(backbone_model_name_or_path=FLAGS.sft10k_path),
        )
        gold_model.eval()
        print("Loaded gold reward model")
    else:
        gold_model = None
        gold_tokenizer = None
    
    @empty_cache_decorator
    @torch.no_grad()
    def get_gold_score(completions, shard_size=1):
        if shard_size >= 1:
            scores, beg = [], 0
            with tqdm(total=len(completions), desc="Computing gold scores") as pbar:
                while beg < len(completions):
                    end = min(beg+shard_size, len(completions))
                    tokenized_completions = gold_tokenizer(
                        completions[beg:end],
                        padding='max_length' if FLAGS.use_tpu else True,
                        max_length=None,
                        truncation=True,
                        return_tensors="pt",
                    )
                    tokenized_completions = accelerate.utils.send_to_device(tokenized_completions, trainer.accelerator.device)
                    empty_cache()
                    scores.append(gold_model(**tokenized_completions, return_dict=False)[0])
                    pbar.update(end-beg)
                    beg = end
            concated_scores = torch.cat(scores, dim=0)
            assert concated_scores.shape[0] == len(completions) and concated_scores.ndim == 1
            return concated_scores
        else:
            tokenized_completions = gold_tokenizer(
                completions,
                padding='max_length' if FLAGS.use_tpu else True,
                max_length=None,
                truncation=True,
                return_tensors="pt",
            )
            tokenized_completions = accelerate.utils.send_to_device(tokenized_completions, trainer.accelerator.device)
            scores = gold_model(**tokenized_completions, return_dict=False)
            assert concated_scores.shape[0] == len(completions) and concated_scores.ndim == 1
            empty_cache()
            return scores

    
    def save_model(checkpoint_dir, epoch_num, add_prefix=True):
        if add_prefix:
            checkpoint_dir = os.path.join(output_dir, checkpoint_dir)
            os.makedirs(checkpoint_dir, exist_ok=True)

        if trainer.accelerator.is_main_process:
            trainer.accelerator.unwrap_model(model).save_pretrained(
                checkpoint_dir,
                save_function=trainer.accelerator.save,
                is_main_process=trainer.accelerator.is_main_process,
                state_dict=trainer.accelerator.get_state_dict(model),
            )
            if trainer.accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
            trainer.accelerator.print(f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")
 
    pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=max(batch_size_pref_data, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    train_as_eval_online_prompt_dataset_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(FLAGS.gold_shard_size * FLAGS.num_eval_batches, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    eval_online_prompt_dataset_dataloader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=max(FLAGS.gold_shard_size * FLAGS.num_eval_batches, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    train_as_eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        pref_dataset,
        batch_size=max(FLAGS.gold_shard_size * FLAGS.num_eval_batches, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    eval_pref_dataset_dataloader = torch.utils.data.DataLoader(
        eval_pref_dataset,
        batch_size=max(FLAGS.gold_shard_size * FLAGS.num_eval_batches, 1),
        collate_fn=None,
        shuffle=True,
        drop_last=True,
    )
    
    all_eval_dataloaders = {
        "train_as_eval_online_prompt": train_as_eval_online_prompt_dataset_dataloader,
        "train_as_eval_pref": train_as_eval_pref_dataset_dataloader,
        "eval_online_prompt": eval_online_prompt_dataset_dataloader,
        "eval_pref": eval_pref_dataset_dataloader,
    }

    if batch_size_online_data == 0:
        zipped_dataloaders = pref_dataset_dataloader
    elif batch_size_pref_data == 0:
        zipped_dataloaders = trainer.dataloader
    else:
        zipped_dataloaders = zip(trainer.dataloader, pref_dataset_dataloader)
    
    if FLAGS.use_length_reward:
        if FLAGS.length_type == 'token':
            def process_completion(text: str) -> int:
                return len(tokenizer(text).input_ids)
        elif FLAGS.length_type == 'char':
            def process_completion(text: str) -> int:
                return len(text)
        elif FLAGS.length_type == 'word':
            def process_completion(text: str) -> int:
                return len(re.findall("[a-zA-Z_]+", text))
        else:
            assert False, f"length_type {FLAGS.length_type} not supported"
        
        def completion_length_reward(texts: str | list[str]) -> float | list[float]:
            if isinstance(texts, str):
                texts = [texts]
            scores = torch.asarray([process_completion(x) for x in texts], dtype=torch.float32)
            scores = accelerate.utils.send_to_device(scores, trainer.accelerator.device)
            return scores
        rew_fn = completion_length_reward
    elif FLAGS.use_gold_reward_model:
        rew_fn = lambda x: get_gold_score(completions=x, shard_size=FLAGS.gold_shard_size)
    else:
        rew_fn = get_pred_reward
    
    @empty_cache_decorator
    @torch.no_grad()
    def process_batch(batch):
        if batch is not None:
            #### Construct query tensors
            query_tensors = tokenizer(batch["query"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=128, return_tensors='pt')
            query_tensors = accelerate.utils.send_to_device(query_tensors, trainer.accelerator.device)

            #### Get generations from SFTModel (including prompt)
            if query_tensors.input_ids.shape[0] > FLAGS.max_gen_batch_size:
                generation_tokens = []
                for i in tqdm(range(0, query_tensors.input_ids.shape[0], FLAGS.max_gen_batch_size), desc=f"Generating for epoch {epoch}"):
                    generation_tokens.append(trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors[i:i+FLAGS.max_gen_batch_size], **generation_kwargs))
                    torch.cuda.empty_cache()
                generation_tokens = torch.cat(generation_tokens, dim=0)
            else:
                generation_tokens = trainer.accelerator.unwrap_model(trainer.model).generate(**query_tensors, **generation_kwargs)
            texts = tokenizer.batch_decode(generation_tokens, skip_special_tokens=True)

            #### Update batch with response
            batch["response"] = [x.split(ASSISTANT_TOKEN)[-1] for x in texts]
            response_tensors = tokenizer(batch["response"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            response_tensors = [response_tensors[i] for i in range(0, len(response_tensors))]

            #### Compute reward score
            rewards = rew_fn(texts)
            rewards = [rewards[i] for i in range(0, len(rewards))]
            ### Reprocess query tensors
            query_tensors = query_tensors.input_ids
            query_tensors = [query_tensors[i] for i in range(0, len(query_tensors))]
        else:
            query_tensors, response_tensors, rewards = [], [], []
        return batch, query_tensors, response_tensors, rewards
    
    @empty_cache_decorator
    @torch.no_grad()
    def process_pref_batch(pref_batch):
        if pref_batch is not None:
            ### Process preference dataset
            pref_query = tokenizer(pref_batch["query"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=128, return_tensors='pt').input_ids
            pref_query_tensors = accelerate.utils.send_to_device(pref_query, trainer.accelerator.device)
            
            pref_response_tensors = tokenizer(pref_batch["response"], padding='max_length' if FLAGS.use_tpu else True, truncation=True, max_length=generation_kwargs['max_new_tokens'], return_tensors='pt').input_ids
            pref_response_tensors = accelerate.utils.send_to_device(pref_response_tensors, trainer.accelerator.device)
            
            #### Compute reward score
            pref_texts = pref_batch["text"]
            pref_rewards = rew_fn(pref_texts)
            
            # now append the preference dataset to the batch
            query_tensors = [pref_query_tensors[i] for i in range(0, len(pref_query_tensors))]
            response_tensors = [pref_response_tensors[i] for i in range(0, len(pref_response_tensors))]
            rewards = [pref_rewards[i] for i in range(0, len(pref_rewards))]
        else:
            query_tensors, response_tensors, rewards = [], [], []
        return pref_batch, query_tensors, response_tensors, rewards

    print("Starting training")
    total_iterations = 0
    columns_to_log: list[str] = ["query", "response"]
    for epoch in tqdm(range(FLAGS.num_train_epochs), desc="Epochs"):
        for sub_iteration, batches in tqdm(enumerate(zipped_dataloaders), desc="Batches", total=len(zipped_dataloaders)):
            empty_cache()

            stats = {}

            if total_iterations % FLAGS.eval_every_steps == 0:
                if FLAGS.use_gold_reward_model and not FLAGS.use_tpu:
                    gold_model = trainer.accelerator.prepare_model(gold_model)
                    
                #### Calculate rewards with train and eval datasets
                for eval_name, eval_dataloader in all_eval_dataloaders.items():
                    print(f"Running evaluation on {eval_name}")
                    batch = next(iter(eval_dataloader))
                    all_rewards = []
                    all_to_log = {}
                    for i in range(FLAGS.num_eval_batches):
                        sbatch = {k: batch[k][i*FLAGS.gold_shard_size:(i+1)*FLAGS.gold_shard_size] for k in batch}
                        sbatch, query_tensors, response_tensors, eval_rewards = process_batch(sbatch)
                        all_rewards.extend([x.cpu().numpy() for x in eval_rewards])
                        for k in columns_to_log:
                            all_to_log[k] = (all_to_log.get(k, []) + [x.cpu().numpy().item() if isinstance(x, torch.Tensor) else x for x in sbatch[k]])
                        del sbatch, query_tensors, response_tensors, eval_rewards
                        empty_cache()

                    eval_rewards = np.array(all_rewards)
                    
                    # log reward stats
                    stats[f"{eval_name}/rewards"] = eval_rewards.mean()
                    stats[f"{eval_name}/rewards_mean"] = eval_rewards.mean()
                    stats[f"{eval_name}/rewards_std"] = eval_rewards.std()
                    stats[f"{eval_name}/rewards_max"] = eval_rewards.max()
                    stats[f"{eval_name}/rewards_min"] = eval_rewards.min()
                    stats[f"{eval_name}/rewards_median"] = np.median(eval_rewards)
                    
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

                    # log table of completions
                    table_rows = list(r for r in zip(*[all_to_log[col] for col in columns_to_log], eval_rewards.tolist()))
                    stats[f"{eval_name}/table"] = wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)

                if FLAGS.use_gold_reward_model and not FLAGS.use_tpu:
                    gold_model = trainer.accelerator.prepare_model(gold_model)

                empty_cache()  

            if isinstance(batches, tuple) and len(batches) == 2:
                batch, pref_batch = batches
            else:
                if batch_size_online_data == 0:
                    batch, pref_batch = None, batches
                elif batch_size_pref_data == 0:
                    batch, pref_batch = batches, None
            
            if FLAGS.use_gold_reward_model:
                gold_model = trainer.accelerator.prepare_model(gold_model)

            batch, query_tensors, response_tensors, rewards = process_batch(batch) # generate using sft model
            pref_batch, pref_query_tensors, pref_response_tensors, pref_rewards = process_pref_batch(pref_batch) # use pref data completions
            query_tensors, response_tensors, rewards = query_tensors + pref_query_tensors, response_tensors + pref_response_tensors, rewards + pref_rewards
            
            des_len = len(rewards)
            
            output_batch = {}
            for k in columns_to_log:
                output_batch[k] = (batch or {}).get(k, []) + (pref_batch or {}).get(k, [])
                assert len(output_batch[k]) == des_len, f"len({k}) = {len(output_batch[k])} != {des_len}, {output_batch[k]}"

            if FLAGS.use_gold_reward_model and not FLAGS.use_tpu:
                gold_model.to(torch.device("cpu"))

            #### Run Trainer step
            train_stats = trainer.step(queries=query_tensors, responses=response_tensors, scores=rewards)
            for key in train_stats:
                stats[key] = train_stats[key]

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
