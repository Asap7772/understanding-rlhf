print('Starting imports')

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
import os

os.environ["WANDB__SERVICE_WAIT"] = "600"
os.environ["WANDB_INIT_TIMEOUT"] = "600"

os.environ["KAGGLE_TPU"] = "yes" # adding a fake env to launch on TPUs
os.environ["TPU_NAME"] = "dummy"
os.environ["XRT_TPU_CONFIG"]="localservice;0;localhost:51011"

import re
from datasets import concatenate_datasets, load_from_disk, Dataset, DatasetDict, load_dataset
from trainers.sft_trainer import SFTTrainer
from trainers.utils import DataCollatorForCompletionOnlyLM
from absl import app, flags
print('Done with imports')

FLAGS = flags.FLAGS
flags.DEFINE_string('dataset_path', "tatsu-lab/alpaca_farm", 'the path to the dataset')
flags.DEFINE_integer('num_samples', 19000, 'the number of samples to use from the dataset')
flags.DEFINE_string('pretrained_dir', "EleutherAI/pythia-1.4b", 'the output directory')
flags.DEFINE_string('output_dir', None, 'the output directory')
flags.DEFINE_integer('batch_size', 4, 'the batch size')
flags.DEFINE_float('learning_rate', 8e-6, 'the learning rate')
flags.DEFINE_float('weight_decay', 0, 'the weight decay')
flags.DEFINE_integer('num_train_epochs', 50, 'the number of training epochs')
flags.DEFINE_integer('max_steps', -1, 'the number of training steps')
flags.DEFINE_bool('push_to_hub', False, 'Push the model to HF Hub')
flags.DEFINE_string('hub_model_id', None, 'The name of the model on HF Hub')
flags.DEFINE_integer("gradient_accumulation_steps", 1, "Gradient accumulation steps")
flags.DEFINE_bool("gradient_checkpointing", False, "Whether to use gradient checkpointing")
flags.DEFINE_bool("mixed_precision", False, "Whether to use mixed precision")
flags.DEFINE_integer("max_seq_length", 512, "The maximum sequence length")
flags.DEFINE_bool("use_tpu", False, "Whether to use TPU")
flags.DEFINE_string("sft_key", 'y_w', "The key to use for SFT")

PROMPT_TOKEN = '<|prompter|>'
ASSISTANT_TOKEN = '<|assistant|>'

TITLE_TOKEN = '<|title|>'
SUMMARY_TOKEN = '<|summary|>'

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

def main(_):
    if FLAGS.dataset_path == "tatsu-lab/alpaca_farm":
        dataset = load_dataset(FLAGS.dataset_path, split="sft")
        eval_dataset = load_dataset(FLAGS.dataset_path, split="val")
    elif FLAGS.dataset_path == "openai/summarize_from_feedback":
        dataset = load_dataset(FLAGS.dataset_path, "comparisons", split='train')
        eval_dataset = load_dataset(FLAGS.dataset_path, "comparisons", split='validation')
    else:
        dataset_name, dataset = construct_dataset(
            path=FLAGS.dataset_path,
            num_samples=FLAGS.num_samples,
            concatenate_prompt=False,
        )
        print('Loaded dataset', dataset_name)
        dataset, eval_dataset = dataset['train'], dataset['test']
    tokenizer = AutoTokenizer.from_pretrained(FLAGS.pretrained_dir, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    eos = tokenizer.eos_token

    if FLAGS.dataset_path == "tatsu-lab/alpaca_farm":
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['instruction'])):
                inst, inp, out = example['instruction'][i], example['input'][i], example['output'][i]
                if inp:
                    text = f"{PROMPT_TOKEN} {inst}\n{inp}{eos}{ASSISTANT_TOKEN}{out}{eos}"
                else:
                    text = f"{PROMPT_TOKEN}{inst}{eos}{ASSISTANT_TOKEN}{out}{eos}"
                output_texts.append(text)
            return output_texts
        instruction_template = PROMPT_TOKEN
        response_template = ASSISTANT_TOKEN
    elif FLAGS.dataset_path == "openai/summarize_from_feedback":
        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example['choice'])):
                infos, summaries = example['info'][i], example['summaries'][i]
                text, title = infos['post'], infos['title']
                summary1, summary2 = summaries[0]['text'], summaries[1]['text']
                
                text1 = f"{TITLE_TOKEN}{title}\n{text}{eos}{SUMMARY_TOKEN}{summary1}{eos}"
                text2 = f"{TITLE_TOKEN}{title}\n{text}{eos}{SUMMARY_TOKEN}{summary2}{eos}"
                
                output_texts.append(text1)
                output_texts.append(text2)
            return output_texts
        instruction_template = TITLE_TOKEN
        response_template = SUMMARY_TOKEN
    else:
        def formatting_prompts_func(example): 
            return example[FLAGS.sft_key]

    model = AutoModelForCausalLM.from_pretrained(FLAGS.pretrained_dir, trust_remote_code=True)
    
    if FLAGS.use_tpu:
        extra_kwargs = dict(
            pad_to_multiple_of=FLAGS.max_seq_length, # seems to be necessary for TPU to ensure batches are the same size 
        )
    else:
        extra_kwargs = {}
    
    collator = DataCollatorForCompletionOnlyLM(
        response_template=response_template,
        instruction_template=instruction_template,
        tokenizer=tokenizer,
        **extra_kwargs
    )

    extra_kwargs = {}
    extra_kwargs['output_dir'] = FLAGS.output_dir

    training_args = TrainingArguments(
        do_predict=True,
        learning_rate=FLAGS.learning_rate,
        weight_decay=FLAGS.weight_decay,
        push_to_hub=FLAGS.push_to_hub,
        gradient_accumulation_steps=FLAGS.gradient_accumulation_steps,
        gradient_checkpointing=FLAGS.gradient_checkpointing,
        fp16=FLAGS.mixed_precision,
        logging_first_step=True,
        optim="adafactor",
        report_to='wandb',
        hub_model_id=FLAGS.hub_model_id,
        per_device_train_batch_size=FLAGS.batch_size,
        per_device_eval_batch_size=FLAGS.batch_size,
        num_train_epochs=FLAGS.num_train_epochs,
        run_name=os.path.basename(FLAGS.output_dir),
        save_strategy='epoch',
        **extra_kwargs
    )

    trainer = SFTTrainer(
        model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        max_seq_length=FLAGS.max_seq_length,
    )

    trainer.train() 
    
if __name__ == "__main__":
    app.run(main)