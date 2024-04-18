"""
Preference Reward Trainer for Transformers.

Based off of https://github.com/CarperAI/autocrit/blob/main/train_reward_model.py
"""
import os

os.environ["WANDB__SERVICE_WAIT"] = "10000"
os.environ["WANDB_INIT_TIMEOUT"] = "10000"

import torch
assert torch.cuda.is_available()
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from absl import flags, app
import accelerate
from accelerate import Accelerator
import datetime
from datasets import concatenate_datasets, load_from_disk, Dataset, DatasetDict, load_dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
)
import numpy as np
import os
import pandas as pd
import pprint
import pprint
import re
from typing import Union
import torch.nn.functional as F
import transformers
import wandb
import tempfile

from transformers.utils import logging
logging.set_verbosity(40)

# for debugging
import ipdb
pdb = ipdb.set_trace


def get_dataset(path, num_samples=-1, return_test_data=True, num_samples_test=1000):
    assert os.path.exists(path), f"Path {path} does not exist"
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


FLAGS = flags.FLAGS
flags.DEFINE_integer("num_samples", -1, "Number of samples")
flags.DEFINE_integer("batch_size", 32, "Batch size")
flags.DEFINE_bool("batched", True, "Whether to use batched processing")
flags.DEFINE_integer("block_size", 128, "Block size")
flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory")
flags.DEFINE_bool("concatenate_prompt", False, "Whether to concatenate prompt")
flags.DEFINE_string("dataset_path","", "Dataset path",)
flags.DEFINE_integer("eval_interval", 100, "Eval interval")
flags.DEFINE_integer("epochs", 1000, "Number of epochs")
flags.DEFINE_bool("downscale_weight", False, "Whether to downscale weight")
flags.DEFINE_bool("gradient_checkpointing", False, "Whether to use gradient checkpointing")
flags.DEFINE_bool("load_in_4bit", False, "Whether to load in 4bit")
flags.DEFINE_float("lr", 6e-4, "Learning rate")
flags.DEFINE_float("min_lr", None, "Minimum learning rate")
flags.DEFINE_bool("mixed_precision", True, "Whether to use mixed precision")
flags.DEFINE_string("model_name", "distilgpt2", "Model name")
flags.DEFINE_integer("num_proc", 32, "Number of processes to use")
flags.DEFINE_integer("num_unfrozen_layers", None, "Number of unfrozen layers")
flags.DEFINE_bool("only_eval", False, "Whether to only eval")
flags.DEFINE_string("output_dir", "model_checkpoints/sft-review-model", "Output directory")
flags.DEFINE_integer("per_device_eval_batch_size", 4, "Per device eval batch size")
flags.DEFINE_integer("per_device_train_batch_size", 4, "Per device train batch size")
flags.DEFINE_bool("push_to_hub", True, "Whether to push to hub")
flags.DEFINE_bool("subset_eval", True, "Whether to subset eval")
flags.DEFINE_integer("num_subsets", 32, "Number of subsets")
flags.DEFINE_integer("seq_length", 512, "Sequence length")
flags.DEFINE_string("wandb_project", "preferences-reward-learning-10-09", "Output directory")
flags.DEFINE_float("weight_decay", 0.1, "Weight decay")
flags.DEFINE_string("description", "", "Description")
flags.DEFINE_integer("device", 0, "Device")
flags.DEFINE_integer("seed", 0, "Seed")
flags.DEFINE_float("label_noise", 0.0, "Noise of labels")
flags.DEFINE_float("label_smoothing_weight", 0.0, "Label smoothing")
flags.DEFINE_integer("gradient_accumulation_steps", 1, "Gradient accumulation steps")


def main(_):
    """
    Seed Setting
    """
    # seed = int(os.environ.get("RANK", 0))
    seed = FLAGS.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    flags_dict = FLAGS.flag_values_dict()
    print("FLAGS:", flags_dict)
    preference_dataset_basename = os.path.basename(FLAGS.dataset_path)
    flags_dict['preference_dataset'] = preference_dataset_basename

    unique_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S.%f") + '-' + str(np.random.randint(100000))
    print("Unique Str:", unique_str)
    
    """
    Dataloader Setup
    """
    if FLAGS.dataset_path.startswith('Asap7772'):
        dataset_name = os.path.basename(FLAGS.dataset_path)
        dataset = load_dataset(FLAGS.dataset_path)
    else:
        dataset_name, dataset = construct_dataset(
            path=FLAGS.dataset_path,
            num_samples=FLAGS.num_samples,
            concatenate_prompt=FLAGS.concatenate_prompt,
        )
    print(f"Loaded: {dataset_name}")

    def process_dataset(batch):
        if FLAGS.label_noise > 0.0:
            is_noise = np.random.rand(len(batch["y_ref"])) < FLAGS.label_noise
            selected = np.where(is_noise, batch["y_l"], batch["y_w"])
            rejected = np.where(is_noise, batch["y_w"], batch["y_l"])
            return dict(
                selected=selected, rejected=rejected, reference=batch["y_ref"]
            )
        else:
            return dict(
                selected=batch["y_w"], rejected=batch["y_l"], reference=batch["y_ref"]
            )

    remove_columns = ["y_ref", "y_w", "y_l", "y_w_score", "y_l_score", "score_diff"]

    dataset = dataset.map(
        process_dataset,
        batched=FLAGS.batched,
        num_proc=FLAGS.num_proc,
        remove_columns=remove_columns,
    )
    
    print('Filtering empty before', len(dataset['train']))
    def filter_empty(batch):
        keep = []
        for i in range(len(batch["selected"])):
            if len(batch["selected"][i]) > 0 and len(batch["rejected"][i]) > 0:
                keep.append(i)
        return keep
    dataset = dataset.filter(filter_empty, batched=True)
    print('Filtering empty after', len(dataset['train']))

    print("Processed dataset")
    print("Keys:", dataset["train"][0].keys())
    print("Sample:", pprint.pprint(dataset["train"][0]))

    tokenizer = AutoTokenizer.from_pretrained(FLAGS.model_name)
    tokenizer.add_special_tokens({"pad_token": "<|padding|>"})
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    def tokenize(prompt, selected, rejected, tokenizer, policy_tokenizer = None):
        # Note: tokenizes prompt + selected/rejected and adds eos token. Can be truncated to seq_length.
        full_tok_dict = {
            "selected_input_ids": tokenizer(selected, truncation=True,max_length=FLAGS.seq_length).input_ids,
            "rejected_input_ids": tokenizer(rejected, truncation=True, max_length=FLAGS.seq_length).input_ids,
        }

        return full_tok_dict  

    def collate_fn(batch):
        # Note aligns input_ids along the batch dimension
        input_ids = sum([[x["rejected_input_ids"], x["selected_input_ids"]] for x in batch], [])
        batch_collate = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        
        return batch_collate

    eval_dataloaders = []
    tokenized = dataset.map(
        tokenize,
        input_columns=["prompt", "selected", "rejected"],
        fn_kwargs=dict(tokenizer=tokenizer, policy_tokenizer=None),
        desc="Tokenizing",
    )
    
    dataloader = torch.utils.data.DataLoader(
        tokenized["train"],
        shuffle=True,
        batch_size=FLAGS.batch_size,
        collate_fn=collate_fn,
    )
    eval_dataloaders.append(
        torch.utils.data.DataLoader(
            tokenized["test"],
            shuffle=False,
            batch_size=FLAGS.batch_size,
            collate_fn=collate_fn,
        )
    )

    """
    Accelerator Setup
    """
    accelerator_args = {"log_with": "wandb", "gradient_accumulation_steps": FLAGS.gradient_accumulation_steps}
    wandb_output_dir = tempfile.mkdtemp()
    accelerator = Accelerator(**accelerator_args)
    accelerator.init_trackers(
        project_name=FLAGS.wandb_project,
        config=flags_dict,
        init_kwargs={
            "wandb": {
                "name": f"{FLAGS.model_name}@{dataset_name}", 
                "id": unique_str,
                "dir": wandb_output_dir,
            }
        },
    )
    accelerator.print(dataset_name, dataset)

    """
    Setup Model Training
    """
    if transformers.__version__ >= "4.30.0":
        kwargs = {"load_in_4bit": FLAGS.load_in_4bit}
    else:
        kwargs = {}

    def create_model():
        model = AutoModelForSequenceClassification.from_pretrained(
            FLAGS.model_name, num_labels=1, **kwargs
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model.resize_token_embeddings(len(tokenizer))

        if FLAGS.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        if FLAGS.downscale_weight:
            model.score.weight.data *= 0.1

        if FLAGS.num_unfrozen_layers is not None and FLAGS.num_unfrozen_layers > 0:
            frozen = False
            try:
                for layer in model.transformer.h[: -FLAGS.num_unfrozen_layers]:
                    layer.requires_grad_(False)
                frozen = True
            except AttributeError:
                pass

            try:
                for layer in model.model.layers[: -FLAGS.num_unfrozen_layers]:
                    layer.requires_grad_(False)
                frozen = True
            except AttributeError:
                pass

            if not frozen:
                raise ValueError(
                    "Could not freeze layers, modify the code to support your architecture."
                )

        opt = torch.optim.AdamW(
            model.parameters(),
            lr=FLAGS.lr,
            betas=(0.9, 0.95),
            eps=1e-08,
            weight_decay=FLAGS.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            opt, T_max=len(dataloader) * FLAGS.epochs, eta_min=FLAGS.min_lr or FLAGS.lr
        )
        return model, opt, scheduler
    
    model, opt, scheduler = create_model()
    model, opt, scheduler, dataloader,*eval_dataloaders = accelerator.prepare(model, opt, scheduler, dataloader, *eval_dataloaders)

    def label_smooth_loss(x, epsilon, reduce='mean'):
        loss = -F.logsigmoid(x) * (1 - epsilon) - F.logsigmoid(-x) * epsilon

        if reduce == 'mean':
            return loss.mean()
        elif reduce == 'sum':
            return loss.sum()
        elif reduce == 'none' or reduce is None:
            return loss
        else:
            raise ValueError(f"Invalid reduce value: {reduce}")

    """
    Process Batch
    """
    def process_batch(batch):
        which_keys_to_keep = ["input_ids", "attention_mask"]
        filtered_batch = {k: v for k, v in batch.items() if k in which_keys_to_keep}
        
        scores = model(**filtered_batch, use_cache=not FLAGS.gradient_checkpointing)[0]

        scores_l = scores.reshape(-1, 2)[:, 0]
        scores_w = scores.reshape(-1, 2)[:, 1]
        scores_dataset = scores.reshape(-1, 2)
        
        loss_pref = label_smooth_loss(scores.reshape(-1, 2).diff(), FLAGS.label_smoothing_weight, reduce=None)
        loss = loss_pref.mean()
        with torch.no_grad():
            loss_pref_unsmooth = label_smooth_loss(scores.reshape(-1, 2).diff(), 0.0, reduce=None)
            loss_unsmooth = loss_pref_unsmooth.mean()
        
        return {
            "loss": loss,
            "loss_unsmooth": loss_unsmooth,
            "scores": scores,
            "scores_l": scores_l,
            "scores_w": scores_w,
            "scores_dataset": scores_dataset,
            "loss_pref": loss_pref,
            "loss_pref_unsmooth": loss_pref_unsmooth,
        }
    
    """
    Model Training Loop
    """
    best_accuracy = 0
    step = 0
    tbar = tqdm(
        range(FLAGS.epochs * len(dataloader)),
        disable=not accelerator.is_main_process or FLAGS.only_eval,
    )

    for epoch_num in range(FLAGS.epochs):
        """
        Epoch Checkpointing (Beginning of Epoch)
        """
        path = (
            f"{FLAGS.model_name}_data{dataset_name}_samp{FLAGS.num_samples}_noise{FLAGS.label_noise}_smooth{FLAGS.label_smoothing_weight}_lr{FLAGS.lr}_wd{FLAGS.weight_decay}_yplus{FLAGS.upweight_yplus_only}".replace("/", "_")
            .replace(":", "_")
            .replace("@", "_")
            + f"/{unique_str}"
        )

        # add checkpointing per epoch
        if accelerator.is_main_process:
            checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, FLAGS.wandb_project, path, f"epoch_{epoch_num}")
            accelerator.unwrap_model(model).save_pretrained(
                checkpoint_dir,
                save_function=accelerator.save,
                is_main_process=accelerator.is_main_process,
                state_dict=accelerator.get_state_dict(model),
            )
            if accelerator.is_main_process:
                tokenizer.save_pretrained(checkpoint_dir)
            accelerator.print(f"Checkpointing Epoch {epoch_num} -> {checkpoint_dir}")
        
        i, total_len = 0, len(dataloader)
        for batch in dataloader:
            epoch_num_float = epoch_num + i / total_len
            i += 1
            if step % FLAGS.eval_interval == 0 or step == tbar.total - 1:
                dict_dataloader = {
                    "train_as_eval": dataloader,
                    "test": eval_dataloaders[0],
                }
                ###
                # Evaluation
                ###
                for eval_dataset_name, eval_dataloader in dict_dataloader.items():
                    model.eval()
                    all_scores, all_scores_completions, all_delta_scores, all_tokens, all_prompt_tokens, all_completion_tokens = [], [], [], [], [], []

                    num_eval_so_far = 0
                    for eval_batch in tqdm(eval_dataloader, desc=f"Evaluating on {eval_dataset_name}", disable=not accelerator.is_main_process, leave=FLAGS.only_eval):
                        with torch.no_grad():
                            values = process_batch(eval_batch)
                        loss, scores, scores_l, scores_w, loss_pref = (
                            values["loss"],
                            values["scores"],
                            values["scores_l"],
                            values["scores_w"],
                            values["loss_pref"],
                        )  

                        delta_scores = scores.reshape(-1, 2).diff().view(-1)
                        delta_scores = accelerator.gather_for_metrics(delta_scores)
                        all_delta_scores.extend(delta_scores.tolist())
                        
                        all_scores.extend(scores.view(-1).tolist())
                        all_tokens.extend(eval_batch["input_ids"].tolist())
                        
                        if FLAGS.subset_eval and num_eval_so_far > FLAGS.num_subsets:
                            break
                        num_eval_so_far += 1

                    delta_scores = np.hstack(all_delta_scores)
                    accuracy = (delta_scores > 0).mean()

                    if accelerator.is_main_process:
                        accelerator.log({'epoch': epoch_num_float}, step=step)
                        texts = [text.replace(tokenizer.pad_token, "") for text in tokenizer.batch_decode(all_tokens)]
                        samples = wandb.Table(["text", "score"], rows=list(zip(texts, all_scores))[:128])
                        
                        postfix = f"_{eval_dataset_name}"
                        log_metrics = {
                            f"accuracy{postfix}": accuracy,
                            f"samples{postfix}": samples,
                            f"loss{postfix}": loss,
                            f"delta_scores{postfix}": delta_scores,
                        }
                        # Deal with bf16 logging
                        for k,v in log_metrics.items():
                            if isinstance(v, torch.Tensor):
                                log_metrics[k] = v.cpu().detach().to(torch.float32).numpy()

                        accelerator.log(log_metrics, step=step)

                    if (
                        accuracy > best_accuracy
                        and eval_dataset_name == "train_as_eval"
                    ):
                        best_accuracy = accuracy
                        accelerator.log({"best_accuracy": best_accuracy}, step=step)

                    if eval_dataset_name == "train_as_eval":
                        tbar.set_postfix(accuracy=accuracy, best_accuracy=best_accuracy)

                accelerator.wait_for_everyone()
                model.train()

            ###
            # Training
            ###
            model.train()
            with accelerator.accumulate(model):
                values = process_batch(batch)
                
                loss, scores, scores_l, scores_w, loss_pref = (
                    values["loss"],
                    values["scores"],
                    values["scores_l"],
                    values["scores_w"],
                    values["loss_pref"],
                )

                accelerator.backward(loss)
                opt.step()
                opt.zero_grad()
                scheduler.step()

            tbar.update()
            description = f"Training {FLAGS.model_name} on {dataset_name}; loss: {loss.item():.4f}"
            
            tbar.set_description(description)
            thing_to_log = {
                "loss": loss.item(), 
                "lr": float(scheduler.get_last_lr()[0]), 
                "score_mean": scores.mean().item(), 
                "score_std": scores.std().item(), 
                "score_min": scores.min().item(), 
                "score_max": scores.max().item(), 
                "scores_l_mean": scores_l.mean().item(),
                "scores_l_std": scores_l.std().item(),
                "scores_l_min": scores_l.min().item(),
                "scores_l_max": scores_l.max().item(),
                "scores_w_mean": scores_w.mean().item(),
                "scores_w_std": scores_w.std().item(),
                "scores_w_min": scores_w.min().item(),
                "scores_w_max": scores_w.max().item(),
                "score_diff_mean": (scores_w - scores_l).mean().item(),
                "score_diff_std": (scores_w - scores_l).std().item(),
                "score_diff_min": (scores_w - scores_l).min().item(),
                "score_diff_max": (scores_w - scores_l).max().item(),
            }
            accelerator.log(thing_to_log, step=step)
            step += 1


if __name__ == "__main__":
    app.run(main)
