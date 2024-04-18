import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import logging
from typing import Optional, Union, Dict, List, Tuple, NamedTuple, Callable, Iterable, Any, Mapping
import numpy as np
import random
import typing
from trainers.reweighted_bc_config import ReweightedBCConfig
from trainers.utils import (
    RunningMoments,
    AdaptiveKLController,
    FixedKLController,
    logprobs_from_logits,
    entropy_from_logits,
    masked_mean,
    masked_mean_sum,
    flatten_dict,
    set_seed,
    is_torch_greater_2_0,
    create_reference_model,
    empty_cache,
    empty_cache_decorator
)
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, is_deepspeed_available
import warnings
from transformers import DataCollatorForLanguageModeling
from torch.optim import Adam
import sys
import inspect
from packaging import version
import datasets
from copy import deepcopy
import tqdm

PreTrainedModelWrapper = typing.Union[nn.Module, nn.DataParallel]

class ReweightedBCTrainer():
    
    def __init__(
        self,
        config: ReweightedBCConfig = None,
        model: PreTrainedModelWrapper = None,
        ref_model: Optional[PreTrainedModelWrapper] = None,
        tokenizer: PreTrainedTokenizerBase = None,
        dataset: Optional[Union[torch.utils.data.Dataset, Dataset]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        data_collator: Optional[typing.Callable] = None,
        num_shared_layers: Optional[int] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        additional_config_kwargs: Optional[dict] = None,
    ):
        self.config = config
        set_seed(self.config.seed)
        
        # Step 1: Initialize Accelerator
        self.accelerator = Accelerator(
            log_with=config.log_with,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            project_config=ProjectConfiguration(**config.project_kwargs),
            **config.accelerator_kwargs,
        )
        
        self.model = model
        if ref_model is None:
            self.ref_model = create_reference_model(self.model, num_shared_layers=num_shared_layers)
        else:
            self.ref_model = ref_model
        self.model_params = filter(lambda p: p.requires_grad, self.model.parameters())
        self.is_encoder_decoder = hasattr(self.model, "is_encoder_decoder")
        if self.is_encoder_decoder: 
            raise ValueError("ReweightedBC does not support encoder-decoder models.")

        self.is_peft_model = getattr(self.model, "is_peft_model", False)
        config.is_encoder_decoder = self.is_encoder_decoder
        config.is_peft_model = self.is_peft_model

        is_using_tensorboard = config.log_with is not None and config.log_with == "tensorboard"
        current_config = dict(trl_rbc_trainer_config=config.to_dict()) if not is_using_tensorboard else config.to_dict()
        current_config.update(flatten_dict(additional_config_kwargs or {}))
        
        self.accelerator.init_trackers(
            config.tracker_project_name,
            config=current_config,
            init_kwargs=config.tracker_kwargs,
        )
        self.is_using_text_environment = getattr(config, "use_text_environment", False)
        self.tokenizer = tokenizer
        
        self.dataset = dataset
        self._signature_columns = None
        if self.dataset is not None:
            self.dataloader = self.prepare_dataloader(self.dataset, data_collator)
        elif self.dataset is None and self.accelerator.num_processes > 1:
            warnings.warn(
                "No dataset is provided. In a multi-GPU setting, this will lead to an error. You should"
                " prepare your dataloader yourself with `dataloader = ppo_trainer.accelerator.prepare(dataloader)`"
                " and using `torch.utils.data.DataLoader`, or pass a dataset to the `PPOTrainer`. Please "
                " refer to the documentation for more details.",
                UserWarning,
            )
            self.dataloader = None
        else:
            self.dataloader = None

        # Step 3: Initialize optimizer and data collator
        self.data_collator = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)
        if optimizer is None:
            self.optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=self.config.learning_rate,
            )
        else:
            self.optimizer = optimizer

        self.lr_scheduler = lr_scheduler
        if self.lr_scheduler is not None:
            lr_scheduler_class = (
                torch.optim.lr_scheduler._LRScheduler
                if not is_torch_greater_2_0()
                else torch.optim.lr_scheduler.LRScheduler
            )

            if not isinstance(self.lr_scheduler, lr_scheduler_class):
                raise ValueError(
                    "lr_scheduler must be a torch.optim.lr_scheduler._LRScheduler or torch.optim.lr_scheduler.LRScheduler (for torch >= 2.0)"
                )

        if self.config.adap_kl_ctrl:
            self.kl_ctl = AdaptiveKLController(self.config.init_kl_coef, self.config.target, self.config.horizon)
        else:
            self.kl_ctl = FixedKLController(self.config.init_kl_coef)

        # Safety checkers for DS integration
        is_deepspeed_used = self.accelerator.distributed_type == "DEEPSPEED" and hasattr(
            self.accelerator.state, "deepspeed_plugin"
        )

        (
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        ) = self.accelerator.prepare(
            self.model,
            self.optimizer,
            self.data_collator,
            self.dataloader,
            self.lr_scheduler,
        )
        if is_deepspeed_used:
            # Quantized models are already set on the correct device
            if not self.is_peft_model and not (
                getattr(self.ref_model.pretrained_model, "is_loaded_in_8bit", False)
                or getattr(self.ref_model.pretrained_model, "is_loaded_in_4bit", False)
            ):
                self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model = self.accelerator.prepare(self.ref_model)

        # In a distributed setup, only logging needs to be performed on the main process
        # check: https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        # or: https://discuss.pytorch.org/t/use-distributed-data-parallel-correctly/82500/11
        self.is_distributed = self.accelerator.distributed_type == "MULTI_GPU"

        # init the current step
        self.current_step = 0

        # init variables for pushing model to hub
        if config.push_to_hub_if_best_kwargs:
            if "repo_id" not in config.push_to_hub_if_best_kwargs:
                raise ValueError("You have to specify repo_id in order to push the model to the hub!")
            self.push_to_hub_kwargs = config.push_to_hub_if_best_kwargs
            self.compare_step = 0
            self.highest_reward = torch.tensor(-float("inf"))

        # post process for PP
        self.current_device = self.accelerator.device
        self.running = RunningMoments(self.accelerator)
        
    def prepare_dataloader(self, dataset: Union[torch.utils.data.Dataset, Dataset], data_collator=None):
        """
        Prepare the dataloader for training.

        Args:
            dataset (Union[`torch.utils.data.Dataset`, `datasets.Dataset`]):
                PyTorch dataset or Hugging Face dataset. If a Hugging Face dataset is passed, the dataset
                will be preprocessed by removing the columns that are not used by the model.
            data_collator (Optional[function]):
                Data collator function.

        Returns:
            `torch.utils.data.DataLoader`: PyTorch dataloader
        """
        if isinstance(dataset, Dataset):
            dataset = self._remove_unused_columns(dataset)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.config.dataloader_batch_size or self.config.batch_size,
            collate_fn=data_collator,
            shuffle=True,
            drop_last=True,
        )
        return dataloader
    
        # Adapted from transformers.Trainer._set_signature_columns_if_needed
    def _set_signature_columns_if_needed(self):
        if self._signature_columns is None:
            # Inspect model forward signature to keep only the arguments it accepts.
            signature = inspect.signature(self.model.forward)
            self._signature_columns = list(signature.parameters.keys())
            # label => sentiment | we need query and response for logging purpose
            self._signature_columns += ["label", "query", "response"]

    # Adapted from transformers.Trainer._remove_unused_columns
    def _remove_unused_columns(self, dataset: "Dataset"):
        if not self.config.remove_unused_columns:
            return dataset
        self._set_signature_columns_if_needed()
        signature_columns = self._signature_columns

        ignored_columns = list(set(dataset.column_names) - set(signature_columns))

        columns = [k for k in signature_columns if k in dataset.column_names]

        if version.parse(datasets.__version__) < version.parse("1.4.0"):
            dataset.set_format(
                type=dataset.format["type"],
                columns=columns,
                format_kwargs=dataset.format["format_kwargs"],
            )
            return dataset
        else:
            return dataset.remove_columns(ignored_columns)

    @empty_cache_decorator
    def _step(
        self,
        queries: torch.LongTensor,
        queries_attention_mask: torch.LongTensor,
        responses: torch.LongTensor,
        responses_attention_mask: torch.LongTensor,
        scores: torch.FloatTensor,
        return_stats: bool = False,
    ):  
        input_ids = torch.cat((queries, responses), dim=1)
        attention_mask = torch.cat((queries_attention_mask, responses_attention_mask), dim=1)
        # mask out query tokens, keep response tokens. Remove last token from response tokens.
        mask = torch.cat((torch.zeros_like(queries), torch.ones_like(responses)), dim=1)[:,:-1]
        input_data = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits, _, _ = self.model(**input_data)
        with torch.no_grad():
            old_logits, _, _ = self.ref_model(**input_data)
            old_logprobs = logprobs_from_logits(old_logits[:, :-1, :], input_ids[:, 1:])

        logprobs = logprobs_from_logits(logits[:, :-1, :], input_ids[:, 1:])
        entropy = entropy_from_logits(logits)
        
        if self.config.use_score_scaling:
            # Score scaling
            scores_mean, scores_std = self.running.update(scores)
            tensor_to_kwargs = dict(dtype=scores.dtype, device=scores.device)
            score_scaling_factor = scores_std.to(**tensor_to_kwargs) + torch.finfo(scores.dtype).eps
            if self.config.use_score_norm:
                scores = (scores - scores_mean.to(**tensor_to_kwargs)) / score_scaling_factor
            else:
                scores /= {score_scaling_factor}
        
        if self.config.score_clip is not None:
            # Score clipping
            scores_dtype = scores.dtype
            scores = torch.clip(scores.float(), -self.config.score_clip, self.config.score_clip).to(dtype=scores_dtype)
        
        batch_mask=None
        if self.config.filter_or_reweight == "filter":
            # filter out bad responses
            if self.config.filter_type == "topk":
                # filter out the top k responses
                topk = self.config.filter_topk
                topk_scores, _ = torch.topk(scores, topk, largest=True, sorted=True)
                baseline_score = topk_scores[-1]
                score_mask = scores >= baseline_score
                score_mask_repeat = score_mask.unsqueeze(-1).repeat(1, mask.shape[-1])
                assert score_mask_repeat.shape == mask.shape
                updated_mask = mask * score_mask_repeat
            elif self.config.filter_type == "threshold":
                # filter out responses with scores below a threshold
                threshold = self.config.filter_threshold
                score_mask = scores >= threshold
                score_mask_repeat = score_mask.unsqueeze(-1).repeat(1, mask.shape[-1])
                assert score_mask_repeat.shape == mask.shape
                updated_mask = mask * score_mask_repeat
            else:
                raise ValueError(f"Filter type {self.config.filter_type} is not supported.")
            batch_mask = score_mask
        elif self.config.filter_or_reweight == "reweight":
            # reweight responses
            if self.config.weighting_type == "exp":
                # reweight responses using softmax
                reweight_scores = torch.exp(scores / self.config.temperature)
                if self.config.clip_weighting:
                    pre_clip_scores, reweight_scores = reweight_scores, torch.clamp(input=reweight_scores, min=self.config.clip_weighting_value_min, max=self.config.clip_weighting_value_max)
                    per_clamped = (reweight_scores != pre_clip_scores).float().mean()
                reweight_scores_repeat = reweight_scores.unsqueeze(-1).repeat(1, mask.shape[-1])
                assert reweight_scores_repeat.shape == mask.shape
                updated_mask = mask * reweight_scores_repeat
            elif self.config.weighting_type == "sigmoid":
                # reweight responses using sigmoid
                reweight_scores = torch.sigmoid(scores / self.config.temperature)
                reweight_scores_repeat = reweight_scores.unsqueeze(-1).repeat(1, mask.shape[-1])
                assert reweight_scores_repeat.shape == mask.shape
                updated_mask = mask * reweight_scores_repeat
            else:
                raise ValueError(f"Reweight type {self.config.reweight_type} is not supported.")
            batch_mask = reweight_scores
        else:
            raise ValueError(f"Filter or reweight type {self.config.filter_or_reweight} is not supported.")

        unweighted_nll_loss = masked_mean(-logprobs, mask)
        if self.config.filter_or_reweight == "reweight":
            # normalize over all masked tokens
            nll_loss = masked_mean(-logprobs, updated_mask, norm_mask=mask)
        else:
            # normalize over all masked tokens + filtered out bad responses
            nll_loss = masked_mean(-logprobs, updated_mask)

        approxkl = 0.5 * masked_mean((logprobs - old_logprobs) ** 2, mask)
        policykl = masked_mean(logprobs - old_logprobs, mask)
        
        sequence_approxkl = 0.5 * masked_mean_sum((logprobs - old_logprobs) ** 2, mask)
        sequence_policykl = masked_mean_sum(logprobs - old_logprobs, mask)
        
        if return_stats:
            stats = dict(
                loss=dict(
                    unweighted_nll_loss=unweighted_nll_loss.detach(),
                    nll_loss=nll_loss.detach()
                ),
                reweighting=dict(
                    weight_mask=batch_mask.detach(),
                    weight_mask_mean=batch_mask.mean().detach(),
                    weight_mask_std=batch_mask.std().detach(),
                    weight_mask_min=batch_mask.min().detach(),
                    weight_mask_max=batch_mask.max().detach(),
                    masked_weight_mask=updated_mask.detach(),
                    masked_weight_mask_mean=updated_mask.mean().detach(),
                    masked_weight_mask_std=updated_mask.std().detach(),
                    masked_weight_mask_min=updated_mask.min().detach(),
                    masked_weight_mask_max=updated_mask.max().detach(),
                    scores=scores.detach(),
                    scores_mean=scores.mean().detach(),
                    scores_std=scores.std().detach(),
                    scores_min=scores.min().detach(),
                    scores_max=scores.max().detach(),
                    reweight_scores_mean=reweight_scores.mean().detach(),
                    reweight_scores_std=reweight_scores.std().detach(),
                    reweight_scores_min=reweight_scores.min().detach(),
                    reweight_scores_max=reweight_scores.max().detach(),
                ),
                policy=dict(
                    entropy=entropy.detach(),
                    approxkl=approxkl.detach(),
                    policykl=policykl.detach(),
                    sequence_approxkl=sequence_approxkl.detach(),
                    sequence_policykl=sequence_policykl.detach(),
                    logprob_mean=logprobs.detach().mean(),
                    masked_logprob_mean=masked_mean(logprobs, mask).detach(),
                )
            )
            
            if self.config.clip_weighting:
                stats['clipping'] = dict(
                    per_clip=per_clamped.detach(),
                    pre_clip_mean=pre_clip_scores.mean().detach(),
                    pre_clip_std=pre_clip_scores.std().detach(),
                    pre_clip_min=pre_clip_scores.min().detach(),
                    pre_clip_max=pre_clip_scores.max().detach(),
                    post_clip_mean=reweight_scores.mean().detach(),
                    post_clip_std=reweight_scores.std().detach(),
                    post_clip_min=reweight_scores.min().detach(),
                    post_clip_max=reweight_scores.max().detach(),
                )
            return nll_loss, flatten_dict(stats)
        else:
            return nll_loss
    
    def step(
        self,
        queries: torch.LongTensor,
        queries_attention_mask: torch.LongTensor,
        responses: torch.LongTensor,
        responses_attention_mask: torch.LongTensor,
        scores: torch.FloatTensor,
    ):
        assert queries.ndim == 2 and responses.ndim == 2 and scores.ndim == 1
        self.model.train()
        bs = self.config.batch_size
        sub_bs = self.config.mini_batch_size
        assert bs % sub_bs == 0
        num_sub_batches = bs // sub_bs
        
        first = True
        for _ in tqdm.tqdm(range(self.config.ppo_epochs), desc="Inner Iteration Steps", leave=False):
            shuffled_indices = torch.randperm(bs)
            queries = queries[shuffled_indices]
            queries_attention_mask = queries_attention_mask[shuffled_indices]
            responses = responses[shuffled_indices]
            responses_attention_mask = responses_attention_mask[shuffled_indices]
            scores = scores[shuffled_indices]

            for i in tqdm.tqdm(range(0, bs, sub_bs), desc="Training with Minibatches", leave=False):
                queries_ = queries[i : i + sub_bs]
                queries_attention_mask_ = queries_attention_mask[i : i + sub_bs]
                responses_ = responses[i : i + sub_bs]
                responses_attention_mask_ = responses_attention_mask[i : i + sub_bs]
                scores_ = scores[i : i + sub_bs]
                if first:
                    loss, stats = self._step(
                        queries=queries_,
                        queries_attention_mask=queries_attention_mask_,
                        responses=responses_,
                        responses_attention_mask=responses_attention_mask_,
                        scores=scores_,
                        return_stats=True
                    )
                    first = False
                else:
                    loss = self._step(
                        queries=queries_,
                        queries_attention_mask=queries_attention_mask_,
                        responses=responses_,
                        responses_attention_mask=responses_attention_mask_,
                        scores=scores_,
                        return_stats=False
                    )
                
                self.optimizer.zero_grad()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.current_step += 1
        return stats

    def log_stats(
            self,
            stats: dict,
            batch: dict,
            rewards: List[torch.FloatTensor],
            columns_to_log: List[str] = ["query", "response"],
        ):
            """
            A function that logs all the training stats. Call it at the end of each epoch.

            Args:
                stats (dict[str, Any]):
                    A dictionary of training stats.
                batch (dict[str, Any]):
                    A dictionary of batch data, this contains the queries and responses.
                rewards (`List[torch.FloatTensor]`):
                    A tensor of rewards.
            """
            # Log only if we are in the main process
            if self.accelerator.is_main_process:
                logs = {}

                # Log stats
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards).to(self.current_device)

                if self.config.log_with == "wandb":
                    import wandb

                    if any([column_to_log not in batch.keys() for column_to_log in columns_to_log]):
                        raise ValueError(f"Columns to log {columns_to_log} are not present in the batch {batch.keys()}.")

                    batch_list = [batch[column_to_log] for column_to_log in columns_to_log]

                    table_rows = [list(r) for r in zip(*batch_list, rewards.cpu().tolist())]
                    logs.update({"game_log": wandb.Table(columns=[*columns_to_log, "reward"], rows=table_rows)})

                logs.update(stats)

                # manually cast in fp32 for bf16 torch tensors
                for k, v in logs.items():
                    if isinstance(v, torch.Tensor) and v.dtype == torch.bfloat16:
                        logs[k] = v.float()
                        
                def to_numpy(x):
                    assert isinstance(x, torch.Tensor)
                    return x.cpu().to(torch.float32).numpy()
                logs["dataset/reward_mean"] = to_numpy(rewards.mean()).item()
                logs["dataset/reward_std"] = to_numpy(rewards.std()).item()
                logs["dataset/reward_dist"] = to_numpy(rewards)

                self.accelerator.log(
                    logs,
                    step=self.current_step if self.config.log_with == "tensorboard" else None,
                )