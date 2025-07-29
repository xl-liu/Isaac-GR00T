# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import Optional, Dict, Union, Any, Tuple, List
import logging 
import sys

import torch
import transformers
from torch.utils.data import Dataset, Sampler
from transformers.trainer import (
    ALL_LAYERNORM_LAYERS,
    TRAINER_STATE_NAME,
    TrainerState,
    get_last_checkpoint,
    get_parameter_names,
    is_sagemaker_mp_enabled,
)


class BaseSampler(Sampler):
    """Sampler for dataset, which enables `set_epoch` for Dataset.
    `set_epoch` will be called by huggingface Trainer at the end of each epoch.
    `shuffle` is also supported for training set shuffling
    """

    def __init__(self, data_source: Dataset, shuffle: bool = False, seed: int = 0):
        self.data_source = data_source
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            # must not add rank here, or randomization will be different for each rank
            return iter(torch.randperm(len(self.data_source), generator=g).tolist())
        return iter(range(len(self.data_source)))

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.data_source, "set_epoch"):
            # this is important for dataset
            self.data_source.set_epoch(epoch)

    def __len__(self):
        return len(self.data_source)


class DualBrainTrainer(transformers.Trainer):
    def __init__(self, **kwargs):
        self.compute_dtype = kwargs.pop("compute_dtype")
        super().__init__(**kwargs)

    def _get_train_sampler(self):
        return BaseSampler(self.train_dataset, shuffle=True, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset):
        return BaseSampler(eval_dataset, shuffle=False)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]

            optimizer_cls, optimizer_kwargs = transformers.Trainer.get_optimizer_cls_and_kwargs(
                self.args
            )
            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            state_dict = self.model.state_dict()

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    def train(
        self,
        resume_from_checkpoint=None,
        trial=None,
        ignore_keys_for_eval=None,
        **kwargs,
    ):
        """Correctly set self.state from checkpoint so get_train_dataloader can read from it."""
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({self.args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            self.state = TrainerState.load_from_json(
                os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
            )
        return super().train(resume_from_checkpoint, trial, ignore_keys_for_eval, **kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[list] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform a prediction step.
        
        Args:
            model: The model to evaluate
            inputs: The inputs for the model
            prediction_loss_only: Whether to return only the loss
            ignore_keys: Keys to ignore when computing the loss
            
        Returns:
            Tuple of (loss, logits, labels)
        """
        has_labels = any(
            inputs.get(k) is not None
            for k in ["labels", "action", "action_mask"]
        )
        
        # Move inputs to device
        inputs = self._prepare_input(inputs)
        
        if ignore_keys is None:
            ignore_keys = []

        with torch.no_grad():
            if has_labels:
                # During evaluation, we want to compute loss
                # Call model with inputs dict (not **inputs)
                outputs = model(inputs)
                
                if isinstance(outputs, dict):
                    loss = outputs.get("loss")
                    # For GR00T, action predictions are in "action_pred" key
                    logits = outputs.get("action_pred")
                else:
                    # Fallback if outputs is not a dict
                    loss = None
                    logits = outputs
            else:
                # During inference, we might not have labels
                # Use get_action method if available (for inference)
                if hasattr(model, 'get_action'):
                    outputs = model.get_action(inputs)
                    loss = None
                    logits = outputs.get("action_pred") if isinstance(outputs, dict) else outputs
                else:
                    # Fallback to regular forward
                    outputs = model(inputs)
                    loss = outputs.get("loss") if isinstance(outputs, dict) else None
                    logits = outputs.get("action_pred") if isinstance(outputs, dict) else outputs
                    
                    # print out the loss
                    logging(f"eval batch loss: {loss.item():.6f}")

        if prediction_loss_only:
            return (loss, None, None)

        # Extract labels - could be "action" or "labels"
        labels = inputs.get("labels")
        if labels is None:
            labels = inputs.get("action")

        # Remove ignored keys from outputs if needed
        if logits is not None and isinstance(logits, dict):
            logits = {k: v for k, v in logits.items() if k not in ignore_keys}

        return (loss, logits, labels)
    
    def evaluation_loop(
        self,
        dataloader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """Override evaluation_loop to add custom logging."""
        
        print(f"=== Starting {description} ===", flush=True)
        
        # Call the parent evaluation_loop
        result = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Extract and print the final eval loss
        eval_loss = result.metrics.get(f"{metric_key_prefix}_loss")
        if eval_loss is not None:
            current_step = getattr(self.state, 'global_step', 'unknown')
            print(f"=== EVALUATION COMPLETE ===", flush=True)
            print(f"Step {current_step}: Final Eval Loss = {eval_loss:.6f}", flush=True)
            print("=" * 40, flush=True)
            sys.stdout.flush()
        
        return result