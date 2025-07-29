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
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Literal
from datetime import datetime
import warnings
warnings.filterwarnings('ignore', module='torchvision')
import torch
import tyro
from transformers import TrainingArguments
import numpy as np
from pathlib import Path
import json

from gr00t.data.dataset import LeRobotMixtureDataset, LeRobotSingleDataset
from gr00t.data.schema import EmbodimentTag
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.experiment.runner import TrainRunner
from gr00t.model.gr00t_n1 import GR00T_N1_5
from gr00t.model.transforms import EMBODIMENT_TAG_MAPPING
from gr00t.utils.peft import get_lora_model


@dataclass
class ArgsConfig:
    """Configuration for GR00T model fine-tuning."""

    # Dataset parameters
    dataset_path: List[str]
    """Path to the dataset directory or directories"""

    output_dir: str = "/tmp/gr00t"
    """Directory to save model checkpoints."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data configuration name from DATA_CONFIG_MAP, we assume all datasets have the same data config"""
    
    # Training parameters
    experiment_name: str = "gr00t_finetune"
    """Project name for wandb logging"""
    
    validation_split: float = 0.1
    """proportion of data reserved for validation"""
    
    eval_steps: int = 500
    """number of steps between validations"""
    
    batch_size: int = 32
    """Batch size per GPU for training."""

    max_steps: int = 10000
    """Maximum number of training steps."""

    num_gpus: int = 1
    """Number of GPUs to use for training."""

    save_steps: int = 2500
    """Number of steps between saving checkpoints."""

    # Model parameters
    base_model_path: str = "nvidia/GR00T-N1.5-3B"
    """Path or HuggingFace model ID for the base model."""

    tune_llm: bool = False
    """Whether to fine-tune the language model backbone."""

    tune_visual: bool = False
    """Whether to fine-tune the vision tower."""

    tune_projector: bool = True
    """Whether to fine-tune the projector."""

    tune_diffusion_model: bool = True
    """Whether to fine-tune the diffusion model."""

    resume: bool = False
    """Whether to resume from a checkpoint."""

    # Advanced training parameters
    learning_rate: float = 1e-4
    """Learning rate for training."""

    weight_decay: float = 1e-5
    """Weight decay for AdamW optimizer."""

    warmup_ratio: float = 0.05
    """Ratio of total training steps used for warmup."""

    lora_rank: int = 0
    """Rank for the LORA model. If 0, no LORA will be used."""

    lora_alpha: int = 16
    """Alpha value for the LORA model."""

    lora_dropout: float = 0.1
    """Dropout rate for the LORA model."""

    lora_full_model: bool = False
    """Whether to use the full model for LORA. If False, only the action head will be trained."""

    dataloader_num_workers: int = 8
    """Number of workers for data loading."""

    report_to: Literal["wandb", "tensorboard", "azure_ml"] = "wandb"
    """Where to report training metrics (e.g., 'wandb', 'tensorboard', 'azure_ml')."""

    # Data loading parameters
    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "new_embodiment"
    """Embodiment tag to use for training. e.g. 'new_embodiment', 'gr1'"""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use for training. [decord, torchvision_av]"""

    # Mixture dataset parameters
    balance_dataset_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, we will balance the dataset weights, by multiplying the total trajectory to each dataset"""

    # Mixture dataset parameters
    balance_trajectory_weights: bool = True
    """Used in LeRobotMixtureDataset. If True, sample trajectories within a dataset weighted by their length; otherwise, equal weighting."""

# helper functions to split the dataset

def split_episodes_by_ratio(trajectory_ids: np.ndarray, val_ratio: float, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Split trajectory IDs into train and validation sets by ratio.
    
    Args:
        trajectory_ids: Array of trajectory IDs
        val_ratio: Ratio of episodes to use for validation (0.0 - 1.0)  
        seed: Random seed for splitting
        
    Returns:
        tuple of (train_trajectory_ids, val_trajectory_ids)
    """
    if val_ratio <= 0.0 or val_ratio >= 1.0:
        return trajectory_ids, np.array([])
    
    rng = np.random.default_rng(seed)
    shuffled_ids = rng.permutation(trajectory_ids)
    
    n_val_episodes = int(len(shuffled_ids) * val_ratio)
    val_ids = shuffled_ids[:n_val_episodes]
    train_ids = shuffled_ids[n_val_episodes:]
    
    return train_ids, val_ids


def create_dataset_subset(dataset: LeRobotSingleDataset, trajectory_ids: np.ndarray) -> LeRobotSingleDataset:
    """Create a subset of LeRobotSingleDataset with only specified trajectory IDs."""
    # Create a copy of the dataset with filtered trajectories
    subset_dataset = LeRobotSingleDataset(
        dataset_path=dataset.dataset_path,
        modality_configs=dataset.modality_configs,
        embodiment_tag=dataset.tag,
        video_backend=dataset.video_backend,
        video_backend_kwargs=dataset.video_backend_kwargs,
        transforms=dataset.transforms,
    )
    
    # Filter trajectories
    mask = np.isin(subset_dataset._trajectory_ids, trajectory_ids)
    subset_dataset._trajectory_ids = subset_dataset._trajectory_ids[mask]
    subset_dataset._trajectory_lengths = subset_dataset._trajectory_lengths[mask]
    
    # Rebuild all_steps for the filtered trajectories
    subset_dataset._all_steps = subset_dataset._get_all_steps()
    
    return subset_dataset


def split_single_dataset(dataset: LeRobotSingleDataset, val_ratio: float, seed: int = 42,
                         output_dir: str = "") -> tuple[LeRobotSingleDataset, LeRobotSingleDataset | None]:
    """Split a single dataset into train and validation sets."""
    if val_ratio <= 0.0:
        return dataset, None
        
    train_ids, val_ids = split_episodes_by_ratio(dataset.trajectory_ids, val_ratio, seed)
    
    train_dataset = create_dataset_subset(dataset, train_ids)
    val_dataset = create_dataset_subset(dataset, val_ids) if len(val_ids) > 0 else None
    
    print(f"Split dataset {dataset.dataset_name}: {len(train_ids)} train episodes, {len(val_ids)} val episodes")
    
    # save the val_ids
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    split_info = {
        "dataset_path": dataset.dataset_path,
        "total_episodes": len(train_ids) + len(val_ids), 
        "split_ratio": val_ratio,
        "val_episodes": val_ids.tolist(),
        "train_episodes": train_ids.tolist()
    }
    with open(output_path / "train_val_split_info.json", "a") as f:
        json.dump(split_info, f, indent=2)
    
    return train_dataset, val_dataset

def split_mixture_dataset(mixture_dataset: LeRobotMixtureDataset, val_ratio: float, seed: int = 42) -> tuple[LeRobotMixtureDataset, LeRobotMixtureDataset | None]:
    """Split a mixture dataset into train and validation sets."""
    if val_ratio <= 0.0:
        return mixture_dataset, None
    
    train_datasets = []
    val_datasets = []
    dataset_weights = []
    
    for dataset, weight in zip(mixture_dataset.datasets, mixture_dataset._dataset_sampling_weights):
        train_subset, val_subset = split_single_dataset(dataset, val_ratio, seed)
        train_datasets.append((train_subset, weight))
        if val_subset is not None:
            val_datasets.append((val_subset, weight))
        dataset_weights.append(weight)
            
    # Create train mixture dataset
    train_mixture = LeRobotMixtureDataset(
        data_mixture=train_datasets,
        mode="train",
        balance_dataset_weights=mixture_dataset.balance_dataset_weights,
        balance_trajectory_weights=mixture_dataset.balance_trajectory_weights,
        seed=mixture_dataset.seed,
        metadata_config={"percentile_mixing_method": "weighted_average"}
    )
    
    # Create validation mixture dataset if we have validation data
    val_mixture = None
    if val_datasets:
        val_mixture = LeRobotMixtureDataset(
            data_mixture=val_datasets,
            mode="val",
            balance_dataset_weights=mixture_dataset.balance_dataset_weights,
            balance_trajectory_weights=mixture_dataset.balance_trajectory_weights,
            seed=mixture_dataset.seed,
            metadata_config={"percentile_mixing_method": "weighted_average"}
        )
    
    return train_mixture, val_mixture


#####################################################################################
# main training function
#####################################################################################


def main(config: ArgsConfig):
    """Main training function."""
    output_dir = config.output_dir + "_" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    
    # ------------ step 1: load dataset ------------
    embodiment_tag = EmbodimentTag(config.embodiment_tag)

    # 1.1 modality configs and transforms
    data_config_cls = DATA_CONFIG_MAP[config.data_config]
    modality_configs = data_config_cls.modality_config()
    transforms = data_config_cls.transform()

    # 1.2 data loader: we will use either single dataset or mixture dataset
    if len(config.dataset_path) == 1:
        full_dataset = LeRobotSingleDataset(
            dataset_path=config.dataset_path[0],
            modality_configs=modality_configs,
            transforms=transforms,
            embodiment_tag=embodiment_tag,  # This will override the dataset's embodiment tag to "new_embodiment"
            video_backend=config.video_backend,
        )
        train_dataset, val_dataset = split_single_dataset(full_dataset, config.validation_split, output_dir=output_dir)
    else:
        single_datasets = []
        for p in config.dataset_path:
            assert os.path.exists(p), f"Dataset path {p} does not exist"
            ## We use the same transforms, modality configs, and embodiment tag for all datasets here,
            ## in reality, you can use dataset from different modalities and embodiment tags
            dataset = LeRobotSingleDataset(
                dataset_path=p,
                modality_configs=modality_configs,
                transforms=transforms,
                embodiment_tag=embodiment_tag,
                video_backend=config.video_backend,
            )
            single_datasets.append(dataset)

        full_dataset = LeRobotMixtureDataset(
            data_mixture=[
                (dataset, 1.0)  # we will use equal weights for all datasets
                for dataset in single_datasets
            ],
            mode="train",
            balance_dataset_weights=config.balance_dataset_weights,
            balance_trajectory_weights=config.balance_trajectory_weights,
            seed=42,
            metadata_config={
                "percentile_mixing_method": "weighted_average",
            },
        )
        print(f"Loaded {len(single_datasets)} datasets, with {config.dataset_path} ")
        train_dataset, val_dataset = split_mixture_dataset(full_dataset, config.validation_split)

    # ------------ step 2: load model ------------
    # First, get the data config to determine action horizon
    data_action_horizon = len(data_config_cls.action_indices)

    # Load model
    model = GR00T_N1_5.from_pretrained(
        pretrained_model_name_or_path=config.base_model_path,
        tune_llm=config.tune_llm,  # backbone's LLM
        tune_visual=config.tune_visual,  # backbone's vision tower
        tune_projector=config.tune_projector,  # action head's projector
        tune_diffusion_model=config.tune_diffusion_model,  # action head's DiT
    )
 
    # Update action_horizon to match data config
    # Need to recreate action head with correct config since it was initialized with old config
    if data_action_horizon != model.action_head.config.action_horizon:
        print(
            f"Recreating action head with action_horizon {data_action_horizon} (was {model.action_head.config.action_horizon})"
        )

        # Update the action head config
        new_action_head_config = model.action_head.config
        new_action_head_config.action_horizon = data_action_horizon

        # Import the FlowmatchingActionHead class
        from gr00t.model.action_head.flow_matching_action_head import (
            FlowmatchingActionHead,
        )

        # Create new action head with updated config
        new_action_head = FlowmatchingActionHead(new_action_head_config)

        # Copy the weights from the old action head to the new one
        new_action_head.load_state_dict(model.action_head.state_dict(), strict=False)

        # Replace the action head
        model.action_head = new_action_head

        # Update model config AND the action_head_cfg dictionary that gets saved
        model.config.action_horizon = data_action_horizon
        model.action_horizon = data_action_horizon
        model.config.action_head_cfg["action_horizon"] = data_action_horizon

        # Set trainable parameters for the new action head
        model.action_head.set_trainable_parameters(
            tune_projector=config.tune_projector, tune_diffusion_model=config.tune_diffusion_model
        )

    # Set the model's compute_dtype to bfloat16
    model.compute_dtype = "bfloat16"
    model.config.compute_dtype = "bfloat16"

    if config.lora_rank > 0:
        model = get_lora_model(
            model,
            rank=config.lora_rank,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            action_head_only=not config.lora_full_model,
        )
    # 2.1 modify training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        run_name=config.experiment_name,
        remove_unused_columns=False,
        deepspeed="",
        gradient_checkpointing=False,
        bf16=True,
        tf32=True,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=1,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=config.dataloader_num_workers > 0,
        optim="adamw_torch",
        adam_beta1=0.95,
        adam_beta2=0.999,
        adam_epsilon=1e-8,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=10.0,
        num_train_epochs=300,
        max_steps=config.max_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=8,
        report_to=config.report_to,
        seed=42,
        eval_strategy="no",
        # eval_strategy="steps" if val_dataset is not None else "no",
        # do_eval=val_dataset is not None,
        # eval_steps=config.eval_steps if val_dataset is not None else None,
        # load_best_model_at_end =True,
        ddp_find_unused_parameters=False,
        ddp_bucket_cap_mb=100,
        torch_compile_mode=None,
    )

    # 2.2 run experiment
    experiment = TrainRunner(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        model=model,
        training_args=training_args,
        resume_from_checkpoint=config.resume,
    )

    # 2.3 run experiment
    experiment.train()


if __name__ == "__main__":
    # Parse arguments using tyro
    config = tyro.cli(ArgsConfig)

    # Print the tyro config
    print("\n" + "=" * 50)
    print("GR00T FINE-TUNING CONFIGURATION:")
    print("=" * 50)
    for key, value in vars(config).items():
        print(f"{key}: {value}")
    print("=" * 50 + "\n")

    available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Validate GPU configuration
    assert (
        config.num_gpus <= available_gpus
    ), f"Number of GPUs requested ({config.num_gpus}) is greater than the available GPUs ({available_gpus})"
    assert config.num_gpus > 0, "Number of GPUs must be greater than 0"
    print(f"Using {config.num_gpus} GPUs")

    if config.num_gpus == 1:
        # Single GPU mode - set CUDA_VISIBLE_DEVICES=0
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        # Run the script normally
        main(config)
    else:
        if os.environ.get("IS_TORCHRUN", "0") == "1":
            main(config)
        else:
            # Multi-GPU mode - use torchrun
            script_path = Path(__file__).absolute()
            # Remove any existing CUDA_VISIBLE_DEVICES from environment
            if "CUDA_VISIBLE_DEVICES" in os.environ:
                del os.environ["CUDA_VISIBLE_DEVICES"]

            # Use subprocess.run instead of os.system
            cmd = [
                "torchrun",
                "--standalone",
                f"--nproc_per_node={config.num_gpus}",
                "--nnodes=1",  # default to 1 node for now
                str(script_path),
            ]

            # Convert config to command line arguments
            for key, value in vars(config).items():
                if isinstance(value, bool):
                    # For boolean values, use --flag or --no-flag format
                    if value:
                        cmd.append(f"--{key.replace('_', '-')}")
                    else:
                        cmd.append(f"--no-{key.replace('_', '-')}")
                else:
                    # For non-boolean values, use --key value format
                    cmd.append(f"--{key.replace('_', '-')}")

                    # if the value is a list (e.g. dataset_path), we need to add each element in the list
                    if isinstance(value, list):
                        for v in value:
                            cmd.append(str(v))
                    else:
                        cmd.append(str(value))
            print("Running torchrun command: ", cmd)
            env = os.environ.copy()
            env["IS_TORCHRUN"] = "1"
            sys.exit(subprocess.run(cmd, env=env).returncode)
