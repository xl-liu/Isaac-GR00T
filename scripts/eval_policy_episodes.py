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

import json
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

import numpy as np
import tyro

from gr00t.data.dataset import LeRobotSingleDataset
from gr00t.data.embodiment_tags import EMBODIMENT_TAG_MAPPING
from gr00t.experiment.data_config import DATA_CONFIG_MAP
from gr00t.model.policy import BasePolicy, Gr00tPolicy
from gr00t.utils.eval import calc_mse_for_single_trajectory

warnings.simplefilter("ignore", category=FutureWarning)
warnings.filterwarnings('ignore', module='torchvision')

"""
Example command:

python scripts/eval_policy_episodes.py \
    --dataset-path ../datasets/G1_BlockStacking/ \
    --video-backend torchvision_av \
    --embodiment-tag new_embodiment \
    --data-config unitree_g1 \
    --modality_keys "left_arm" "left_hand" \
    --val_config_file ../checkpoints/<>/train_val_split_info.json
    --plot --save_plot_path pretrained_inference_g1_block_steps10000.png

Where train_val_split_info.json contains:
{
    "dataset_path": "block_stacking/",
    "total_episodes": 300,
    "val_episodes": [1, 2, 3, 4, 5],
    "train_episodes": []
}
"""

@dataclass
class ArgsConfig:
    """Configuration for evaluating a policy on validation episodes."""

    model_path: str
    """Path to the finetuned model checkpoint."""

    val_config_file: str
    """Path to JSON file with validation config (dataset_path and val_episodes)."""

    data_config: Literal[tuple(DATA_CONFIG_MAP.keys())] = "fourier_gr1_arms_only"
    """Data config to use."""

    embodiment_tag: Literal[tuple(EMBODIMENT_TAG_MAPPING.keys())] = "gr1"
    """Embodiment tag to use."""

    modality_keys: List[str] = field(default_factory=lambda: ["right_arm", "left_arm"])
    """Modality keys to evaluate."""

    steps: int = 150
    """Number of steps to evaluate per trajectory."""

    action_horizon: int = None
    """Action horizon to evaluate. If None, will use the data config's action horizon."""

    video_backend: Literal["decord", "torchvision_av"] = "decord"
    """Video backend to use."""

    denoising_steps: int = 4
    """Number of denoising steps to use."""

    plot: bool = False
    """Whether to plot the trajectories."""

    save_plot_path: str = None
    """Path to save plots."""


def load_val_config(config_file: str) -> dict:
    """Load validation configuration from JSON file."""
    config_path = Path(config_file)
    if not config_path.exists():
        raise FileNotFoundError(f"Validation config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Validate required keys
    if "dataset_path" not in config:
        raise KeyError("Missing 'dataset_path' in validation config file")
    if "val_episodes" not in config:
        raise KeyError("Missing 'val_episodes' in validation config file")
    
    print(f"Loaded validation config:")
    print(f"  Dataset path: {config['dataset_path']}")
    print(f"  Validation episodes: {len(config['val_episodes'])} episodes")
    print(f"  Episode IDs: {config['val_episodes']}")
    
    return config


def create_validation_dataset(dataset_path: str, val_episodes: List[int], args: ArgsConfig) -> LeRobotSingleDataset:
    """Create a dataset filtered to only validation episodes."""
    
    data_config = DATA_CONFIG_MAP[args.data_config]
    modality_config = data_config.modality_config()
    
    # Create full dataset first
    full_dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,  # We'll handle transforms through the policy
        embodiment_tag=args.embodiment_tag,
    )
    
    # Filter to only validation episodes
    validation_ids = np.array(val_episodes)
    mask = np.isin(full_dataset._trajectory_ids, validation_ids)
    
    if not mask.any():
        raise ValueError(f"No validation episodes found in dataset. "
                        f"Available episodes: {full_dataset._trajectory_ids}, "
                        f"Requested validation episodes: {validation_ids}")
    
    # Create filtered dataset
    filtered_dataset = LeRobotSingleDataset(
        dataset_path=dataset_path,
        modality_configs=modality_config,
        video_backend=args.video_backend,
        video_backend_kwargs=None,
        transforms=None,
        embodiment_tag=args.embodiment_tag,
    )
    
    # Apply filtering
    filtered_dataset._trajectory_ids = full_dataset._trajectory_ids[mask]
    filtered_dataset._trajectory_lengths = full_dataset._trajectory_lengths[mask]
    filtered_dataset._all_steps = filtered_dataset._get_all_steps()
    
    print(f"Created validation dataset with {len(filtered_dataset._trajectory_ids)} episodes")
    
    return filtered_dataset


def main(args: ArgsConfig):
    """Main evaluation function."""
    
    if args.model_path and not args.val_config_file:
        val_file_path = args.modal_path + "train_val_split_info.json"
    else:
        val_file_path = args.val_config_file
    
    val_config = load_val_config(val_file_path)
    dataset_path = val_config["dataset_path"]
    val_episodes = val_config["val_episodes"]
    
    # Set up data config
    data_config = DATA_CONFIG_MAP[args.data_config]
    
    # Set action_horizon from data config if not provided
    if args.action_horizon is None:
        args.action_horizon = len(data_config.action_indices)
        print(f"Using action_horizon={args.action_horizon} from data config '{args.data_config}'")
    
    import torch
    
    modality_config = data_config.modality_config()
    modality_transform = data_config.transform()
    
    policy: BasePolicy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=modality_config,
        modality_transform=modality_transform,
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    
    # Create validation dataset
    dataset = create_validation_dataset(dataset_path, val_episodes, args)
    
    print(f"\nRunning evaluation on {len(dataset.trajectory_ids)} validation trajectories")
    print(f"Modality keys: {args.modality_keys}")
    
    # Run evaluation on all validation trajectories
    all_mse = []
    
    for idx, traj_id in enumerate(dataset.trajectory_ids):
        print(f"\n=== Evaluating trajectory {idx+1}/{len(dataset.trajectory_ids)} (ID: {traj_id}) ===")
        
        # Find the index of this trajectory in the filtered dataset
        traj_idx = np.where(dataset.trajectory_ids == traj_id)[0][0]
        
        if idx == 0:
            # only plot one episode
            mse = calc_mse_for_single_trajectory(
                policy,
                dataset,
                traj_idx,
                modality_keys=args.modality_keys,
                steps=args.steps,
                action_horizon=args.action_horizon,
                plot=args.plot,
                save_plot_path=args.save_plot_path,
            )
        else:
                mse = calc_mse_for_single_trajectory(
                policy,
                dataset,
                traj_idx,
                modality_keys=args.modality_keys,
                # TODO: steps = len(trajs)
                steps=args.steps,
                action_horizon=args.action_horizon,
                plot=False,
            )
        
        print(f"Trajectory {traj_id} MSE: {mse:.6f}")
        all_mse.append(mse)
    
    # Print summary
    avg_mse = np.mean(all_mse)
    std_mse = np.std(all_mse)
    min_mse = np.min(all_mse)
    max_mse = np.max(all_mse)
    
    print(f"\n{'='*60}")
    print("VALIDATION EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Dataset: {dataset_path}")
    print(f"Number of validation trajectories: {len(all_mse)}")
    print(f"Average MSE: {avg_mse:.6f} ± {std_mse:.6f}")
    print(f"Min MSE: {min_mse:.6f}")
    print(f"Max MSE: {max_mse:.6f}")
    print(f"Per-trajectory MSE: {[f'{mse:.6f}' for mse in all_mse]}")
    print(f"{'='*60}")
    
    print("Validation evaluation complete!")


if __name__ == "__main__":
    config = tyro.cli(ArgsConfig)
    main(config) 