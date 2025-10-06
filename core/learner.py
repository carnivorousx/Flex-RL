import dataclasses
import torch
import torch.nn as nn
import gymnasium as gym
import numpy as np
import torch.optim as optim
import torch.utils.tensorboard as tb
from typing import Callable, Any


@dataclasses.dataclass
class LearnerConfig:
    experiment_name: str = "default_experiment"
    log_dir_name: str = "logs"
    save_checkpoint: bool = False
    load_checkpoint: bool = False
    load_optimizer_state: bool = False
    load_checkpoint_path: str = ""
    log_every_n: int = 1
    # Environment
    env_name: str = "CartPole-v1"
    state_dim: int = -1
    action_dim: int = -1
    # Framework
    state_processor_fn: callable = None
    action_processor_fn: callable = None
    reward_processor_fn: callable = None
    # Experience Collection
    sample_model_name: str = "policy"
    num_episodes_per_global_step: int = 20
    epsilon_greedy: float = 0.0
    action_bound: float = 1.0
    is_continuous_action: bool = False
    # Algorithm
    algorithm: str = "ppo"


@dataclasses.dataclass
class Experience:
    state: list[float]
    action: int | list[float]
    log_prob: float | list[float]
    reward: float
    value: float
    value_target: float
    td_error: float
    gae: float
    return_: float
    return_tg: float
    # metadata
    timestep: int
    episode_id: int
    episode_axis_value_target_std: float
    episode_axis_value_target_var: float
    horizon: int
    global_step: int


def sample_action(env: gym.Env, state: np.ndarray, 
                  policy: nn.Module, q_model: nn.Module, 
                  config: LearnerConfig) -> Any:
    pass
