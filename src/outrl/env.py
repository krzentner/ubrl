from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional
import torch


@dataclass
class Step:
    observations: torch.Tensor
    rewards: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    env_state: Any
    infos: Dict[str, torch.Tensor] = field(default_factory=dict)

    @property
    def done(self) -> torch.Tensor:
        return self.terminated | self.truncated


@dataclass
class EnvSpec:
    observation_low: torch.Tensor
    observation_high: torch.Tensor

    action_low: torch.Tensor
    action_high: torch.Tensor

    action_type: Optional[str]

    max_episode_length: int

    @property
    def observation_shape(self):
        return self.observation_low.shape

    @property
    def observation_dtype(self):
        return self.observation_low.dtype

    @property
    def action_shape(self):
        return self.action_low.shape

    @property
    def action_dtype(self):
        return self.action_low.dtype


class VecEnv:
    @property
    def spec(self) -> EnvSpec:
        raise NotImplementedError()

    def reset(self, prev_step: Optional[Step], mask: torch.Tensor) -> Step:
        del prev_step, mask
        raise NotImplementedError()

    def step(self, prev_step: Step, actions: torch.Tensor, mask: torch.Tensor) -> Step:
        del prev_step, actions, mask
        raise NotImplementedError()


EnvConstructor = Callable[[int], VecEnv]
