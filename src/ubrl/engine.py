"""Utilities for accelerating / distributing training.

Mostly a wrapper around HuggingFace accelerate, with the potential to use other
libraries.
"""

import logging
from typing import Any
import torch
import ubrl

_LOGGER = logging.getLogger("ubrl")

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None


class Engine:
    def __init__(self, cfg: "ubrl.TrainerConfig"):
        pass

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError()

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def shard_episodes(
        self, episodes: list["ubrl._EpisodeData"], shuffle: bool
    ) -> list["ubrl._EpisodeData"]:
        """This function determines how the episodes from the replay buffer (or
        some portion of the replay buffer) should be distributed between
        machines, handling shuffling as appropriate.

        Locally, this just means shuffling episodes when requested.
        The agent forward() method is responsible for actually collating and
        placing episode contents as appropriate.
        """
        raise NotImplementedError()

    def backward(self, loss: torch.Tensor):
        raise NotImplementedError()

    def save(self, checkpoint: dict[str, Any], path: str):
        raise NotImplementedError()

    def load(self, path: str) -> dict[str, Any]:
        raise NotImplementedError()


class LocalEngine(Engine):
    """Engine that just uses torch directly. Supports local cpu and gpu training."""

    def __init__(self, cfg: "ubrl.TrainerConfig"):
        _LOGGER.info(f"Using LocalEngine(device={cfg.device!r})")
        self.device = cfg.device

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        module = module.to(device=self.device)
        return module

    def prepare_module_opt(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        module = module.to(device=self.device)
        return module, optimizer

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device)

    def shard_episodes(
        self, episodes: list["ubrl._EpisodeData"], shuffle: bool
    ) -> list["ubrl._EpisodeData"]:
        if shuffle:
            shuffled_indices = torch.randperm(len(episodes))
            episodes = [episodes[i] for i in shuffled_indices]
        return episodes

    def clip_grad_norm_(
        self,
        params: list[torch.nn.Parameter],
        max_norm: float,
        error_if_nonfinite: bool,
    ):
        torch.nn.utils.clip_grad_norm_(
            params, max_norm, error_if_nonfinite=error_if_nonfinite
        )

    def backward(self, loss: torch.Tensor):
        loss.backward()

    def save(self, checkpoint: dict[str, Any], path: str):
        torch.save(checkpoint, path)

    def load(self, path: str) -> dict[str, Any]:
        return torch.load(path, map_location=self.device)


class AcceleratorEngine(Engine):
    def __init__(self, cfg: "ubrl.TrainerConfig"):
        print("Using AcceleratorEngine")
        from accelerate import Accelerator

        self.accelerator = Accelerator()

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return self.accelerator.prepare(module)

    def prepare_module_opt(
        self, module: torch.nn.Module, optimizer: torch.optim.Optimizer
    ) -> tuple[torch.nn.Module, torch.optim.Optimizer]:
        return self.accelerator.prepare(module, optimizer)

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.accelerator.prepare(tensor)

    def shard_episodes(
        self, episodes: list["ubrl._EpisodeData"], shuffle: bool
    ) -> list["ubrl._EpisodeData"]:
        return self.accelerator.prepare(episodes)

    def clip_grad_norm_(
        self,
        params: list[torch.nn.Parameter],
        max_norm: float,
        error_if_nonfinite: bool,
    ):
        self.accelerator.clip_grad_norm_(params, max_norm)

    def backward(self, loss: torch.Tensor):
        self.accelerator.backward(loss)

    def save(self, checkpoint: dict[str, Any], path: str):
        self.accelerator.wait_for_everyone()
        self.accelerator.save(checkpoint, path)

    def load(self, path: str) -> dict[str, Any]:
        return self.accelerator.load(path)


if Accelerator is None:
    DefaultEngine = LocalEngine
else:
    DefaultEngine = AcceleratorEngine
