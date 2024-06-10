"""Utilities for accelerating / distributing training.

Mostly a wrapper around HuggingFace accelerate, with the potential to use other
libraries.
"""

import torch


class Accel:

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        raise NotImplementedError()

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def shard_episodes(
        self, episodes: list["ubrl._EpisodeData"], shuffle: bool
    ) -> list["ubrl._EpisodeData"]:
        raise NotImplementedError()


class LocalAccel(Accel):
    """Accel that just uses torch directly. Supports local cpu and gpu training."""

    def __init__(self, cfg: "ubrl.TrainerConfig"):
        self.device = cfg.device

    def prepare_module(self, module: torch.nn.Module) -> torch.nn.Module:
        return module.to(device=self.device)

    def prepare_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device)

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
        if shuffle:
            shuffled_indices = torch.randperm(len(episodes))
            episodes = [episodes[i] for i in shuffled_indices]
        return episodes


DefaultAccel = LocalAccel
