from typing import Any, Dict, List
import torch
import warnings
import outrl.agent as agent
import outrl.fragment_buffer as fragment_buffer
import outrl.nn as nn
import outrl.env as env

FragmentBuffer = fragment_buffer.FragmentBuffer

from outrl.agent import Agent
from outrl.stochastic_mlp_agent import StochasticMLPAgent
from outrl.sampler import Sampler


def as_torch_dict(info: Dict[str, Any]):
    info_torch = {}
    for k, v in info.items():
        try:
            info_torch[k] = torch.tensor(v)
        except TypeError:
            warnings.warn(f"Could not convert info {k!r} ({v!r}) to torch.Tensor")
            pass
    return info_torch


def stack_dicts(infos: List[Dict[str, torch.Tensor]]):
    common_keys = set(infos[0].keys())
    all_keys = set(infos[0].keys())
    for info in infos:
        new_keys = set(info.keys())
        common_keys = common_keys.intersection(new_keys)
        all_keys = all_keys.union(new_keys)
    for key in all_keys - common_keys:
        warnings.warn(f"Discarding info not present in every step: {key}")
    return {k: torch.stack([info[k] for info in infos]) for k in common_keys}


class IterCallback(torch.utils.data.IterableDataset):

    """Converts a function that returns an iterator into an iterator."""

    def __init__(self, callback):
        self.callback = callback

    def __iter__(self):
        return self.callback()
