from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

import outrl
from outrl.dists import DistConstructor
from outrl.utils import copy_default


@dataclass(frozen=True)
class Step:
    actions: torch.Tensor

    # prev actions encoded should go in here
    hidden_states: torch.Tensor

    # Should be -log pi(a|s) / Z for some constant Z
    action_energy: torch.Tensor

    # Either V(s) or Q(s, *)
    predicted_returns: Optional[torch.Tensor]

    action_logits: Dict[str, torch.Tensor] = copy_default({})

    infos: Dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        assert self.action_energy.shape == (self.actions.shape[0],)


class Agent(nn.Module):
    def initial_hidden_states(self, batch_size: int):
        del batch_size
        raise NotImplementedError()

    def forward(self, observations: torch.Tensor, hidden_states: torch.Tensor) -> Step:
        del observations, hidden_states
        raise NotImplementedError()


class StochasticAgent(Agent):
    def __init__(self, action_dist_cons: DistConstructor):
        super().__init__()
        if isinstance(action_dist_cons, type):
            action_dist_cons = action_dist_cons()
        assert isinstance(action_dist_cons, outrl.dists.DistConstructor)
        self.action_dist_cons = action_dist_cons
