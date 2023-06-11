from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass(frozen=True)
class Step:

    action_dists: Any
    actions: torch.Tensor
    actions_encoded: torch.Tensor

    # prev actions encoded should go in here
    hidden_states: torch.Tensor

    # Should be -log pi(a|s) / Z for some constant Z
    action_energy: torch.Tensor

    # Used by actor-critic algorithms
    predicted_returns: Optional[torch.Tensor]

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

    def action_energy(self, action_dists: Any, actions: torch.Tensor) -> torch.Tensor:
        del action_dists, actions
        """Should return -log pi(a|s) / Z for some constant Z"""
        raise NotImplementedError()

    def cross_energy(self, pi1: Any, pi2: Any, actions: torch.Tensor) -> torch.Tensor:
        """Should return a value that approximates log (pi_1(a|s) / pi_2(a|s)).

        This should be D_{KL}(pi_1, pi_2) when possible.
        """
        return self.action_energy(pi1, actions) - self.action_energy(pi2, actions)
