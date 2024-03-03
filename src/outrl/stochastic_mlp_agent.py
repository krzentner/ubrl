from typing import List, Any

import torch
import outrl.agent
import outrl.nn
from outrl.torch_utils import pack_tensors, pack_tensors_check
import numpy as np


class StochasticMLPAgent(outrl.agent.StochasticAgent):
    def __init__(
        self,
        observation_shape: outrl.nn.Shape,
        action_shape: outrl.nn.Shape,
        hidden_sizes: List[int],
        *,
        vf_hidden_sizes: List[int] = [],
        pi_hidden_sizes: List[int] = [],
        hidden_nonlinearity: outrl.nn.SupportsNonlinearity = torch.relu,
        hidden_w_init: outrl.nn.Initializer = torch.nn.init.xavier_normal_,
        hidden_b_init: outrl.nn.Initializer = torch.nn.init.zeros_,
        output_w_init: outrl.nn.Initializer = torch.nn.init.xavier_normal_,
        output_b_init: outrl.nn.Initializer = torch.nn.init.zeros_,
        layer_normalization: bool = False,
        action_dist_cons: outrl.dists.DistConstructor = outrl.dists.NormalConstructor(),
    ):
        super().__init__(action_dist_cons=action_dist_cons)

        self.shared_layers = outrl.nn.make_mlp(
            input_size=outrl.nn.flatten_shape(observation_shape),
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            layer_normalization=layer_normalization,
        )

        self.vf_layers = outrl.nn.make_mlp(
            input_size=hidden_sizes[-1],
            hidden_sizes=vf_hidden_sizes,
            output_size=1,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            # output_w_init=output_w_init,
            # output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

        self.pi_layers = outrl.nn.make_mlp(
            input_size=hidden_sizes[-1],
            hidden_sizes=pi_hidden_sizes,
            output_size=self.action_dist_cons.get_input_size(action_shape),
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

    def get_actions(
        self,
        observations: np.ndarray,
        reset_mask: np.ndarray,
        best_action: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del reset_mask
        del best_action
        observations = torch.from_numpy(observations)
        B = observations.shape[0]
        shared_x = self.shared_layers(observations.reshape(B, -1))
        pi_x = self.pi_layers(shared_x)

        dists = self.action_dist_cons(pi_x)
        actions = dists.sample()

        action_ll = dists.log_prob(actions).unsqueeze(-1)
        assert action_ll.shape == (observations.shape[0], 1)
        action_mean = dists.mean
        action_stddev = dists.stddev
        return np.asarray(actions), {
            "action_ll": action_ll,
            "action_mean": action_mean,
            "action_stddev": action_stddev,
        }

    def step(
        self,
        observations: torch.Tensor,
        hidden_states: torch.Tensor,
        prev_reward: torch.Tensor,
    ):
        del prev_reward
        B = observations.shape[0]
        shared_x = self.shared_layers(observations.reshape(B, -1))
        pi_x = self.pi_layers(shared_x)
        vf_x = self.vf_layers(shared_x).squeeze(-1)

        dists = self.action_dist_cons(pi_x)
        actions = dists.sample()

        return outrl.agent.Step(
            predicted_returns=vf_x,
            actions=actions,
            action_energy=-dists.log_prob(actions),
            hidden_states=hidden_states,
        )

    def forward_both(
        self, observations: list[torch.Tensor], actions: list[torch.Tensor]
    ):
        obs_packed, obs_lens = pack_tensors(observations)
        act_packed = pack_tensors_check(actions, obs_lens)
        shared_x = self.shared_layers(obs_packed)
        pi_x = self.pi_layers(shared_x)
        vf_x = self.vf_layers(shared_x).squeeze(-1)

        dists = self.action_dist_cons(pi_x)

        return self.action_energy(dists, act_packed), vf_x

    def vf_forward(self, observations: torch.Tensor):
        B = observations.shape[0]
        T = observations.shape[1]
        shared_x = self.shared_layers(observations.reshape(B * T, -1))
        vf_x = self.vf_layers(shared_x).squeeze(-1)
        return vf_x.reshape(B, T)

    def action_energy(self, action_dists, actions):
        res = -action_dists.log_prob(actions)
        return res

    def initial_hidden_states(self, batch_size: int):
        return torch.zeros((batch_size,))


def test_smoke():
    B = 7
    A = (5,)
    O = (3, 2)
    agent = StochasticMLPAgent(
        observation_shape=O, action_shape=A, hidden_sizes=[128, 128]
    )
    hidden_states = agent.initial_hidden_states(B)
    step = agent(
        observations=torch.randn(B, *O),
        hidden_states=hidden_states,
        prev_reward=torch.zeros(B),
    )
    energy = agent.action_energy(step.action_dists, step.actions)
    assert energy.shape == (B,)

    agent = StochasticMLPAgent(
        observation_shape=O,
        action_shape=A,
        hidden_sizes=[128, 128],
        action_dist_cons=outrl.dists.CategoricalConstructor,
    )
    energy = agent.action_energy(step.action_dists, step.actions)
    assert energy.shape == (B,)
