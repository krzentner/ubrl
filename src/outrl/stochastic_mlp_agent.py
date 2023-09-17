from typing import List

import torch
import outrl


class StochasticMLPAgent(outrl.agent.Agent):
    def __init__(
        self,
        observation_shape: outrl.nn.Shape,
        action_shape: outrl.nn.Shape,
        hidden_sizes: List[int],
        action_dist: outrl.nn.DistConstructor = outrl.nn.NormalConstructor(),
        *,
        vf_hidden_sizes: List[int] = [],
        pi_hidden_sizes: List[int] = [],
        hidden_nonlinearity: outrl.nn.SupportsNonlinearity = torch.relu,
        hidden_w_init: outrl.nn.Initializer = torch.nn.init.xavier_normal_,
        hidden_b_init: outrl.nn.Initializer = torch.nn.init.zeros_,
        output_w_init: outrl.nn.Initializer = torch.nn.init.xavier_normal_,
        output_b_init: outrl.nn.Initializer = torch.nn.init.zeros_,
        layer_normalization: bool = False,
    ):
        super().__init__()

        if isinstance(action_dist, type):
            action_dist = action_dist()

        assert isinstance(action_dist, outrl.nn.DistConstructor)

        self.distribution_constructor = action_dist

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
            output_size=self.distribution_constructor.get_input_size(action_shape),
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
        )

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

        dists = self.distribution_constructor(pi_x)
        actions = dists.sample()

        # Encode the actions that were actually taken
        actions_encoded = self.distribution_constructor.encode_actions(actions, dists)

        return outrl.agent.Step(
            predicted_returns=vf_x,
            action_dists=dists,
            actions=actions,
            action_energy=-dists.log_prob(actions),
            actions_encoded=actions_encoded,
            hidden_states=hidden_states,
        )

    def forward(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor
    ):
        B = observations.shape[0]
        T = observations.shape[1]
        # At least B, T, X dimensions
        assert len(observations.shape) >= 3
        shared_x = self.shared_layers(observations.reshape(B * T, -1))
        pi_x = self.pi_layers(shared_x)
        vf_x = self.vf_layers(shared_x).squeeze(-1)

        dists = self.distribution_constructor(pi_x)

        return self.action_energy(dists, actions.reshape(B * T, -1).squeeze(-1)), vf_x

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
        action_dist=outrl.nn.CategoricalConstructor,
    )
    energy = agent.action_energy(step.action_dists, step.actions)
    assert energy.shape == (B,)
