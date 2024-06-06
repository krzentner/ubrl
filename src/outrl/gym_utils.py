"""Utilities for working with the OpenAI Gym API.

These utilities are not considered part of the core OutRL library, and OutRL does not depend on anything in this module.

If you're using OutRL, the main reason is probably that you want to implement
custom neural networks for your environment, which you can base off of the code
in this module. We recommend you copy freely from this file instead of
inheriting from the classes defined here.

## Example

```python
.. include:: ../../examples/gym_example.py
```
"""
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import TransformedDistribution, AffineTransform
from tqdm import tqdm

from outrl import Agent, AgentOutput
from outrl.torch_utils import (
    force_concat,
    make_mlp,
    flatten_shape,
    RunningMeanVar,
    pack_tensors,
    unpack_tensors,
    Sizes,
    clamp_identity_grad,
    soft_clamp,
)


class RobustTransformedDist(TransformedDistribution):
    """A hack to avoid the situation where
    dist.log_prob(dist.sample()) gives non-finite values.

    """

    def log_prob(self, value: torch.Tensor):
        # Sometimes, sampling from a transformed distribution
        # produces a value that the transformed distribution
        # claims has 0 probability of being sampled.

        # This is usually due to sampling at the edge of the
        # distribution, so compensate by slightly shrinking the
        # input.

        log_probs = super().log_prob(value)
        if not torch.isfinite(log_probs).all():
            value = value / (1 + 1e-6)
        log_probs = super().log_prob(value)
        assert torch.isfinite(log_probs).all()
        return log_probs



class GymAgent(Agent):
    """Agent intended to be used with the Gym API."""

    def __init__(
        self,
        *,
        obs_size: int,
        hidden_sizes: Sizes,
        pi_hidden_sizes: Sizes,
    ):
        """

        Args:

            obs_size (int): Flattened size of observations.
            hidden_sizes (Sizes): Sizes of latent representations for the shared layers.
            pi_hidden_sizes (Sizes): Sizes of latent representations for the policy specific layers.
        """
        super().__init__(hidden_sizes[-1])

        self.shared_layers = nn.Sequential(
            RunningMeanVar(
                init_mean=torch.zeros(obs_size), init_var=torch.ones(obs_size)
            ),
            make_mlp(
                input_size=obs_size,
                hidden_sizes=hidden_sizes,
                use_dropout=False,
            ),
        )

        self.pi_layers = make_mlp(
            input_size=hidden_sizes[-1], hidden_sizes=pi_hidden_sizes, use_dropout=False
        )

    @property
    def _input_device(self):
        return self.shared_layers[0].mean.device

    @property
    def _input_dtype(self):
        return self.shared_layers[0].mean.dtype

    def run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        """Main "forward pass" method used during both collect and optimization.

        Returns a tuple of:

            A torch.Tensor containing state_encodings (not used during collect).

            A dictionary of distribution params (will be passed to construct_dist()).

            A dictionary of infos (will be logged).
        """
        raise NotImplementedError()

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        """Construct a torch.distribution.Distribution from a dictionary of
        parameters returned by run_net() (or a concatenation thereof).
        """
        raise NotImplementedError()

    def get_actions(
        self,
        observations: np.ndarray,
        reset_mask: np.ndarray,
        best_action: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Compute a batch of actions from a batch of observations.

        If an index in reset_mask is True, this is the first observation in the
        episode (only used for Agents with memory).

        When best_action is True, takes the most likely action (mode) of the
        action distributions.

        Returns a tuple of:

            A np.ndarray containing a batch of (sampled) actions.

            A dict of infos to log.
        """
        del reset_mask
        assert len(observations.shape) == 2
        with torch.no_grad():
            _, params, infos = self.run_net(
                torch.from_numpy(observations)
                .to(dtype=self._input_dtype, device=self._input_device)
            )
        dist, mode = self.construct_dist(params)
        if best_action:
            action = mode
        else:
            action = dist.sample()
        action_ll = dist.log_prob(action)
        infos.update({f"action_params.{k}": v for (k, v) in params.items()})
        infos.update(
            {
                "action_ll": action_ll,
            }
        )
        return np.asarray(action.cpu()), infos

    def forward(self, episodes: list[dict[str, torch.Tensor]]) -> list[AgentOutput]:
        """Implements the OutRL forward pass API in terms of run_net().

        Args:

            episodes: A list of "episode" objects, as produced by collect().

        Returns:

            A list of AgentOutput objects.
        """
        observations, pack_lens = pack_tensors([ep["observations"] for ep in episodes])
        observations = observations.to(dtype=self._input_dtype, device=self._input_device)
        # Add a trailing fake action so each observation has an action afterwards
        actions, action_pack_lens = pack_tensors(
            [
                torch.cat([ep["actions"], ep["actions"][-1].unsqueeze(0)])
                for ep in episodes
            ]
        )
        actions = actions.to(dtype=self._input_dtype, device=self._input_device)
        assert action_pack_lens == pack_lens
        state_encodings, params, infos = self.run_net(observations)
        del infos

        batch_dist, _ = self.construct_dist(params)
        state_encodings = unpack_tensors(state_encodings, pack_lens)
        packed_action_ll = batch_dist.log_prob(actions).squeeze(-1)
        assert torch.isfinite(packed_action_ll).all()

        unpacked_params = {k: unpack_tensors(v, pack_lens) for (k, v) in params.items()}
        dists = [
            self.construct_dist({k: v[i][:-1] for (k, v) in unpacked_params.items()})[0]
            for i in range(len(episodes))
        ]
        action_lls = [
            act_lls[:-1] for act_lls in unpack_tensors(packed_action_ll, pack_lens)
        ]
        return [
            AgentOutput(
                state_encodings=state_enc, action_lls=act_lls, action_dists=dist
            )
            for (state_enc, act_lls, dist) in zip(state_encodings, action_lls, dists)
        ]


class GymBoxGaussianAgent(GymAgent):
    """GymAgent producing a continuous action distribution suitable for the
    "Box" action space.

    Despite this library's name, this class cannot be fully avoided.
    """

    def __init__(
        self,
        *,
        obs_size: int,
        action_size: int,
        hidden_sizes: Sizes,
        pi_hidden_sizes: Sizes,
        init_std: float,
        min_std: float,
        loc: np.ndarray,
        scale: np.ndarray,
    ):
        super().__init__(
            obs_size=obs_size,
            hidden_sizes=hidden_sizes,
            pi_hidden_sizes=pi_hidden_sizes,
        )

        self.action_mean = nn.Linear(pi_hidden_sizes[-1], action_size)

        self.action_logstd = nn.Linear(pi_hidden_sizes[-1], action_size)
        nn.init.constant_(self.action_logstd.bias, torch.tensor(init_std).log().item())
        nn.init.orthogonal_(
            self.action_logstd.weight, gain=torch.tensor(init_std).log().item()
        )
        nn.init.zeros_(self.action_logstd.weight)
        nn.init.zeros_(self.action_mean.bias)
        self.action_mean.weight.data.copy_(0.01 * self.action_mean.weight.data)
        self.min_std = min_std
        self.loc = torch.from_numpy(loc)
        self.scale = torch.from_numpy(scale)

    def run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        state_encodings = self.shared_layers(obs)
        pi_x = self.pi_layers(state_encodings)
        mean = self.action_mean(pi_x)
        mean_adjusted = mean + self.loc
        std_pre_clamp = self.action_logstd(pi_x).exp()
        std = clamp_identity_grad(std_pre_clamp, min=self.min_std)
        std_adjusted = std * self.scale
        params = dict(mean=mean_adjusted, std=std_adjusted)
        return (
            state_encodings,
            params,
            {
                "action_stddev_unclamped": std_pre_clamp.mean(dim=-1),
            },
        )

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:
        mean, std = params["mean"], params["std"]
        dist = torch.distributions.Normal(mean, std)
        if len(mean.shape) > 1:
            dist = torch.distributions.Independent(dist, 1)
        return dist, dist.mode


class GymBoxBetaAgent(GymAgent):
    """GymAgent producing a continuous action distribution suitable for the
    "Box" action space.

    Despite this library's name, this class cannot be fully avoided.
    """

    def __init__(
        self,
        *,
        obs_size: int,
        action_size: int,
        hidden_sizes: Sizes,
        pi_hidden_sizes: Sizes,
        init_std: float,
        min_std: float,
        loc: np.ndarray,
        scale: np.ndarray,
    ):
        super().__init__(
            obs_size=obs_size,
            hidden_sizes=hidden_sizes,
            pi_hidden_sizes=pi_hidden_sizes,
        )

        self.action_mean = nn.Linear(pi_hidden_sizes[-1], action_size)

        self.action_logstd = nn.Linear(pi_hidden_sizes[-1], action_size)
        nn.init.constant_(self.action_logstd.bias, torch.tensor(init_std / 2).log().item())
        nn.init.orthogonal_(
            self.action_logstd.weight, gain=torch.tensor(init_std / 2).log().item()
        )
        nn.init.zeros_(self.action_logstd.weight)
        nn.init.zeros_(self.action_mean.bias)
        self.action_mean.weight.data.copy_(0.01 * self.action_mean.weight.data)
        self.min_std = min_std
        self.loc = torch.from_numpy(loc)
        self.scale = torch.from_numpy(scale)

    def run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        state_encodings = self.shared_layers(obs)
        pi_x = self.pi_layers(state_encodings)

        # Center the mean output in the beta distribution and
        # clamp to 0.0 to 1.0 range
        mean = clamp_identity_grad(
                self.action_mean(pi_x) + 0.5,
                min=0.0, max=1.0)

        std = self.action_logstd(pi_x).exp()
        var = std ** 2

        # Constraint of the beta distribution:
        #  mean * (1 - mean) > var
        max_allowed_var = mean * (1 - mean)
        var_clipped = soft_clamp(
            var,
            min=self.min_std ** 2,
            max=max_allowed_var.detach())

        v = mean * (1 - mean) / var_clipped - 1
        alpha = mean * v
        beta = (1 - mean) * v

        assert (0 <= alpha).all()
        assert (0 <= beta).all()

        mean_adjusted = (mean - 0.5) + self.loc
        std_adjusted = var_clipped.sqrt() * 2 * self.scale
        params = dict(mean=mean_adjusted, std=std_adjusted,
                      alpha=alpha, beta=beta)
        return (
            state_encodings,
            params,
            {
                "action_stddev_unclamped": std.mean(dim=-1),
            },
        )

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> tuple[torch.distributions.Distribution, torch.Tensor]:

        alpha, beta = params["alpha"], params["beta"]
        dist = torch.distributions.Beta(alpha, beta)
        if len(alpha.shape) > 1:
            dist = torch.distributions.Independent(dist, 1)
        scale = 2 * self.scale
        loc = scale * (self.loc - 0.5)
        transform = AffineTransform(loc=loc,
                                    scale=scale)
        assert transform(torch.zeros_like(self.scale)) == -self.scale
        assert transform(torch.ones_like(self.scale)) == self.scale
        mode_untransformed = dist.mode
        mode_untransformed[~torch.isfinite(mode_untransformed)] = dist.mean[~torch.isfinite(mode_untransformed)]
        assert torch.isfinite(mode_untransformed).all()

        mode = (mode_untransformed + self.loc - 0.5) * 2 * self.scale

        transformed_dist = RobustTransformedDist(dist, [transform])
        return transformed_dist, mode


class GymBoxCategorialAgent(GymAgent):
    """GymAgent producing a discrete action distribution suitable for the
    "Categorical" action space.
    """

    def __init__(
        self,
        *,
        obs_size: int,
        action_size: int,
        hidden_sizes: Sizes,
        pi_hidden_sizes: Sizes,
    ):
        super().__init__(
            obs_size=obs_size,
            hidden_sizes=hidden_sizes,
            pi_hidden_sizes=pi_hidden_sizes,
        )

        self.action_logits = nn.Linear(pi_hidden_sizes[-1], action_size)

    def run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        state_encodings = self.shared_layers(obs)
        pi_x = self.pi_layers(state_encodings)
        action_logits = self.action_logits(pi_x)
        params = dict(logits=action_logits)
        dist, _ = self.construct_dist(params)
        infos = {
            f"action_{i}_prob": prob
            for (i, prob) in enumerate(dist.probs.transpose(0, 1))
        }
        assert len(infos["action_0_prob"]) == len(dist.probs)
        return state_encodings, params, infos

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> [torch.distributions.Categorical, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=params["logits"])
        return dist, dist.mode


def make_gym_agent(
    env,
    hidden_sizes: Sizes,
    pi_hidden_sizes: Sizes,
    init_std: float = 0.5,
    min_std: float = 1e-6,
):
    """Attempt to construct a GymAgent appropriate for the passed in environment.

    Not all observation and action spaces are supported.

    If you know specifically what type of environment you have, it is probably
    simpler to construct the appropriate GymAgent directly.

    Args:

        env (gym.Env or gymnasium.Env): Environment. Used for its
            observation_space and action_space fields.
        hidden_sizes (Sizes): Shared hidden sizes.
        pi_hidden_sizes (Sizes): Action tail hidden sizes.
        init_std (float): Initial standard deviation when using a Box action
            space. Ignored for other action spaces.
        min_std (float): Minimum standard deviation when using a Box action
            space. Ignored for other action spaces.

    """
    while isinstance(env, list):
        env = env[0]
    act_space = type(env.action_space).__name__
    obs_space = type(env.observation_space).__name__
    obs_size = flatten_shape(env.observation_space.shape)

    if obs_space == "Box" and act_space == "Box":
        action_size = flatten_shape(env.action_space.shape)
        loc = (env.action_space.high + env.action_space.low) / 2
        scale = (env.action_space.high - env.action_space.low) / 2
        return GymBoxGaussianAgent(
            obs_size=obs_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            pi_hidden_sizes=pi_hidden_sizes,
            init_std=init_std,
            min_std=min_std,
            loc=loc,
            scale=scale,
        )
    elif obs_space == "Box" and act_space == "Discrete":
        action_size = env.action_space.n
        return GymBoxCategorialAgent(
            obs_size=obs_size,
            action_size=action_size,
            hidden_sizes=hidden_sizes,
            pi_hidden_sizes=pi_hidden_sizes,
        )
    else:
        raise NotImplementedError(
            f"No GymAgent for observation_space={env.observation_space}, "
            f"action_space={env.action_space}"
        )


def process_episode(episode: dict[str, Any]) -> dict[str, Any]:
    """Convert an episode from the temporary form used in collect() into a
    "durable" form used in GymAgent.forward().

    Mostly converts a bunch of lists of np.ndarrays to torch.Tensors.
    """
    agent_info_keys = episode["agent_infos"][0].keys()
    agent_infos = {
        k: force_concat(agent_i[k] for agent_i in episode["agent_infos"])
        for k in agent_info_keys
    }
    env_info_keys = episode["env_infos"][0].keys()
    env_infos = {
        k: force_concat(env_i[k] for env_i in episode["env_infos"])
        for k in env_info_keys
    }

    action_lls = torch.stack(
        [agent_i["action_ll"] for agent_i in episode["agent_infos"]]
    )
    return {
        "observations": torch.from_numpy(np.array(episode["observations"])),
        "env_infos": env_infos,
        "agent_infos": agent_infos,
        "actions": torch.from_numpy(np.array(episode["actions"])),
        "rewards": torch.from_numpy(np.array(episode["rewards"])),
        "terminated": any(episode["terminals"]),
        "action_lls": action_lls,
        "action_dists": episode["action_dists"],
        "any_actions_possible": torch.ones(len(action_lls), dtype=torch.bool),
    }


def collect(
    num_timesteps: int,
    envs: list["Env"],
    agent: GymAgent,
    best_action: bool = False,
    full_episodes_only: bool = True,
    max_episode_length: Optional[int] = None,
):
    """Minimal data collector for Gym(nasium) environments.

    Batches observations and actions to the agent, and performs no other
    optimizations.

    Attempts to collect num_timesteps total timesteps using some number of
    parallel environments. Will produce extra if num_timesteps is not a
    multiple of len(envs).

    If best_action is true, asks the agent to perform the most likely actions
    ("deterministic eval").
    If full_episodes_only is true, only returns full episodes, and potentially
    more than num_timesteps. Usually used with the previous argument for policy
    evaluation.

    Will avoid collecting more than max_episode_length timesteps per episode,
    if specified. Episodes truncated by max_episode_length will by used when
    full_episodes_only is set.
    """
    agent.train(mode=False)
    n_envs = len(envs)

    # Wrap around gym inconsistencies
    if len(envs[0].reset()) == 2:

        def reset_new(env):
            obs, extra = env.reset()
            return obs, extra

        reset = reset_new
    else:

        def reset_old(env):
            return env.reset(), {}

        reset = reset_old

    if len(envs[0].step(envs[0].action_space.sample())) == 4:

        def step_old(env, act):
            obs, reward, done, info = env.step(act)
            terminated = done
            if "TimeLimit.truncated" in info:
                terminated = False
            return obs, reward, terminated, info

        step = step_old
    else:

        def step_new(env, act):
            step_ret = env.step(act)
            # Skip truncated, we don't need it
            return step_ret[0], step_ret[1], step_ret[2], step_ret[4]

        step = step_new

    # Final return value
    episodes = []

    # Populated during an episode
    observations = [[] for _ in range(n_envs)]
    env_infos = [[] for _ in range(n_envs)]
    actions = [[] for _ in range(n_envs)]
    agent_infos = [[] for _ in range(n_envs)]
    rewards = [[] for _ in range(n_envs)]
    terminals = [[] for _ in range(n_envs)]

    for i, env in enumerate(envs):
        obs, e_info = reset(env)
        observations[i].append(obs)
        env_infos[i].append(e_info)
    reset_mask = np.ones(len(envs), dtype=np.bool_)
    with tqdm(desc="Collecting timesteps", total=num_timesteps) as pbar:
        timestep = 0
        end_now = False
        while not end_now:
            timestep += 1
            # run the agent
            acts, agent_i = agent.get_actions(
                np.asarray([obs[-1] for obs in observations]),
                best_action=best_action,
                reset_mask=reset_mask,
            )

            # step all the enivronments
            for i, act in enumerate(acts):
                obs, reward, term, info = step(envs[i], act)
                observations[i].append(obs)
                env_infos[i].append(info)
                actions[i].append(act)
                agent_infos[i].append({k: v[i] for k, v in agent_i.items()})
                rewards[i].append(reward)
                terminals[i].append(term)

            total_timesteps = n_envs * timestep

            # Check if we are allowed to end now
            end_now = (total_timesteps >= num_timesteps) and not full_episodes_only

            # handle episodes ending
            for i, terms in enumerate(terminals):
                if (
                    end_now
                    or terms[-1]
                    or (
                        max_episode_length is not None
                        and len(terms) >= max_episode_length
                    )
                ):
                    # Construct a single batch distribution
                    # covering the whole episode
                    action_dist_params = {}
                    for key in agent_infos[i][0].keys():
                        if key.startswith("action_params."):
                            k = key.split(".", 1)[1]
                            action_dist_params[k] = torch.stack(
                                [agent_i[key] for agent_i in agent_infos[i]]
                            )
                    action_dists, _ = agent.construct_dist(action_dist_params)

                    episodes.append(
                        process_episode(
                            {
                                "observations": observations[i],
                                "env_infos": env_infos[i],
                                "actions": actions[i],
                                "agent_infos": agent_infos[i],
                                "rewards": rewards[i],
                                "terminals": terminals[i],
                                "action_dists": action_dists,
                            }
                        )
                    )
                    observations[i] = []
                    env_infos[i] = []
                    actions[i] = []
                    agent_infos[i] = []
                    rewards[i] = []
                    terminals[i] = []

                    if not end_now:
                        reset_mask[i] = True
                        obs, e_info = reset(envs[i])
                        observations[i].append(obs)
                        env_infos[i].append(e_info)

            # Check if we are completely done
            if sum(len(ep["rewards"]) for ep in episodes) >= num_timesteps:
                end_now = True

            if full_episodes_only:
                pbar.n = sum(len(ep["rewards"]) for ep in episodes)
            else:
                pbar.n = total_timesteps
            pbar.refresh()
    return episodes


def episode_stats(episodes: list[dict[str, Any]]) -> dict[str, float]:
    """Compute evaluation stats from episodes returned by collect().

    Return value is typically passed to Trainer.add_eval_stats().

    If you're using a custom environment, you likely want to copy and modify
    this function to report important statistics about your environment besides
    just reward.
    """
    returns = [ep["rewards"].sum() for ep in episodes]
    terminations = [ep["terminated"] for ep in episodes]
    lengths = [len(ep["rewards"]) for ep in episodes]
    sampled_action_mean = [ep["actions"].float().mean() for ep in episodes]
    sampled_action_std = [ep["actions"].float().std() for ep in episodes]
    stats = {
        "episode_total_rewards": float(np.mean(returns)),
        "episode_termination_rate": float(np.mean(terminations)),
        "episode_length": float(np.mean(lengths)),
        "sampled_action_mean": sampled_action_mean,
        "sampled_action_std": sampled_action_std,
    }
    if "success" in episodes[0]["env_infos"]:
        stats["episode_success_rate"] = float(
            np.mean([float(np.any(ep["env_infos"]["success"])) for ep in episodes])
        )
    return stats
