from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from outrl.rl import ActorOutput
from outrl.torch_utils import (
    make_mlp,
    flatten_shape,
    as_2d,
    maybe_sample,
    RunningMeanVar,
    pack_tensors,
    unpack_tensors,
)


class GymActor(nn.Module):
    def __init__(
        self,
        env: "GymEnv",
        hidden_sizes: list[int],
    ):
        super().__init__()

        obs_dim = flatten_shape(env.observation_space.shape)
        self.shared_layers = nn.Sequential(
            RunningMeanVar(
                init_mean=torch.zeros(obs_dim), init_var=torch.ones(obs_dim)
            ),
            make_mlp(
                input_size=obs_dim,
                hidden_sizes=hidden_sizes,
            ),
        )

        self.dtype = torch.float32
        self.device = "cpu"
        self.observation_latent_size = hidden_sizes[-1]

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        raise NotImplementedError()

    def get_actions(
        self,
        observations: np.ndarray,
        reset_mask: np.ndarray,
        best_action: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del reset_mask
        assert len(observations.shape) == 2
        with torch.no_grad():
            dist = self._run_net(
                torch.from_numpy(observations)
                .to(dtype=self.dtype)
                .to(device=self.device)
            )[1]
        action = maybe_sample(dist, best_action)
        action_ll = dist.log_prob(action)
        action_mean = dist.mean
        action_stddev = dist.stddev
        return np.asarray(action), {
            "action_ll": action_ll,
            "action_mean": action_mean,
            "action_stddev": action_stddev,
        }

    def forward(self, episodes: list[dict[str, torch.Tensor]]) -> list[ActorOutput]:
        observations, pack_lens = pack_tensors([ep["observations"] for ep in episodes])
        observations = observations.to(dtype=self.dtype).to(device=self.device)
        # Add a trailing fake action so each observation has an action afterwards
        actions, action_pack_lens = pack_tensors(
            [
                torch.cat([ep["actions"], ep["actions"][-1].unsqueeze(0)])
                for ep in episodes
            ]
        )
        actions = actions.to(dtype=self.dtype).to(device=self.device)
        assert action_pack_lens == pack_lens
        observation_latents, dist = self._run_net(observations)
        observation_latents = unpack_tensors(observation_latents, pack_lens)
        packed_action_ll = dist.log_prob(actions).squeeze(-1)
        action_lls = [
            act_lls[:-1] for act_lls in unpack_tensors(packed_action_ll, pack_lens)
        ]
        return [
            ActorOutput(
                observation_latents=obs_latents,
                action_lls=act_lls,
            )
            for (obs_latents, act_lls) in zip(observation_latents, action_lls)
        ]


class GymBoxActor(GymActor):
    def __init__(
        self,
        env: "GymEnv",
        hidden_sizes: list[int],
        init_std: float = 0.5,
        min_std: float = 1e-6,
    ):
        super().__init__(env, hidden_sizes=hidden_sizes)

        self.action_mean = nn.Linear(
            hidden_sizes[-1], flatten_shape(env.action_space.shape)
        )

        self.action_logstd = nn.Linear(
            hidden_sizes[-1], flatten_shape(env.action_space.shape)
        )
        nn.init.constant_(self.action_logstd.bias, torch.tensor(init_std).log().item())
        self.min_std = min_std

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        observation_latents = self.shared_layers(as_2d(obs))
        mean = self.action_mean(observation_latents)
        std = torch.clamp(
            self.action_logstd(observation_latents).exp(), min=self.min_std
        )
        dist = torch.distributions.Normal(mean, std)
        return observation_latents, dist


class GymBoxCategorialActor(GymActor):
    def __init__(
        self,
        env: "GymEnv",
        hidden_sizes: list[int],
    ):
        super().__init__(env, hidden_sizes=hidden_sizes)

        self.action_logits = nn.Linear(
            hidden_sizes[-1], flatten_shape(env.action_space.shape)
        )

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.Distribution]:
        observation_latents = self.shared_layers(as_2d(obs))
        action_logits = self.action_logits(observation_latents)
        dist = torch.distributions.Categorical(logits=action_logits)
        return observation_latents, dist


def make_gym_actor(env, hidden_sizes, **kwargs):
    while isinstance(env, list):
        env = env[0]
    act_space = type(env.action_space).__name__
    obs_space = type(env.observation_space).__name__

    if obs_space == "Box" and act_space == "Box":
        return GymBoxActor(env, hidden_sizes, **kwargs)
    elif obs_space == "Box" and act_space == "Discrete":
        return GymBoxCategorialActor(env, hidden_sizes, **kwargs)
    else:
        raise NotImplementedError(
            f"No GymActor for observation_space={env.observation_space}, "
            f"action_space={env.action_space}"
        )


def process_episode(episode: dict[str, Any]) -> dict[str, Any]:
    action_lls = torch.cat([agent_i["action_ll"] for agent_i in episode["agent_infos"]])
    return {
        "observations": torch.from_numpy(np.array(episode["observations"])),
        "env_infos": episode["env_infos"],
        "agent_infos": episode["agent_infos"],
        "actions": torch.from_numpy(np.array(episode["actions"])),
        "rewards": torch.from_numpy(np.array(episode["rewards"])),
        "terminated": any(episode["terminals"]),
        "action_lls": action_lls,
        "actions_possible": torch.ones(len(action_lls), dtype=torch.bool),
    }


def collect(
    num_timesteps: int,
    envs,
    actor,
    best_action: bool = False,
    full_episodes_only: bool = True,
    max_episode_length: Optional[int] = None,
):
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
            acts, agent_i = actor.get_actions(
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
                    episodes.append(
                        process_episode(
                            {
                                "observations": observations[i],
                                "env_infos": env_infos[i],
                                "actions": actions[i],
                                "agent_infos": agent_infos[i],
                                "rewards": rewards[i],
                                "terminals": terminals[i],
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
    returns = [ep["rewards"].sum() for ep in episodes]
    terminations = [ep["terminated"] for ep in episodes]
    lengths = [len(ep["rewards"]) for ep in episodes]
    return {
        "AverageReturn": float(np.mean(returns)),
        "TerminationRate": float(np.mean(terminations)),
        "AverageEpisodeLengths": float(np.mean(lengths)),
    }