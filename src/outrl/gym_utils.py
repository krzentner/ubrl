from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from outrl.rl import ActorOutput
from outrl.torch_utils import (
    concat,
    make_mlp,
    flatten_shape,
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
        pi_hidden_sizes: list[int],
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
                use_dropout=False,
            ),
        )

        self.pi_layers = make_mlp(
            input_size=hidden_sizes[-1], hidden_sizes=pi_hidden_sizes, use_dropout=False
        )

        self.dtype = torch.float32
        self.device = "cpu"
        self.observation_latent_size = hidden_sizes[-1]

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        raise NotImplementedError()

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> torch.distributions.Distribution:
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
            _, params, infos = self._run_net(
                torch.from_numpy(observations)
                .to(dtype=self.dtype)
                .to(device=self.device)
            )
        dist = self.construct_dist(params)
        action = maybe_sample(dist, best_action)
        action_ll = dist.log_prob(action)
        infos.update({f"action_params.{k}": v for (k, v) in params.items()})
        infos.update(
            {
                "action_ll": action_ll,
            }
        )
        return np.asarray(action), infos

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
        obs_latents, params, infos = self._run_net(observations)
        del infos

        batch_dist = self.construct_dist(params)
        obs_latents = unpack_tensors(obs_latents, pack_lens)
        packed_action_ll = batch_dist.log_prob(actions).squeeze(-1)
        unpacked_params = {k: unpack_tensors(v, pack_lens) for (k, v) in params.items()}
        dists = [
            self.construct_dist({k: v[i][:-1] for (k, v) in unpacked_params.items()})
            for i in range(len(episodes))
        ]
        action_lls = [
            act_lls[:-1] for act_lls in unpack_tensors(packed_action_ll, pack_lens)
        ]
        return [
            ActorOutput(
                observation_latents=obs_lat, action_lls=act_lls, action_dists=dist
            )
            for (obs_lat, act_lls, dist) in zip(obs_latents, action_lls, dists)
        ]


class GymBoxActor(GymActor):
    def __init__(
        self,
        env: "GymEnv",
        hidden_sizes: list[int],
        pi_hidden_sizes: list[int],
        init_std: float,
        min_std: float,
    ):
        super().__init__(
            env, hidden_sizes=hidden_sizes, pi_hidden_sizes=pi_hidden_sizes
        )

        self.action_mean = nn.Linear(
            pi_hidden_sizes[-1], flatten_shape(env.action_space.shape)
        )

        self.action_logstd = nn.Linear(
            pi_hidden_sizes[-1], flatten_shape(env.action_space.shape)
        )
        nn.init.constant_(self.action_logstd.bias, torch.tensor(init_std).log().item())
        nn.init.orthogonal_(
            self.action_logstd.weight,
            gain=torch.tensor(init_std).log().item()
        )
        nn.init.zeros_(self.action_logstd.weight)
        nn.init.zeros_(self.action_mean.bias)
        self.action_mean.weight.data.copy_(0.01 * self.action_mean.weight.data)
        self.min_std = min_std

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        observation_latents = self.shared_layers(obs)
        pi_x = self.pi_layers(observation_latents)
        mean = self.action_mean(pi_x)
        std_pre_clamp = self.action_logstd(pi_x).exp()
        std = torch.clamp(std_pre_clamp, min=self.min_std)
        params = dict(mean=mean, std=std)
        return (
            observation_latents,
            params,
            {
                # "action_mean": mean.mean(dim=-1),
                # "action_stddev": std.mean(dim=-1),
                "action_stddev_unclamped": std_pre_clamp.mean(dim=-1),
            },
        )

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> torch.distributions.Distribution:
        mean, std = params["mean"], params["std"]
        dist = torch.distributions.Normal(mean, std)
        if len(mean.shape) > 1:
            dist = torch.distributions.Independent(dist, 1)
        return dist


class GymBoxCategorialActor(GymActor):
    def __init__(
        self,
        env: "GymEnv",
        hidden_sizes: list[int],
        pi_hidden_sizes: list[int],
    ):
        super().__init__(
            env, hidden_sizes=hidden_sizes, pi_hidden_sizes=pi_hidden_sizes
        )

        self.action_logits = nn.Linear(pi_hidden_sizes[-1], env.action_space.n)

    def _run_net(
        self, obs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, Any]]:
        observation_latents = self.shared_layers(obs)
        pi_x = self.pi_layers(observation_latents)
        action_logits = self.action_logits(pi_x)
        params = dict(logits=action_logits)
        dist = self.construct_dist(params)
        infos = {
            f"action_{i}_prob": prob
            for (i, prob) in enumerate(dist.probs.transpose(0, 1))
        }
        assert len(infos["action_0_prob"]) == len(dist.probs)
        return observation_latents, params, infos

    def construct_dist(
        self, params: dict[str, torch.Tensor]
    ) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=params["logits"])


def make_gym_actor(env, hidden_sizes, pi_hidden_sizes, init_std: float = 0.5, min_std: float = 1e-6):
    while isinstance(env, list):
        env = env[0]
    act_space = type(env.action_space).__name__
    obs_space = type(env.observation_space).__name__

    if obs_space == "Box" and act_space == "Box":
        return GymBoxActor(env, hidden_sizes, pi_hidden_sizes, init_std=init_std, min_std=min_std)
    elif obs_space == "Box" and act_space == "Discrete":
        return GymBoxCategorialActor(env, hidden_sizes, pi_hidden_sizes)
    else:
        raise NotImplementedError(
            f"No GymActor for observation_space={env.observation_space}, "
            f"action_space={env.action_space}"
        )


def process_episode(episode: dict[str, Any]) -> dict[str, Any]:
    agent_info_keys = episode["agent_infos"][0].keys()
    agent_infos = {
        k: concat(agent_i[k] for agent_i in episode["agent_infos"])
        for k in agent_info_keys
    }
    env_info_keys = episode["env_infos"][0].keys()
    env_infos = {
        k: concat(env_i[k] for env_i in episode["env_infos"]) for k in env_info_keys
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
    envs,
    actor,
    best_action: bool = False,
    full_episodes_only: bool = True,
    max_episode_length: Optional[int] = None,
):
    actor.train(mode=False)
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
                    # Construct a single batch distribution
                    # covering the whole episode
                    action_dist_params = {}
                    for key in agent_infos[i][0].keys():
                        if key.startswith("action_params."):
                            k = key.split(".", 1)[1]
                            action_dist_params[k] = torch.stack(
                                [agent_i[key] for agent_i in agent_infos[i]]
                            )
                    action_dists = actor.construct_dist(action_dist_params)

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
    returns = [ep["rewards"].sum() for ep in episodes]
    terminations = [ep["terminated"] for ep in episodes]
    lengths = [len(ep["rewards"]) for ep in episodes]
    return {
        "AverageReturn": float(np.mean(returns)),
        "TerminationRate": float(np.mean(terminations)),
        "AverageEpisodeLengths": float(np.mean(lengths)),
    }
