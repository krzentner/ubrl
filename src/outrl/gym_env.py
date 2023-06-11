from dataclasses import dataclass
from typing import Any, Callable, Optional
from functools import cached_property
import numpy as np
import torch

import outrl


@dataclass
class GymEnvCons:

    env_name: str
    # If not provided here, env.max_path_length must be defined
    max_episode_length: Optional[int] = None
    make_env: Optional[Callable[[str], "gym.Env"]] = None

    def __call__(self, step_batch: int):
        make_env = self.make_env
        if make_env is None:
            import gym

            make_env = gym.make
        return GymEnv(
            env_name=self.env_name,
            make_env=make_env,
            n_envs=step_batch,
            max_episode_length=self.max_episode_length,
        )


class GymEnv(outrl.env.VecEnv):
    def __init__(
        self,
        env_name: str,
        make_env: Callable[[str], Any],
        n_envs: int,
        max_episode_length: Optional[int] = None,
    ):
        self.envs = [make_env(env_name) for _ in range(n_envs)]
        self.step_counts = [0 for _ in range(n_envs)]
        if max_episode_length is None:
            max_episode_length = self.envs[0].max_path_length
        assert max_episode_length
        self.max_episode_length: int = max_episode_length

    @property
    def n_envs(self):
        return len(self.envs)

    @cached_property
    def spec(self) -> outrl.env.EnvSpec:
        act_space = self.envs[0].action_space
        if hasattr(act_space, "low") and hasattr(act_space, "high"):
            act_low = torch.from_numpy(act_space.low)
            act_high = torch.from_numpy(act_space.high)
            action_type = "continuous"
        elif hasattr(act_space, "n"):
            act_high = torch.zeros(act_space.n)
            act_low = torch.ones(act_space.n)
            action_type = "discrete"
        else:
            raise NotImplementedError(f"Unsupported action space type {act_space}")
        obs_space = self.envs[0].observation_space
        if hasattr(obs_space, "low") and hasattr(obs_space, "high"):
            obs_low = torch.from_numpy(obs_space.low)
            obs_high = torch.from_numpy(obs_space.high)
        elif hasattr(obs_space, "n"):
            obs_high = torch.zeros(obs_space.n)
            obs_low = torch.ones(obs_space.n)
        else:
            raise NotImplementedError(f"Unsupported observation space type {obs_space}")
        return outrl.env.EnvSpec(
            observation_low=obs_low,
            observation_high=obs_high,
            action_low=act_low,
            action_high=act_high,
            action_type=action_type,
            max_episode_length=self.max_episode_length,
        )

    def reset(
        self, prev_step: Optional[outrl.env.Step], mask: torch.Tensor
    ) -> outrl.env.Step:
        assert mask.shape == (len(self.envs),)
        if not mask.any():
            assert prev_step
            return prev_step
        spec = self.spec
        obs = torch.zeros(
            self.n_envs, *spec.observation_shape, dtype=spec.observation_dtype
        )
        infos = []
        terminal = torch.zeros(self.n_envs, dtype=torch.bool)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool)
        rewards = torch.zeros(self.n_envs, dtype=torch.float32)
        for i, env in enumerate(self.envs):
            should_reset = mask[i]
            if should_reset:
                first_obs = env.reset()
                if isinstance(first_obs, tuple) and len(first_obs) == 2:
                    # Probably gymnasium
                    first_obs, info = first_obs
                else:
                    info = {}
                obs[i] = torch.from_numpy(first_obs)
                infos.append(outrl.as_torch_dict(info))
                terminal[i] = False
                truncated[i] = False
                rewards[i] = 0
                self.step_counts[i] = 0
            else:
                # If mask is False anywhere, there must be a previous step
                assert prev_step
                obs[i] = prev_step.observations[i]
                infos.append({k: v[i] for (k, v) in prev_step.infos})
                terminal[i] = prev_step.terminated[i]
                truncated[i] = prev_step.truncated[i]
                rewards[i] = prev_step.rewards[i]

        return outrl.env.Step(
            observations=obs,
            rewards=rewards,
            terminated=terminal,
            truncated=truncated,
            env_state=None,
            infos=outrl.stack_dicts(infos),
        )

    def step(
        self, prev_step: outrl.env.Step, actions: torch.Tensor, mask: torch.Tensor
    ) -> outrl.env.Step:
        spec = self.spec
        obs = torch.zeros(
            self.n_envs, *spec.observation_shape, dtype=spec.observation_dtype
        )
        infos = []
        terminal = torch.zeros(self.n_envs, dtype=torch.bool)
        truncated = torch.zeros(self.n_envs, dtype=torch.bool)
        rewards = torch.zeros(self.n_envs, dtype=torch.float32)

        for i, env in enumerate(self.envs):
            if not mask[i]:
                infos.append({k: v[i] for (k, v) in prev_step.infos.items()})
                continue
            self.step_counts[i] += 1
            step_res = env.step(actions[i].detach().cpu().numpy())
            if len(step_res) == 4:
                observation, reward, done, info = step_res
                trunc = info.get("TimeLimit.truncated", False)
                terminated = done and not trunc
            elif len(step_res) == 6:
                observation, reward, terminated, trunc, info, done = step_res
                del done
            else:
                raise ValueError("Unexpected size of step result tuple")
            if self.step_counts[i] >= self.max_episode_length:
                trunc = True
            obs[i] = torch.from_numpy(observation)
            rewards[i] = reward
            truncated[i] = trunc
            terminal[i] = terminated
            infos.append(info)

        return outrl.env.Step(
            observations=obs,
            rewards=rewards,
            terminated=terminal,
            truncated=truncated,
            env_state=None,
            infos=outrl.stack_dicts(infos),
        )


def test_smoke():
    B = 2
    cons = GymEnvCons("CartPole-v0", max_episode_length=200)
    env = cons(B)
    step = env.reset(None, mask=torch.ones(B, dtype=bool))
    for _ in range(10):
        actions = torch.randint(2, size=(B,))
        step = env.step(step, actions, torch.ones(B, dtype=bool))
        step = env.reset(step, mask=step.done)
