import math
from typing import Optional
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataclasses import dataclass
from tqdm import tqdm

import stick

import outrl
from outrl.nn import compute_advantages, discount_cumsum
from outrl.utils import (
    get_config,
    to_yaml,
    save_yaml,
    Serializable,
    copy_default,
    RunningMeanVar,
)
from outrl.torch_utils import (
    DictDataset,
    pack_tensors,
    pack_tensors_check,
    pad_tensors,
    unpad_tensors,
)
from outrl.gym_utils import collect


@dataclass
class AlgoConfig(Serializable):
    seed: int = 0
    exp_name: Optional[str] = None
    log_dir: Optional[str] = None
    max_episode_length: Optional[int] = None


@dataclass
class PPOConfig(AlgoConfig):
    n_envs: int = 10
    steps_per_epoch: int = 4096
    preprocess_hidden_sizes: list[int] = copy_default([128])
    pi_hidden_sizes: list[int] = copy_default([64, 64])
    vf_hidden_sizes: list[int] = copy_default([64, 64])
    minibatch_size: int = 64
    learning_rate: float = 1e-3
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    fragment_length: int = 1
    epochs_per_policy_step: int = 10
    vf_loss_coeff: float = 0.1
    vf_clip_ratio: Optional[float] = None
    norm_rewards: bool = False
    norm_observations: bool = False


class PPO(nn.Module):
    def __init__(self, cfg: PPOConfig, envs):
        super().__init__()
        self.cfg = cfg
        self.envs = envs
        # self.sampler = outrl.Sampler(env_cons, self.cfg.n_envs)

        self.env_spec = self.envs[0]

        act_space = self.env_spec.action_space
        if hasattr(act_space, "low") and hasattr(act_space, "high"):
            act_shape = act_space.shape
            action_type = "continuous"
        elif hasattr(act_space, "n"):
            act_shape = (act_space.n,)
            action_type = "discrete"
        else:
            raise NotImplementedError(f"Unsupported action space type {act_space}")
        self.agent = outrl.StochasticMLPAgent(
            observation_shape=self.env_spec.observation_space.shape,
            action_shape=act_shape,
            hidden_sizes=self.cfg.preprocess_hidden_sizes,
            vf_hidden_sizes=self.cfg.vf_hidden_sizes,
            pi_hidden_sizes=self.cfg.pi_hidden_sizes,
            action_dist_cons=outrl.dists.DEFAULT_DIST_TYPES[action_type],
        )

        self.obs_normalizer = RunningMeanVar(
            mean=torch.zeros(self.env_spec.observation_space.shape),
            var=torch.ones(self.env_spec.observation_space.shape),
        )
        self.reward_normalizer = RunningMeanVar()

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(), lr=self.cfg.learning_rate
        )

    def _loss_function(self, mb: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        advantages, adv_len = pack_tensors(mb["advantages"])
        advantages -= advantages.mean()
        advantages /= advantages.std()

        total_timesteps = sum(adv_len)

        lr_adjustment = total_timesteps / self.cfg.minibatch_size

        actor_out, critic_out = self.agent.forward_both(
            mb["observations"], mb["actions"]
        )
        log_prob = -actor_out
        minibatch_size = sum(adv_len)
        assert log_prob.shape == (minibatch_size,)
        old_log_prob = -pack_tensors_check(mb["action_lls"], adv_len)
        assert old_log_prob.shape == (minibatch_size,)
        assert not old_log_prob.grad_fn

        ratio = torch.exp(log_prob - old_log_prob)
        ratio_clipped = torch.clamp(
            ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio
        )

        assert ratio.shape == (minibatch_size,)
        assert ratio_clipped.shape == (minibatch_size,)

        policy_gradient = ratio * advantages
        clip_policy_gradient = ratio_clipped * advantages
        assert policy_gradient.shape == (minibatch_size,)
        assert clip_policy_gradient.shape == (minibatch_size,)

        pi_loss = -torch.min(policy_gradient, clip_policy_gradient).mean()

        disc_returns = pack_tensors_check(mb["discounted_returns"], adv_len)
        if self.cfg.vf_clip_ratio:
            # This clipping logic matches the logic from Tianshou
            # But it isn't scale invariant, so it's probably wrong?
            # Maybe it's fine with reward scaling? But the return scale still
            # depends on the discount even with reward norm?
            # TODO(krzentner): Compare to paper: arXiv:1811.02553v3 Sec. 4.1
            diff = (critic_out - mb["vf_returns"]).clamp(
                -self.cfg.vf_clip_ratio, self.cfg.vf_clip_ratio
            )
            v_clip = mb["vf_returns"] + diff
            vf_loss_1 = (disc_returns - critic_out) ** 2
            vf_loss_2 = (disc_returns - v_clip) ** 2
            # Mean *after* taking max over each element
            vf_loss = torch.max(vf_loss_1, vf_loss_2).mean()
        else:
            vf_loss = F.mse_loss(critic_out, disc_returns)

        self.log_training(
            mb,
            dict(
                clip_portion=(ratio_clipped != ratio).mean(dtype=torch.float32),
                mb_ev=outrl.nn.explained_variance(critic_out, disc_returns),
            )
            | locals(),
        )

        loss = pi_loss + self.cfg.vf_loss_coeff * vf_loss
        return loss * lr_adjustment

    def train_step(self):
        packed_data, lengths = self.collect()
        packed_data["length"] = torch.as_tensor(lengths)
        dataset = DictDataset(packed_data)
        self.log_dataset(dataset)
        for mb in tqdm(
            dataset.episode_minibatches(self.cfg.minibatch_size, drop_last=False)
        ):
            loss = self._loss_function(mb)
            self.optimizer.zero_grad()
            loss.backward()
            try:
                clip_grad_norm_(
                    self.agent.parameters(), max_norm=10.0, error_if_nonfinite=True
                )
                self.optimizer.step()
            except RuntimeError as ex:
                # This seems to only trigger if the batch is so small the loss
                # is nan or so big we OOM
                warnings.warn(ex)

    def log_training(
        self, mb: dict[str, torch.Tensor], train_locals: dict[str, torch.Tensor]
    ):
        training_stats = {
            k: train_locals[k].item()
            for k in ["vf_loss", "pi_loss", "mb_ev", "clip_portion"]
        }

        training_stats["lr_adjustment"] = train_locals["lr_adjustment"]

        training_stats["average_timestep_reward"] = (
            pack_tensors(mb["rewards"])[0].mean().item()
        )
        training_stats["disc_mean"] = train_locals["disc_returns"].mean()
        training_stats["critic_out_mean"] = train_locals["critic_out"].mean()

        if not self.logged_dataset:
            self.logged_dataset = True
            stick.log("training_stats", training_stats, level=stick.RESULTS)
        else:
            stick.log("training_stats", training_stats, level=stick.INFO)

    def log_dataset(self, dataset: DictDataset):
        full_episode_rewards = pad_tensors([ep["rewards"] for ep in dataset])
        vf_returns = pack_tensors([ep["vf_returns"] for ep in dataset])[0]
        discounted_returns = pack_tensors([ep["discounted_returns"] for ep in dataset])[
            0
        ]
        if len(full_episode_rewards):
            stick.log(
                "dataset_stats",
                {
                    "ep_rew": full_episode_rewards.sum(dim=-1).mean(dim=0),
                    "disc_mean": discounted_returns.mean(),
                    "vf_mean": vf_returns.mean(),
                    "ev": outrl.nn.explained_variance(
                        vf_returns,
                        discounted_returns,
                    ),
                },
                level=stick.RESULTS,
            )

    def collect(self):
        self.logged_dataset = False
        with torch.no_grad():
            self._replay_buffer = collect(
                self.cfg.steps_per_epoch,
                self.envs,
                self.agent,
                max_episode_length=self.cfg.max_episode_length,
            )

        terminated = [ep["terminated"] for ep in self._replay_buffer]
        keys = ["actions", "rewards", "action_lls"]
        padded_data = {
            k: pad_tensors(
                [ep[k].to(dtype=torch.float32) for ep in self._replay_buffer]
            )
            for k in keys
        }
        episode_lengths = [len(ep["rewards"]) for ep in self._replay_buffer]
        if padded_data["actions"].var() == 0:
            warnings.warn("Action variance is zero")

        assert self.cfg.steps_per_epoch <= sum(episode_lengths)
        if self.cfg.max_episode_length:
            upper_bound = (
                self.cfg.n_envs
                * self.cfg.max_episode_length
                * math.ceil(
                    self.cfg.steps_per_epoch
                    / (self.cfg.n_envs * self.cfg.max_episode_length)
                )
            )
            assert upper_bound >= sum(episode_lengths)

        obs_lens = [len(ep["observations"]) for ep in self._replay_buffer]
        padded_obs = pad_tensors(
            [ep["observations"].to(dtype=torch.float32) for ep in self._replay_buffer]
        )
        padded_data["observations"] = padded_obs
        packed_obs = pack_tensors_check(unpad_tensors(padded_obs, obs_lens), obs_lens)

        self.obs_normalizer.update(packed_obs)
        if self.cfg.norm_observations:
            padded_obs = self.reward_normalizer.normalize_batch(padded_obs)

        # Compute vf_returns, filter based on episode termination
        with torch.no_grad():
            vf_returns = self.agent.vf_forward(padded_obs)
        # Can't use valids mask, since vf_returns goes to t + 1
        for i, episode_length in enumerate(episode_lengths):
            episode_length = episode_lengths[i]
            # Everything after last valid observation should be empty
            vf_returns[i, episode_length + 1 :] = 0.0
            # zero vf_{t+1} in terminated episodes
            if terminated[i]:
                vf_returns[i, episode_length] = 0.0
            else:
                assert vf_returns[i, episode_length] != 0

        padded_rewards = padded_data["rewards"]
        packed_rewards = pack_tensors_check(
            unpad_tensors(padded_rewards, episode_lengths), episode_lengths
        )
        self.reward_normalizer.update(packed_rewards)
        if self.cfg.norm_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)
        else:
            rewards_normed = padded_rewards

        advantages = compute_advantages(
            discount=self.cfg.discount,
            gae_lambda=self.cfg.gae_lambda,
            vf_returns=vf_returns,
            rewards=rewards_normed,
        )

        discounted_returns = discount_cumsum(padded_rewards, discount=self.cfg.discount)

        padded_data.update(
            {
                "vf_returns": vf_returns[:, :-1],
                "advantages": advantages,
                "discounted_returns": discounted_returns,
                "valid_rewards": padded_rewards,
            }
        )
        episode_data = {
            k: unpad_tensors(b, episode_lengths) for (k, b) in padded_data.items()
        }
        return episode_data, episode_lengths


@dataclass
class GymConfig(PPOConfig):
    env_name: str = "CartPole-v1"
    algorithm: str = "ppo"


def gym_ppo(cfg: GymConfig):
    config_yaml = to_yaml(cfg)
    print("CONFIG START")
    print(config_yaml)
    print("CONFIG END")

    if cfg.exp_name is None:
        cfg.exp_name = "gym_ppo"

    if cfg.log_dir is None:
        import hashlib

        hexdigest = hashlib.sha1(config_yaml.encode()).hexdigest()
        cfg.log_dir = f"outputs/{cfg.algorithm}/{hexdigest}"

    if cfg.max_episode_length is None:
        cfg.max_episode_length = 200

    import os

    os.makedirs(cfg.log_dir, exist_ok=True)
    save_yaml(cfg, f"{cfg.log_dir}/cfg.yaml")
    stick.init_extra(log_dir=cfg.log_dir, run_name=cfg.exp_name, config=cfg)

    stick.seed_all_imported_modules(cfg.seed)

    # Log RESULTS level and higher to stdout
    from stick.pprint_output import PPrintOutputEngine

    stick.add_output(PPrintOutputEngine("stdout"))

    import outrl.gym_env

    env_cons = outrl.gym_env.GymEnvCons(
        cfg.env_name, max_episode_length=cfg.max_episode_length
    )
    import gymnasium as gym

    envs = [gym.make(cfg.env_name) for _ in range(cfg.n_envs)]
    model = PPO(cfg, envs)
    while True:
        model.train_step()


if __name__ == "__main__":
    gym_ppo(get_config(GymConfig))
