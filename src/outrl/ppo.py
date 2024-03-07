import math
from typing import Any, Optional
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
    make_mlp,
    pack_tensors,
    pack_tensors_check,
    pad_tensors,
    unpad_tensors,
    explained_variance,
    unpack_tensors,
)
from outrl.gym_utils import collect, make_gym_actor
from outrl.stochastic_mlp_agent import StochasticMLPAgent


@dataclass(eq=False)
class ActorOutput:
    """Result from one episode when calling .forward() on an actor.

    Differentiating both observation_latents and action_lls should affect
    earlier layers in the actor.
    """

    observation_latents: torch.Tensor
    """Differentiable representations of the observations in the episode."""

    action_lls: torch.Tensor
    """Differentiable log-likelihood of actions taken in the episode."""


@dataclass(eq=False)
class EpisodeData:
    """Wrapper around an episode that maintains metadata and caches computed values."""

    episode: Any
    """The episode (treated as an opaque object)."""

    episode_number: int
    """Records with add_episode() call this episode came from."""

    num_timesteps: int
    """Number of steps in episode."""

    terminated: bool
    """Boolean indicating if the episode reached a terminal state. Always False
    in infinite horizon MDPs or when and episode reaches a timeout."""

    original_action_lls: torch.Tensor
    """Original action lls provided when the episode was added."""

    rewards: torch.Tensor
    """Rewards provided when the episode was added."""

    actions_possible: torch.Tensor
    """Boolean actions_possible mask."""

    sample_priority: float = 1.0
    """Priority for sampling episode. Decreased every train_step()."""

    def __lt__(self, other):
        return self.sort_key() < other.sort_key()

    def sort_key(self):
        """Retain highest sampling priority episodes, tie-break to prefer newer episodes."""
        return (self.sample_priority, self.episode_number)

    def __post_init__(self):
        assert self.original_action_lls.shape == (self.num_timesteps,)
        assert self.rewards.shape == (self.num_timesteps,)
        assert self.actions_possible.shape == (self.num_timesteps,)


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

        self.new_agent = make_gym_actor(self.envs[0], self.cfg.preprocess_hidden_sizes, self.cfg.pi_hidden_sizes)

        print('new_agent', self.new_agent)

        self.observation_latent_size = self.new_agent.observation_latent_size

        self.vf = make_mlp(
            input_size=self.observation_latent_size,
            hidden_sizes=self.cfg.vf_hidden_sizes,
            output_size=0,
            use_dropout=True,
        )
        vf_output = self.vf.get_submodule("output_linear")
        vf_output.weight.data.copy_(0.01 * vf_output.weight.data)

        self.reward_normalizer = RunningMeanVar()

        self.optimizer = torch.optim.Adam(
            list(self.vf.parameters()) + list(self.new_agent.parameters()),
            lr=self.cfg.learning_rate,
        )

        self._replay_buffer: list[EpisodeData] = []
        self.total_env_steps: int = 0
        self._next_episode_number: int = 0
        self._dtype = torch.float32

    def _loss_function(self, mb: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        advantages, adv_len = pack_tensors(mb["advantages"])
        advantages -= advantages.mean()
        advantages /= advantages.std()

        total_timesteps = sum(adv_len)

        lr_adjustment = total_timesteps / self.cfg.minibatch_size

        agent_outputs = self.new_agent(mb["episodes"])
        obs_latents_packed, obs_lens = pack_tensors([agent_out.observation_latents[:-1] for agent_out in agent_outputs])
        critic_out = self.vf(obs_latents_packed)
        action_ll_packed = pack_tensors_check([agent_out.action_lls for agent_out in agent_outputs], adv_len)
        # log_prob = -actor_out
        log_prob = action_ll_packed
        minibatch_size = sum(adv_len)
        assert log_prob.shape == (minibatch_size,)
        old_log_prob = pack_tensors_check(mb["original_action_lls"], adv_len)
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
        vf_loss = F.mse_loss(critic_out, disc_returns)

        self.log_training(
            mb,
            dict(
                clip_portion=(ratio_clipped != ratio).mean(dtype=torch.float32),
                mb_ev=explained_variance(critic_out, disc_returns),
            )
            | locals(),
        )

        loss = pi_loss + self.cfg.vf_loss_coeff * vf_loss
        return loss * lr_adjustment

    def train_step(self):
        packed_data, lengths = self.preprocess()
        packed_data["length"] = torch.as_tensor(lengths)
        dataset = DictDataset(packed_data)
        self.log_dataset(dataset)
        self.new_agent.train(mode=True)
        self.vf.train(mode=True)
        for mb in tqdm(
            dataset.episode_minibatches(self.cfg.minibatch_size, drop_last=False)
        ):
            loss = self._loss_function(mb)
            self.optimizer.zero_grad()
            loss.backward()
            try:
                clip_grad_norm_(
                    self.new_agent.parameters(), max_norm=10.0, error_if_nonfinite=True
                )
                clip_grad_norm_(
                    self.vf.parameters(), max_norm=10.0, error_if_nonfinite=True
                )
                self.optimizer.step()
            except RuntimeError as ex:
                # This seems to only trigger if the batch is so small the loss
                # is nan or so big we OOM
                warnings.warn(ex)
        self.new_agent.train(mode=False)
        self.vf.train(mode=False)

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
                    "ev": explained_variance(
                        vf_returns,
                        discounted_returns,
                    ),
                    "total_env_steps": self.total_env_steps,
                },
                level=stick.RESULTS,
            )

    def collect(self):
        self.logged_dataset = False
        self._replay_buffer = []
        with torch.no_grad():
            self.new_agent.train(mode=False)
            collect_res = collect(
                self.cfg.steps_per_epoch,
                self.envs,
                self.new_agent,
                max_episode_length=self.cfg.max_episode_length,
            )
            self.new_agent.train(mode=True)
        for episode in collect_res:
            self.add_episode(
                episode,
                rewards=episode["rewards"],
                action_lls=episode["action_lls"],
                terminated=episode["terminated"],
                actions_possible=episode["actions_possible"],
            )

    def preprocess(self):
        terminated = [data.terminated for data in self._replay_buffer]
        episode_lengths = [len(data.rewards) for data in self._replay_buffer]

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

        obs_lens = [len(data.episode["observations"]) for data in self._replay_buffer]
        padded_obs = pad_tensors(
            [data.episode["observations"] for data in self._replay_buffer]
        )

        # Compute vf_returns, filter based on episode termination
        self.new_agent.train(mode=False)
        self.vf.train(mode=False)
        with torch.no_grad():
            agent_outputs = self.new_agent([data.episode for data in self._replay_buffer])
            obs_latents_packed = pack_tensors_check([agent_out.observation_latents for agent_out in agent_outputs], obs_lens)
            vf_returns_packed = self.vf(obs_latents_packed)
        self.new_agent.train(mode=True)
        self.vf.train(mode=True)

        vf_returns = pad_tensors(unpack_tensors(vf_returns_packed, obs_lens))
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

        padded_rewards = pad_tensors([data.rewards for data in self._replay_buffer])
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

        padded_data = {
            "vf_returns": vf_returns[:, :-1],
            "advantages": advantages,
            "discounted_returns": discounted_returns,
            "valid_rewards": padded_rewards,
        }
        episode_data = {
            k: unpad_tensors(b, episode_lengths) for (k, b) in padded_data.items()
        }
        episode_data["original_action_lls"] = [
            data.original_action_lls for data in self._replay_buffer
        ]
        episode_data["rewards"] = [data.rewards for data in self._replay_buffer]
        episode_data["episodes"] = [data.episode for data in self._replay_buffer]
        episode_data["observations"] = [
            data.episode["observations"][:-1] for data in self._replay_buffer
        ]
        episode_data["actions"] = [
            data.episode["actions"] for data in self._replay_buffer
        ]
        return episode_data, episode_lengths

    def add_episode(
        self,
        episode: Any,
        rewards: torch.Tensor,
        action_lls: torch.Tensor,
        terminated: bool,
        actions_possible: Optional[torch.Tensor] = None,
        sample_priority: float = 1.0,
    ):
        """Add a new episode to the replay buffer.

        Arguments:
            rewards (torch.Tensor): Float Tensor containing rewards achieved
                after taking each action.
            terminated (bool): True if the episode ended in a terminal state,
                and False if the episode timed-out, was abandoned, or is still
                ongoing
            actions_possible (torch.Tensor?): Boolean Tensor indicating if any
                actions were possible to take at this state. This allows
                ignoring states where no action was possible (e.g. the initial
                prompt in an LLM, or intermediate frames in a video input when
                using frame-skip). Assumes always true if not provided.
            sample_priority (float): Optional initial sample priority weight.
                Will be adjusted by the learner.

        """
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(action_lls, torch.Tensor)
        assert not action_lls.requires_grad
        num_timesteps = len(rewards)
        self.total_env_steps += num_timesteps
        if actions_possible is None:
            actions_possible = torch.ones(num_timesteps, dtype=torch.bool)
        else:
            assert isinstance(actions_possible, torch.Tensor)

        rewards = rewards.to(dtype=self._dtype)
        self.reward_normalizer.update(rewards)

        self._replay_buffer.append(
            EpisodeData(
                episode,
                episode_number=self._next_episode_number,
                num_timesteps=num_timesteps,
                terminated=terminated,
                original_action_lls=action_lls.to(dtype=self._dtype),
                rewards=rewards,
                actions_possible=actions_possible,
                sample_priority=sample_priority,
            )
        )
        self._next_episode_number += 1


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

    import gymnasium as gym

    envs = [gym.make(cfg.env_name) for _ in range(cfg.n_envs)]
    model = PPO(cfg, envs)
    while True:
        model.collect()
        model.train_step()


if __name__ == "__main__":
    gym_ppo(get_config(GymConfig))
