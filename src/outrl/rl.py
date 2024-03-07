from dataclasses import dataclass, replace
from textwrap import dedent
from typing import Any, Optional
import copy
import os
import random
import sys
import warnings
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
)

import stick
from stick import log

from outrl.torch_utils import (
    DictDataset,
    concat,
    make_mlp,
    pack_tensors,
    pack_tensors_check,
    pad_tensors,
    unpad_tensors,
    explained_variance,
    unpack_tensors,
    RunningMeanVar,
)
from outrl.config import Config, tunable, IntListDistribution, default_run_name


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

    infos: dict[str, Any]
    """User infos. Will be logged if possible."""

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
class TrainerConfig(Config):
    """Config structure for the Trainer.

    Can be saved to / loaded from yaml.
    All fields have default values.
    Most fields can be tuned via optuna.

    Can be subclassed to add more options.
    """

    seed: int = random.randrange(10000)
    """Random seed for this experiment. Set to a random int < 10k if not specified.

    Set to -1 to disable setting the random seed.
    """

    run_name: str = default_run_name()
    """The name of this trainer run.

    Generated uniquely from the name of the main module and time by default."""

    log_dir: str = "runs"
    """Directory to log into. Full path will be {log_dir}/{run_name}.

    Set to None to disable logging."""

    pprint_logging: bool = True
    """Log to stdout using pprint. Because the pprint output engine defaults to
    the RESULTS log level, this defaults to True."""

    parquet_logging: bool = False
    """Log to parquet files using pyarrow."""

    tb_log_level: str = "INFO"
    """Log level to log to TensorBoard. Defaults to INFO to avoid slowing down
    TensorBoard with too many keys."""

    max_buffer_episodes: int = 128
    """Maximum number of episodes to keep in replay buffer."""

    episode_priority_decay: float = tunable(0.5, FloatDistribution(0.0, 1.0))
    """Probability to sample an episode from a prior actor version.

    Set to 0 to ensure sampling is completely on-policy.
    """

    max_timesteps_per_forward: Optional[int] = None
    """Maximum number of timesteps to include in a forward pass to the actor.
    Usually used to avoid out-of-memory errors.

    Defaults to no limit. Automatically decreases on (most) memory errors.
    """

    minibatch_size: int = tunable(64, IntDistribution(1, 50000, log=True))

    policy_epochs_per_train_step: int = tunable(3, IntDistribution(1, 100, log=True))
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    normalize_rewards: bool = tunable(True, CategoricalDistribution([True, False]))
    """Normalize rewards to have zero mean and unit variance."""

    normalize_minibatch_advantages: bool = tunable(
        False, CategoricalDistribution([True, False])
    )
    """Normalize advantages of each minibatch."""

    # TODO: Add cosine schedule, etc.
    # TODO: Implement policy LR scheduling.
    actor_lr_schedule: str = tunable(
        "linear", CategoricalDistribution([None, "linear"])
    )
    actor_lr_start: float = tunable(1e-3, FloatDistribution(1e-6, 1e-2, log=True))
    actor_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-3, log=True))
    actor_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))
    actor_grad_max_norm: float = tunable(1.0, FloatDistribution(0.01, 10.0, log=True))
    actor_clip_ratio: float = tunable(0.2, FloatDistribution(1e-3, 10.0, log=True))

    # TODO: Implement VF LR scheduling.
    vf_lr_schedule: str = tunable("linear", CategoricalDistribution([None, "linear"]))
    vf_lr_start: float = tunable(3e-4, FloatDistribution(1e-6, 1e-2, log=True))
    vf_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-4, log=True))
    vf_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))
    vf_grad_max_norm: float = tunable(1.0, FloatDistribution(0.01, 10.0, log=True))
    vf_minibatch_size: int = tunable(512, IntDistribution(16, 10000))
    vf_training_epochs: int = tunable(5, IntDistribution(1, 100))

    vf_recompute_targets: bool = tunable(False, CategoricalDistribution([True, False]))

    vf_loss_coeff: float = tunable(0.1, FloatDistribution(0.01, 2.0))

    vf_hidden_sizes: list[int] = tunable(
        [128, 128],
        IntListDistribution(
            [
                16,
            ],
            [256, 256, 256],
        ),
    )

    discount_inv: float = tunable(0.01, FloatDistribution(1e-5, 0.1, log=True))
    """Discount, expresseed such that gamma = 1 - discount_inv."""

    v_trace_lambda: float = tunable(0.95, FloatDistribution(0.0, 1.0))
    v_trace_rho_max: float = tunable(1.0, FloatDistribution(1.0, 1e3, log=True))
    v_trace_c_max: float = tunable(1.0, FloatDistribution(1.0, 1e3, log=True))

    initial_temperature: float = tunable(10.0, FloatDistribution(1e-2, 1e3, log=True))
    temperature_lr: float = tunable(0.01, FloatDistribution(1e-4, 1.0, log=True))
    temperature_min: float = tunable(0.01, FloatDistribution(0.0, 1.0))
    temperature_max: float = tunable(1e5, FloatDistribution(0.1, 1e10, log=True))

    initial_kl_coef: float = tunable(10.0, FloatDistribution(0.0, 100.0))
    kl_coef_lr: float = tunable(0.01, FloatDistribution(1e-4, 1.0, log=True))
    kl_coef_min: float = tunable(0.01, FloatDistribution(0.0, 1.0))
    kl_coef_max: float = tunable(1e5, FloatDistribution(0.1, 1e10, log=True))

    entropy_target: float = tunable(-10.0, FloatDistribution(-100.0, 0.0))
    kl_soft_target: float = tunable(0.25, FloatDistribution(1e-3, 10.0, log=True))
    kl_fixup_coef: float = tunable(3, FloatDistribution(1.0, 20.0, log=True))

    use_top_half_advantages: bool = tunable(
        False, CategoricalDistribution([True, False])
    )

    def fill_defaults(self):
        """Fill in values with non-constant defaults. Called after construction."""
        if self.seed < -1:
            raise ValueError("seed should be positive or exactly -1")
        log_dir = os.path.abspath(self.log_dir)
        return replace(self, log_dir=log_dir)


class Trainer(nn.Module):
    def __init__(self, cfg: TrainerConfig, agent):
        super().__init__()
        self.cfg = cfg

        self.agent = agent

        self.observation_latent_size = self.agent.observation_latent_size

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
            list(self.vf.parameters()) + list(self.agent.parameters()),
            lr=self.cfg.actor_lr_start,
        )

        self._replay_buffer: list[EpisodeData] = []
        self.total_env_steps: int = 0
        self._next_episode_number: int = 0
        self._dtype = torch.float32

        self.last_training_stats = {}

    def _loss_function(self, mb: dict[str, list[torch.Tensor]]) -> torch.Tensor:
        advantages, adv_len = pack_tensors(mb["advantages"])
        if self.cfg.normalize_minibatch_advantages:
            advantages -= advantages.mean()
            advantages /= advantages.std()

        this_minibatch_size = sum(adv_len)

        lr_adjustment = this_minibatch_size / self.cfg.minibatch_size

        agent_outputs = self.agent(mb["episodes"])
        obs_latents_packed = pack_tensors_check(
            [agent_out.observation_latents[:-1] for agent_out in agent_outputs], adv_len
        )
        critic_out = self.vf(obs_latents_packed)
        action_ll_packed = pack_tensors_check(
            [agent_out.action_lls for agent_out in agent_outputs], adv_len
        )
        log_prob = action_ll_packed
        minibatch_size = sum(adv_len)
        assert log_prob.shape == (minibatch_size,)
        old_log_prob = pack_tensors_check(mb["original_action_lls"], adv_len)
        assert old_log_prob.shape == (minibatch_size,)
        assert not old_log_prob.grad_fn

        ratio = torch.exp(log_prob - old_log_prob)
        ratio_clipped = torch.clamp(
            ratio, 1 / (1 + self.cfg.actor_clip_ratio), 1 + self.cfg.actor_clip_ratio
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
        self.agent.train(mode=True)
        self.vf.train(mode=True)
        with tqdm(
            desc="train_step",
            total=packed_data["length"].sum().item()
            * self.cfg.policy_epochs_per_train_step,
        ) as pbar:
            for _ in range(self.cfg.policy_epochs_per_train_step):
                for mb in dataset.episode_minibatches(
                    self.cfg.minibatch_size, drop_last=False
                ):
                    loss = self._loss_function(mb)
                    self.optimizer.zero_grad()
                    loss.backward()
                    try:
                        clip_grad_norm_(
                            self.agent.parameters(),
                            max_norm=10.0,
                            error_if_nonfinite=True,
                        )
                        clip_grad_norm_(
                            self.vf.parameters(), max_norm=10.0, error_if_nonfinite=True
                        )
                        self.optimizer.step()
                    except RuntimeError as ex:
                        # This seems to only trigger if the batch is so small the loss
                        # is nan or so big we OOM
                        warnings.warn(ex)
                    pbar.n += mb["length"].sum().item()
                    pbar.refresh()
        self.agent.train(mode=False)
        self.vf.train(mode=False)
        stick.log("last_training_stats", self.last_training_stats, level=stick.RESULTS)
        self._replay_buffer = []

    def log_training(
        self, mb: dict[str, torch.Tensor], train_locals: dict[str, torch.Tensor]
    ):
        training_stats = {
            k: train_locals[k].item()
            for k in ["vf_loss", "pi_loss", "mb_ev", "clip_portion"]
        }

        training_stats["lr_adjustment"] = train_locals["lr_adjustment"]
        training_stats["ratio"] = train_locals["ratio"]

        training_stats["average_timestep_reward"] = (
            pack_tensors(mb["rewards"])[0].mean().item()
        )
        training_stats["disc_mean"] = train_locals["disc_returns"].mean()
        training_stats["critic_out_mean"] = train_locals["critic_out"].mean()

        self.last_training_stats = training_stats
        stick.log("training_stats", training_stats, level=stick.INFO)

    def log_dataset(self, dataset: DictDataset):
        full_episode_rewards = pad_tensors([ep["rewards"] for ep in dataset])
        vf_returns = pack_tensors([ep["vf_returns"] for ep in dataset])[0]
        discounted_returns = pack_tensors([ep["discounted_returns"] for ep in dataset])[
            0
        ]
        if len(full_episode_rewards):
            dataset_stats = {
                "ep_rew": full_episode_rewards.sum(dim=-1).mean(dim=0),
                "disc_mean": discounted_returns.mean(),
                "vf_mean": vf_returns.mean(),
                "ev": explained_variance(
                    vf_returns,
                    discounted_returns,
                ),
                "total_env_steps": self.total_env_steps,
            }
            for k in self._replay_buffer[0].infos.keys():
                dataset_stats[k] = concat(data.infos[k] for data in self._replay_buffer)

            stick.log(
                "dataset_stats",
                dataset_stats,
                level=stick.RESULTS,
                step=self.total_env_steps,
            )

    def preprocess(self):
        terminated = [data.terminated for data in self._replay_buffer]
        episode_lengths = [len(data.rewards) for data in self._replay_buffer]

        obs_lens = [len(data.episode["observations"]) for data in self._replay_buffer]

        # Compute vf_returns, filter based on episode termination
        self.agent.train(mode=False)
        self.vf.train(mode=False)
        with torch.no_grad():
            agent_outputs = self.agent([data.episode for data in self._replay_buffer])
            obs_latents_packed = pack_tensors_check(
                [agent_out.observation_latents for agent_out in agent_outputs], obs_lens
            )
            vf_returns_packed = self.vf(obs_latents_packed)
        self.agent.train(mode=True)
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
        if self.cfg.normalize_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)
        else:
            rewards_normed = padded_rewards

        discount = 1 - self.cfg.discount_inv
        advantages = compute_advantages(
            discount=discount,
            gae_lambda=self.cfg.v_trace_lambda,
            vf_returns=vf_returns,
            rewards=rewards_normed,
        )

        discounted_returns = discount_cumsum(padded_rewards, discount=discount)

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
        infos: Optional[dict[str, Any]] = None,
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
        if infos is None:
            infos = {}

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
                infos=infos,
            )
        )
        self._next_episode_number += 1

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        logging.info(f"Eval primary stat ({primary}):", stats[primary])
        log("eval_stats", stats, step=self.total_env_steps, level=stick.RESULTS)
        hparams = self.cfg.to_dict()
        hparams["metric-primary"] = stats[primary]
        for k, v in stats.items():
            hparams[k] = v
        log("hparams", hparams, step=self.total_env_steps)


@torch.jit.script
def compute_advantages(
    *,
    discount: float,
    gae_lambda: float,
    vf_returns: torch.Tensor,
    rewards: torch.Tensor,
):
    """Calculate advantages.

    Advantages are a discounted cumulative sum.

    Calculate advantages using a baseline according to Generalized Advantage
    Estimation (GAE)

    The discounted cumulative sum can be computed using conv2d with filter.
    filter:
        [1, (discount * gae_lambda), (discount * gae_lambda) ^ 2, ...]
        where the length is same with max_episode_length.

    expected_returns and rewards should have the same shape.
        expected_returns:
        [ [b_11, b_12, b_13, ... b_1n],
          [b_21, b_22, b_23, ... b_2n],
          ...
          [b_m1, b_m2, b_m3, ... b_mn] ]
        rewards:
        [ [r_11, r_12, r_13, ... r_1n],
          [r_21, r_22, r_23, ... r_2n],
          ...
          [r_m1, r_m2, r_m3, ... r_mn] ]

    Args:
        discount (float): RL discount factor (i.e. gamma).
        gae_lambda (float): Lambda, as used for Generalized Advantage
            Estimation (GAE).
        vf_returns (torch.Tensor): A 2D tensor of value function
            estimates with shape (N, T + 1), where N is the batch dimension
            (number of episodes) and T is the maximum episode length
            experienced by the agent. If an episode terminates in fewer than T
            time steps, the remaining elements in that episode should be set to
            0.
        rewards (torch.Tensor): A 2D tensor of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        episode_lengths (torch.Tensor): A 1D vector of episode lengths.
    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining values
            in that episode should be set to 0.

    """
    n_episodes = rewards.shape[0]
    max_episode_length = rewards.shape[1]
    assert vf_returns.shape == (n_episodes, max_episode_length + 1)

    delta = -vf_returns[:, :-1] + rewards + discount * vf_returns[:, 1:]
    adv_gae = torch.zeros((n_episodes, max_episode_length))
    adv_gae[:, max_episode_length - 1] = delta[:, max_episode_length - 1]
    for t in range(max_episode_length - 2, 0, -1):
        adv_gae[:, t] = delta[:, t] + discount * gae_lambda * adv_gae[:, t + 1]
    return adv_gae.squeeze()


def discount_cumsum(x: torch.Tensor, discount: float):
    B, L = x.shape
    discount_x = discount * torch.ones_like(x[0])
    discount_x[0] = 1.0
    # Compute discount weights.
    weights = torch.cumprod(discount_x, dim=0)
    # Add channel in dimensions and channel out dimensions
    weights = weights.reshape(1, 1, L)
    x = x.reshape(B, 1, L)
    # Add pad end of episodes to zero
    # Only need 2l - 1 timesteps to make index L valid
    x_pad = torch.cat([x, torch.zeros_like(x[:, :, :-1])], axis=-1)
    returns = F.conv1d(x_pad, weights, stride=1)
    assert returns.shape == (B, 1, L)
    return returns.squeeze()
