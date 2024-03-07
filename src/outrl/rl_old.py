"""All Reinforcement Learning related functionality."""
# pylint: ignore=invalid-name,import-error

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
from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
)
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

import stick
from stick import log
from outrl.config import Config, tunable, IntListDistribution, default_run_name
from outrl.torch_utils import (
    average_modules,
    explained_variance,
    make_mlp,
    RunningMeanVar,
    pack_tensors,
    pack_tensors_check,
    split_shuffled_indices,
    unpack_tensors,
    pad_tensors,
    unpad_tensors,
    DictDataset,
    approx_kl_div,
)


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

    parquet_logging: bool = True
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

    policy_epochs_per_train_step: int = 10
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    # TODO: Add cosine schedule, etc.
    # TODO: Implement policy LR scheduling.
    actor_lr_schedule: str = tunable(
        "linear", CategoricalDistribution([None, "linear"])
    )
    actor_lr_start: float = tunable(1e-4, FloatDistribution(1e-6, 1e-2, log=True))
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

    discount_inv: float = tunable(0.1, FloatDistribution(1e-5, 0.1, log=True))
    """Discount, expresseed such that gamma = 1 - discount_inv."""

    v_trace_lambda: float = tunable(0.9, FloatDistribution(0.0, 1.0))
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
    """This is the Trainer, the primary entrypoint into OutRL.

    The Trainer implements an offline-ish RL algorithm similar to V-MPO, with
    many hyper-parameters in TrainerConfig.

    The Trainer does not implement data collection (i.e. "sampling"), instead
    episodes, which are treated as opaque objects, should be provided via
    add_episode(), and then train_step() should be called to perform training.
    """

    def __init__(
        self,
        cfg: TrainerConfig,
        actor: nn.Module,
        actor_optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Construct a trainer. To perform training, add episodes using
        add_episode(), and then call train_step().

        Arguments:
            cfg (TrainerConfig): Contains all configuration options. Can be
                automatically tuned using hyper-parameter optimization.
            actor (nn.Module): The deep neural network policy to optimize. Must
                have a forward method that takes in a list of episodes, and
                produces a list of ActorOutput.
                - observation_latents: Differentiable representations of the
                    observations in the episode.
                - action_lls: Differentiable log-likelihood of actions taken in the
                    episode.
                - rewards: Float Tensor containing rewards achieved after taking
                    each action.
                - terminals: Boolean Tensor recording if a state is a terminal
                    state. Should contain at most one true value, and only at the
                    last index.
                - action_possible: Boolean Tensor indicating if any actions were
                    possible to take at this state. This allows ignoring states
                    where no action was possible (e.g. the initial prompt in an
                    LLM, or intermediate frames in a video input when using
                    frame-skip).
            actor_optimizer (Optimizer): Optional optimizer for the actor. Will be
                constructed if not provided.

        """
        super().__init__()
        self.cfg = cfg
        self.actor = actor
        if actor_optimizer is None:
            actor_optimizer = torch.optim.AdamW(
                self.actor.parameters(),
                lr=self.cfg.actor_lr_start,
                weight_decay=self.cfg.actor_weight_decay,
            )
        self.actor_optimizer = actor_optimizer

        self._replay_buffer: list[EpisodeData] = []
        self._next_episode_number: int = 0
        self._episode_number_at_last_train_step: int = 0

        self._reward_normalizer = RunningMeanVar()

        self._dtype = torch.float32

        self._kl_coef = nn.Parameter(torch.tensor(self.cfg.initial_kl_coef))
        self._kl_coef_opt = torch.optim.Adam([self._kl_coef], lr=self.cfg.kl_coef_lr)
        self._temperature = nn.Parameter(torch.tensor(self.cfg.initial_temperature))
        self._temperature_opt = torch.optim.Adam(
            [self._temperature], lr=self.cfg.temperature_lr
        )

        self.observation_latent_size = actor.observation_latent_size

        # Don't expand the variance of the returns
        self._vf_normalizer = RunningMeanVar()
        self._vf = make_mlp(
            input_size=self.observation_latent_size,
            hidden_sizes=self.cfg.vf_hidden_sizes,
            output_size=0,
        )
        self._vf_optimizer = torch.optim.AdamW(
            self._vf.parameters(),
            lr=self.cfg.vf_lr_start,
            weight_decay=self.cfg.vf_weight_decay,
        )

        self.total_env_steps = 0

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
        self._reward_normalizer.update(rewards)

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

    def _run_forward(
        self, episodes: list[EpisodeData], desc: Optional[str] = None, epochs: int = 1
    ):
        """Split a large number of episodes into batches to avoid out of memory error.

        Yields: list[tuple[EpisodeData, ForwardResult]]
        """
        with tqdm(
            total=epochs * sum([ep.num_timesteps for ep in episodes]), desc=desc
        ) as pbar:
            for _ in range(epochs):
                next_ep_index = 0
                while next_ep_index < len(episodes):
                    start_batch_ep_index = next_ep_index
                    batch = []
                    num_batch_steps = 0
                    # Accumulate episodes into batch until we run out of space
                    while next_ep_index < len(episodes) and (
                        self.cfg.max_timesteps_per_forward is None
                        or num_batch_steps + episodes[next_ep_index].num_timesteps
                        <= self.cfg.max_timesteps_per_forward
                    ):
                        batch.append(episodes[next_ep_index])
                        num_batch_steps += episodes[next_ep_index].num_timesteps
                        next_ep_index += 1

                    if len(batch) == 0:
                        # We can't fit even a single forward pass in memory!
                        # Crash in this case (maybe the user can decrease the episode
                        # length, decrease the model size, enable gradient
                        # checkpointing / implement BPT, or buy a bigger GPU).
                        ep_steps = episodes[next_ep_index].num_timesteps
                        max_steps = self.cfg.max_timesteps_per_forward
                        raise RuntimeError(
                            dedent(
                                f"""\
                            Cannot run .forward() on episode of length:
                            f{ep_steps} > f{max_steps} = cfg.max_timesteps_per_forward
                            Increase cfg.max_timesteps_per_forward, decrease model size,
                            or find another way of increasing available memory.
                            """
                            )
                        )

                    try:
                        f_res = self.actor([ep.episode for ep in batch])
                        pbar.update(sum([ep.num_timesteps for ep in batch]))
                        # TODO: Filter NaNs in actor forward pass?
                        yield batch, f_res
                    except RuntimeError:
                        # Decrease to just one below the current size, which will
                        # prevent the last episode from being in the batch.
                        # This avoids dropping the max_timesteps_per_forward too low
                        # from one unusually large trailing episode.
                        # Note that this may still decrease by a large number of steps
                        # (or set max_timesteps_per_forward when it was previously
                        # None).
                        self.cfg.max_timesteps_per_forward = num_batch_steps - 1
                        warnings.warn(
                            f"Decreasing cfg.max_timesteps_per_forward to "
                            f"{self.cfg.max_timesteps_per_forward}",
                        )
                        # Retry the batch
                        next_ep_index = start_batch_ep_index

    def train_step(self):
        """Run training to produce a better policy.
        Should be called repeatedly after providing (new) data via
        add_episode()."""
        assert (
            len(self._replay_buffer) > 0
        ), "Please call add_episode() before training."

        episode_number_before = torch.tensor(
            [ep.episode_number for ep in self._replay_buffer]
        )
        # Discard old episodes from replay buffer
        if len(self._replay_buffer) > self.cfg.max_buffer_episodes:
            logging.debug(
                f"Trimming replay buffer "
                f"({len(self._replay_buffer)} > {self.cfg.max_buffer_episodes})"
            )
            self._replay_buffer.sort(key=EpisodeData.sort_key, reverse=True)
            self._replay_buffer = self._replay_buffer[: self.cfg.max_buffer_episodes]
            assert len(self._replay_buffer) == self.cfg.max_buffer_episodes

        episode_number = torch.tensor([ep.episode_number for ep in self._replay_buffer])

        observation_latents, action_lls_now = self._cache_actor_results()

        # v_s_packed, V_x_packed, episode_lengths = self._update_vf(
        #     observation_latents, action_lls_now
        # )
        v_s_packed, V_x_packed, episode_lengths = self._compute_vf_returns(
            observation_latents, action_lls_now
        )
        assert not v_s_packed.requires_grad
        assert not V_x_packed.requires_grad
        assert not v_s_packed.isnan().any()
        assert not V_x_packed.isnan().any()

        self._update_actor(v_s_packed, V_x_packed, episode_lengths)

        v_s_packed, V_x_packed, episode_lengths = self._update_vf(
            observation_latents, action_lls_now
        )

    def _update_actor(self, v_s_packed, V_x_packed, episode_lengths):
        vf_target_dict = self._packed_to_dict(
            self._vf_normalizer.normalize_batch(v_s_packed), episode_lengths
        )

        # See page 3 of V-MPO paper: https://arxiv.org/abs/1909.12238
        advantages_packed = v_s_packed - V_x_packed
        if self.cfg.use_top_half_advantages:
            median_advantage = torch.median(advantages_packed)
            v_mpo_mask = advantages_packed > median_advantage
        else:
            v_mpo_mask = torch.ones(
                advantages_packed.shape,
                device=advantages_packed.device,
                dtype=torch.bool,
            )

        assert not advantages_packed.requires_grad

        # Compute psi (eq 3.2)
        exp_adv_packed = torch.exp(
            torch.clamp(advantages_packed / self._temperature.detach(), min=-8, max=8)
        )
        total_exp_adv = torch.sum(exp_adv_packed * v_mpo_mask)
        # psi_packed = v_mpo_mask * exp_adv_packed / total_exp_adv
        psi_packed = v_mpo_mask * exp_adv_packed

        assert not exp_adv_packed.requires_grad
        assert not v_mpo_mask.requires_grad
        assert not psi_packed.requires_grad
        # psi_dict = self._packed_to_dict(psi_packed, episode_lengths)
        # mask_dict = self._packed_to_dict(v_mpo_mask, episode_lengths)
        exp_adv_dict = self._packed_to_dict(exp_adv_packed, episode_lengths)
        adv_dict = self._packed_to_dict(advantages_packed, episode_lengths)

        log(table="actor_before_updates")

        self._vf.train(mode=True)
        self.actor.train(mode=True)
        for batch, forward_res in self._run_forward(
            self._replay_buffer,
            desc="Actor Update",
            epochs=self.cfg.policy_epochs_per_train_step,
        ):
            action_ll_packed, lengths = pack_tensors(
                [f_res.action_lls for f_res in forward_res]
            )
            advantages = pack_tensors_check(
                [adv_dict[ep.episode_number][:-1] for ep in batch], lengths
            )
            batch_original_lls = pack_tensors_check(
                [ep.original_action_lls for ep in batch], lengths
            )
            assert not batch_original_lls.requires_grad
            assert action_ll_packed.requires_grad
            assert not advantages.requires_grad
            kl_div = approx_kl_div(batch_original_lls, action_ll_packed)
            assert kl_div.requires_grad

            likelihood_ratio = torch.exp(action_ll_packed - batch_original_lls)
            ratio_clipped = torch.clamp(
                likelihood_ratio,
                1 / (1 + self.cfg.actor_clip_ratio),
                1 + self.cfg.actor_clip_ratio,
            )
            policy_gradient = likelihood_ratio * advantages
            clip_policy_gradient = ratio_clipped * advantages

            L_ppo = -torch.min(policy_gradient, clip_policy_gradient)

            L_pi = L_ppo
            # L_pi = -(action_ll_packed - batch_original_lls).exp() * adv_packed
            # L_pi = -(action_ll_packed - batch_original_lls).exp() * adv_packed + 10 * kl_div
            # L_pi = -action_ll_packed * (adv_packed / self._temperature.detach()).exp()
            # L_pi = -action_ll_packed * (adv_packed > 0)
            # (adv_packed / self._temperature.detach()).exp()
            self.actor_optimizer.zero_grad()
            # L_pi.mean().backward()
            L_pi.sum().backward()
            log(table="actor_grad_step")
            self.actor_optimizer.step()
            self.actor_optimizer.zero_grad()

            # assert action_ll_packed.requires_grad
            # batch_psi_packed = pack_tensors_check(
            #     [psi_dict[ep.episode_number][:-1] for ep in batch], lengths
            # )
            # batch_mask_packed = pack_tensors_check(
            #     [mask_dict[ep.episode_number][:-1] for ep in batch], lengths
            # )
            # if not batch_mask_packed.any():
            #     continue

            # batch_original_lls = pack_tensors_check(
            #     [ep.original_action_lls for ep in batch], lengths
            # )
            # assert not batch_original_lls.requires_grad

            # # Compute L_pi (eq 3.1)
            # L_pi = -(batch_mask_packed * batch_psi_packed * action_ll_packed).sum()

            # # Compute L_eta (eq 4)
            # # TODO: Maybe move this up, since it doesn't depend on the actor output?
            # # TODO: Check if this should depend on actor output somehow? Seems off.
            # # TODO: This is supposed to be enforcing an "entropy
            # # TODO: constraint", but it currently doesn't do that within a batch.
            # batch_exp_adv = pack_tensors_check(
            #     [exp_adv_dict[ep.episode_number][:-1] for ep in batch], lengths
            # )
            # batch_mean_exp_adv = (
            #     batch_mask_packed * batch_exp_adv
            # ).sum() / batch_mask_packed.sum()
            # L_eta = (
            #     self._temperature * self.cfg.entropy_target
            #     + self._temperature * torch.log(batch_mean_exp_adv)
            # )

            # # Compute L_alpha
            # D_KL = approx_kl_div(batch_original_lls, action_ll_packed)
            # assert D_KL.requires_grad
            # L_alpha = self._kl_coef * (self.cfg.kl_soft_target - D_KL.detach())

            # # self._kl_coef_opt.zero_grad()
            # # L_alpha.backward()
            # # clip_grad_norm_([self._kl_coef], max_norm=10.0, error_if_nonfinite=True)
            # # self._kl_coef_opt.step()

            # if self._kl_coef < self.cfg.kl_coef_min:
            #     with torch.no_grad():
            #         self._kl_coef.copy_(torch.tensor(self.cfg.kl_coef_min))
            # elif self._kl_coef > self.cfg.kl_coef_max:
            #     with torch.no_grad():
            #         self._kl_coef.copy_(torch.tensor(self.cfg.kl_coef_max))

            # # self._temperature_opt.zero_grad()
            # # L_eta.backward()
            # # clip_grad_norm_([self._temperature], max_norm=10.0, error_if_nonfinite=True)
            # # self._temperature_opt.step()

            # if self._temperature < self.cfg.temperature_min:
            #     with torch.no_grad():
            #         self._temperature.copy_(torch.tensor(self.cfg.temperature_min))
            # elif self._temperature > self.cfg.temperature_max:
            #     with torch.no_grad():
            #         self._temperature.copy_(torch.tensor(self.cfg.temperature_max))

            # # TODO: Maybe perform these gradient steps together?
            # vf_loss = self._vf_loss(
            #     batch,
            #     forward_res,
            #     [vf_target_dict[ep.episode_number] for ep in batch],
            # )
            # # TODO: Add entropy "constraint"
            # L_theta = L_pi
            # # L_theta = (
            # #     L_pi + self.cfg.vf_loss_coeff * vf_loss + self._kl_coef.detach() * D_KL
            # # )
            # self.actor_optimizer.zero_grad()

            # L_theta.backward()

            # # TODO: add grad clipping
            # self.actor_optimizer.step()
            # # clip_grad_norm_(
            # #     self._vf.parameters(),
            # #     max_norm=self.cfg.vf_grad_max_norm,
            # #     error_if_nonfinite=True,
            # # )
            # # clip_grad_norm_(
            # #     self.actor.parameters(),
            # #     max_norm=self.cfg.actor_grad_max_norm,
            # #     error_if_nonfinite=True,
            # # )
            # log(table="actor_grad_step")
            # self._vf_optimizer.step()
            # self._vf_optimizer.zero_grad()
            # self.actor_optimizer.zero_grad()
            # Free grads before beginning next forward pass
        self._vf.train(mode=False)
        self.actor.train(mode=False)

    def _packed_to_dict(self, packed_tensor, lengths):
        as_list = unpack_tensors(packed_tensor, lengths)
        as_dict = {}
        for i, ep in enumerate(self._replay_buffer):
            as_dict[ep.episode_number] = as_list[i]
        return as_dict

    def _cache_actor_results(
        self,
    ) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
        new_episode_count = 0
        new_rb = []
        for episode in self._replay_buffer:
            if episode.episode_number >= self._episode_number_at_last_train_step:
                new_episode_count += 1
                new_rb.append(episode)
            else:
                ep_new = replace(
                    episode,
                    sample_priority=self.cfg.episode_priority_decay
                    * episode.sample_priority,
                )
                new_rb.append(ep_new)

        assert len(self._replay_buffer) == len(new_rb)
        self._replay_buffer = new_rb

        if new_episode_count == 0:
            warnings.warn("No new episodes added since last train_step() call.")

        # episode_number to Tensor
        observation_latents = {}
        action_lls_now = {}

        # Pre-compute all observation latents
        with torch.no_grad():
            for batch, forward_res in self._run_forward(
                self._replay_buffer, desc="Caching Latents"
            ):
                for episode, f_res in zip(batch, forward_res):
                    observation_latents[
                        episode.episode_number
                    ] = f_res.observation_latents
                    action_lls_now[episode.episode_number] = f_res.action_lls

        return observation_latents, action_lls_now

    def _compute_vf_returns(
        self,
        observation_latents: dict[int, torch.Tensor],
        action_lls_now: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        gamma = 1 - self.cfg.discount_inv

        gammas = torch.clamp(
            pad_tensors([ep.actions_possible for ep in self._replay_buffer]),
            min=gamma,
        )
        all_obs_latents = []
        for episode in self._replay_buffer:
            all_obs_latents.append(observation_latents[episode.episode_number])
        obs_latents_packed, episode_lengths = pack_tensors(all_obs_latents)
        all_rewards = pad_tensors(
            [
                self._reward_normalizer.normalize_batch(ep.rewards)
                for ep in self._replay_buffer
            ]
        )
        all_action_lls = pad_tensors(
            [action_lls_now[ep.episode_number] for ep in self._replay_buffer],
        )
        all_original_action_lls = pad_tensors(
            [ep.original_action_lls for ep in self._replay_buffer],
        )
        all_terminated = torch.as_tensor(
            [ep.terminated for ep in self._replay_buffer], dtype=torch.bool
        )
        all_episode_lengths = torch.as_tensor(episode_lengths, dtype=torch.long)

        with torch.no_grad():
            V_x_norm = self._vf(obs_latents_packed)
        V_x_packed = self._vf_normalizer.denormalize_batch(V_x_norm)
        V_x_list = unpack_tensors(V_x_packed, episode_lengths)

        # Compute VF target v_s
        v_s = v_trace_return_estimation(
            lmbda=self.cfg.v_trace_lambda,
            rho_max=self.cfg.v_trace_rho_max,
            c_max=self.cfg.v_trace_c_max,
            gammas=gammas,
            vf_returns=pad_tensors(V_x_list),
            rewards=all_rewards,
            action_lls=all_action_lls,
            original_action_lls=all_original_action_lls,
            terminated=all_terminated,
            episode_lengths=all_episode_lengths,
        )

        v_s_packed = pack_tensors_check(
            unpad_tensors(v_s, episode_lengths), episode_lengths
        )
        monte_carlo_returns = pack_tensors_check(
            unpad_tensors(
                torch.cat(
                    [
                        discount_cumsum(all_rewards, gamma),
                        torch.zeros(
                            all_rewards.shape[0],
                            1,
                            device=all_rewards.device,
                            dtype=all_rewards.dtype,
                        ),
                    ],
                    dim=1,
                ),
                episode_lengths,
            ),
            episode_lengths,
        )

        vf_explained_variance = explained_variance(v_s_packed, monte_carlo_returns)

        log("compute_vf_targets")

        return v_s_packed, V_x_packed, episode_lengths

    def _update_vf(
        self,
        observation_latents: dict[int, torch.Tensor],
        action_lls_now: dict[int, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        vf1 = copy.deepcopy(self._vf)
        vf1_opt = torch.optim.Adam(vf1.parameters())
        vf1_opt.load_state_dict(self._vf_optimizer.state_dict())

        vf2 = copy.deepcopy(self._vf)
        vf2_opt = torch.optim.Adam(vf2.parameters())
        vf2_opt.load_state_dict(self._vf_optimizer.state_dict())

        gamma = 1 - self.cfg.discount_inv

        gammas = torch.clamp(
            pad_tensors([ep.actions_possible for ep in self._replay_buffer]),
            min=gamma,
        )
        all_obs_latents = []
        for episode in self._replay_buffer:
            all_obs_latents.append(observation_latents[episode.episode_number])
        obs_latents_packed, episode_lengths = pack_tensors(all_obs_latents)
        all_rewards = pad_tensors(
            [
                self._reward_normalizer.normalize_batch(ep.rewards)
                for ep in self._replay_buffer
            ]
        )
        all_action_lls = pad_tensors(
            [action_lls_now[ep.episode_number] for ep in self._replay_buffer],
        )
        all_original_action_lls = pad_tensors(
            [ep.original_action_lls for ep in self._replay_buffer],
        )
        all_terminated = torch.as_tensor(
            [ep.terminated for ep in self._replay_buffer], dtype=torch.bool
        )
        all_episode_lengths = torch.as_tensor(episode_lengths, dtype=torch.long)

        monte_carlo_returns = pack_tensors_check(
            unpad_tensors(
                torch.cat(
                    [
                        discount_cumsum(all_rewards, gamma),
                        torch.zeros(
                            all_rewards.shape[0],
                            1,
                            device=all_rewards.device,
                            dtype=all_rewards.dtype,
                        ),
                    ],
                    dim=1,
                ),
                episode_lengths,
            ),
            episode_lengths,
        )

        total_timesteps = int(all_episode_lengths.sum())
        assert total_timesteps == sum(
            ep.num_timesteps + 1 for ep in self._replay_buffer
        )

        vf1_indices, vf2_indices = split_shuffled_indices(total_timesteps, p_right=0.5)

        v_s_packed = None

        # Enable training mode of vf1 and vf2
        vf1.train(mode=True)
        vf2.train(mode=True)
        with tqdm(
            total=self.cfg.vf_training_epochs * total_timesteps, desc="VF Update"
        ) as pbar:
            for epoch in range(self.cfg.vf_training_epochs + 1):
                # Re-Compute V(x)
                if (
                    epoch == 0
                    or epoch == self.cfg.vf_training_epochs
                    or self.cfg.vf_recompute_targets
                ):
                    # Compute targets ot start and end of training (or every epoch
                    # if cfg.vf_recompute_targets is true).

                    # self._vf.load_state_dict(vf1.state_dict())
                    # self._vf.train(mode=False)
                    # self._vf_optimizer.load_state_dict(vf1_opt.state_dict())
                    self._vf.load_state_dict(average_modules(vf1, vf2))
                    self._vf.train(mode=False)
                    self._vf_optimizer.load_state_dict(
                        average_modules(vf1_opt, vf2_opt)
                    )
                    with torch.no_grad():
                        V_x_norm = self._vf(obs_latents_packed)
                    V_x_packed = self._vf_normalizer.denormalize_batch(V_x_norm)
                    V_x_list = unpack_tensors(V_x_packed, episode_lengths)

                    # Compute VF target v_s
                    v_s = v_trace_return_estimation(
                        lmbda=self.cfg.v_trace_lambda,
                        rho_max=self.cfg.v_trace_rho_max,
                        c_max=self.cfg.v_trace_c_max,
                        gammas=gammas,
                        vf_returns=pad_tensors(V_x_list),
                        rewards=all_rewards,
                        action_lls=all_action_lls,
                        original_action_lls=all_original_action_lls,
                        terminated=all_terminated,
                        episode_lengths=all_episode_lengths,
                    )

                    v_s_packed = pack_tensors_check(
                        unpad_tensors(v_s, episode_lengths), episode_lengths
                    )

                    vf_explained_variance = explained_variance(
                        v_s_packed, monte_carlo_returns
                    )

                    if epoch == 0:
                        # Computed only for debug purposes
                        log("vf_update_start_trace")
                        log(
                            "vf_update_start",
                            {"vf_explained_variance": vf_explained_variance},
                        )

                    if epoch == self.cfg.vf_training_epochs:
                        # We're done pre-training the vf, return the final results
                        log("vf_update_final_trace")
                        log(
                            "vf_update_final",
                            {"vf_explained_variance": vf_explained_variance},
                        )
                        return v_s_packed, V_x_packed, episode_lengths
                assert v_s_packed is not None, "v_s_packed was set at first epoch"

                v_s_norm_packed = self._vf_normalizer(v_s_packed)
                mc_returns_norm = self._vf_normalizer.normalize_batch(
                    monte_carlo_returns
                )
                v_s_norm_packed = mc_returns_norm

                # train vfs using frozen observation latents
                vf1_dataset = DictDataset(
                    vf_targets=v_s_norm_packed[vf1_indices],
                    obs_latents=obs_latents_packed[vf1_indices],
                    mc_returns=mc_returns_norm[vf1_indices],
                )
                vf2_dataset = DictDataset(
                    vf_targets=v_s_norm_packed[vf2_indices],
                    obs_latents=obs_latents_packed[vf2_indices],
                    mc_returns=mc_returns_norm[vf2_indices],
                )
                for batch1, batch2 in zip(
                    vf1_dataset.minibatches(self.cfg.vf_minibatch_size),
                    vf2_dataset.minibatches(self.cfg.vf_minibatch_size),
                ):
                    vf1_opt.zero_grad()
                    vf1_out = vf1(batch1["obs_latents"])
                    vf1_loss = F.mse_loss(vf1_out, batch1["vf_targets"])
                    vf1_loss.backward()
                    clip_grad_norm_(
                        vf1.parameters(),
                        max_norm=self.cfg.vf_grad_max_norm,
                        error_if_nonfinite=True,
                    )

                    with torch.no_grad():
                        vf1_cross_pred = vf1(batch2["obs_latents"])
                        vf1_cross_val_loss = F.mse_loss(
                            vf1_cross_pred, batch2["vf_targets"]
                        )
                        vf1_cross_explained_variance = explained_variance(
                            vf1_cross_pred, batch2["vf_targets"]
                        )
                        vf1_cross_mc_ev = explained_variance(
                            vf1_cross_pred, batch2["mc_returns"]
                        )

                    vf2_opt.zero_grad()
                    vf2_out = vf2(batch2["obs_latents"])
                    vf2_loss = F.mse_loss(vf2_out, batch2["vf_targets"])
                    vf2_loss.backward()
                    clip_grad_norm_(
                        vf2.parameters(),
                        max_norm=self.cfg.vf_grad_max_norm,
                        error_if_nonfinite=True,
                    )

                    with torch.no_grad():
                        vf2_cross_pred = vf1(batch1["obs_latents"])
                        vf2_cross_val_loss = F.mse_loss(
                            vf2_cross_pred, batch1["vf_targets"]
                        )
                        vf2_cross_explained_variance = explained_variance(
                            vf2_cross_pred, batch1["vf_targets"]
                        )
                        vf2_cross_mc_ev = explained_variance(
                            vf2_cross_pred, batch1["mc_returns"]
                        )
                    vf_cross_explained_variance = (
                        vf1_cross_explained_variance + vf2_cross_explained_variance
                    ) / 2
                    log("vf_update_grad")

                    vf1_opt.step()
                    vf2_opt.step()
                    pbar.update(int(len(vf1_out) + len(vf2_out)))

        assert False, "Should have exited after last training epoch"

    def _vf_loss(
        self,
        episodes: list[EpisodeData],
        actor_outputs: list[ActorOutput],
        vf_targets: list[torch.Tensor],
    ) -> torch.Tensor:
        obs_latents_packed, episode_lengths = pack_tensors(
            [actor_out.observation_latents for actor_out in actor_outputs]
        )
        v_s_packed, v_s_lengths = pack_tensors(vf_targets)
        assert episode_lengths == v_s_lengths

        vf_out = self._vf(obs_latents_packed)
        L_vf = F.mse_loss(vf_out, v_s_packed)
        log("vf_loss")
        return L_vf

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        logging.info(f"Eval primary stat ({primary}):", stats[primary])
        log("eval_stats", stats, step=self.total_env_steps, level=stick.RESULTS)
        hparams = self.cfg.to_dict()
        hparams["metric-primary"] = stats[primary]
        for k, v in stats.items():
            hparams[k] = v
        log("hparams", hparams, step=self.total_env_steps)


@torch.jit.script
def v_trace_return_estimation(
    *,
    lmbda: float,
    rho_max: float,
    c_max: float,
    gammas: torch.Tensor,
    vf_returns: torch.Tensor,
    rewards: torch.Tensor,
    action_lls: torch.Tensor,
    original_action_lls: torch.Tensor,
    terminated: torch.Tensor,
    episode_lengths: torch.Tensor,
):
    """Calculate value function targets using a V-Trace like estimator.

    When rho_max and c_max are set infinitely high (or the data is "completely
    on-policy"), this function is equivalent to TD(lambda) with a discount
    mask.

    See page 3 of "IMPALA" from https://arxiv.org/abs/1802.01561

    Args:
        lmbda (float): Lambda parameter that controls the bias-variance tradeoff.
        rho_max (float): The "VF truncation" importance weight clip.
        c_max (float): The "trace-cutting" importance weight clip.
        gammas (torch.Tensor): A 2D tensor of per-timestep discount factors.
            Used to avoid discounting across states where no action was
            possible.
        vf_returns (torch.Tensor): A 2D tensor of value function
            estimates with shape (N, T + 1), where N is the batch dimension
            (number of episodes) and T is the maximum episode length
            experienced by the agent. If an episode terminates in fewer than T
            time steps, the remaining elements in that episode should be set to
            0.
        rewards (torch.Tensor): A 2D tensor of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent.
        episode_lengths (torch.Tensor) A 1D tensor indicating the episode length.
        terminated (torch.Tensor): A 1D tensor indicating if the episode
            ended in a terminal state.
        apply_discount: A 2D Tensor indicating if the discount should be
            applied over this timestep.
    Returns:
        torch.Tensor: A 2D vector of calculated advantage values with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining values
            in that episode should be set to 0.

    """
    # The paper says to assume rho_max >= c_max, but the math should still work
    # either way.

    assert len(rewards.shape) == 2
    n_episodes = rewards.shape[0]
    max_episode_length = rewards.shape[1]
    v_xs = vf_returns
    assert v_xs.shape == (n_episodes, max_episode_length + 1)
    assert action_lls.shape == (n_episodes, max_episode_length)
    assert original_action_lls.shape == (n_episodes, max_episode_length)

    importance_weight = (action_lls - original_action_lls).exp()
    rho = torch.clamp(importance_weight, max=rho_max)

    # Multiply in the lambda term (not present in standard V-Trace, but matches
    # TD(lambda)).
    c = lmbda * torch.clamp(importance_weight, max=c_max)

    delta_V = rho * (rewards + gammas * v_xs[:, 1:] - v_xs[:, :-1])

    v_xs_delta_V = v_xs[:, :-1] + delta_V

    v_s = torch.zeros((n_episodes, max_episode_length + 1))
    # Fill final trailing state with 0 terminal states and V(x_s) for non-terminal states.
    # In other words, the target for non-terminal states will be the current
    # value, and zero for terminal states.
    v_s[~terminated, max_episode_length] = v_xs[~terminated, max_episode_length]

    # Can't use reversed in torchscript
    # Start at max_episode_length - 1 and go down to 0
    for t in range(max_episode_length - 1, -1, -1):
        # Main V-Trace update
        prior = c[:, t] * (v_s[:, t + 1] - v_xs[:, t + 1])
        v_s[:, t] = v_xs_delta_V[:, t] + gammas[:, t] * prior

        # If we have a terminal state, just try to estimate the final step rewards.
        # If we have a non-terminal final state, fill with v_xs_delta_V (which
        # includes v_xs[:, t + 1] but with the importance sampling
        # corrections).
        done = t == episode_lengths
        v_s[done, t] = rewards[done, t] + ~terminated[done] * v_xs_delta_V[done, t]

    return v_s


@torch.jit.script
def discount_cumsum(x: torch.Tensor, discount: float):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)
    B, T = x.shape
    discount_x = discount * torch.ones_like(x[0])
    discount_x[0] = 1.0
    # Compute discount weights.
    weights = torch.cumprod(discount_x, dim=0)
    # Add channel in dimensions and channel out dimensions
    weights = weights.reshape(1, 1, T)
    x = x.reshape(B, 1, T)
    # Add pad end of episodes to zero
    # Only need 2T - 1 timesteps to make index T valid
    z = torch.zeros_like(x[:, :, :-1])
    x_pad = torch.cat([x, z], dim=-1)
    returns = F.conv1d(x_pad, weights, stride=1)
    assert returns.shape == (B, 1, T)
    return returns.squeeze(dim=1)


def test_discount_cumsum():
    B = 7
    L = 9
    discount = 0.9
    rewards = torch.randn(B, L)
    expected_result = torch.zeros_like(rewards)
    expected_result[:, -1] = rewards[:, -1]
    for i in range(L - 2, -1, -1):
        expected_result[:, i] = rewards[:, i] + discount * expected_result[:, i + 1]
    actual_result = discount_cumsum(rewards, discount)
    assert torch.allclose(actual_result, expected_result)
