"""
.. include:: ../../README.md
"""

import dataclasses
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Optional, TypeVar, Generator, Literal
import os
import random
import warnings
import logging
import pickle
from glob import glob
import time
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import simple_parsing

try:
    from accelerate import Accelerator
except ImportError:
    Accelerator = None

from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
)

import noko

from outrl.torch_utils import (
    approx_entropy_of,
    entropy_of,
    force_concat,
    make_mlp,
    pack_recursive,
    pack_tensors,
    pack_tensors_check,
    pad_packed,
    pad_tensors,
    softmax_clip,
    unpad_tensors,
    pack_padded,
    explained_variance,
    unpack_tensors,
    RunningMeanVar,
    make_scheduler,
    DictDataset,
    ActionDist,
    kl_div_of,
    approx_kl_div_of,
    used_for_logging,
    discount_cumsum,
    pack_dataclass,
    truncate_packed,
    concat_lists,
)
from outrl.config import tunable, IntListDistribution, default_run_name

_LOGGER = logging.getLogger("outrl")

T = TypeVar("T")

EpisodeID = int


@dataclass(eq=False)
class _EpisodeData:
    """Wrapper around an episode that maintains metadata and caches computed values."""

    episode: Any
    """The episode (treated as an opaque object). Will be passed back to the
    agent during optimization."""

    episode_id: EpisodeID
    """Records which add_episode() call this episode came from."""

    n_timesteps: int
    """Number of time steps in episode."""

    terminated: bool
    """Boolean indicating if the episode reached a terminal state. Always False
    in infinite horizon MDPs or when an episode reaches a timeout."""

    original_action_lls: torch.Tensor
    """Original action lls provided when the episode was added. Used for
    v-trace off-policy correction and in the PPO loss (when enabled).

    Do not require gradients.
    """

    original_action_dists: Optional[ActionDist]
    """Original action distributions optionally provided when the episode was
    added."""

    rewards: torch.Tensor
    """Rewards provided when the episode was added."""

    any_actions_possible: torch.Tensor
    """Boolean any_actions_possible mask."""

    infos: dict[str, Any]
    """User infos. Will be logged if possible."""

    weight: float = 1.0
    """Importance of this episode."""

    def __post_init__(self):
        assert self.original_action_lls.shape == (self.n_timesteps,)
        assert self.rewards.shape == (self.n_timesteps,)
        assert self.any_actions_possible.shape == (self.n_timesteps,)
        assert self.n_timesteps > 0

        requires_grad_msg = dedent(
            """\
            This will retain a gradient tape for as long as the
            episode is in the replay buffer, and is almost
            certainly not intended.
            """
        )

        if self.rewards.requires_grad:
            raise ValueError(
                "rewards passed to replay buffer requires grad." + requires_grad_msg
            )

        if self.original_action_lls.requires_grad:
            raise ValueError(
                "action_lls passed to replay buffer requires grad." + requires_grad_msg
            )

        if self.any_actions_possible.requires_grad:
            raise ValueError(
                "any_actions_possible passed to replay buffer requires grad."
                + requires_grad_msg
            )

        # Check if action_dists requires_grad
        if self.original_action_dists is not None:
            if isinstance(self.original_action_dists, list):
                act_dist_list = self.original_action_dists
            else:
                act_dist_list = [self.original_action_dists]
            for act_dist in act_dist_list:
                for k, v in act_dist.__dict__.items():
                    if getattr(v, "requires_grad", None):
                        raise ValueError(
                            dedent(
                                f"""\
                            action_dists passed to replay buffer requires grad
                            through field {k}.
                            """
                            )
                            + requires_grad_msg
                        )


@dataclass(eq=False)
class LossInput:
    """Input values to the loss function not present in AgentOutput.

    Analogous to "targets" or "labels" in supervised learning.
    These values are recomputed every train_step()."""

    advantages: torch.Tensor
    """Advantages for this episode. Used in PPO loss (when enabled)."""

    exp_advantages: torch.Tensor
    """Coefficients used in the AWR loss."""

    vf_targets: torch.Tensor
    """Target values for the value function loss."""

    vf_returns: torch.Tensor
    """Estimated returns from the value function. Used in logging."""

    original_action_lls: torch.Tensor
    """Copied from Trainer.add_episode(), these are the original log
    likelihoods of the actions taken when the data was collected."""

    episode_lengths: list[int]

    original_action_dists: Optional[list[ActionDist]]
    """Copied from Trainer.add_episode()."""

    def __post_init__(self):
        assert not self.original_action_lls.requires_grad
        assert not self.vf_returns.requires_grad
        assert not self.vf_targets.requires_grad
        assert not self.exp_advantages.requires_grad
        assert not self.advantages.requires_grad

    @classmethod
    def pack(cls, loss_inputs: list["LossInput"]) -> "LossInput":
        new_fields = {}
        for field in dataclasses.fields(cls):
            field_vals = [getattr(li, field.name) for li in loss_inputs]
            if field.name == "original_action_dists":
                if None in field_vals:
                    assert field_vals == [None] * len(field_vals)
                    new_fields[field.name] = None
                else:
                    new_fields[field.name] = concat_lists(field_vals)
            elif field.name == "episode_lengths":
                new_fields[field.name] = concat_lists(field_vals)
            else:
                new_fields[field.name] = torch.cat(
                    [getattr(li, field.name) for li in loss_inputs]
                )
        return cls(**new_fields)


_OPTIMIZER_FIELDS = [
    "agent_optimizer",
    "vf_optimizer",
    "vf_lr_scheduler",
    "agent_lr_scheduler",
    "kl_coef_opt",
]

_SUBMODULE_FIELDS = [
    "agent",
    "vf",
    "reward_normalizer",
]

_PARAM_FIELDS = [
    "kl_coef",
]

_IGNORED_FIELDS = ["_is_full_backward_hook"]


class Trainer:
    """An implementation of a bespoke reinforcement learning algorithm.

    The trainer trains an Agent to acquire higher rewards according to the data
    provided to it. To allow repeatedly adding new data to the Trainer, it is a
    class instead of a function.

    The typical sequence of methods called on this class are as follows:

        1. Construction using a TrainerConfig (the default value is typically
           fine) and Agent. Optionally, call Trainer.attempt_resume() to resume
           from a prior checkpoint.

        2. Repeated calls to Trainer.add_episode() to add new data and
           Trainer.train_step() to process the data. Typically multiple
           episodes (on the order of 10-100) should be added between each
           Trainer.train_step() call.

        3. Periodic calls to Trainer.add_eval_stats() and
           Trainer.maybe_checkpoint().
    """

    def __init__(self, cfg: "TrainerConfig", agent: "Agent"):
        """Constructs a Trainer."""

        super().__init__()
        self.cfg: "TrainerConfig" = cfg
        """The configuration used to contruct the Trainer.

        Modifying this field after constructing the Trainer *may* change the
        Trainer's behavior, but is not guaranteed to and should be avoided
        (except via load_state_dict()).
        """

        if Accelerator is not None:
            self.accelerator = Accelerator()
            self.device = self.accelerator.device
        else:
            self.accelerator = None
            self.device = self.cfg.device

        self.agent: "Agent" = self._prepare(agent)
        """The agent being optimized. Provides action (log-likelihoods) and
        state encodings."""

        self._state_encoding_size = self.agent.state_encoding_size

        self.vf: nn.Module = self._prepare(
            make_mlp(
                input_size=self._state_encoding_size,
                hidden_sizes=self.cfg.vf_hidden_sizes,
                output_size=0,
                use_dropout=True,
            )
        )
        """The value function. Feed-forward networks that predicts future
        rewards from state_encodings."""

        # Zero the initial VF output to stabilize training
        vf_output = self.vf.get_submodule("output_linear")
        vf_output.weight.data.copy_(0.01 * vf_output.weight.data)

        self.reward_normalizer: RunningMeanVar = RunningMeanVar(use_mean=False).to(
            self.device
        )
        """Normalized used to make rewards have unit variance if
        cfg.normalize_rewards is True."""

        self.total_env_steps: int = 0
        """Total number of environment steps passed to add_episode().
        May not count timesteps only used to compute statistics passed to
        add_eval_stats()."""

        self.train_steps_so_far: int = 0
        """Number of times train_step() has been called."""

        self.last_eval_stats: dict[str, float] = {}
        """Last stats passd to add_eval_stats()."""

        self.primary_performance: float = float("-inf")
        """Value of the "primary_performance" stat at the last add_eval_stats() call."""

        self.best_checkpoint_primary_performance: float = float("-inf")
        """Largest value of the "primary_performance" stat among all add_eval_stats() calls."""

        self.train_steps_so_far_at_last_checkpoint: int = 0
        """Number of (completed) train_step() calls at last periodic checkpoint."""

        self.kl_coef: nn.Parameter = nn.Parameter(
            torch.tensor(float(self.cfg.kl_coef_init), device=self.device)
        )
        """Dynamically adjusted parameter used for KL regularization."""

        self.starting_entropy: Optional[float] = None
        """Initial mean entropy of action distributions, measured immediately
        at the start of cfg.entropy_schedule_start_train_step."""

        self._replay_buffer: list[_EpisodeData] = []
        self._next_episode_id: EpisodeID = 0
        self._dtype = torch.float32

        # This method is also called after loading the state dict to re-attach
        # parameters
        self._setup_optimizers()

    def _prepare(self, torch_object):
        if self.accelerator is not None:
            return self.accelerator.prepare(torch_object)
        else:
            return torch_object.to(device=self.device)

    def _setup_optimizers(self):
        """(Re)create all of the optimizers to use the current parameters.

        This method is called in __init__ and also in load_state_dict().
        """
        self.agent_optimizer = torch.optim.AdamW(
            self.agent.parameters(),
            lr=self.cfg.agent_lr_start,
            weight_decay=self.cfg.agent_weight_decay,
        )
        self.agent_lr_scheduler = make_scheduler(
            self.agent_optimizer,
            self.cfg.agent_lr_schedule,
            self.cfg.agent_lr_start,
            self.cfg.agent_lr_end,
            self.cfg.expected_train_steps,
        )
        self.vf_optimizer = torch.optim.AdamW(
            self.vf.parameters(),
            lr=self.cfg.vf_lr_start,
            weight_decay=self.cfg.vf_weight_decay,
        )
        self.vf_lr_scheduler = make_scheduler(
            self.vf_optimizer,
            self.cfg.vf_lr_schedule,
            self.cfg.vf_lr_start,
            self.cfg.vf_lr_end,
            self.cfg.expected_train_steps,
        )

        self.kl_coef_opt = torch.optim.AdamW([self.kl_coef], lr=self.cfg.kl_coef_lr)

        self.total_agent_grad_steps = 0
        self.agent_grad_steps_at_start_of_train_step = 0
        self.agent_grad_steps_last_train_step = 0

    def _primary_loss_function(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reinforcement learning loss function.

        Combines together losses from many sources to train both the agent and
        value function.

        Returns a loss which is usually near unit scale and a dictionary of
        infos to log for this gradient step.

        Note that lagrangian losses (for the KL and entropy) are optimized in
        their related loss functions.
        """
        kl_loss, kl_infos = self._kl_loss(loss_input, agent_output)
        entropy_loss, entropy_infos = self._entropy_loss(loss_input, agent_output)
        ppo_loss, ppo_infos = self._ppo_loss(loss_input, agent_output)
        awr_loss, awr_infos = self._awr_loss(loss_input, agent_output)
        vf_loss, vf_infos = self._vf_loss(loss_input, agent_output)

        loss = ppo_loss + awr_loss + vf_loss + kl_loss + entropy_loss

        # *_infos will all get included in locals of this method
        used_for_logging(kl_infos, ppo_infos, awr_infos, vf_infos, entropy_infos)
        return loss, locals()

    def _ppo_loss(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        ratio = torch.exp(agent_output.action_lls - loss_input.original_action_lls)
        ratio_clipped = torch.clamp(
            ratio, 1 / (1 + self.cfg.ppo_clip_epsilon), 1 + self.cfg.ppo_clip_epsilon
        )

        policy_gradient = ratio * loss_input.advantages
        clip_policy_gradient = ratio_clipped * loss_input.advantages

        # Mean is handled in the primary_loss_function
        ppo_loss = -torch.min(policy_gradient, clip_policy_gradient)
        norm_coef = self.cfg.ppo_loss_coef / self.cfg.minibatch_target_timesteps
        ppo_loss_scaled = norm_coef * ppo_loss.sum()
        infos = locals()
        del infos["self"]
        infos["clip_portion"] = (ratio_clipped != ratio).mean(dtype=torch.float32)
        return (ppo_loss_scaled, infos)

    def _awr_loss(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Dividing by temperature and normalizing are handled in Trainer.preprocess()
        awr_loss = -agent_output.action_lls * loss_input.exp_advantages

        log_probs_scale = agent_output.action_lls.detach().abs().mean()
        used_for_logging(log_probs_scale)
        norm_coef = self.cfg.awr_loss_coef / (self.cfg.minibatch_target_timesteps)

        awr_loss_scaled = norm_coef * awr_loss.sum()
        # Don't want to log these here
        infos = locals()
        del infos["self"]
        return (awr_loss_scaled, infos)

    def _vf_loss(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        state_enc_packed = truncate_packed(
            agent_output.state_encodings, loss_input.episode_lengths, 1
        )
        critic_out = self.vf(state_enc_packed)
        vf_loss = F.mse_loss(critic_out, loss_input.vf_targets)

        norm_coef = self.cfg.vf_loss_coef / self.cfg.minibatch_target_timesteps
        vf_loss_scaled = norm_coef * vf_loss
        infos = locals()
        del infos["self"]
        return (vf_loss_scaled, infos)

    def _kl_loss(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        log_probs = agent_output.action_lls.clone()
        old_log_probs = loss_input.original_action_lls
        # Normalize to form a proper distribution across actions P(a)
        log_probs -= log_probs.exp().sum().log()
        old_log_probs -= old_log_probs.exp().sum().log()

        # Approximate per-timestep KL by multiplying back in the number of timesteps

        total_timesteps = sum(loss_input.episode_lengths)
        approx_kl = total_timesteps * approx_kl_div_of(old_log_probs, log_probs)
        total_approx_kl = approx_kl.sum()
        used_for_logging(total_approx_kl)

        # Compute KL Divergence
        if self.cfg.use_approx_kl is False:
            new_dists = agent_output.action_dists
            old_dists = loss_input.original_action_dists
            # TODO: Add options for other KL directions
            try:
                kl = kl_div_of(old_dists, new_dists)
            except NotImplementedError:
                kl = approx_kl
        else:
            kl = approx_kl

        assert kl.shape == (total_timesteps,)
        assert approx_kl.shape == (total_timesteps,)
        kl = kl[torch.isfinite(kl)]

        # Update KL loss coefficient
        if self.cfg.kl_target_stat == "max":
            kl_coef_loss = self.kl_coef * (self.cfg.kl_soft_target - kl.detach().max())
        elif self.cfg.kl_target_stat == "mean":
            kl_coef_loss = self.kl_coef * (self.cfg.kl_soft_target - kl.detach().mean())
        else:
            raise ValueError(f"Unknown kl_target_stat {self.cfg.kl_target_stat}")
        self.kl_coef_opt.zero_grad()
        kl_coef_loss.backward()
        try:
            clip_grad_norm_(
                [self.kl_coef], max_norm=self.cfg.grad_norm_max, error_if_nonfinite=True
            )
            self.kl_coef_opt.step()
        except RuntimeError:
            # Probably inf gradients, don't apply them
            pass

        if self.kl_coef < self.cfg.kl_coef_min:
            with torch.no_grad():
                self.kl_coef.copy_(self.cfg.kl_coef_min)
        # If the KL coef has become non-finite, it's probably because of
        # infinite KL, so set to maximum.
        if self.kl_coef > self.cfg.kl_coef_max or not torch.isfinite(self.kl_coef):
            with torch.no_grad():
                self.kl_coef.copy_(self.cfg.kl_coef_max)
        kl_coef = self.kl_coef.detach()

        kl_mean = kl.mean()
        kl_max = kl.max()
        used_for_logging(kl_mean, kl_max)

        norm_coef = kl_coef / self.cfg.minibatch_target_timesteps

        kl_loss_scaled = (norm_coef * kl).sum()
        infos = locals()
        del infos["self"]
        return kl_loss_scaled, infos

    def _entropy_target(self) -> Optional[float]:
        agent_grad_steps_this_train_step = (
            self.total_agent_grad_steps - self.agent_grad_steps_at_start_of_train_step
        )
        progress_at_train_step = min(
            1,
            agent_grad_steps_this_train_step
            / max(1, self.agent_grad_steps_last_train_step),
        )

        if self.starting_entropy is not None:
            # 0 at cfg.entropy_schedule_start_train_step
            # 1 at (and after) cfg.expected_train_steps
            step_fraction = min(
                1,
                (
                    (progress_at_train_step + self.train_steps_so_far)
                    - self.cfg.entropy_schedule_start_train_step
                )
                / (
                    self.cfg.expected_train_steps
                    - self.cfg.entropy_schedule_start_train_step
                ),
            )
            assert step_fraction >= 0
            assert step_fraction <= 1
            if self.cfg.entropy_schedule_end_target is not None:
                final_entropy = self.cfg.entropy_schedule_end_target
            else:
                if self.starting_entropy < 0:
                    final_entropy = (
                        self.starting_entropy / self.cfg.entropy_schedule_end_fraction
                    )
                else:
                    final_entropy = (
                        self.cfg.entropy_schedule_end_fraction * self.starting_entropy
                    )
            if self.cfg.entropy_schedule == "linear":
                mix = step_fraction
            elif self.cfg.entropy_schedule == "cosine":
                # Cosine curve from 1 to 0
                mix = 1 - 0.5 * (1 + math.cos(step_fraction * math.pi))
            else:
                raise NotImplementedError(
                    f"Unknown entropy schedule {self.cfg.entropy_schedule}"
                )
            assert mix >= 0
            assert mix <= 1
            target = (1 - mix) * self.starting_entropy + mix * final_entropy
            return target
        else:
            return None

    def _entropy_loss(
        self, loss_input: LossInput, agent_output: "AgentOutput"
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        approx_entropy, entropy = self._entropy_of(
            agent_output.action_lls,
            agent_output.action_dists,
        )
        entropy_target = self._entropy_target()
        if entropy_target is not None:
            entropy_loss = self.cfg.entropy_loss_coef * (
                (entropy - entropy_target).sum() ** 2
            )
        else:
            entropy_target = entropy.mean()
            entropy_loss = torch.tensor(0.0)

        used_for_logging(approx_entropy)
        infos = locals()
        del infos["self"]
        return entropy_loss, infos

    def _agent_minibatches(
        self,
        episodes: Optional[list[_EpisodeData]] = None,
        minibatch_target_timesteps: Optional[int] = None,
        desc: Optional[str] = None,
        epochs: int = 1,
        shuffle: bool = False,
        need_full: bool = True,
    ) -> Generator[tuple[int, int, list[_EpisodeData], "AgentOutput"], None, None]:
        """Runs the agent forward pass on minibatches drawn from the replay buffer.

        Minibatches are a list of tuples of _EpisodeData from the input, data T
        from the extra_data (None if not provided), and AgentOutput from the
        agent forward pass.

        This method handles sizing of minibatches to avoid OOM, shuffling of
        episodes, rendering the progress bar, and running for multiple epochs.
        Basically, this method plays a similar role to "Trainer" classes in
        supervised learning libraries.

        Yields: list[tuple[_EpisodeData, T, ForwardResult]]
        """
        if episodes is None:
            episodes = self._replay_buffer

        if shuffle:
            shuffled_indices = torch.randperm(len(episodes))
            episodes = [episodes[i] for i in shuffled_indices]

        minibatch_hard_cap = 2**64
        if self.cfg.max_timesteps_per_forward is not None:
            minibatch_hard_cap = self.cfg.max_timesteps_per_forward

        minibatch_soft_cap = minibatch_hard_cap
        if minibatch_target_timesteps is not None:
            minibatch_soft_cap = min(minibatch_target_timesteps, minibatch_hard_cap)

        with tqdm(
            total=epochs * sum([ep.n_timesteps for ep in episodes]), desc=desc
        ) as pbar:
            for epoch in range(epochs):
                next_ep_index = 0
                minibatch_number = 0
                while next_ep_index < len(episodes):
                    start_batch_ep_index = next_ep_index
                    batch = []
                    n_batch_steps = 0
                    # Accumulate episodes into batch until we run out of space
                    while next_ep_index < len(episodes) and (
                        n_batch_steps + episodes[next_ep_index].n_timesteps
                        <= minibatch_hard_cap
                    ):
                        batch.append(episodes[next_ep_index])
                        n_batch_steps += episodes[next_ep_index].n_timesteps
                        next_ep_index += 1
                        if n_batch_steps >= minibatch_soft_cap:
                            break

                    if len(batch) == 0:
                        # We can't fit even a single forward pass in memory!
                        # Crash in this case (maybe the user can decrease the episode
                        # length, decrease the model size, enable gradient
                        # checkpointing / implement BPT, or buy a bigger GPU).
                        ep_steps = episodes[next_ep_index].n_timesteps
                        max_steps = self.cfg.max_timesteps_per_forward
                        raise RuntimeError(
                            dedent(
                                f"""\
                            Cannot run .forward() on episode of length:
                            {ep_steps} > {max_steps} = cfg.max_timesteps_per_forward
                            Increase cfg.max_timesteps_per_forward, decrease model size,
                            or find another way of increasing available memory.
                            """
                            )
                        )

                    try:
                        agent_output = _call_agent_forward(
                            self.agent,
                            [data.episode for data in batch],
                            need_full=need_full,
                            expected_lengths=[data.n_timesteps for data in batch],
                        )
                        yield (
                            epoch,
                            minibatch_number,
                            batch,
                            agent_output,
                        )
                        minibatch_number += 1
                        pbar.update(sum([data.n_timesteps for data in batch]))
                    except RuntimeError as ex:
                        if "Cannot allocate memory" not in str(ex):
                            raise ex
                        # Decrease to just one below the current size, which will
                        # prevent the last episode from being in the batch.
                        # This avoids dropping the max_timesteps_per_forward too low
                        # from one unusually large trailing episode.
                        # Note that this may still decrease by a large number of steps
                        # (or set max_timesteps_per_forward when it was previously
                        # None).
                        self.cfg.max_timesteps_per_forward = n_batch_steps - 1
                        minibatch_hard_cap = self.cfg.max_timesteps_per_forward
                        warnings.warn(
                            f"Decreasing cfg.max_timesteps_per_forward to "
                            f"{self.cfg.max_timesteps_per_forward}",
                        )
                        # Retry the batch
                        next_ep_index = start_batch_ep_index

    def train_step(self):
        """Runs one "policy step" of training.

        This method should be called repeatedly until training is complete.
        Unless training is completely off-policy, new episodes should be added
        between each call to this method using add_episode().

        All options for tuning this method are present in the TrainerConfig
        passed on Trainer initialization.
        """
        self.agent_grad_steps_at_start_of_train_step = self.total_agent_grad_steps
        if len(self._replay_buffer) > self.cfg.replay_buffer_episodes:
            self._replay_buffer = list(
                self._replay_buffer[-self.cfg.replay_buffer_episodes :]
            )
            assert len(self._replay_buffer) == self.cfg.replay_buffer_episodes
        self._maybe_record_starting_entropy()

        errors = 0
        start_time = time.monotonic()
        if (
            self.cfg.train_step_timeout_seconds is not None
            and self.cfg.train_step_timeout_seconds > 0
        ):
            # Allow extra time for first train_step
            if self.train_steps_so_far == 0:
                deadline = (
                    start_time
                    + self.cfg.first_train_step_timeout_coef
                    * self.cfg.train_step_timeout_seconds
                )
            else:
                deadline = start_time + self.cfg.train_step_timeout_seconds
        else:
            deadline = start_time + float("inf")

        timed_out = False

        self.agent.train(mode=True)
        self.vf.train(mode=True)

        # Cached policy outputs for VF training.
        # This avoids performing a full pass on the agent network when tuning
        # the VF outside of the primary loss.
        # The VF loss still tunes the full network in the primary loss.
        state_encoding_cache = {}
        action_lls_cache = {}

        pre_train_epochs = self.cfg.vf_pre_training_epochs
        if self.train_steps_so_far == 0:
            pre_train_epochs = max(self.cfg.vf_warmup_training_epochs, pre_train_epochs)

        with torch.no_grad():
            for _, _, episode_data, agent_output in self._agent_minibatches(
                desc="Caching latents", need_full=True
            ):
                for episode_id, state_encodings, action_lls in _split_agent_output(
                    episode_data, agent_output
                ):
                    state_encoding_cache[episode_id] = state_encodings.detach()
                    action_lls_cache[episode_id] = action_lls.detach()

        # Pre-train VF (usually only used for off-policy algorithms)
        if pre_train_epochs > 0:
            timed_out = self._train_vf(
                state_encoding_cache,
                action_lls_cache,
                pre_train_epochs,
                desc="Pre Training VF",
                deadline=deadline,
            )

        loss_inputs = self._prepare_loss_inputs(state_encoding_cache, action_lls_cache)
        self._log_dataset(loss_inputs)

        state_encoding_cache = {}
        action_lls_cache = {}

        # Run primary training loop.
        for _, batch_i, episode_data, agent_output in self._agent_minibatches(
            desc="Training Agent",
            epochs=self.cfg.policy_epochs_per_train_step,
            shuffle=True,
            minibatch_target_timesteps=self.cfg.minibatch_target_timesteps,
            need_full=False,
        ):
            if time.monotonic() > deadline:
                timed_out = True
                break
            if agent_output.full_valid():
                for episode_id, state_encodings, action_lls in _split_agent_output(
                    episode_data, agent_output
                ):
                    state_encoding_cache[episode_id] = state_encodings.detach()
                    action_lls_cache[episode_id] = action_lls.detach()
            loss_input = LossInput.pack(
                [loss_inputs[ep_data.episode_id] for ep_data in episode_data]
            )
            loss, loss_infos = self._primary_loss_function(loss_input, agent_output)
            self.vf_optimizer.zero_grad()
            self.agent_optimizer.zero_grad()
            loss.backward()
            self.total_agent_grad_steps += 1
            if (
                self.cfg.log_grad_step_period > 0
                and batch_i % self.cfg.log_grad_step_period == 0
            ):
                noko.log_row(
                    "train_locals",
                    loss_infos,
                    level=noko.TRACE,
                    step=self.total_agent_grad_steps,
                )
            try:
                clip_grad_norm_(
                    self.agent.parameters(),
                    max_norm=self.cfg.grad_norm_max,
                    error_if_nonfinite=True,
                )
                clip_grad_norm_(
                    self.vf.parameters(),
                    max_norm=self.cfg.grad_norm_max,
                    error_if_nonfinite=True,
                )
                self.vf_optimizer.step()
                self.agent_optimizer.step()
            except RuntimeError as ex:
                # This seems to only trigger if the batch is so small the loss
                # is NaN or so big we OOM.
                # Because we checked for non-finite in grad_norm, this should
                # reliably prevent corrupting the network, at the cost of
                # potentially extremely slowing training on e.g. an over-fit
                # VF or excessively off-policy data.
                _LOGGER.error(f"RuntimeError in agent optimizations: {ex}")
                errors += 1
                if errors > self.cfg.max_permitted_errors_per_train_step:
                    raise ex
            if self.cfg.recompute_loss_inputs:
                loss_inputs = self._preprocess()

        # Extra VF tuning after the primary training loop.
        # This achieves similar objectives to PPG by simply not doing updates
        # on the agent network in this phase.
        # Inputs are guaranteed to be cached, since we ran at least one full
        # epoch in the primary loop.
        if self.cfg.vf_post_training_epochs > 0 and not timed_out:
            missing_episodes = [
                episode_data
                for episode_data in self._replay_buffer
                if episode_data.episode_id not in state_encoding_cache
            ]
            with torch.no_grad():
                for _, _, episode_data, agent_output in self._agent_minibatches(
                    episodes=missing_episodes, desc="Caching latents", need_full=True
                ):
                    for episode_id, state_encodings, action_lls in _split_agent_output(
                        episode_data, agent_output
                    ):
                        state_encoding_cache[episode_id] = state_encodings.detach()
                        action_lls_cache[episode_id] = action_lls.detach()
            timed_out = self._train_vf(
                state_encoding_cache,
                action_lls_cache,
                self.cfg.vf_post_training_epochs,
                desc="Post Training VF",
                deadline=deadline,
            )

        # Update all the statistics.
        self.agent_lr_scheduler.step()
        self.vf_lr_scheduler.step()
        self.agent.train(mode=False)
        self.vf.train(mode=False)
        self.train_steps_so_far += 1
        self._maybe_record_starting_entropy()
        self.agent_grad_steps_last_train_step = (
            self.total_agent_grad_steps - self.agent_grad_steps_at_start_of_train_step
        )
        self.agent_grad_steps_at_start_of_train_step = self.total_agent_grad_steps

        if timed_out:
            _LOGGER.error("train_step() timed out")

    def _train_vf(
        self,
        state_encodings: dict[EpisodeID, torch.Tensor],
        action_lls: dict[EpisodeID, torch.Tensor],
        training_epochs: int,
        desc: str,
        deadline: float,
    ):
        """Train just the VF using cached agent outputs.

        state_encodings and action_lls are indexed by the `episode_id`
        field of _EpisodeData, and should contain non-differentiable cached
        components from each episode's AgentOutput.

        This method does not tune the parameters of the agent.

        Because this training is only tuning the memoryless VF tail, it uses
        smaller minibatches of shuffled timesteps from across multiple
        episodes.
        """
        state_enc_packed, obs_lens = pack_tensors(
            [state_encodings[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        action_lls_now = pad_tensors(
            [action_lls[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        assert not state_enc_packed.requires_grad
        assert not action_lls_now.requires_grad

        padded_rewards = pad_tensors([data.rewards for data in self._replay_buffer]).to(
            device=self.device
        )
        if self.cfg.normalize_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)
        else:
            rewards_normed = padded_rewards

        original_action_lls = pad_tensors(
            [data.original_action_lls for data in self._replay_buffer]
        )

        discount = 1 - self.cfg.discount_inv
        gammas = discount * torch.ones_like(rewards_normed)

        episode_lengths = [len(data.rewards) for data in self._replay_buffer]
        terminated = torch.zeros(len(self._replay_buffer), dtype=torch.bool)

        with tqdm(desc=desc, total=training_epochs * sum(episode_lengths)) as pbar:
            for epoch in range(training_epochs):
                if (
                    epoch == 0
                    or epoch == training_epochs
                    or self.cfg.vf_recompute_targets
                ):
                    vf_x_packed = self.vf(state_enc_packed)
                    vf_x = pad_packed(vf_x_packed, obs_lens)

                    for i, episode_length in enumerate(episode_lengths):
                        # zero vf_{t+1} in terminated episodes
                        if self._replay_buffer[i].terminated:
                            vf_x[i, episode_length] = 0.0
                            terminated[i] = True
                    terminated = terminated.to(device=self.device)

                    with torch.no_grad():
                        _, vf_targets = _v_trace_estimation(
                            lmbda=self.cfg.v_trace_lambda,
                            rho_max=self.cfg.v_trace_rho_max,
                            c_max=self.cfg.v_trace_c_max,
                            gammas=gammas,
                            vf_x=vf_x,
                            rewards=rewards_normed,
                            action_lls=action_lls_now,
                            original_action_lls=original_action_lls,
                            terminated=terminated,
                            episode_lengths=torch.tensor(episode_lengths),
                        )

                    vf_targets_packed = pack_padded(vf_targets, obs_lens)

                # pyright doesn't understand that epoch 0 is guaranteed to happen first
                dataset = DictDataset(
                    state_encodings=state_enc_packed, vf_targets=vf_targets_packed
                )
                for batch in dataset.minibatches(self.cfg.vf_minibatch_size):
                    self.vf_optimizer.zero_grad()
                    vf_out = self.vf(batch["state_encodings"])
                    vf_loss = self.cfg.vf_loss_coef * F.mse_loss(
                        vf_out, batch["vf_targets"]
                    )
                    vf_loss.backward()
                    # If we have a NaN in this update, it's probably best to
                    # just crash, since something is very wrong with the
                    # training run.
                    clip_grad_norm_(
                        self.vf.parameters(),
                        max_norm=self.cfg.grad_norm_max,
                        error_if_nonfinite=True,
                    )
                    self.vf_optimizer.step()
                    pbar.n += len(batch["state_encodings"])
                    pbar.refresh()
                    if time.monotonic() > deadline:
                        return True
        return False

    def _log_dataset(self, loss_inputs: dict[EpisodeID, LossInput]):
        loss_input = LossInput.pack(list(loss_inputs.values()))
        full_episode_rewards = pad_tensors(
            [data.rewards for data in self._replay_buffer]
        )
        ep_lengths = [len(data.rewards) for data in self._replay_buffer]
        discounted_returns = pack_padded(
            discount_cumsum(full_episode_rewards, discount=1 - self.cfg.discount_inv),
            ep_lengths,
        )
        dataset_stats = {
            "episode_total_rewards": full_episode_rewards.sum(dim=-1).mean(dim=0),
            "vf_explained_variance": explained_variance(
                loss_input.vf_returns,
                discounted_returns,
            ),
            "discounted_rewards": discounted_returns.mean(),
            "vf_mean": loss_input.vf_returns.mean(),
            "vf_target_mean": loss_input.vf_targets.mean(),
            "total_env_steps": self.total_env_steps,
        }
        for k in self._replay_buffer[0].infos.keys():
            dataset_stats[k] = force_concat(
                data.infos[k] for data in self._replay_buffer
            )

        noko.log_row(
            "dataset_stats",
            dataset_stats,
            level=noko.RESULTS,
            step=self.total_env_steps,
        )

    def _prepare_loss_inputs(
        self,
        state_encodings: dict[EpisodeID, torch.Tensor],
        action_lls: dict[EpisodeID, torch.Tensor],
    ) -> dict[EpisodeID, LossInput]:
        """Compute training inputs from the replay buffer and value function.

        Mostly this consists of the advantages and value function
        targets. See the LossInput class for more
        information.
        """
        state_enc_packed, obs_lens = pack_tensors(
            [state_encodings[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        action_lls_now = pad_tensors(
            [action_lls[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        assert not state_enc_packed.requires_grad
        assert not action_lls_now.requires_grad

        episode_lengths = [len(data.rewards) for data in self._replay_buffer]
        obs_lens = [len(data.episode["observations"]) for data in self._replay_buffer]
        assert [ep_len + 1 for ep_len in episode_lengths] == obs_lens

        # Compute vf_returns
        self.agent.train(mode=False)
        self.vf.train(mode=False)

        with torch.no_grad():
            # TODO: split up this call if necessary
            vf_returns_packed = self.vf(state_enc_packed)
        self.agent.train(mode=True)
        self.vf.train(mode=True)

        original_action_lls = pad_tensors(
            [data.original_action_lls for data in self._replay_buffer]
        ).to(device=self.device)
        terminated = torch.zeros(len(self._replay_buffer), dtype=torch.bool)

        vf_returns = pad_packed(vf_returns_packed, obs_lens)
        # Can't use valids mask, since vf_returns goes to t + 1
        for i, episode_length in enumerate(episode_lengths):
            # Everything after last valid observation should have been padded to zero
            assert (vf_returns[i, episode_length + 1 :] == 0.0).all()

            # zero vf_{t+1} in terminated episodes
            if self._replay_buffer[i].terminated:
                vf_returns[i, episode_length] = 0.0
                terminated[i] = True
        terminated = terminated.to(device=self.device)

        padded_rewards = pad_tensors([data.rewards for data in self._replay_buffer]).to(
            device=self.device
        )
        if self.cfg.normalize_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)
        else:
            rewards_normed = padded_rewards

        discount = 1 - self.cfg.discount_inv
        gammas = discount * torch.ones_like(rewards_normed)

        padded_advantages, vf_targets = _v_trace_estimation(
            lmbda=self.cfg.v_trace_lambda,
            rho_max=self.cfg.v_trace_rho_max,
            c_max=self.cfg.v_trace_c_max,
            gammas=gammas,
            vf_x=vf_returns,
            rewards=rewards_normed,
            action_lls=action_lls_now,
            original_action_lls=original_action_lls,
            terminated=terminated,
            episode_lengths=torch.tensor(episode_lengths, device=self.device),
        )

        adv_packed = pack_padded(padded_advantages, episode_lengths)
        assert not adv_packed.requires_grad

        if self.cfg.normalize_awr_advantages:
            adv_packed = adv_packed - adv_packed.mean()
            adv_packed = adv_packed / adv_packed.std()

        # This 1000 is documented in the TrainerConfig.awr_temperature
        # docstring.
        if self.cfg.awr_temperature < 1000:
            heated_adv = adv_packed / self.cfg.awr_temperature
            max_exp_adv = torch.tensor(self.cfg.advantage_clip).exp()
            softmax_adv = softmax_clip(heated_adv, max_exp_adv)
            awr_clip_ratio = (softmax_adv == max_exp_adv).mean(dtype=torch.float32)
            normed_exp_adv = softmax_adv * len(softmax_adv)
            assert 0.9 <= normed_exp_adv.mean() <= 1.1
        else:
            awr_clip_ratio = 0.0
            normed_exp_adv = torch.ones_like(adv_packed)

        loss_inputs = {
            episode_data.episode_id: LossInput(
                advantages=adv,
                vf_returns=vf_ret,
                vf_targets=vf_target,
                exp_advantages=exp_adv,
                original_action_lls=episode_data.original_action_lls,
                episode_lengths=[episode_data.n_timesteps],
                original_action_dists=[episode_data.original_action_dists],
            )
            for (episode_data, adv, vf_ret, vf_target, exp_adv) in zip(
                self._replay_buffer,
                unpack_tensors(adv_packed, episode_lengths),
                unpad_tensors(vf_returns, episode_lengths),
                unpad_tensors(vf_targets, episode_lengths),
                unpack_tensors(normed_exp_adv, episode_lengths),
            )
        }
        assert len(loss_inputs) == len(self._replay_buffer)
        for data in self._replay_buffer:
            assert len(loss_inputs[data.episode_id].advantages) == len(data.rewards)

        used_for_logging(awr_clip_ratio)
        infos = locals()
        del infos["self"]
        del infos["episode_lengths"]
        del infos["obs_lens"]
        del infos["loss_inputs"]
        noko.log_row(
            "preprocess",
            infos,
            level=noko.TRACE,
            step=self.total_env_steps,
        )
        return loss_inputs

    def add_episode(
        self,
        episode: Any,
        rewards: torch.Tensor,
        action_lls: torch.Tensor,
        terminated: bool,
        action_dists: Optional[ActionDist] = None,
        any_actions_possible: Optional[torch.Tensor] = None,
        weight: float = 1.0,
        infos: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Add a new episode to the replay buffer.

        Args:

            episode (Any): The episode. Can be any value that the agent accepts
                (in a list) to its forward method.
            rewards (torch.Tensor): Float Tensor containing rewards achieved
                after taking each action.
            terminated (bool): True if the episode ended in a terminal state,
                and False if the episode timed-out, was abandoned, or is still
                ongoing.
            action_dists (Optional[ActionDist]): Optional original action
                distributions. Used to compute exact KL divergence and entropy
                regularization depending on the config.
            any_actions_possible (Optional[torch.Tensor]): Boolean Tensor
                indicating if any actions were possible to take at this state.
                This allows ignoring states where no action was possible (e.g.
                the initial prompt in an LLM, or intermediate frames in a video
                input when using frame-skip). Assumes always true if not
                provided.
            weight (float): Optional weight indicating importance of the
                episode. May be adjusted by the learner.
            infos (Optional[dict[str,torch.Tensor]]): extra information about
                the episode. Will be summarized and logged every training step.

        """
        assert isinstance(rewards, torch.Tensor)
        assert isinstance(action_lls, torch.Tensor)
        assert not action_lls.requires_grad
        n_timesteps = len(rewards)
        self.total_env_steps += n_timesteps
        if any_actions_possible is None:
            any_actions_possible = torch.ones(n_timesteps, dtype=torch.bool)
        else:
            assert isinstance(any_actions_possible, torch.Tensor)

        rewards = rewards.to(dtype=self._dtype)
        self.reward_normalizer.update(rewards)
        if infos is None:
            infos = {}

        self._replay_buffer.append(
            _EpisodeData(
                episode,
                episode_id=self._next_episode_id,
                n_timesteps=n_timesteps,
                terminated=terminated,
                original_action_lls=action_lls.to(dtype=self._dtype),
                original_action_dists=action_dists,
                rewards=rewards,
                any_actions_possible=any_actions_possible,
                weight=weight,
                infos=infos,
            )
        )
        self._next_episode_id += 1

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        """Add evaluation statistics for the current agent.

        Will be logged to {cfg.runs_dir}/{cfg.run_name}/eval_stats.csv.

        The primary stat should be present in stats and indicates how to choose
        the "best" agent for purposes of checkpointing and hyper-parameter
        tuning.
        """
        assert "primary" not in stats
        stats["primary"] = stats[primary]
        noko.log_row("eval_stats", stats, step=self.total_env_steps, level=noko.RESULTS)
        self.primary_performance = stats[primary]
        self.last_eval_stats = stats
        hparams = self.cfg.to_dict()
        hparams["metric-primary"] = stats[primary]
        for k, v in stats.items():
            hparams[k] = v
        noko.log_row("hparams", hparams, step=self.total_env_steps)
        _LOGGER.info(f"Eval primary stat ({primary}): {stats[primary]}")

    def attempt_resume(
        self, prefer_best: bool = False, checkpoint_dir: Optional[str] = None
    ):
        """Attempt to resume from the checkpoint directory.

        If checkpoint_dir is not passed, defaults to the current run directory:
        {cfg.runs_dir}/{cfg.run_name}.

        Prefers the most recent checkpoint, according to the train_step_[i}.pkl
        name, or best.pkl if it exists and prefer_best is True.
        """
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.cfg.runs_dir, self.cfg.run_name)
        best_ckpt = os.path.join(checkpoint_dir, "best.pkl")
        if prefer_best and os.path.exists(best_ckpt):
            try:
                with open(best_ckpt, "rb") as f:
                    state = pickle.load(f)
                    self.load_state_dict(state)
                    _LOGGER.critical(
                        f"Resuming from checkpoint {best_ckpt} which had {self.train_steps_so_far} train_steps"
                    )
                    return True
            except (pickle.UnpicklingError, ValueError) as ex:
                _LOGGER.error(f"Could not load {best_ckpt}: {ex}")
        checkpoints = glob(f"{checkpoint_dir}/train_step_*.pkl")
        with_idx = [
            (int(f_name.rsplit("_", 1)[-1].split(".", 1)[0]), f_name)
            for f_name in checkpoints
        ]
        for _, f_name in sorted(with_idx, reverse=True):
            try:
                with open(f_name, "rb") as f:
                    state = pickle.load(f)
                    self.load_state_dict(state)
                    _LOGGER.critical(
                        f"Resuming from checkpoint {f_name} which had {self.train_steps_so_far} train_steps"
                    )
                    return True
            except (pickle.UnpicklingError, ValueError) as ex:
                _LOGGER.error(f"Could not load {f_name}: {ex}")
        return False

    def maybe_checkpoint(self):
        """Checkpoint to the run directory, depending on config values.

        If cfg.checkpoint_best and the primary stat passed to add_eval_stats()
        is at a maximal value, checkpoints to {cfg.runs_dir}/{cfg.run_name}/best.pkl.

        Also periodically checkpoints to train_step_{i}.pkl every
        cfg.checkpoint_interval train_step() calls.
        """
        checkpoint_interval = (
            self.cfg.checkpoint_interval >= 0
            and self.train_steps_so_far - self.train_steps_so_far_at_last_checkpoint
            >= self.cfg.checkpoint_interval
        )
        checkpoint_best = (
            self.cfg.checkpoint_best
            and self.primary_performance > self.best_checkpoint_primary_performance
        )
        if checkpoint_interval or checkpoint_best:
            state_dict = self.state_dict()
            if checkpoint_interval:
                f_name = os.path.join(
                    self.cfg.runs_dir,
                    self.cfg.run_name,
                    f"train_step_{self.train_steps_so_far}.pkl",
                )
                if not os.path.exists(f_name):
                    _LOGGER.info(f"Checkpointing to {f_name!r}")
                    with open(
                        f_name,
                        "wb",
                    ) as f:
                        pickle.dump(state_dict, f)
                    self.train_steps_so_far_at_last_checkpoint = self.train_steps_so_far
                else:
                    _LOGGER.info(f"Checkpoint {f_name!r} already exists")
            if checkpoint_best:
                f_name = os.path.join(self.cfg.runs_dir, self.cfg.run_name, f"best.pkl")
                _LOGGER.info(f"Checkpointing to {f_name!r}")
                with open(f_name, "wb") as f:
                    pickle.dump(state_dict, f)
                self.best_checkpoint_primary_performance = self.primary_performance
            return True
        else:
            return False

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the Trainer as a dictionary.

        Return value shares Tensors with the Trainer fields.

        Note that the Trainer is *not* an nn.Module.

        The returned dictionary is not "flat", and contains nested state
        dictionaries for submodules, parameters, optimizers, and learning rate
        schedulers.
        """

        state = {}
        for k in _SUBMODULE_FIELDS + _OPTIMIZER_FIELDS:
            state[k] = getattr(self, k).state_dict()
        for k in _PARAM_FIELDS:
            state[k] = getattr(self, k).data
        for k, v in self.__dict__.items():
            if (
                k in _IGNORED_FIELDS
                or k in _OPTIMIZER_FIELDS
                or k in _SUBMODULE_FIELDS
                or k in _PARAM_FIELDS
            ):
                continue
            elif k == "cfg":
                state[k] = v.to_dict()
            elif k == "replay_buffer":
                if self.cfg.checkpoint_replay_buffer:
                    state[k] = v
            else:
                if hasattr(v, "state_dict"):
                    _LOGGER.error(
                        f"Field {k} was not expected to have a state_dict method"
                    )
                state[k] = v
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the state of the trainer from a dictionary.

        Note that the Trainer is *not* an nn.Module.
        """
        state = state_dict
        for k in _PARAM_FIELDS + _SUBMODULE_FIELDS + _OPTIMIZER_FIELDS:
            assert k in state, f"Missing {k!r} from state dict"
        for k, v in state.items():
            if k in _OPTIMIZER_FIELDS:
                # We need to handle these fields once all parameters are loaded
                continue
            elif k == "cfg":
                self.cfg = type(self.cfg).from_dict(v)
            elif k in _PARAM_FIELDS:
                setattr(self, k, nn.Parameter(v))
            elif k in _SUBMODULE_FIELDS:
                getattr(self, k).load_state_dict(v)
            else:
                missing = object()
                field_now = getattr(self, k, missing)
                if field_now is missing:
                    _LOGGER.error(f"Attempting to set unknown field {k}")
                else:
                    if hasattr(field_now, "load_state_dict"):
                        _LOGGER.error(
                            f"Field {k} was not expected to have a load_state_dict method"
                        )
                setattr(self, k, v)

        # Attach all of the optimizers again
        # This method depends cfg, and all parameter and sub-module fields
        self._setup_optimizers()

        # Now we can load the optimizer state dictionaries
        for k in _OPTIMIZER_FIELDS:
            getattr(self, k).load_state_dict(state[k])

        # Make sure optimizers are attached to parameters
        assert self.kl_coef_opt.param_groups[0]["params"][0] is self.kl_coef
        assert self.vf_optimizer.param_groups[0]["params"][0] is next(
            self.vf.parameters()
        )
        assert self.agent_optimizer.param_groups[0]["params"][0] is next(
            self.agent.parameters()
        )

    def _entropy_of(
        self, action_lls: torch.Tensor, action_dists: Optional[ActionDist]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the approximate entropy (and exact entropy, if possible)."""
        approx_entropy = approx_entropy_of(action_lls)
        entropy = None
        if not self.cfg.use_approx_entropy:
            try:
                # Mixed case is warned on add_episode()
                if action_dists is not None:
                    entropy = entropy_of(action_dists)
            except NotImplementedError:
                pass
        if entropy is None:
            entropy = approx_entropy
        return approx_entropy, entropy

    def _maybe_record_starting_entropy(self):
        """Record starting_etropy if train_steps_so_far >= cfg.entropy_schedule_start_train_step."""
        if (
            self.starting_entropy is None
            and self.train_steps_so_far >= self.cfg.entropy_schedule_start_train_step
        ):
            original_action_lls = torch.cat(
                [ep_data.original_action_lls for ep_data in self._replay_buffer]
            )
            original_dists = [
                ep_data.original_action_dists for ep_data in self._replay_buffer
            ]
            approx_entropy, entropy = self._entropy_of(
                original_action_lls, original_dists
            )
            del approx_entropy
            self.starting_entropy = entropy.mean().item()


@noko.declare_summarizer(Trainer)
def summarize_trainer(trainer, key, dst):
    """Summarize fields for the noko logging library."""
    for k in _SUBMODULE_FIELDS + _OPTIMIZER_FIELDS + _PARAM_FIELDS:
        noko.summarize(getattr(trainer, k), f"{key}.{k}", dst)
    for k, v in trainer.__dict__.items():
        if k not in _IGNORED_FIELDS and k not in "last_eval_stats":
            noko.summarize(v, f"{key}.{k}", dst)


class Agent(nn.Module):
    """Agent API optimized by OutRL.

    You do not need to actually inherit from this class, it exists for
    documentation purposes.
    """

    def __init__(self, state_encoding_size: int):
        super().__init__()

        self.state_encoding_size = state_encoding_size
        """Species the dimensionality of encoded states in the
        AgentOutput.state_encodings field."""

    def forward(self, inputs: "AgentInput") -> "AgentOutput":
        """Run the agent's forward pass. This is the main API entry point
        between your agent and OutRL's trainer.

        If you already need to use the forward pass for another purpose, you
        can define outrl_forward() instead, but note that this will not invoke
        torch forward hooks.

        Episodes can be any value passed to Trainer.add_episode().

        The inside of this function will usually look something like:
        ```
        logits, latents = self.net(torch.cat([ep["observations"] for ep in inputs.episodes]))
        dist = CategoricalDistribution(logits)
        action_lls = dist.log_prob(torch.cat([ep["actions"] for ep in inputs.episodes]))
        return AgentOutput(
            state_encodings=latents,
            action_lls=action_lls,
            action_dists=dist,
            valid_mask=torch.ones(sum([len(ep["observations"]) for ep in inputs.episodes])),
        )
        ```
        """
        del inputs
        raise NotImplementedError()


@dataclass(eq=False)
class AgentOutput:
    """Result from all episodes concatenated together when calling .forward()
    on an agent.

    Differentiating both state_encodings and action_lls should affect
    earlier layers in the agent.

    See `AgentInput` for the input type.
    """

    state_encodings: torch.Tensor
    """Differentiable representation of the observations. Because there should
    be a trailing final state, this tensor should be longer than action_lls by
    the number of episodes in AgentInput."""

    action_lls: torch.Tensor
    """Differentiable log-likelihood of actions taken."""

    action_dists: Optional[ActionDist] = None
    """Distribution used to generate actions.

    Will be used for the KL penalty if not None and cfg.use_approx_kl is False.

    Will be used for entropy regularization if not None and cfg.approx_entropy
    is False.
    """

    valid_mask: Optional[torch.Tensor] = None
    """Mask of timesteps where `state_encodings` and `action_lls` were computed
    and can be differentiated for.

    Mostly used for rescaling losses.
    If None, all timesteps are presumed to be valid.
    """

    def __post_init__(self):
        # The number of state_encoding dimensions is not constant and cannot be
        # checked here.
        assert len(self.state_encodings.shape) == 2
        assert len(self.action_lls.shape) == 1
        assert (
            self.state_encodings.shape[0] != self.action_lls.shape[0]
        ), "There should be one more state_encoding per episode than action_ll. "
        if self.valid_mask is not None:
            assert len(self.valid_mask.shape) == 1
            assert self.valid_mask.shape == self.state_encodings.shape
            assert self.valid_mask.dtype == torch.bool

    def full_valid(self):
        """If the AgentOutput has contains all timesteps for the requested
        episodes."""
        return self.valid_mask is None or self.valid_mask.all()


def _split_agent_output(
    episode_data: list[_EpisodeData], agent_output: AgentOutput
) -> list[tuple[EpisodeID, torch.Tensor, torch.Tensor]]:
    action_lens = [ep_data.n_timesteps for ep_data in episode_data]
    obs_lens = [ep_data.n_timesteps + 1 for ep_data in episode_data]
    state_encodings = unpack_tensors(agent_output.state_encodings, obs_lens)
    action_lls = unpack_tensors(agent_output.action_lls, action_lens)
    episode_ids = [ep_data.episode_id for ep_data in episode_data]
    assert len(episode_ids) == len(state_encodings)
    assert len(episode_ids) == len(action_lls)
    for i, ep_data in enumerate(episode_data):
        assert ep_data.n_timesteps == action_lls[i].shape[0]
    return list(zip(episode_ids, state_encodings, action_lls))


@dataclass(eq=False)
class AgentInput:
    """Input to Agent forward() method.

    See `AgentOutput` for the return type.
    """

    episodes: list[Any]
    """List of episodes. Will be whatever type was passed in to
    `Trainer.add_episode`.
    """

    need_full: bool
    """If True, the agent must return action_lls and state_encodings for the
    entire episode (i.e. `AgentOutput.valid_mask` should be None or all True).

    If False, the agent can optionally only propagate a portion of timesteps.
    """


def _call_agent_forward(
    agent: Agent, episodes: list[Any], need_full: bool, expected_lengths: list[int]
) -> AgentOutput:
    """Calls the outrl_forward method if available, otherwise calls the normal
    forward method via __call__.

    Does some basic checking of the shapes of the AgentOutput.
    """
    agent_input = AgentInput(episodes=episodes, need_full=need_full)
    if hasattr(agent, "outrl_forward"):
        output = agent.outrl_forward(agent_input)
    else:
        output = agent(agent_input)
    assert output.state_encodings.shape[0] == sum(expected_lengths) + len(
        expected_lengths
    )
    assert output.action_lls.shape[0] == sum(expected_lengths)
    return output


_GENERATED_FROM_TIME = "GENERATED_FROM_TIME"


# TrainerConfig is after Trainer because that improves the documentation order.


@dataclass
class TrainerConfig(simple_parsing.Serializable):
    """Config structure for the Trainer.

    Can be saved to and loaded from yaml.
    All fields have default values.
    Most fields can be tuned via optuna.

    Can be subclassed to add more options.
    """

    seed: int = random.randrange(10000)
    """Random seed for this experiment. Set to a random int < 10k if not
    specified.

    Set to -1 to disable setting the random seed.
    """

    run_name: str = _GENERATED_FROM_TIME
    """The name of this trainer run.

    Generated uniquely from the name of the main module and time by default."""

    runs_dir: str = "runs"
    """Directory to log into. Full path will be {runs_dir}/{run_name}.

    Set to None to disable logging."""

    stderr_log_level: noko.LogLevels = noko.LogLevels.INFO
    """Log level to stderr for noko and python logging."""

    pprint_logging: bool = True
    """Log to stdout using pprint. Because the pprint output engine defaults to
    the RESULTS log level, this defaults to True."""

    parquet_logging: bool = False
    """Log to parquet files using pyarrow."""

    tb_log_level: noko.LogLevels = noko.LogLevels.INFO
    """Log level to log to TensorBoard. Defaults to INFO to avoid slowing down
    TensorBoard with too many keys."""

    replay_buffer_episodes: int = 128
    """Maximum number of episodes to keep in replay buffer."""

    max_timesteps_per_forward: Optional[int] = None
    """Maximum number of timesteps to include in a forward pass to the agent.
    Used to avoid out-of-memory errors.

    Defaults to no limit. Automatically decreases on (most) memory errors.
    """

    minibatch_target_timesteps: int = tunable(1024, IntDistribution(1, 50000, log=True))
    """Attempt to keep timesteps in each minibatch to this number of timesteps.
    In practice, this acts a divisor on most losses.

    Will still run whole episodes if they exceed this cap.
    """

    policy_epochs_per_train_step: int = tunable(15, IntDistribution(1, 100, log=True))
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    normalize_rewards: bool = tunable(True, CategoricalDistribution([True, False]))
    """Normalize rewards to have zero mean and unit variance."""

    expected_train_steps: int = 1000
    """Expected number of training steps. Used for controlling scheduled parameters."""

    train_step_timeout_seconds: Optional[float] = None
    """train_step() will exit early if this number of seconds of wall-clock
    time is exceeded during it. The current gradient step will still finish
    first, so this timeout is only approximately enforced.

    See also `first_train_step_timeout_coef`, which increases the
    timeout for the very first train step to accomodate JIT
    compilation and VF warmup.
    """

    first_train_step_timeout_coef: float = 5.0
    """Multiple of the normal train_step() timeout to use for the
    first train_step. Useful for accomodating additional time
    needed for JIT compilation or VF warmup (see
    `vf_warmup_training_epochs`).

    Has no effect if train_step_timeout_seconds is None.
    """

    ppo_loss_coef: float = tunable(0.0, FloatDistribution(0.0, 1000.0))
    """Loss coefficient for the PPO loss. Usually unused, since the AWR loss is
    more flexible.
    """

    awr_loss_coef: float = tunable(1.0, FloatDistribution(0.0, 1000.0))
    """Loss coefficient for the main RL loss. Usually does not need to be
    tuned."""

    agent_lr_schedule: Literal[None, "linear", "cosine"] = tunable(
        "cosine", CategoricalDistribution([None, "linear", "cosine"])
    )
    """Learning rate schedule for the agent. Typically used to decrease the
    learning rate to near-zero near the end of training."""

    agent_lr_start: float = tunable(2e-4, FloatDistribution(1e-5, 5e-2, log=True))
    """Initial learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will be used throughout training. """

    agent_lr_end: float = tunable(1e-5, FloatDistribution(1e-8, 1e-3, log=True))
    """Final learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will not be used."""

    agent_weight_decay: float = tunable(1e-8, FloatDistribution(1e-10, 1e-2, log=True))
    """Weight decay for the agent using AdamW."""

    ppo_clip_epsilon: float = tunable(0.2, FloatDistribution(0.05, 2.0, log=True))
    """PPO loss will be clipped to only apply the loss when the log-likelihood
    is between 1 / (1 + ppo_clip_epsilon) and 1 + ppo_clip_epsilon.

    Because the ppo_loss is disabled by default, this field also has no effect by default.
    Because OutRL uses regularized VF training, VF clipping is not used.
    """

    vf_lr_schedule: Literal[None, "linear", "cosine"] = tunable(
        "cosine", CategoricalDistribution([None, "linear", "cosine"])
    )
    """Learning rate schedule for the value function parameters. Typically used
    to decrease the learning rate to near-zero near the end of training."""

    vf_lr_start: float = tunable(2e-3, FloatDistribution(1e-4, 0.1, log=True))
    """Initial learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will be used throughout
    training. """

    vf_lr_end: float = tunable(1e-5, FloatDistribution(1e-8, 1e-4, log=True))
    """Final learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will not be used. """

    vf_weight_decay: float = tunable(1e-5, FloatDistribution(1e-8, 1e-2, log=True))
    """Weight decay for the value function parameters using AdamW."""

    vf_minibatch_size: int = tunable(64, IntDistribution(1, 2**32, log=True))
    """Number of timesteps used in minibatches when pre-training and
    post-training the value function from frozen state encodings.

    Because the value function uses a memoryless architecture (the VF relies on
    the agent to encode memory, if necessary), this minibatch size is typically
    met precisely (except for one trailing odd-sized minibatch).
    """

    vf_warmup_training_epochs: int = tunable(30, IntDistribution(0, 1000))
    """Number of epochs of value function training to run from
    frozen state encodings on the very first train_step() before
    training the agent.

    Because OutRL uses an AWR-style loss, training the VF before
    the policy is expected.
    """

    vf_pre_training_epochs: int = tunable(3, IntDistribution(0, 20))
    """Number of epochs of value function training to run from frozen state
    encodings each train_step() before training the agent.

    Because OutRL uses an AWR-style loss, training the VF before the policy is
    expected.
    """

    vf_post_training_epochs: int = tunable(3, IntDistribution(0, 20))
    """Number of epochs of value function training to run from frozen state
    encodings each train_step() after training the agent.

    Because the value function is also tuned with the agent, this pass mostly
    serves to allow the value function to "catch up" to changing state
    encodings before the next train_step begins.
    """

    vf_recompute_targets: bool = tunable(True, CategoricalDistribution([True, False]))
    """If true, value function targets will be recomputed every epoch of value
    function optimization.

    This allows the value function to make predictions based on "mixing"
    advantages from multiple episodes, as is typical in Q-learning based
    algorithms.

    If your environment is partially observable, disabling this option may
    improve training reliability.
    """

    vf_loss_coef: float = tunable(0.1, FloatDistribution(1e-6, 1.0, log=True))
    """Coefficient to apply to the value function loss.

    Losses are usually around unit-scale by default. This coefficient being
    smaller than the agent loss coefficient(s) encourages the agent to focus on
    performing well, and only producing good state encodings as a secondary
    priority.

    Contrary to comments in some other frameworks, this hyper parameter has a
    very large effect!

    To keep the gradient scale on the value function consistent between the
    value function and agent training phases, this coefficient is applied in
    both cases.
    """

    vf_hidden_sizes: list[int] = tunable(
        [128, 128],
        IntListDistribution(
            [
                16,
            ],
            [256, 256, 256],
        ),
    )
    """Size of latent representations used in the value function to predict
    future returns from state encodings.

    Value function training is regularized with dropout, and value function
    training is relatively fast, so there is little disadvantage to making the
    value function wider.

    Defaults to [128, 128].
    """

    discount_inv: float = tunable(0.01, FloatDistribution(1e-5, 0.1, log=True))
    """Discount, expresseed such that gamma = 1 - discount_inv to allow
    hyper-parameter tuning in log space.

    This hyper parameter can have a very significant effect.
    """

    v_trace_lambda: float = tunable(0.33, FloatDistribution(0.0, 1.0))
    """Lambda parameter to v-trace advantage estimation.

    Controls the bias-variance tradeoff between pure belmann bootstraps and
    monte-carlo estimates. A value of 1 corresponds to minimal bias, and
    matches v-trace as originally proposed.
    """

    v_trace_rho_max: float = tunable(3.0, FloatDistribution(1.0, 1e3, log=True))
    """The "value function truncation" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which rewards from increased likelihood
    off-policy actions can contribute to estimated advantages.
    """

    v_trace_c_max: float = tunable(3.0, FloatDistribution(1.0, 1e3, log=True))
    """The "trace-cutting" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which future value function estimates
    from increased likelihood off-policy states can contribute to estimated
    advantages.
    """

    kl_coef_init: float = tunable(0.1, FloatDistribution(0.0, 100.0))
    """Initial loss coefficient for KL penalty / regularization of the agent.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).

    This penalty is applied exactly if action_dists is provided by the agent,
    or applied approximately using action_lls otherwise.
    """

    kl_coef_lr: float = tunable(0.01, FloatDistribution(1e-6, 0.1, log=True))
    """How quickly to adapt the loss coefficient for KL penalty.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).
    """

    kl_coef_min: float = tunable(0.01, FloatDistribution(1e-3, 1.0, log=True))
    """Minimum value of the KL coefficient. Setting this to a non-zero value
    can help stabilize training very low-entropy continuous action space
    policies using the PPO loss, but is typically unnecessary.
    """

    kl_coef_max: float = tunable(100.0, FloatDistribution(1e-2, 1e6, log=True))
    """Maximum value of the KL coefficient. Necessary to ensure eventual
    convergence of the KL penalty.

    If you are experiencing crashes due to NaNs when the kl_coef is high,
    decrease this value.
    """

    kl_target_stat: Literal["mean", "max"] = tunable(
        "mean", CategoricalDistribution(["mean", "max"])
    )
    """What statistic of the KL divergence to constrain.

    Constraining the mean KL divergence is typical, but constraining the max KL
    can improve stability during long runs with little disadvantage.
    """

    kl_soft_target: float = tunable(0.5, FloatDistribution(1e-3, 10.0, log=True))
    """Target per-timestep KL divergence per train-step.

    If this value is exceeded, the kl_coef will become non-zero to limit the
    training step size.

    Because this value is enforced using a lagrangian, KL steps are often 2-3x
    this target.
    """

    kl_fixup_coef: float = tunable(3, FloatDistribution(1.1, 20.0, log=True))
    """Multiple of the kl_soft_target to strictly enforce when kl_use_fixup is
    True.

    Low values of this parameter may drastically increase the compute time used
    by the fixup phase.
    """

    kl_use_fixup: bool = False
    """Strictly enforce a KL limit using a fixup phase."""

    use_approx_kl: bool = False
    """Approximate the KL divergence using action log-likelihoods even if exact
    action distributions are provided by the agent."""

    use_approx_entropy: bool = False
    """Approximate the action entropy using action log-likelihoods even if
    exact action distributions are provided by the agent."""

    entropy_schedule: Literal[None, "linear", "cosine"] = tunable(
        "cosine", CategoricalDistribution([None, "linear", "cosine"])
    )
    """Whether to schedule an entropy loss.

    With None, no entropy loss will be applied.

    With "linear", entropy will be scaled down from the initial entropy at
    start of training to a fraction of that `entropy_schedule_end_fraction`.
    """

    entropy_schedule_end_target: Optional[float] = None
    """Target entropy at end of schedule.

    Overrides entropy_schedule_end_fraction.
    """

    entropy_schedule_end_fraction: float = tunable(
        0.01, FloatDistribution(1e-6, 1.0, log=True)
    )
    """Portion of "starting entropy" to attempt to maintain at end of
    training.

    Only used if entropy_schedule_end_target is None."""

    entropy_schedule_start_train_step: int = 1
    """Train step at which to measure the "starting entropy".

    This indicates at the end of which train step entropy should be measured.
    The default value measures the entropy after one train step.
    """

    entropy_loss_coef: float = tunable(1e-4, FloatDistribution(0.0, 1.0))
    """Entropy coefficient.

    Coefficient to apply to entropy loss.
    By default the entropy loss is a mean squared error relative to
    the entropy schedule.
    """

    awr_temperature: float = tunable(0.01, FloatDistribution(1e-2, 1e3, log=True))
    """AWR temperature.

    Very low values result in a sparse loss that only attempts to repeat
    very high-advantage actions.

    High values cause the AWR loss to ignore advantages and just perform
    behavioral cloning.

    If set to >=1000, will literally just perform behavioral cloning.
    """

    normalize_awr_advantages: bool = tunable(
        True, CategoricalDistribution([False, True])
    )
    """Whether to normalize the advantages across the batch before computing
    the AWR coefficients.

    Note that this is not used in the PPO loss.
    """

    advantage_clip: float = tunable(8.0, FloatDistribution(0.1, 12.0))
    """Max exponent value for advantages in AWR loss.

    Large values can lead to NaN errors.

    Can have a significant effect, since many timesteps will have clipped
    coefficients at low temperatures.
    """

    grad_norm_max: float = tunable(5.0, FloatDistribution(1.0, 1e2, log=True))
    """Grad norm to clip the actor and value function parameters to in the main
    training loop.

    Small values will consistently lower the loss.
    """

    recompute_loss_inputs: bool = tunable(False, CategoricalDistribution([False, True]))
    """Recompute advantages and VF targets every epoch inside a train_step().

    This causes full off-policy steps to be taken every train_step().
    """

    checkpoint_interval: int = 1
    """Number of train_step() calls between checkpoints when calling
    maybe_checkpoint().

    If set to 0, maybe_checkpoint() will always checkpoint if the checkpoint
    file does not already exist for the current train step.

    Disable periodic checkpointing by setting to a negative value."""

    checkpoint_best: bool = True
    """Whether to checkpoint in maybe_checkpoint() after an improvement in the
    primary performance statistic passed to add_eval_stats()."""

    checkpoint_replay_buffer: bool = True
    """Whether to include the replay_buffer in the checkpoint.

    Needed for fully reproducible resume, but can have significant time and
    disk-space costs if the replay buffer is larger than the agent and
    checkpointing is being performed frequently.
    """

    log_grad_step_period: int = 20
    """Log information every n training gradient steps.

    This has moderate time costs if set to very low values (e.g. 1).

    Set to -1 to disable logging train_locals (likely a good idea if training
    large models on GPU).
    """

    max_permitted_errors_per_train_step: int = 10
    """Number of times to permit RuntimeError in each train_step() before
    re-raising the error.

    By default, occasional errors are caught and logged. Occasional anomalies
    from extremely unlikely events and torch bugs are usually prevented from
    crashing the run by clipping the grad norm.

    Many errors at once usually indicate that training cannot continue, so the
    error should be re-raised to avoid wasting time.
    """

    device: str = "cpu"
    """PyTorch device to use for optimization."""

    def __post_init__(self):
        """Fill in values with non-constant defaults. Called after construction."""
        if self.seed < -1:
            raise ValueError("seed should be positive or exactly -1")
        if self.run_name == _GENERATED_FROM_TIME:
            object.__setattr__(self, "run_name", default_run_name())
        if self.checkpoint_interval < -1:
            raise ValueError("checkpoint_interval should be positive or exactly -1")
        if self.kl_target_stat == "max" and self.use_approx_kl:
            raise ValueError("Cannot used kl_target_stat='max' with approximated KL.")
        if self.log_grad_step_period <= 0:
            assert self.log_grad_step_period == -1
        runs_dir = os.path.abspath(self.runs_dir)
        object.__setattr__(self, "runs_dir", runs_dir)
        if isinstance(self.stderr_log_level, str):
            stderr_log_level = noko.LOG_LEVELS[self.stderr_log_level]
            object.__setattr__(self, "stderr_log_level", stderr_log_level)


@torch.jit.script
def _v_trace_estimation(
    *,
    lmbda: float,
    rho_max: float,
    c_max: float,
    gammas: torch.Tensor,
    vf_x: torch.Tensor,
    rewards: torch.Tensor,
    action_lls: torch.Tensor,
    original_action_lls: torch.Tensor,
    terminated: torch.Tensor,
    episode_lengths: torch.Tensor,
):
    """Calculate value function targets using a V-Trace like estimator.

    Most inputs are 2D tensors.
    All 2D tensor inputs have batch as first dimension and time as second dimension.

    When rho_max and c_max are set infinitely high (or the data is "completely
    on-policy"), this function is equivalent to TD(lambda).

    See page 3 of "IMPALA" from https://arxiv.org/abs/1802.01561

    Args:
        lmbda (float): Lambda parameter that controls the bias-variance tradeoff.
        rho_max (float): The "VF truncation" importance weight clip.
        c_max (float): The "trace-cutting" importance weight clip.
        gammas (torch.Tensor): A 2D tensor of per-timestep discount
            coefficients. Used to avoid discounting across states where no
            action was possible.
        vf_x (torch.Tensor): A 2D tensor of value function estimates with shape
            (N, T + 1), where N is the batch dimension (number of episodes) and
            T is the maximum episode length experienced by the agent. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        rewards (torch.Tensor): A 2D tensor of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the agent.
        episode_lengths (torch.Tensor) A 1D tensor indicating the episode length.
        terminated (torch.Tensor): A 1D tensor indicating if the episode
            ended in a terminal state.

    Returns:
        A tuple containing:
            torch.Tensor: A 2D tensor of estimated advantages.
            torch.Tensor: A 2D tensor of VF targets.
    """
    # The paper says to assume rho_max >= c_max, but the math should still work
    # either way.

    assert len(rewards.shape) == 2
    n_episodes = rewards.shape[0]
    max_episode_length = rewards.shape[1]
    assert vf_x.shape == (n_episodes, max_episode_length + 1)
    assert action_lls.shape == (n_episodes, max_episode_length)
    assert original_action_lls.shape == (n_episodes, max_episode_length)
    assert gammas.shape == (n_episodes, max_episode_length)

    importance_weight = (action_lls - original_action_lls).exp()
    rho = importance_weight.clamp_max(rho_max)

    # Set gamma = 0 on the last timestep of terminated episodes
    ep_indices = torch.arange(n_episodes)
    gammas[ep_indices, episode_lengths - 1] *= (~terminated).to(dtype=gammas.dtype)
    assert bool(gammas[0, episode_lengths[0] - 1]) == (not terminated[0])

    # Multiply in the lambda term (not present in standard V-Trace, but matches
    # TD(lambda)).
    c = lmbda * importance_weight.clamp_max(c_max)

    gamma_c = gammas * c

    delta_V = rho * (rewards + gammas * vf_x[:, 1:] - vf_x[:, :-1])

    # In the paper: v_{s_t} - V(x_t)
    # Will be modified in-place except for last time index
    v_diff = torch.zeros_like(vf_x)

    # Can't use reversed in torchscript :(
    # Start at max_episode_length - 1 and go down to 0
    for t in range(max_episode_length - 1, -1, -1):
        # Main V-Trace update
        v_diff[:, t] = delta_V[:, t] + gamma_c[:, t] * v_diff[:, t + 1]

    v_s = v_diff + vf_x

    # Note the time offset. Can't use v_diff here!
    advantages = rho * (rewards + gammas * v_s[:, 1:] - vf_x[:, :-1])
    return advantages, v_s
