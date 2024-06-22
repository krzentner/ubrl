"""
All purely reinforcement learning code for PyTorch should be in this file.

TorchTrainer contains the primary interface to ubrl.
"""

import dataclasses
from dataclasses import dataclass
from textwrap import dedent
from typing import Any, Optional, Generator
import os
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

import noko

from ubrl.torch_utils import (
    approx_entropy_of,
    entropy_of,
    force_concat,
    make_mlp,
    pack_tensors,
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
    truncate_packed,
    concat_lists,
)
import ubrl.torch_cluster
from ubrl.config import TrainerConfig

_LOGGER = logging.getLogger("ubrl")

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
    """Copied from TorchTrainer.add_episode(), these are the original log
    likelihoods of the actions taken when the data was collected."""

    episode_lengths: list[int]

    original_action_dists: Optional[list[ActionDist]]
    """Copied from TorchTrainer.add_episode()."""

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

_IGNORED_FIELDS = ["_is_full_backward_hook", "cluster"]


class TorchTrainer:
    """An implementation of a bespoke reinforcement learning algorithm.

    The trainer trains an Agent to acquire higher rewards according to the data
    provided to it. To allow repeatedly adding new data to the TorchTrainer, it is a
    class instead of a function.

    The typical sequence of methods called on this class are as follows:

        1. Construction using a TrainerConfig (the default value is typically
           fine) and Agent. Optionally, call TorchTrainer.attempt_resume() to resume
           from a prior checkpoint.

        2. Repeated calls to TorchTrainer.add_episode() to add new data and
           TorchTrainer.train_step() to process the data. Typically multiple
           episodes (on the order of 10-100) should be added between each
           TorchTrainer.train_step() call.

        3. Periodic calls to TorchTrainer.add_eval_stats() and
           TorchTrainer.maybe_checkpoint().
    """

    def __init__(self, cfg: "TrainerConfig", agent: "Agent"):
        """Constructs a TorchTrainer."""

        super().__init__()

        n_params = sum(p.numel() for p in agent.parameters())
        cfg = cfg.choose_device(n_params=n_params)

        self.cfg: "TrainerConfig" = cfg
        """The configuration used to contruct the TorchTrainer.

        Modifying this field after constructing the TorchTrainer *may* change the
        TorchTrainer's behavior, but is not guaranteed to and should be avoided
        (except via load_state_dict()).
        """
        self.cluster = ubrl.torch_cluster.DefaultCluster(self.cfg)

        # Will be passed through accelerator in _setup_optimizers
        self.agent: "Agent" = agent
        """The agent being optimized. Provides action (log-likelihoods) and
        state encodings."""

        self._state_encoding_size = self.agent.state_encoding_size

        # Will be passed through accelerator in _setup_optimizers
        self.vf: nn.Module = make_mlp(
            input_size=self._state_encoding_size,
            hidden_sizes=self.cfg.vf_hidden_sizes,
            output_size=0,
            use_dropout=True,
        )
        """The value function. Feed-forward networks that predicts future
        rewards from state_encodings."""

        # Zero the initial VF output to stabilize training
        vf_output = self.vf.get_submodule("output_linear")
        vf_output.weight.data.copy_(0.01 * vf_output.weight.data)

        self.reward_normalizer: RunningMeanVar = self.cluster.prepare_module(
            RunningMeanVar(use_mean=False)
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

        # TODO(krzentner): Figure out if kl_coef should be on cpu or used through cluster
        # Maybe it would be simpler to just make kl_coef a nn.Linear(1, 1, bias=False)
        self.kl_coef: nn.Parameter = nn.Parameter(
            torch.tensor(float(self.cfg.kl_coef_init))
        )
        """Dynamically adjusted parameter used for KL regularization."""

        self.starting_entropy: Optional[float] = None
        """Initial mean entropy of action distributions, measured immediately
        at the start of cfg.entropy_schedule_start_train_step."""

        self._replay_buffer: list[_EpisodeData] = []
        self._next_episode_id: EpisodeID = 0
        self._loss_dtype = torch.float32

        self.total_agent_grad_steps = 0
        self.agent_grad_steps_at_start_of_train_step = 0
        self.agent_grad_steps_last_train_step = 0

        # This method is also called after loading the state dict to re-attach
        # parameters
        self._setup_optimizers()

    def _setup_optimizers(self):
        """(Re)create all of the optimizers to use the current parameters.

        This method is called in __init__ and also in load_state_dict().
        """
        self.agent, self.agent_optimizer = self.cluster.prepare_module_opt(
            self.agent,
            torch.optim.AdamW(
                self.agent.parameters(),
                lr=self.cfg.agent_lr_start,
                weight_decay=self.cfg.agent_weight_decay,
            ),
        )
        self.agent_lr_scheduler = make_scheduler(
            self.agent_optimizer,
            self.cfg.agent_lr_schedule,
            self.cfg.agent_lr_start,
            self.cfg.agent_lr_end,
            self.cfg.expected_train_steps,
        )
        self.vf, self.vf_optimizer = self.cluster.prepare_module_opt(
            self.vf,
            torch.optim.AdamW(
                self.vf.parameters(),
                lr=self.cfg.vf_lr_start,
                weight_decay=self.cfg.vf_weight_decay,
            ),
        )
        self.vf_lr_scheduler = make_scheduler(
            self.vf_optimizer,
            self.cfg.vf_lr_schedule,
            self.cfg.vf_lr_start,
            self.cfg.vf_lr_end,
            self.cfg.expected_train_steps,
        )

        self.kl_coef_opt = torch.optim.AdamW([self.kl_coef], lr=self.cfg.kl_coef_lr)

    def _primary_loss_function(
        self, agent_output: "AgentOutput", loss_input: LossInput
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reinforcement learning loss function.

        Combines together losses from many sources to train both the agent and
        value function.

        Returns a loss which is usually near unit scale and a dictionary of
        infos to log for this gradient step.

        Note that lagrangian losses (for the KL and entropy) are optimized in
        their related loss functions.
        """
        kl_loss, kl_infos = self._kl_loss(agent_output, loss_input)
        entropy_loss, entropy_infos = self._entropy_loss(agent_output, loss_input)
        ppo_loss, ppo_infos = self._ppo_loss(agent_output, loss_input)
        awr_loss, awr_infos = self._awr_loss(agent_output, loss_input)
        vf_loss, vf_infos = self._vf_loss(agent_output, loss_input)

        if agent_output.inherent_loss is not None:
            inherent_loss = (
                self.cfg.inherent_loss_coef * agent_output.inherent_loss.mean()
            )
        else:
            inherent_loss = 0.0

        loss = ppo_loss + awr_loss + vf_loss + kl_loss + entropy_loss + inherent_loss

        # *_infos will all get included in locals of this method
        used_for_logging(kl_infos, ppo_infos, awr_infos, vf_infos, entropy_infos)
        return loss, locals()

    def _ppo_loss(
        self, agent_output: "AgentOutput", loss_input: LossInput
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
        self, agent_output: "AgentOutput", loss_input: LossInput
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Dividing by temperature and normalizing are handled in TorchTrainer.preprocess()
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
        self, agent_output: "AgentOutput", loss_input: LossInput
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
        self, agent_output: "AgentOutput", loss_input: LossInput
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
        if self.starting_entropy is None:
            return None

        agent_grad_steps_this_train_step = (
            self.total_agent_grad_steps - self.agent_grad_steps_at_start_of_train_step
        )
        progress_at_train_step = min(
            1,
            agent_grad_steps_this_train_step
            / max(1, self.agent_grad_steps_last_train_step),
        )

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
        elif self.starting_entropy < 0:
            # We must be in an (at least partly) continuous action space.
            # Decrease the entropy by -log(fraction) instead.
            final_entropy = self.starting_entropy + math.log(
                self.cfg.entropy_schedule_end_fraction
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

    def _entropy_loss(
        self, agent_output: "AgentOutput", loss_input: LossInput
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

    def train_step(self):
        """Runs one "policy step" of training.

        This method should be called repeatedly until training is complete.
        Unless training is completely off-policy, new episodes should be added
        between each call to this method using add_episode().

        All options for tuning this method are present in the TrainerConfig
        passed on TorchTrainer initialization.
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

        agent_w = _AgentWrapper(self.agent, self.cfg, self.cluster)

        pre_train_epochs = self.cfg.vf_pre_training_epochs
        if self.train_steps_so_far == 0:
            pre_train_epochs = max(self.cfg.vf_warmup_training_epochs, pre_train_epochs)

        # Pre-train VF (usually only used for off-policy algorithms)
        if pre_train_epochs > 0:
            agent_w.fill_caches(self._replay_buffer)
            timed_out = self._train_vf(
                agent_w.state_encodings_cache,
                agent_w.action_lls_cache,
                pre_train_epochs,
                desc="Pre Training VF",
                deadline=deadline,
            )

        loss_inputs = self._prepare_loss_inputs(
            agent_w.state_encodings_cache, agent_w.action_lls_cache
        )
        self._log_dataset(loss_inputs)
        agent_w.clear_caches()

        # Run primary training loop.
        for batch_i, minibatch in enumerate(
            _minibatch_episodes(
                self.cluster,
                self._replay_buffer,
                desc="Training Agent",
                epochs=self.cfg.policy_epochs_per_train_step,
                shuffle=True,
                minibatch_target_timesteps=self.cfg.minibatch_target_timesteps,
                minibatch_max_timesteps=self.cfg.minibatch_max_timesteps,
            )
        ):
            if time.monotonic() > deadline:
                timed_out = True
                break
            loss_input = LossInput.pack(
                [loss_inputs[ep_data.episode_id] for ep_data in minibatch]
            )
            agent_output = agent_w.agent_forward(minibatch, need_full=False)
            loss, loss_infos = self._primary_loss_function(agent_output, loss_input)
            self.vf_optimizer.zero_grad()
            self.agent_optimizer.zero_grad()
            self.cluster.backward(loss)
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
                self.cluster.clip_grad_norm_(
                    self.agent.parameters(),
                    max_norm=self.cfg.grad_norm_max,
                    error_if_nonfinite=True,
                )
                self.cluster.clip_grad_norm_(
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
                loss_inputs = self._prepare_loss_inputs(
                    agent_w.state_encodings_cache, agent_w.action_lls_cache
                )

        # Extra VF tuning after the primary training loop.
        # This achieves similar objectives to PPG by simply not doing updates
        # on the agent network in this phase.
        # Inputs are guaranteed to be cached, since we ran at least one full
        # epoch in the primary loop.
        if self.cfg.vf_post_training_epochs > 0 and not timed_out:
            agent_w.fill_caches(self._replay_buffer)
            timed_out = self._train_vf(
                agent_w.state_encodings_cache,
                agent_w.action_lls_cache,
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
    ) -> bool:
        """Train just the VF using cached agent outputs.

        state_encodings and action_lls are indexed by the `episode_id`
        field of _EpisodeData, and should contain non-differentiable cached
        components from each episode's AgentOutput.

        This method does not tune the parameters of the agent.

        Because this training is only tuning the memoryless VF tail, it uses
        smaller minibatches of shuffled timesteps from across multiple
        episodes.

        Returns True iff the deadline is reached.
        """
        state_enc_packed, obs_lens = pack_tensors(
            [state_encodings[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        action_lls_now = pad_tensors(
            [action_lls[ep_data.episode_id] for ep_data in self._replay_buffer]
        )
        assert not state_enc_packed.requires_grad
        assert not action_lls_now.requires_grad

        padded_rewards = self.cluster.prepare_tensor(
            pad_tensors([data.rewards for data in self._replay_buffer])
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

                    # TODO(krzentner): Refactor / eliminate this loop
                    for i, episode_length in enumerate(episode_lengths):
                        # zero vf_{t+1} in terminated episodes
                        if self._replay_buffer[i].terminated:
                            vf_x[i, episode_length] = 0.0
                            terminated[i] = True
                    terminated = self.cluster.prepare_tensor(terminated)

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
                    self.cluster.clip_grad_norm_(
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
        full_episode_rewards = self.cluster.prepare_tensor(
            pad_tensors([data.rewards for data in self._replay_buffer])
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
            level=noko.INFO,
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
            # TODO(krzentner): split up this call if necessary
            vf_returns_packed = self.vf(state_enc_packed)
        self.agent.train(mode=True)
        self.vf.train(mode=True)

        original_action_lls = self.cluster.prepare_tensor(
            pad_tensors([data.original_action_lls for data in self._replay_buffer])
        )
        terminated = torch.zeros(len(self._replay_buffer), dtype=torch.bool)

        vf_returns = pad_packed(vf_returns_packed, obs_lens)

        # TODO(krzentner): Refactor / eliminate this loop
        # Can't use valids mask, since vf_returns goes to t + 1
        for i, episode_length in enumerate(episode_lengths):
            # Everything after last valid observation should have been padded to zero
            assert (vf_returns[i, episode_length + 1 :] == 0.0).all()

            # zero vf_{t+1} in terminated episodes
            if self._replay_buffer[i].terminated:
                vf_returns[i, episode_length] = 0.0
                terminated[i] = True
        terminated = self.cluster.prepare_tensor(terminated)

        padded_rewards = self.cluster.prepare_tensor(
            pad_tensors([data.rewards for data in self._replay_buffer])
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
            episode_lengths=self.cluster.prepare_tensor(torch.tensor(episode_lengths)),
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

        rewards = rewards.to(dtype=self._loss_dtype)
        self.reward_normalizer.update(rewards)
        if infos is None:
            infos = {}

        self._replay_buffer.append(
            _EpisodeData(
                episode,
                episode_id=self._next_episode_id,
                n_timesteps=n_timesteps,
                terminated=terminated,
                original_action_lls=action_lls.to(dtype=self._loss_dtype),
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
                state = self.cluster.load(best_ckpt)
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
                state = self.cluster.load(best_ckpt)
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
                    self.cluster.save(state_dict, f_name)
                    self.train_steps_so_far_at_last_checkpoint = self.train_steps_so_far
                else:
                    _LOGGER.info(f"Checkpoint {f_name!r} already exists")
            if checkpoint_best:
                f_name = os.path.join(self.cfg.runs_dir, self.cfg.run_name, "best.pkl")
                _LOGGER.info(f"Checkpointing to {f_name!r}")
                self.cluster.save(state_dict, f_name)
                self.best_checkpoint_primary_performance = self.primary_performance
            return True
        else:
            return False

    def state_dict(self) -> dict[str, Any]:
        """Return the state of the TorchTrainer as a dictionary.

        Return value shares Tensors with the TorchTrainer fields.

        Note that the TorchTrainer is *not* an nn.Module.

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
                _LOGGER.debug(
                    f"Adding {k!r}: {v!r} to ubrl.TorchTrainer.state_dict() verbatim"
                )
                if hasattr(v, "state_dict"):
                    _LOGGER.error(
                        f"Field {k} was not expected to have a state_dict method"
                    )
                state[k] = v
        return state

    def load_state_dict(self, state_dict: dict[str, Any]):
        """Load the state of the trainer from a dictionary.

        Note that the TorchTrainer is *not* an nn.Module.
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


@noko.declare_summarizer(TorchTrainer)
def summarize_trainer(trainer, key, dst):
    """Summarize fields for the noko logging library."""
    for k in _SUBMODULE_FIELDS + _OPTIMIZER_FIELDS + _PARAM_FIELDS:
        noko.summarize(getattr(trainer, k), f"{key}.{k}", dst)
    for k, v in trainer.__dict__.items():
        if k not in _IGNORED_FIELDS and k not in "last_eval_stats":
            noko.summarize(v, f"{key}.{k}", dst)


class Agent(nn.Module):
    """Agent API optimized by ubrl.

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
        between your agent and ubrl's trainer.

        If you already need to use the forward pass for another purpose, you
        can define ubrl_forward() instead, but note that this will not invoke
        torch forward hooks.

        Episodes can be any value passed to TorchTrainer.add_episode().

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

    inherent_loss: Optional[torch.Tensor] = None
    """Loss computed by the model itself. Can be used for regularization or
    supervised co-training.

    Typically should be near unit scale when used.
    """

    infos: Optional[dict[str, Any]] = None
    """Not used by ubrl, but can be used if agent.forward() needs to return
    additional values for other use cases."""

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


# This function is here because of its close relationship to AgentOutput
# It is not part of AgentOutput to avoid making it part of the API
def _split_agent_output(
    episode_data: list[_EpisodeData], agent_output: AgentOutput
) -> list[tuple[EpisodeID, torch.Tensor, torch.Tensor]]:
    """Split an AgentOutput into separate state_encodings and action_lls for
    each episode.
    """
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
    `TorchTrainer.add_episode`.
    """

    need_full: bool
    """If True, the agent must return action_lls and state_encodings for the
    entire episode (i.e. `AgentOutput.valid_mask` should be None or all True).

    If False, the agent can optionally only propagate a portion of timesteps.
    """


class _AgentWrapper:
    """Performs checking of the Agent API and caches outputs."""

    def __init__(
        self, agent: Agent, cfg: "TrainerConfig", cluster: "ubrl.cluster.Cluster"
    ):
        self.agent: Agent = agent
        self.cfg = cfg
        self.cluster = cluster
        self.state_encodings_cache: dict[EpisodeID, torch.Tensor] = {}
        self.action_lls_cache: dict[EpisodeID, torch.Tensor] = {}

    def clear_caches(self):
        self.state_encodings_cache = {}
        self.action_lls_cache = {}

    def fill_caches(self, episode_data: list[_EpisodeData]):
        missing_episodes = [
            ep_data
            for ep_data in episode_data
            if ep_data.episode_id not in self.state_encodings_cache
        ]
        with torch.no_grad():
            for minibatch in _minibatch_episodes(
                self.cluster,
                missing_episodes,
                desc="Caching state encodings",
                minibatch_max_timesteps=self.cfg.minibatch_max_timesteps,
            ):
                self.agent_forward(minibatch, need_full=True)

    def agent_forward(
        self, episode_data: list[_EpisodeData], need_full: bool
    ) -> AgentOutput:
        """Wrapper around the Agent forward() method.

        Handles delegating to ubrl_forward() if necessary, checking that
        need_full is respected, and caching.
        """
        episodes = [ep_data.episode for ep_data in episode_data]
        expected_lengths = [ep_data.n_timesteps for ep_data in episode_data]
        agent_input = AgentInput(episodes=episodes, need_full=need_full)
        if hasattr(self.agent, "ubrl_forward"):
            agent_output = self.agent.ubrl_forward(agent_input)
        else:
            agent_output = self.agent(agent_input)
        assert agent_output.state_encodings.shape[0] == sum(expected_lengths) + len(
            expected_lengths
        )
        assert agent_output.action_lls.shape[0] == sum(expected_lengths)
        if agent_output.full_valid():
            for episode_id, state_encodings, action_lls in _split_agent_output(
                episode_data, agent_output
            ):
                self.state_encodings_cache[episode_id] = state_encodings.detach()
                self.action_lls_cache[episode_id] = action_lls.detach()
        else:
            total_valid = agent_output.valid_mask.sum().item()
            total_timesteps = sum(expected_lengths)
            if need_full:
                raise ValueError(
                    f"Requested fully valid output but Agent only provided valid output for {total_valid}/{total_timesteps}"
                )
        return agent_output


# From here to the end of the file there should be only private helper / utlity
# functions.


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


def _group_episodes_to_minibatches(
    episodes: list["ubrl._EpisodeData"],
    *,
    minibatch_target_timesteps: Optional[int] = None,
    minibatch_max_timesteps: Optional[int] = None,
) -> list[list["ubrl._EpisodeData"]]:
    """Group a list of episodes into a list of list of episodes.
    Each minibatch (list of episodes) will have at least
    minibatch_target_timesteps timesteps unless adding the next episode would
    make it longer than minibatch_max_timesteps, or it is the last minibatch.

    Raises ValueError if any epsidoe is longer than minibatch_max_timesteps.
    """
    all_minibatches = []

    minibatch_now = []
    minibatch_now_size = 0
    for episode in episodes:
        # If we would go over the maximum
        if (
            minibatch_max_timesteps is not None
            and minibatch_now_size + episode.n_timesteps > minibatch_max_timesteps
        ) or (
            minibatch_target_timesteps is not None
            and minibatch_now_size >= minibatch_target_timesteps
        ):
            if len(minibatch_now) == 0:
                raise ValueError(
                    f"Episode length ({episode.n_timesteps}) exceeds max "
                    f"allowed timesteps in a minibatch ({minibatch_max_timesteps})"
                )
            all_minibatches.append(minibatch_now)
            minibatch_now = []
            minibatch_now_size = 0
        minibatch_now.append(episode)
        minibatch_now_size += episode.n_timesteps
    if minibatch_now_size > 0:
        all_minibatches.append(minibatch_now)

    return all_minibatches


def _minibatch_episodes(
    cluster: "ubrl.cluster.Cluster",
    episodes: list[_EpisodeData],
    minibatch_target_timesteps: Optional[int] = None,
    minibatch_max_timesteps: Optional[int] = None,
    desc: Optional[str] = None,
    epochs: int = 1,
    shuffle: bool = False,
) -> Generator[list[_EpisodeData], None, None]:
    """Top-level wrapper for iterating through minibatches from the replay buffer.

    Handles rendering a progress bar, shuffling episodes, and grouping into
    minibatches.
    Mostly this function is here to avoid adding two more indentation levels to
    the main loss loop.
    """
    with tqdm(
        total=epochs * sum([ep.n_timesteps for ep in episodes]),
        desc=desc,
        disable=(desc is None),
    ) as pbar:
        for _ in range(epochs):
            episodes = cluster.shard_episodes(episodes, shuffle=shuffle)

            minibatches = _group_episodes_to_minibatches(
                episodes=episodes,
                minibatch_target_timesteps=minibatch_target_timesteps,
                minibatch_max_timesteps=minibatch_max_timesteps,
            )
            for minibatch in minibatches:
                yield minibatch
                pbar.update(sum(ep_data.n_timesteps for ep_data in minibatch))


__all__ = [
    "TorchTrainer",
    "Agent",
    "AgentInput",
    "AgentOutput",
    "LossInput",
]
