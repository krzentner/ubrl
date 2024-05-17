"""
.. include:: ../../README.md
"""

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

from optuna.distributions import (
    FloatDistribution,
    CategoricalDistribution,
    IntDistribution,
)

import stick

from outrl.torch_utils import (
    entropy_of,
    force_concat,
    make_mlp,
    pack_recursive,
    pack_tensors,
    pack_tensors_check,
    pad_packed,
    pad_tensors,
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
)
from outrl.config import tunable, IntListDistribution, default_run_name

_LOGGER = logging.getLogger("outrl")

T = TypeVar("T")


@dataclass(eq=False)
class _EpisodeData:
    """Wrapper around an episode that maintains metadata and caches computed values."""

    episode: Any
    """The episode (treated as an opaque object). Will be passed back to the
    agent during optimization."""

    episode_number: int
    """Records which add_episode() call this episode came from."""

    num_timesteps: int
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
        assert self.original_action_lls.shape == (self.num_timesteps,)
        assert self.rewards.shape == (self.num_timesteps,)
        assert self.any_actions_possible.shape == (self.num_timesteps,)
        assert self.num_timesteps > 0

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
class _PolicyTrainingInputs:
    """Values used to train the policy that are not present in _EpisodeData.

    An instance of this class accompanies each _EpisodeData during policy
    optimization.

    These values are recomputed every train_step()."""

    advantages: torch.Tensor
    """Advantages for this episode. Used in PPO loss (when enabled)."""

    exp_advantages: torch.Tensor
    """Coefficients used in the AWR loss."""

    vf_targets: torch.Tensor
    """Target values for the value function loss."""

    discounted_returns: torch.Tensor
    """Discounted sum of future rewards. Used in logging."""

    vf_returns: torch.Tensor
    """Estimated returns from the value function. Used in logging."""


_OPTIMIZER_FIELDS = [
    "agent_optimizer",
    "vf_optimizer",
    "vf_lr_scheduler",
    "agent_lr_scheduler",
    "kl_coef_opt",
    "awr_temperature_opt",
    "entropy_coef_opt",
]

_SUBMODULE_FIELDS = [
    "agent",
    "vf",
    "reward_normalizer",
]

_PARAM_FIELDS = [
    "kl_coef",
    "awr_temperature",
    "entropy_coef",
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

        self.agent: "Agent" = agent
        """The agent being optimized. Provides action (log-likelihoods) and
        state encodings."""

        self._state_encoding_size = self.agent.state_encoding_size

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

        self.reward_normalizer: RunningMeanVar = RunningMeanVar(use_mean=False)
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
            torch.tensor(float(self.cfg.kl_coef_init))
        )
        """Dynamically adjusted parameter used for KL regularization."""

        self.entropy_coef: nn.Parameter = nn.Parameter(
            torch.tensor(0.0)
        )
        """Dynamically adjusted parameter used for entropy enforcing."""

        self.awr_temperature: nn.Parameter = nn.Parameter(
            torch.tensor(float(self.cfg.temperature_init))
        )
        """Dynamically adjusted parameter used for temperature in the AWR loss."""

        self.starting_entropy: Optional[float] = None
        """Initial mean entropy of action distributions, measured immediately
        at the start of cfg.entropy_schedule_start_train_step."""

        self._replay_buffer: list[_EpisodeData] = []
        self._next_episode_number: int = 0
        self._dtype = torch.float32

        self._last_training_stats = {}

        # This method is also called after loading the state dict to re-attach
        # parameters
        self._setup_optimizers()

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
        self.awr_temperature_opt = torch.optim.AdamW(
            [self.awr_temperature], lr=self.cfg.temperature_lr
        )
        self.entropy_coef_opt = torch.optim.AdamW(
            [self.entropy_coef], lr=self.cfg.entropy_coef_lr
        )

        self.total_agent_grad_steps = 0

    def _primary_loss_function(
        self, batch: list[tuple[_EpisodeData, _PolicyTrainingInputs, "AgentOutput"]]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reinforcement learning loss function.

        Combines together losses from many sources to train both the agent and
        value function.

        Returns a loss which is usually near unit scale and a dictionary of
        infos to log for this gradient step.

        Note that lagrangian losses (for the KL and entropy) are optimized in
        their related loss functions.
        """

        episode_data: list[_EpisodeData] = [b[0] for b in batch]
        train_inputs: list[_PolicyTrainingInputs] = [b[1] for b in batch]
        agent_outputs: list["AgentOutput"] = [b[2] for b in batch]

        kl_loss, kl_infos = self._kl_loss(episode_data, train_inputs, agent_outputs)

        entropy_loss, entropy_infos = self._entropy_loss(episode_data, train_inputs, agent_outputs)

        ppo_loss, ppo_infos = self._ppo_loss(episode_data, train_inputs, agent_outputs)

        awr_loss, awr_infos = self._awr_loss(episode_data, train_inputs, agent_outputs)

        # TODO: Tune awr_temperature

        vf_loss, vf_infos = self._vf_loss(episode_data, train_inputs, agent_outputs)

        loss = ppo_loss + awr_loss + vf_loss + kl_loss + entropy_loss

        # *_infos will all get included in locals of this method
        used_for_logging(kl_infos, ppo_infos, awr_infos, vf_infos,
                         entropy_infos)
        return loss, locals()

    def _ppo_loss(
        self,
        episode_data: list[_EpisodeData],
        train_inputs: list[_PolicyTrainingInputs],
        agent_outputs: list["AgentOutput"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        advantages, adv_len = pack_tensors(
            [train_input.advantages for train_input in train_inputs]
        )
        log_probs = pack_tensors_check(
            [agent_out.action_lls for agent_out in agent_outputs], adv_len
        )

        old_log_probs = pack_tensors_check(
            [data.original_action_lls for data in episode_data], adv_len
        )
        assert not old_log_probs.requires_grad

        ratio = torch.exp(log_probs - old_log_probs)
        ratio_clipped = torch.clamp(
            ratio, 1 / (1 + self.cfg.ppo_clip_epsilon), 1 + self.cfg.ppo_clip_epsilon
        )

        policy_gradient = ratio * advantages
        clip_policy_gradient = ratio_clipped * advantages

        # Mean is handled in the primary_loss_function
        ppo_loss = -torch.min(policy_gradient, clip_policy_gradient)
        norm_coef = self.cfg.ppo_loss_coef / self.cfg.minibatch_target_timesteps
        ppo_loss_scaled = norm_coef * ppo_loss.sum()
        infos = locals()
        del infos["self"]
        del infos["adv_len"]
        infos["clip_portion"] = (ratio_clipped != ratio).mean(dtype=torch.float32)
        return (ppo_loss_scaled, infos)

    def _awr_loss(
        self,
        episode_data: list[_EpisodeData],
        train_inputs: list[_PolicyTrainingInputs],
        agent_outputs: list["AgentOutput"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        del episode_data

        # Dividing by temperature and normalizing are handled in preprocess
        # TODO: Should advantage exponentiation happen every minibatch or not?
        # temperature tuning happens every minibatch, so it seems like the
        # temperature should be used every minibatch as well.
        # But also it seems like having a constant temperature throughout a
        # batch might be ideal.

        # We'll probably add entropy regularization using a minibatch
        # lagrangian anyways, so maybe it doesn't matter.
        exp_adv, adv_len = pack_tensors(
            [train_input.exp_advantages for train_input in train_inputs]
        )

        log_probs = pack_tensors_check(
            [agent_out.action_lls for agent_out in agent_outputs], adv_len
        )

        awr_loss = -log_probs * exp_adv

        log_probs_scale = log_probs.detach().abs().mean()
        used_for_logging(log_probs_scale)
        norm_coef = self.cfg.awr_loss_coef / (self.cfg.minibatch_target_timesteps)

        awr_loss_scaled = norm_coef * awr_loss.sum()
        # Don't want to log these here
        infos = locals()
        del infos["self"]
        del infos["adv_len"]
        return (awr_loss_scaled, infos)

    def _vf_loss(
        self,
        episode_data: list[_EpisodeData],
        train_inputs: list[_PolicyTrainingInputs],
        agent_outputs: list["AgentOutput"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # For consistency with other loss functions, we still take
        # episode_data, but don't use it.
        del episode_data

        vf_targets_packed, adv_len = pack_tensors(
            [train_input.vf_targets for train_input in train_inputs]
        )
        state_enc_packed = pack_tensors_check(
            [agent_out.state_encodings[:-1] for agent_out in agent_outputs], adv_len
        )

        critic_out = self.vf(state_enc_packed)
        vf_loss = F.mse_loss(critic_out, vf_targets_packed)

        discounted_returns = pack_tensors_check(
            [train_input.discounted_returns for train_input in train_inputs], adv_len
        )
        minibatch_ev = explained_variance(critic_out, discounted_returns)
        used_for_logging(minibatch_ev)

        norm_coef = self.cfg.vf_loss_coef / self.cfg.minibatch_target_timesteps
        vf_loss_scaled = norm_coef * vf_loss
        infos = locals()
        del infos["self"]
        del infos["adv_len"]
        return (vf_loss_scaled, infos)

    def _kl_loss(
        self,
        episode_data: list[_EpisodeData],
        train_inputs: list[_PolicyTrainingInputs],
        agent_outputs: list["AgentOutput"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # For consistency with other loss functions, we still take
        # train_inputs, but don't use it.
        del train_inputs

        original_kl_coef = self.kl_coef.detach().item()
        used_for_logging(original_kl_coef)

        log_probs, lengths = pack_tensors(
            [agent_out.action_lls for agent_out in agent_outputs]
        )
        old_log_probs = pack_tensors_check(
            [data.original_action_lls for data in episode_data], lengths
        )
        # Normalize to form a proper distribution across actions P(a)
        log_probs -= log_probs.exp().sum().log()
        old_log_probs -= old_log_probs.exp().sum().log()

        # Approximate per-timestep KL by multiplying back in the number of timesteps
        total_timesteps = sum(lengths)
        approx_kl = total_timesteps * approx_kl_div_of(old_log_probs, log_probs)
        total_approx_kl = approx_kl.sum()
        used_for_logging(total_approx_kl)

        # Compute KL Divergence
        if self.cfg.use_approx_kl is False:
            new_dists = [agent_out.action_dists for agent_out in agent_outputs]
            assert new_dists[0] is not None
            old_dists = [ep_data.original_action_dists for ep_data in episode_data]
            assert old_dists[0] is not None
            # TODO: Add options for other KL directions
            kl = kl_div_of(old_dists, new_dists)
        else:
            kl = approx_kl

        assert kl.shape == (sum(ep.num_timesteps for ep in episode_data),)
        assert approx_kl.shape == (sum(ep.num_timesteps for ep in episode_data),)
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
        self.kl_coef_opt.step()
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

        # Gradient will be dL/dKL / (KL + 1) (less sensitive to
        # outliers)
        # Some approx KL values can be negative, so use the proper
        # symlog
        kl_symlog = torch.sign(kl) * torch.log(kl.abs() + 1)

        kl_loss_scaled = (norm_coef * kl_symlog).sum()
        infos = locals()
        del infos["lengths"]
        del infos["self"]
        return kl_loss_scaled, infos

    def _entropy_target(self) -> Optional[float]:
        if self.starting_entropy is not None:
            # 0 at cfg.entropy_schedule_start_train_step
            # 1 at (and after) cfg.expected_train_steps
            step_fraction = min(1,
                                  ((1 + self.train_steps_so_far) - self.cfg.entropy_schedule_start_train_step) / (self.cfg.expected_train_steps - self.cfg.entropy_schedule_start_train_step))
            assert step_fraction >= 0
            assert step_fraction <= 1
            final_entropy = self.cfg.entropy_schedule_end_fraction * self.starting_entropy
            if self.cfg.entropy_schedule == "linear":
                mix = step_fraction
            elif self.cfg.entropy_schedule == "cosine":
                # Cosine curve from 1 to 0
                mix = 0.5 * (1 + math.cos(step_fraction * math.pi))
            else:
                raise NotImplementedError(f"Unknown entropy schedule {self.cfg.entropy_schedule}")
            assert mix >= 0
            assert mix <= 1
            target = (1 - mix) * self.starting_entropy + mix * final_entropy
            return target
        else:
            return None

    def _entropy_loss(
        self,
        episode_data: list[_EpisodeData],
        train_inputs: list[_PolicyTrainingInputs],
        agent_outputs: list["AgentOutput"],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        approx_entropy, entropy = self._entropy_of(
            [agent_out.action_lls for agent_out in agent_outputs],
            [agent_out.action_dists for agent_out in agent_outputs])
        entropy_target = self._entropy_target()
        if entropy_target is not None:

            # Tune temperature
            exp_adv = torch.cat([train_input.exp_advantages for train_input in train_inputs])
            temperature_loss = (self.awr_temperature * entropy_target + self.awr_temperature * torch.log(exp_adv.mean()))
            temperature_loss_norm = temperature_loss * len(exp_adv) / self.cfg.minibatch_target_timesteps
            self.awr_temperature_opt.zero_grad()
            temperature_loss_norm.backward()
            self.awr_temperature_opt.step()

            if self.awr_temperature < self.cfg.temperature_min:
                with torch.no_grad():
                    self.awr_temperature.copy_(self.cfg.temperature_min)
            if self.awr_temperature > self.cfg.temperature_max or not torch.isfinite(self.awr_temperature):
                with torch.no_grad():
                    self.awr_temperature.copy_(self.cfg.temperature_max)

            # Tune entropy_coef
            entropy_coef_loss = self.entropy_coef * (entropy_target - entropy.detach().mean())
            self.entropy_coef_opt.zero_grad()
            entropy_coef_loss.backward()
            self.entropy_coef_opt.step()

            if self.entropy_coef < self.cfg.entropy_coef_min:
                with torch.no_grad():
                    self.entropy_coef.copy_(self.cfg.entropy_coef_min)
            if self.entropy_coef > self.cfg.entropy_coef_max:
                with torch.no_grad():
                    self.entropy_coef.copy_(self.cfg.entropy_coef_max)
            # Unclear why entropy_coef became non-finite, so set back
            # to zero
            if not torch.isfinite(self.entropy_coef):
                with torch.no_grad():
                    self.entropy_coef.copy_(0.0)

            entropy_coef = self.entropy_coef.detach()
            norm_coef = entropy_coef / self.cfg.minibatch_target_timesteps
            entropy_loss = (norm_coef * entropy).sum()
        else:
            entropy_loss = torch.tensor(0.0)

        used_for_logging(approx_entropy)
        infos = locals()
        del infos["self"]
        return entropy_loss, infos

    def _agent_minibatches(
        self,
        episodes: Optional[list[_EpisodeData]] = None,
        extra_data: Optional[list[T]] = None,
        minibatch_target_timesteps: Optional[int] = None,
        desc: Optional[str] = None,
        epochs: int = 1,
        shuffle: bool = False,
    ) -> Generator[
        tuple[int, int, list[tuple[_EpisodeData, T, "AgentOutput"]]], None, None
    ]:
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

        if extra_data is None or len(extra_data) == 0:
            # pyright doesn't understand that extra_data = None -> T = None
            extra_data = [None for _ in episodes]
        assert extra_data is not None and len(extra_data) == len(episodes)

        if shuffle:
            shuffled_indices = torch.randperm(len(episodes))
            episodes = [episodes[i] for i in shuffled_indices]
            extra_data = [extra_data[i] for i in shuffled_indices]

        minibatch_hard_cap = 2**64
        if self.cfg.max_timesteps_per_forward is not None:
            minibatch_hard_cap = self.cfg.max_timesteps_per_forward

        minibatch_soft_cap = minibatch_hard_cap
        if minibatch_target_timesteps is not None:
            minibatch_soft_cap = min(minibatch_target_timesteps, minibatch_hard_cap)

        with tqdm(
            total=epochs * sum([ep.num_timesteps for ep in episodes]), desc=desc
        ) as pbar:
            for epoch in range(epochs):
                next_ep_index = 0
                minibatch_number = 0
                while next_ep_index < len(episodes):
                    start_batch_ep_index = next_ep_index
                    batch = []
                    num_batch_steps = 0
                    # Accumulate episodes into batch until we run out of space
                    while next_ep_index < len(episodes) and (
                        num_batch_steps + episodes[next_ep_index].num_timesteps
                        <= minibatch_hard_cap
                    ):
                        batch.append(
                            (episodes[next_ep_index], extra_data[next_ep_index])
                        )
                        num_batch_steps += episodes[next_ep_index].num_timesteps
                        next_ep_index += 1
                        if num_batch_steps >= minibatch_soft_cap:
                            break

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
                            {ep_steps} > {max_steps} = cfg.max_timesteps_per_forward
                            Increase cfg.max_timesteps_per_forward, decrease model size,
                            or find another way of increasing available memory.
                            """
                            )
                        )

                    try:
                        forward_result = _call_agent_forward(
                            self.agent, [data[0].episode for data in batch]
                        )
                        yield (
                            epoch,
                            minibatch_number,
                            [
                                (data[0], data[1], f_res)
                                for (data, f_res) in zip(batch, forward_result)
                            ],
                        )
                        minibatch_number += 1
                        pbar.update(sum([data[0].num_timesteps for data in batch]))
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
                        self.cfg.max_timesteps_per_forward = num_batch_steps - 1
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
        observation_latent_cache = {}
        action_lls_cache = {}

        # Pre-train VF (usually only used for off-policy algorithms)
        if self.cfg.vf_pre_training_epochs > 0:
            with torch.no_grad():
                for _, _, batch in self._agent_minibatches(desc="Caching latents"):
                    for ep_data, _, agent_res in batch:
                        observation_latent_cache[
                            ep_data.episode_number
                        ] = agent_res.state_encodings
                        action_lls_cache[ep_data.episode_number] = agent_res.action_lls
            timed_out = self._train_vf(
                observation_latent_cache,
                action_lls_cache,
                self.cfg.vf_pre_training_epochs,
                desc="Pre Training VF",
                deadline=deadline,
            )

        train_inputs = self._preprocess()
        self._log_dataset(train_inputs)

        # Run primary training loop.
        for _, batch_i, batch in self._agent_minibatches(
            desc="Training Agent",
            extra_data=train_inputs,
            epochs=self.cfg.policy_epochs_per_train_step,
            shuffle=True,
            minibatch_target_timesteps=self.cfg.minibatch_target_timesteps,
        ):
            if time.monotonic() > deadline:
                timed_out = True
                break
            for ep_data, _, agent_res in batch:
                observation_latent_cache[
                    ep_data.episode_number
                ] = agent_res.state_encodings.detach()
                action_lls_cache[ep_data.episode_number] = agent_res.action_lls.detach()
            loss, loss_infos = self._primary_loss_function(batch)
            self.vf_optimizer.zero_grad()
            self.agent_optimizer.zero_grad()
            loss.backward()
            self.total_agent_grad_steps += 1
            if batch_i == 0 or (
                self.cfg.log_grad_step_period > 0
                and batch_i % self.cfg.log_grad_step_period == 0
            ):
                self._log_training_infos(loss_infos)
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
            if self.cfg.recompute_train_inputs:
                train_inputs = self._preprocess()

        # Extra VF tuning after the primary training loop.
        # This achieves similar objectives to PPG by simply not doing updates
        # on the agent network in this phase.
        # Inputs are guaranteed to be cached, since we ran at least one full
        # epoch in the primary loop.
        if self.cfg.vf_post_training_epochs > 0 and not timed_out:
            timed_out = self._train_vf(
                observation_latent_cache,
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
        stick.log_row("last_training_stats", self._last_training_stats, level=stick.RESULTS)
        self.train_steps_so_far += 1
        self._maybe_record_starting_entropy()

        if timed_out:
            _LOGGER.error("train_step() timed out")

    def _train_vf(
        self,
        state_encodings: dict[int, torch.Tensor],
        action_lls: dict[int, torch.Tensor],
        training_epochs: int,
        desc: str,
        deadline: float,
    ):
        """Train just the VF using cached agent outputs.

        state_encodings and action_lls are indexed by the `episode_number`
        field of _EpisodeData, and should contain non-differentiable cached
        components from each episode's AgentOutput.

        This method does not tune the parameters of the agent.

        Because this training is only tuning the memoryless VF tail, it uses
        smaller minibatches of shuffled timesteps from across multiple
        episodes.
        """
        state_enc_packed, obs_lens = pack_tensors(
            [state_encodings[ep_data.episode_number] for ep_data in self._replay_buffer]
        )
        assert not state_enc_packed.requires_grad

        padded_rewards = pad_tensors([data.rewards for data in self._replay_buffer])
        if self.cfg.normalize_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)
        else:
            rewards_normed = padded_rewards

        action_lls_now = pad_tensors(
            [action_lls[ep_data.episode_number] for ep_data in self._replay_buffer]
        )
        assert not action_lls_now.requires_grad

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

    def _log_training_infos(self, train_locals: dict[str, Any]):
        training_stats = {
            k: train_locals[k].item()
            for k in ["loss", "vf_loss", "ppo_loss", "kl_loss", "awr_loss"]
            if k in train_locals
        }
        training_stats["clip_portion"] = train_locals["ppo_infos"]["clip_portion"]
        training_stats["kl_mean"] = train_locals["kl_infos"]["kl_mean"]
        training_stats["kl_max"] = train_locals["kl_infos"]["kl_max"]
        stick.log_row(
            "training_stats",
            training_stats,
            level=stick.INFO,
            step=self.total_agent_grad_steps,
        )
        self._last_training_stats = training_stats

        stick.log_row(
            "train_locals",
            train_locals,
            level=stick.TRACE,
            step=self.total_agent_grad_steps,
        )

    def _log_dataset(self, train_inputs: list[_PolicyTrainingInputs]):
        full_episode_rewards = pad_tensors(
            [data.rewards for data in self._replay_buffer]
        )
        t_inputs = pack_recursive(train_inputs)[0]
        dataset_stats = {
            "ep_rew": full_episode_rewards.sum(dim=-1).mean(dim=0),
            "ev": explained_variance(
                t_inputs.vf_returns,
                t_inputs.discounted_returns,
            ),
            "disc_mean": t_inputs.discounted_returns.mean(),
            "vf_mean": t_inputs.vf_returns.mean(),
            "vf_target_mean": t_inputs.vf_targets.mean(),
            "total_env_steps": self.total_env_steps,
        }
        for k in self._replay_buffer[0].infos.keys():
            dataset_stats[k] = force_concat(
                data.infos[k] for data in self._replay_buffer
            ).mean()

        stick.log_row(
            "dataset_stats",
            dataset_stats,
            level=stick.RESULTS,
            step=self.total_env_steps,
        )

        for k in self._replay_buffer[0].infos.keys():
            dataset_stats[k] = force_concat(
                data.infos[k] for data in self._replay_buffer
            )

        stick.log_row(
            "dataset_stats_trace",
            dataset_stats,
            level=stick.TRACE,
            step=self.total_env_steps,
        )

    def _preprocess(self) -> list[_PolicyTrainingInputs]:
        """Compute training inputs from the replay buffer and value function.

        Mostly this consists of the advantages and value function
        targets. See the _PolicyTrainingInputs class for more
        information.
        """

        episode_lengths = [len(data.rewards) for data in self._replay_buffer]
        obs_lens = [len(data.episode["observations"]) for data in self._replay_buffer]
        assert [ep_len + 1 for ep_len in episode_lengths] == obs_lens

        # Compute vf_returns
        self.agent.train(mode=False)
        self.vf.train(mode=False)
        with torch.no_grad():
            agent_outputs = _call_agent_forward(
                self.agent, [data.episode for data in self._replay_buffer]
            )
            state_enc_packed = pack_tensors_check(
                [agent_out.state_encodings for agent_out in agent_outputs], obs_lens
            )
            vf_returns_packed = self.vf(state_enc_packed)
        self.agent.train(mode=True)
        self.vf.train(mode=True)

        action_lls_now = pad_tensors(
            [agent_out.action_lls for agent_out in agent_outputs]
        )
        original_action_lls = pad_tensors(
            [data.original_action_lls for data in self._replay_buffer]
        )
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

        padded_rewards = pad_tensors([data.rewards for data in self._replay_buffer])
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
            episode_lengths=torch.tensor(episode_lengths),
        )

        adv_packed = pack_padded(padded_advantages, episode_lengths)
        assert not adv_packed.requires_grad

        if self.cfg.normalize_awr_advantages:
            adv_packed = adv_packed - adv_packed.mean()
            adv_packed = adv_packed / adv_packed.std()

        heated_adv = adv_packed / self.awr_temperature.detach()
        exp_advantages = heated_adv.exp()

        max_exp_adv = torch.tensor(self.cfg.advantage_clip).exp()
        exp_advantages_clip_mask = ~torch.isfinite(exp_advantages)
        exp_advantages[exp_advantages_clip_mask] = 1 + max_exp_adv
        rect_adv = exp_advantages - exp_advantages.min()

        large_exp_adv = ~torch.isfinite(rect_adv) | (rect_adv > max_exp_adv)
        clipped_exp_adv = rect_adv.clone()
        clipped_exp_adv[large_exp_adv] = max_exp_adv

        discounted_returns = _discount_cumsum(padded_rewards, discount=discount)

        train_inputs = [
            _PolicyTrainingInputs(
                advantages=adv,
                discounted_returns=disc_ret,
                vf_returns=vf_ret,
                vf_targets=vf_target,
                exp_advantages=exp_adv,
            )
            for (adv, disc_ret, vf_ret, vf_target, exp_adv) in zip(
                unpack_tensors(adv_packed, episode_lengths),
                unpad_tensors(discounted_returns, episode_lengths),
                unpad_tensors(vf_returns, episode_lengths),
                unpad_tensors(vf_targets, episode_lengths),
                unpack_tensors(clipped_exp_adv, episode_lengths),
            )
        ]
        assert len(train_inputs) == len(self._replay_buffer)
        assert [
            len(train_input.advantages) for train_input in train_inputs
        ] == episode_lengths
        assert [
            len(train_input.discounted_returns) for train_input in train_inputs
        ] == episode_lengths

        infos = locals()
        del infos["self"]
        del infos["episode_lengths"]
        del infos["obs_lens"]
        stick.log_row(
            "preprocess",
            infos,
            level=stick.TRACE,
            step=self.total_env_steps,
        )

        return train_inputs

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
        num_timesteps = len(rewards)
        self.total_env_steps += num_timesteps
        if any_actions_possible is None:
            any_actions_possible = torch.ones(num_timesteps, dtype=torch.bool)
        else:
            assert isinstance(any_actions_possible, torch.Tensor)

        rewards = rewards.to(dtype=self._dtype)
        self.reward_normalizer.update(rewards)
        if infos is None:
            infos = {}

        self._replay_buffer.append(
            _EpisodeData(
                episode,
                episode_number=self._next_episode_number,
                num_timesteps=num_timesteps,
                terminated=terminated,
                original_action_lls=action_lls.to(dtype=self._dtype),
                original_action_dists=action_dists,
                rewards=rewards,
                any_actions_possible=any_actions_possible,
                weight=weight,
                infos=infos,
            )
        )
        self._next_episode_number += 1

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        """Add evaluation statistics for the current agent.

        Will be logged to {cfg.runs_dir}/{cfg.run_name}/eval_stats.csv.

        The primary stat should be present in stats and indicates how to choose
        the "best" agent for purposes of checkpointing and hyper-parameter
        tuning.
        """
        assert "primary" not in stats
        stats["primary"] = stats[primary]
        stick.log_row("eval_stats", stats, step=self.total_env_steps, level=stick.RESULTS)
        self.primary_performance = stats[primary]
        self.last_eval_stats = stats
        hparams = self.cfg.to_dict()
        hparams["metric-primary"] = stats[primary]
        for k, v in stats.items():
            hparams[k] = v
        stick.log_row("hparams", hparams, step=self.total_env_steps)
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
        for idx, f_name in sorted(with_idx, reverse=True):
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
                field_now = getattr(self, k, None)
                if field_now is None:
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

    def _entropy_of(self,
                    action_lls: list[torch.Tensor],
                    action_dists: list[Optional[ActionDist]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the approximate entropy (and exact entropy, if possible).
        """
        approx_entropy = -torch.cat(action_lls)
        entropy = None
        if not self.cfg.use_approx_entropy:
            try:
                # Mixed case is warned on add_episode()
                if None not in action_dists:
                    entropy = entropy_of(action_dists)
            except NotImplementedError:
                pass
        if entropy is None:
            entropy = approx_entropy
        return approx_entropy, entropy

    def _maybe_record_starting_entropy(self):
        """Record starting_etropy if train_steps_so_far >= cfg.entropy_schedule_start_train_step.
        """
        if self.starting_entropy is None and self.train_steps_so_far >= self.cfg.entropy_schedule_start_train_step:

            original_action_lls = [ep_data.original_action_lls
                                   for ep_data in self._replay_buffer]
            original_dists = [ep_data.original_action_dists
                              for ep_data in self._replay_buffer]
            approx_entropy, entropy = self._entropy_of(
                original_action_lls, original_dists)
            del approx_entropy
            self.starting_entropy = entropy.mean().item()


@stick.declare_summarizer(Trainer)
def summarize_trainer(trainer, key, dst):
    """Summarize fields for the stick logging library."""
    for k in _SUBMODULE_FIELDS + _OPTIMIZER_FIELDS + _PARAM_FIELDS:
        stick.summarize(getattr(trainer, k), f"{key}.{k}", dst)
    for k, v in trainer.__dict__.items():
        stick.summarize(v, f"{key}.{k}", dst)


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

    def forward(self, episodes: list[Any]) -> list["AgentOutput"]:
        """Run the agent's forward pass. This is the main API entry point
        between your agent and OutRL's trainer.

        If you already need to use the forward pass for another purpose, you
        can define outrl_forward() instead, but note that this will not invoke
        torch forward hooks.

        Episodes can be any value passed to Trainer.add_episode().
        """
        del episodes
        raise NotImplementedError()


@dataclass(eq=False)
class AgentOutput:
    """Result from one episode when calling .forward() on an agent.

    Differentiating both state_encodings and action_lls should affect
    earlier layers in the agent.
    """

    state_encodings: torch.Tensor
    """Differentiable representation of the observations in the episode."""

    action_lls: torch.Tensor
    """Differentiable log-likelihood of actions taken in the episode."""

    action_dists: Optional[ActionDist] = None
    """Distribution used to generate actions.

    Will be used for the KL penalty if not None and cfg.use_approx_kl is False.

    Will be used for entropy regularization if not None and cfg.approx_entropy
    is False.
    """


def _call_agent_forward(agent: Agent, episodes: list[Any]) -> list[AgentOutput]:
    """Calls the outrl_forward method if available, otherwise calls the normal
    forward method via __call__.
    """
    if hasattr(agent, "outrl_forward"):
        return agent.outrl_forward(episodes)
    else:
        return agent(episodes)


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

    stderr_log_level: stick.LogLevels = stick.LogLevels.INFO
    """Log level to stderr for stick and python logging."""

    pprint_logging: bool = True
    """Log to stdout using pprint. Because the pprint output engine defaults to
    the RESULTS log level, this defaults to True."""

    parquet_logging: bool = False
    """Log to parquet files using pyarrow."""

    tb_log_level: stick.LogLevels = stick.LogLevels.INFO
    """Log level to log to TensorBoard. Defaults to INFO to avoid slowing down
    TensorBoard with too many keys."""

    replay_buffer_episodes: int = 128
    """Maximum number of episodes to keep in replay buffer."""

    max_timesteps_per_forward: Optional[int] = None
    """Maximum number of timesteps to include in a forward pass to the agent.
    Used to avoid out-of-memory errors.

    Defaults to no limit. Automatically decreases on (most) memory errors.
    """

    minibatch_target_timesteps: int = tunable(64, IntDistribution(1, 50000, log=True))
    """Attempt to keep timesteps in each minibatch to this number of timesteps.
    In practice, this acts a divisor on most losses.

    Will still run whole episodes if they exceed this cap.
    """

    policy_epochs_per_train_step: int = tunable(3, IntDistribution(1, 100, log=True))
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    normalize_rewards: bool = tunable(False, CategoricalDistribution([True, False]))
    """Normalize rewards to have zero mean and unit variance."""

    expected_train_steps: int = 1000
    """Expected number of training steps. Used for controlling scheduled parameters."""

    train_step_timeout_seconds: Optional[float] = None
    """train_step() will exit early if this number of seconds of wall-clock
    time is exceeded during it. The current gradient step will still finish
    first, so this timeout is only approximately enforced.
    """

    ppo_loss_coef: float = tunable(0.0, FloatDistribution(0.0, 1000.0))
    """Loss coefficient for the PPO loss. Usually unused, since the AWR loss is
    more flexible.
    """

    awr_loss_coef: float = tunable(1.0, FloatDistribution(0.0, 1000.0))
    """Loss coefficient for the main RL loss. Usually does not need to be
    tuned."""

    agent_lr_schedule: Literal[None, "linear"] = tunable(
        "linear", CategoricalDistribution([None, "linear"])
    )
    """Learning rate schedule for the agent. Typically used to decrease the
    learning rate to near-zero near the end of training."""

    agent_lr_start: float = tunable(1e-3, FloatDistribution(1e-4, 5e-2, log=True))
    """Initial learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will be used throughout training. """

    agent_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-3, log=True))
    """Final learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will not be used."""

    agent_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))
    """Weight decay for the agent using AdamW."""

    ppo_clip_epsilon: float = tunable(0.2, FloatDistribution(0.05, 2.0, log=True))
    """PPO loss will be clipped to only apply the loss when the log-likelihood
    is between 1 / (1 + ppo_clip_epsilon) and 1 + ppo_clip_epsilon.

    Because the ppo_loss is disabled by default, this field also has no effect by default.
    Because OutRL uses regularized VF training, VF clipping is not used.
    """

    vf_lr_schedule: Literal[None, "linear"] = tunable(
        "linear", CategoricalDistribution([None, "linear"])
    )
    """Learning rate schedule for the value function parameters. Typically used
    to decrease the learning rate to near-zero near the end of training."""

    vf_lr_start: float = tunable(3e-3, FloatDistribution(1e-4, 0.1, log=True))
    """Initial learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will be used throughout
    training. """

    vf_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-4, log=True))
    """Final learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will not be used. """

    vf_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))
    """Weight decay for the value function parameters using AdamW."""

    vf_minibatch_size: int = tunable(512, IntDistribution(1, 2**32, log=True))
    """Number of timesteps used in minibatches when pre-training and
    post-training the value function from frozen state encodings.

    Because the value function uses a memoryless architecture (the VF relies on
    the agent to encode memory, if necessary), this minibatch size is typically
    met precisely (except for one trailing odd-sized minibatch).
    """

    vf_pre_training_epochs: int = tunable(10, IntDistribution(0, 20))
    """Number of epochs of value function training to run from frozen state
    encodings each train_step() before training the agent.

    Because OutRL uses an AWR-style loss, training the VF before the policy is
    expected.
    """
    vf_post_training_epochs: int = tunable(1, IntDistribution(0, 20))
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

    v_trace_lambda: float = tunable(0.95, FloatDistribution(0.0, 1.0))
    """Lambda parameter to v-trace advantage estimation.

    Controls the bias-variance tradeoff between pure belmann bootstraps and
    monte-carlo estimates. A value of 1 corresponds to minimal bias, and
    matches v-trace as originally proposed.
    """

    v_trace_rho_max: float = tunable(1.0, FloatDistribution(1.0, 1e3, log=True))
    """The "value function truncation" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which rewards from increased likelihood
    off-policy actions can contribute to estimated advantages.
    """

    v_trace_c_max: float = tunable(1.0, FloatDistribution(1.0, 1e3, log=True))
    """The "trace-cutting" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which future value function estimates
    from increased likelihood off-policy states can contribute to estimated
    advantages.
    """

    kl_coef_init: float = tunable(0.0, FloatDistribution(0.0, 100.0))
    """Initial loss coefficient for KL penalty / regularization of the agent.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).

    This penalty is applied exactly if action_dists is provided by the agent,
    or applied approximately using action_lls otherwise.
    """

    kl_coef_lr: float = tunable(0.1, FloatDistribution(1e-4, 1.0, log=True))
    """How quickly to adapt the loss coefficient for KL penalty.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).
    """

    kl_coef_min: float = tunable(0.0, FloatDistribution(0.0, 1.0))
    """Minimum value of the KL coefficient. Setting this to a non-zero value
    can help stabilize training very low-entropy continuous action space
    policies using the PPO loss, but is typically unnecessary.
    """

    kl_coef_max: float = tunable(1e3, FloatDistribution(0.1, 1e6, log=True))
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

    kl_soft_target: float = tunable(0.25, FloatDistribution(1e-3, 10.0, log=True))
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

    entropy_schedule: Literal[None, "linear", "cosine"] = None
    """Whether to schedule an entropy loss.

    With None, no entropy schedule will be applied, and entropy_coef_init will
    be used throughout training.

    With "linear", entropy will be scaled down from the initial entropy at
    start of training to a fraction of that `entropy_schedule_end_fraction`.
    """

    entropy_schedule_end_fraction: float = tunable(0.1, FloatDistribution(0.0, 1.0))
    """Portion of "starting entropy" to attempt to maintain at end of
    training."""

    entropy_schedule_start_train_step: int = 1
    """Train step at which to measure the "starting entropy".

    This indicates at the end of which train step entropy should be measured.
    The default value measures the entropy after one train step.
    """

    entropy_coef_init: float = tunable(0.01, FloatDistribution(0, 1.0))
    """Entropy regularizer coefficient.

    Will be adjusted by the entropy schedule if entropy_schedule is set.

    If both entropy_coef_init and entropy_coef_lr are 0, entropy regularization
    is disabled.
    """

    entropy_coef_lr: float = tunable(0.001, FloatDistribution(1e-5, 0.01, log=True))
    """How quickly to adapt the loss coefficient for the entropy regularizer.

    If set to 0, the entropy regularization coefficient will not be tuned
    during training.
    """

    entropy_coef_min: float = tunable(-10, FloatDistribution(-100, 0))
    """Minimum value for the loss coefficient of the entropy regularizer.

    Negative values actively decrease the entropy.
    """

    entropy_coef_max: float = tunable(10, FloatDistribution(0, 100))
    """Maximum value for the loss coefficient of the entropy regularizer.
    """

    temperature_init: float = tunable(1.0, FloatDistribution(1e-2, 1e3, log=True))
    """Initial AWR temperature. If temperature_lr is 0, this value will be used
    throughout training.

    Very low values result in a sparse loss that only attempts to repeat
    very high-advantage actions.

    High values cause the AWR loss to ignore advantages and just perform
    behavioral cloning.
    """

    temperature_lr: float = tunable(0.0, FloatDistribution(0, 0.1))
    """How quickly to tune the AWR temperature.

    The temperature is tuned to follow the entropy schedule and complements
    entropy regularization. If set to 0, the temperature will not be adjusted
    during training.
    """

    temperature_min: float = tunable(0.05, FloatDistribution(0.0, 1.0))
    """Minimum AWR temperature. Small values exponentially increase the AWR
    loss.

    Very low values result in a sparse loss that only attempts to repeat
    very high-advantage actions.
    """

    # Does this parameter need to exist? Seems kinda useless
    temperature_max: float = tunable(100, FloatDistribution(0.1, 1e5, log=True))
    """Maximum AWR temperature. Large temperature values cause the
    AWR loss to ignore advantages and just perform behavioral
    cloning.
    """

    normalize_awr_advantages: bool = tunable(True, CategoricalDistribution([False, True]))
    """Whether to normalize the advantages across the batch before computing
    the AWR coefficients.

    Note that this is not used in the PPO loss.
    """

    advantage_clip: float = tunable(8.0, FloatDistribution(0.1, 8.0))
    """Max exponent value for advantages in AWR loss.

    Large values can lead to NaN errors.

    Can have a significant effect, since many timesteps will have clipped
    coefficients at low temperatures.
    """

    grad_norm_max: float = tunable(10.0, FloatDistribution(1.0, 1e2, log=True))
    """Grad norm to clip the actor and value function parameters to in the main
    training loop.

    Small values will consistently lower the loss.
    """

    recompute_train_inputs: bool = tunable(True, CategoricalDistribution([False, True]))
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
        runs_dir = os.path.abspath(self.runs_dir)
        object.__setattr__(self, "runs_dir", runs_dir)
        if isinstance(self.stderr_log_level, str):
            stderr_log_level = stick.LOG_LEVELS[self.stderr_log_level]
            object.__setattr__(self, "stderr_log_level", stderr_log_level)


def _discount_cumsum(x: torch.Tensor, discount: float):
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
