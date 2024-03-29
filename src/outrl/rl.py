from dataclasses import dataclass, replace
from textwrap import dedent
from typing import Any, Callable, Optional, TypeVar, Generator, Union, Literal
import os
import random
import warnings
import logging
import pickle
from glob import glob

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
    concat,
    make_mlp,
    pack_tensors,
    pack_tensors_check,
    pad_tensors,
    unpad_tensors,
    explained_variance,
    unpack_tensors,
    RunningMeanVar,
    make_scheduler,
    DictDataset,
    ActionDist,
    kl_div,
    approx_kl_div,
)
from outrl.config import Config, tunable, IntListDistribution, default_run_name

LOGGER = logging.getLogger("outrl")

T = TypeVar("T")


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

    action_dists: Optional[ActionDist] = None
    """Distribution used to generate actions.

    Will be used for the KL penalty if not None, cfg.kl_dist_penalty > 0, and
    cfg.use_approx_kl is False.

    Will be used for entropy regularization if not None and cfg.approx_entropy is False.
    """


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

    original_action_dists: Optional[ActionDist]
    """Original action distributions provided when the episode was added."""

    rewards: torch.Tensor
    """Rewards provided when the episode was added."""

    any_actions_possible: torch.Tensor
    """Boolean any_actions_possible mask."""

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

    stderr_log_level: stick.LogLevels = stick.LogLevels.INFO

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
    Used to avoid out-of-memory errors.

    Defaults to no limit. Automatically decreases on (most) memory errors.
    """

    minibatch_target_timesteps: int = tunable(64, IntDistribution(1, 50000, log=True))
    """Attempt to keep timesteps in each minibatch to this number of timesteps.

    Will still run whole episodes if they exceed this cap.
    """

    policy_epochs_per_train_step: int = tunable(3, IntDistribution(1, 100, log=True))
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    normalize_rewards: bool = tunable(False, CategoricalDistribution([True, False]))
    """Normalize rewards to have zero mean and unit variance."""

    expected_train_steps: int = 1000
    """Expected number of training steps. Used for controlling scheduled parameters."""

    actor_lr_schedule: str = tunable(
        "linear", CategoricalDistribution([None, "linear"])
    )
    actor_lr_start: float = tunable(1e-3, FloatDistribution(1e-6, 1e-2, log=True))
    actor_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-3, log=True))

    actor_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))

    actor_clip_ratio: float = tunable(0.2, FloatDistribution(1e-3, 10.0, log=True))

    vf_lr_schedule: str = tunable("linear", CategoricalDistribution([None, "linear"]))
    vf_lr_start: float = tunable(3e-3, FloatDistribution(1e-6, 1e-2, log=True))
    vf_lr_end: float = tunable(1e-6, FloatDistribution(1e-8, 1e-4, log=True))

    vf_weight_decay: float = tunable(1e-6, FloatDistribution(1e-8, 1e-2, log=True))

    vf_minibatch_size: int = tunable(512, IntDistribution(1, 2**32, log=True))
    vf_pre_training_epochs: int = tunable(0, IntDistribution(0, 100))
    vf_post_training_epochs: int = tunable(10, IntDistribution(0, 100))

    vf_recompute_targets: bool = tunable(False, CategoricalDistribution([True, False]))

    vf_loss_coef: float = tunable(0.1, FloatDistribution(0.01, 2.0))

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

    kl_coef_init: float = tunable(0.0, FloatDistribution(0.0, 100.0))
    kl_coef_lr: float = tunable(0.01, FloatDistribution(1e-4, 1.0, log=True))
    kl_coef_min: float = tunable(0.0, FloatDistribution(0.0, 1.0))
    kl_coef_max: float = tunable(1e5, FloatDistribution(0.1, 1e10, log=True))
    kl_target_stat: Literal["max", "mean"] = tunable(
        "max", CategoricalDistribution(["max", "mean"])
    )
    kl_soft_target: float = tunable(0.25, FloatDistribution(1e-3, 10.0, log=True))
    kl_fixup_coef: float = tunable(3, FloatDistribution(1.0, 20.0, log=True))
    kl_use_fixup: bool = False

    use_approx_kl: bool = False
    """Approximate the KL divergence using the log-likelihoods."""

    use_approx_entropy: bool = False
    """Approximate the action entropy using the log-likelihoods."""

    entropy_target: float = tunable(-10.0, FloatDistribution(-100.0, 0.0))

    use_top_half_advantages: bool = tunable(
        False, CategoricalDistribution([True, False])
    )

    grad_norm_max: float = tunable(10.0, FloatDistribution(1.0, 1e3, log=True))

    checkpoint_interval: int = 1
    """Number of train_step calls between checkpoints when calling maybe_checkpoint().

    Disable checkpointing by setting to a negative value."""

    checkpoint_best: bool = True
    """Whether to checkpoint in maybe_checkpoint() after an improvement in the
    primary performance statistic passed to add_eval_stats."""

    checkpoint_replay_buffer: bool = True
    """Whether to checkpoint the replay_buffer."""

    def fill_defaults(self):
        """Fill in values with non-constant defaults. Called after construction."""
        if self.seed < -1:
            raise ValueError("seed should be positive or exactly -1")
        if self.checkpoint_interval < -1:
            raise ValueError("checkpoint_interval should be positive or exactly -1")
        log_dir = os.path.abspath(self.log_dir)
        if isinstance(self.stderr_log_level, str):
            stderr_log_level = stick.LOG_LEVELS[self.stderr_log_level]
        else:
            stderr_log_level = self.stderr_log_level
        return replace(self, log_dir=log_dir, stderr_log_level=stderr_log_level)


@dataclass
class PolicyTrainingInputs:
    """Values used to train the policy that are not present in EpisodeData.

    An instance of this class accompanies each EpisodeData during policy optimization.

    These values are recomputed every train_step()."""

    advantages: torch.Tensor
    discounted_returns: torch.Tensor
    vf_returns: torch.Tensor
    vf_targets: torch.Tensor


SUB_STATE_DICT_FIELDS = [
    "actor",
    "vf",
    "reward_normalizer",
    "actor_optimizer",
    "vf_optimizer",
    "vf_lr_scheduler",
    "actor_lr_scheduler",
    "kl_coef_opt",
]

IGNORED_FIELDS = ["_is_full_backward_hook"]


class Trainer(nn.Module):
    def __init__(self, cfg: TrainerConfig, actor):
        super().__init__()
        self.cfg = cfg

        self.actor = actor

        self.observation_latent_size = self.actor.observation_latent_size

        self.vf = make_mlp(
            input_size=self.observation_latent_size,
            hidden_sizes=self.cfg.vf_hidden_sizes,
            output_size=0,
            use_dropout=True,
        )
        vf_output = self.vf.get_submodule("output_linear")
        vf_output.weight.data.copy_(0.01 * vf_output.weight.data)

        self.reward_normalizer = RunningMeanVar()
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=self.cfg.actor_lr_start,
            weight_decay=self.cfg.actor_weight_decay,
        )
        self.actor_lr_scheduler = make_scheduler(
            self.actor_optimizer,
            self.cfg.actor_lr_schedule,
            self.cfg.actor_lr_start,
            self.cfg.actor_lr_end,
            self.cfg.expected_train_steps,
        )

        self.vf_optimizer = torch.optim.Adam(
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

        self._replay_buffer: list[EpisodeData] = []
        self._next_episode_number: int = 0
        self._dtype = torch.float32

        self.total_env_steps: int = 0
        self.train_steps_so_far: int = 0

        self.last_training_stats = {}
        self.last_eval_stats = {}
        self.primary_performance: float = float("-inf")
        self.primary_performance_at_last_checkpoint: float = float("-inf")
        self.train_steps_so_far_at_last_checkpoint = 0

        self.kl_coef = nn.Parameter(torch.tensor(float(self.cfg.kl_coef_init)))
        self.kl_coef_opt = torch.optim.Adam([self.kl_coef], lr=self.cfg.kl_coef_lr)

    def _primary_loss_function(
        self, batch: list[tuple[EpisodeData, PolicyTrainingInputs, ActorOutput]]
    ) -> torch.Tensor:
        episode_data: list[EpisodeData] = [b[0] for b in batch]
        train_inputs: list[PolicyTrainingInputs] = [b[1] for b in batch]
        actor_outputs: list[ActorOutput] = [b[2] for b in batch]

        kl_loss, kl_info = self._kl_loss(episode_data, train_inputs, actor_outputs)

        ppo_loss, ppo_info = self._ppo_loss(episode_data, train_inputs, actor_outputs)
        vf_loss, vf_info = self._vf_loss(episode_data, train_inputs, actor_outputs)

        loss = ppo_loss + vf_loss + kl_loss

        # kl_info, vf_info, and ppo_info will all get included in locals
        self._log_training_infos(locals())
        return loss

    def _ppo_loss(
        self,
        episode_data: list[EpisodeData],
        train_inputs: list[PolicyTrainingInputs],
        actor_outputs: list[ActorOutput],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        advantages, adv_len = pack_tensors(
            [train_input.advantages for train_input in train_inputs]
        )
        log_probs = pack_tensors_check(
            [actor_out.action_lls for actor_out in actor_outputs], adv_len
        )

        old_log_probs = pack_tensors_check(
            [data.original_action_lls for data in episode_data], adv_len
        )
        assert not old_log_probs.requires_grad

        ratio = torch.exp(log_probs - old_log_probs)
        ratio_clipped = torch.clamp(
            ratio, 1 / (1 + self.cfg.actor_clip_ratio), 1 + self.cfg.actor_clip_ratio
        )

        policy_gradient = ratio * advantages
        clip_policy_gradient = ratio_clipped * advantages

        ppo_loss = -torch.min(policy_gradient, clip_policy_gradient).mean()
        return (
            ppo_loss,
            dict(
                clip_portion=(ratio_clipped != ratio).mean(dtype=torch.float32),
            )
            | locals(),
        )

    def _vf_loss(
        self,
        episode_data: list[EpisodeData],
        train_inputs: list[PolicyTrainingInputs],
        actor_outputs: list[ActorOutput],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # For consistency with other loss functions, we still take
        # episode_data, but don't use it.
        del episode_data
        vf_targets_packed, adv_len = pack_tensors(
            [train_input.vf_targets for train_input in train_inputs]
        )
        obs_latents_packed = pack_tensors_check(
            [actor_out.observation_latents[:-1] for actor_out in actor_outputs], adv_len
        )

        critic_out = self.vf(obs_latents_packed)
        vf_loss = F.mse_loss(critic_out, vf_targets_packed)

        discounted_returns = pack_tensors_check(
            [train_input.discounted_returns for train_input in train_inputs], adv_len
        )

        return (
            self.cfg.vf_loss_coef * vf_loss,
            dict(
                mb_ev=explained_variance(critic_out, discounted_returns),
            )
            | locals(),
        )

    def _kl_loss(
        self,
        episode_data: list[EpisodeData],
        train_inputs: list[PolicyTrainingInputs],
        actor_outputs: list[ActorOutput],
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # For consistency with other loss functions, we still take
        # train_inputs, but don't use it.
        del train_inputs
        # Computed for logging purposes
        original_kl_coef = self.kl_coef.detach().item()

        # Compute KL Divergence
        if (
            not self.cfg.use_approx_kl
            and episode_data[0].original_action_dists is not None
        ):
            new_dists = [actor_out.action_dists for actor_out in actor_outputs]
            assert new_dists[0] is not None
            old_dists = [ep_data.original_action_dists for ep_data in episode_data]
            assert old_dists[0] is not None
            # TODO: Add options for other KL directions
            kl = kl_div(old_dists, new_dists)
        else:
            log_probs, lengths = pack_tensors(
                [actor_out.action_lls for actor_out in actor_outputs]
            )
            old_log_probs = pack_tensors_check(
                [data.original_action_lls for data in episode_data], lengths
            )
            kl = approx_kl_div(old_log_probs, log_probs)

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
        if self.kl_coef > self.cfg.kl_coef_max:
            with torch.no_grad():
                self.kl_coef.copy_(self.cfg.kl_coef_max)
        kl_coef = self.kl_coef.detach()

        return kl_coef * kl, locals()

    def _actor_minibatches(
        self,
        episodes: Optional[list[EpisodeData]] = None,
        extra_data: Optional[list[T]] = None,
        minibatch_target_timesteps: Optional[int] = None,
        desc: Optional[str] = None,
        epochs: int = 1,
        shuffle: bool = False,
        start_of_epoch_callback: Optional[Callable[[int], None]] = None,
    ) -> Generator[list[tuple[EpisodeData, T, ActorOutput]], None, None]:
        """Runs the actor forward pass on minibatches drawn from the replay buffer.

        Minibatches are a list of tuples of EpisodeData from the input, data T
        from the extra_data (None if not provided), and ActorOutput from the
        actor forward pass.

        This method handles sizing of minibatches to avoid OOM, shuffling of
        episodes, rendering the progress bar, and running for multiple epochs.
        Basically, this method plays a similar role to "Trainer" classes in
        supervised learning libraries.

        Yields: list[tuple[EpisodeData, T, ForwardResult]]
        """
        if episodes is None:
            episodes = self._replay_buffer

        if len(extra_data) == 0 or extra_data is None:
            extra_data = [None for _ in episodes]
        assert len(extra_data) == len(episodes)

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
                if start_of_epoch_callback is not None:
                    start_of_epoch_callback(epoch)
                next_ep_index = 0
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
                        forward_result = self.actor([data[0].episode for data in batch])
                        yield [
                            (data[0], data[1], f_res)
                            for (data, f_res) in zip(batch, forward_result)
                        ]
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

        train_inputs = self.preprocess()
        self.log_dataset(train_inputs)
        self.actor.train(mode=True)
        self.vf.train(mode=True)

        # Cached policy outputs for VF training.
        # This avoids performing a full pass on the actor network when tuning
        # the VF outside of the primary loss.
        # The VF loss still tunes the full network in the primary loss.
        observation_latent_cache = {}
        action_lls_cache = {}

        # Pre-train VF (usually only used for off-policy algorithms)
        if self.cfg.vf_pre_training_epochs > 0:
            with torch.no_grad():
                for batch in self._actor_minibatches(desc="Caching latents"):
                    for ep_data, _, actor_res in batch:
                        observation_latent_cache[
                            ep_data.episode_number
                        ] = actor_res.observation_latents
                        action_lls_cache[ep_data.episode_number] = actor_res.action_lls
            self._train_vf(
                observation_latent_cache,
                action_lls_cache,
                self.cfg.vf_pre_training_epochs,
                desc="Pre Training VF",
            )

        # Run primary training loop.
        for batch in self._actor_minibatches(
            desc="Training Actor",
            extra_data=train_inputs,
            epochs=self.cfg.policy_epochs_per_train_step,
            shuffle=True,
            minibatch_target_timesteps=self.cfg.minibatch_target_timesteps,
        ):
            for ep_data, _, actor_res in batch:
                observation_latent_cache[
                    ep_data.episode_number
                ] = actor_res.observation_latents.detach()
                action_lls_cache[ep_data.episode_number] = actor_res.action_lls.detach()
            loss = self._primary_loss_function(batch)
            self.vf_optimizer.zero_grad()
            self.actor_optimizer.zero_grad()
            loss.backward()
            try:
                clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.cfg.grad_norm_max,
                    error_if_nonfinite=True,
                )
                clip_grad_norm_(
                    self.vf.parameters(),
                    max_norm=self.cfg.grad_norm_max,
                    error_if_nonfinite=True,
                )
                self.vf_optimizer.step()
                self.actor_optimizer.step()
            except RuntimeError as ex:
                # This seems to only trigger if the batch is so small the loss
                # is NaN or so big we OOM.
                # Because we checked for non-finite in grad_norm, this should
                # reliably prevent corrupting the network, at the cost of
                # potentially extremely slowing training on e.g. an over-fit
                # VF or excessively off-policy data.
                # TODO: Crash if we catch too many errors here
                LOGGER.error(f"RuntimeError in actor optimizations: {ex}")

        # Extra VF tuning after the primary training loop.
        # This achieves similar objectives to PPG by simply not doing updates
        # on the actor network in this phase.
        # Inputs are guaranteed to be cached, since we ran at least one full
        # epoch in the primary loop.
        if self.cfg.vf_post_training_epochs > 0:
            self._train_vf(
                observation_latent_cache,
                action_lls_cache,
                self.cfg.vf_post_training_epochs,
                desc="Post Training VF",
            )

        # Update all the statistics.
        self.actor_lr_scheduler.step()
        self.vf_lr_scheduler.step()
        self.actor.train(mode=False)
        self.vf.train(mode=False)
        stick.log("last_training_stats", self.last_training_stats, level=stick.RESULTS)
        self._replay_buffer = []
        self.train_steps_so_far += 1

    def _train_vf(
        self,
        observation_latents: dict[int, torch.Tensor],
        action_lls: dict[int, torch.Tensor],
        training_epochs: int,
        desc: str,
    ):
        """Train just the VF using cached actor outputs.

        observation_latents and action_lls are indexed by the `episode_number`
        field of EpisodeData, and should contain non-differentiable cached
        components from each episode's ActorOutput.

        This method does not tune the parameters of the actor.

        Because this training is only tuning the memoryless VF tail, it uses
        smaller minibatches of shuffled timesteps from across multiple
        episodes.
        """
        obs_latents_packed, obs_lens = pack_tensors(
            [
                observation_latents[ep_data.episode_number]
                for ep_data in self._replay_buffer
            ]
        )
        assert not obs_latents_packed.requires_grad

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
                    vf_x_packed = self.vf(obs_latents_packed)
                    vf_x = pad_tensors(unpack_tensors(vf_x_packed, obs_lens))

                    for i, episode_length in enumerate(episode_lengths):
                        # zero vf_{t+1} in terminated episodes
                        if self._replay_buffer[i].terminated:
                            vf_x[i, episode_length] = 0.0
                            terminated[i] = True

                    with torch.no_grad():
                        _, vf_targets = v_trace_estimation(
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

                    vf_targets_packed = pack_tensors_check(
                        unpad_tensors(vf_targets, obs_lens), obs_lens
                    )

                dataset = DictDataset(
                    observation_latents=obs_latents_packed, vf_targets=vf_targets_packed
                )
                for batch in dataset.minibatches(self.cfg.vf_minibatch_size):
                    self.vf_optimizer.zero_grad()
                    vf_out = self.vf(batch["observation_latents"])
                    vf_loss = F.mse_loss(vf_out, batch["vf_targets"])
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
                    pbar.n += len(batch["observation_latents"])
                    pbar.refresh()

    def _log_training_infos(self, train_locals: dict[str, Any]):
        training_stats = {
            k: train_locals[k].item() for k in ["vf_loss", "ppo_loss", "kl_loss"]
        }
        training_stats["clip_portion"] = train_locals["ppo_info"]["clip_portion"]
        stick.log("training_stats", training_stats, level=stick.INFO)
        self.last_training_stats = training_stats

        stick.log("train_locals", train_locals, level=stick.TRACE)

    def log_dataset(self, train_inputs: list[PolicyTrainingInputs]):
        full_episode_rewards = pad_tensors(
            [data.rewards for data in self._replay_buffer]
        )
        vf_returns, vf_lens = pack_tensors(
            [train_input.vf_returns for train_input in train_inputs]
        )
        discounted_returns = pack_tensors_check(
            [train_input.discounted_returns for train_input in train_inputs], vf_lens
        )
        vf_targets = pack_tensors_check(
            [train_input.vf_targets for train_input in train_inputs], vf_lens
        )
        dataset_stats = {
            "ep_rew": full_episode_rewards.sum(dim=-1).mean(dim=0),
            "ev": explained_variance(
                vf_returns,
                discounted_returns,
            ),
            "disc_mean": discounted_returns.mean(),
            "vf_mean": vf_returns.mean(),
            "vf_target_mean": vf_targets.mean(),
            "total_env_steps": self.total_env_steps,
        }
        # for k in self._replay_buffer[0].infos.keys():
        #     dataset_stats[k] = concat(data.infos[k] for data in self._replay_buffer)

        stick.log(
            "dataset_stats",
            dataset_stats,
            level=stick.RESULTS,
            step=self.total_env_steps,
        )

        for k in self._replay_buffer[0].infos.keys():
            dataset_stats[k] = concat(data.infos[k] for data in self._replay_buffer)

        stick.log(
            "dataset_stats_trace",
            dataset_stats,
            level=stick.TRACE,
            step=self.total_env_steps,
        )

    def preprocess(self):
        episode_lengths = [len(data.rewards) for data in self._replay_buffer]
        obs_lens = [len(data.episode["observations"]) for data in self._replay_buffer]
        assert [ep_len + 1 for ep_len in episode_lengths] == obs_lens

        # Compute vf_returns
        self.actor.train(mode=False)
        self.vf.train(mode=False)
        with torch.no_grad():
            actor_outputs = self.actor([data.episode for data in self._replay_buffer])
            obs_latents_packed = pack_tensors_check(
                [actor_out.observation_latents for actor_out in actor_outputs], obs_lens
            )
            vf_returns_packed = self.vf(obs_latents_packed)
        self.actor.train(mode=True)
        self.vf.train(mode=True)

        action_lls_now = pad_tensors(
            [actor_out.action_lls for actor_out in actor_outputs]
        )
        original_action_lls = pad_tensors(
            [data.original_action_lls for data in self._replay_buffer]
        )
        terminated = torch.zeros(len(self._replay_buffer), dtype=torch.bool)

        vf_returns = pad_tensors(unpack_tensors(vf_returns_packed, obs_lens))
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

        advantages, vf_targets = v_trace_estimation(
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

        discounted_returns = discount_cumsum(padded_rewards, discount=discount)

        train_inputs = [
            PolicyTrainingInputs(
                advantages=adv,
                discounted_returns=disc_ret,
                vf_returns=vf_ret,
                vf_targets=vf_target,
            )
            for (adv, disc_ret, vf_ret, vf_target) in zip(
                unpad_tensors(advantages, episode_lengths),
                unpad_tensors(discounted_returns, episode_lengths),
                unpad_tensors(vf_returns, episode_lengths),
                unpad_tensors(vf_targets, episode_lengths),
                # unpad_tensors(discounted_returns, episode_lengths),
            )
        ]
        assert len(train_inputs) == len(self._replay_buffer)
        assert [
            len(train_input.advantages) for train_input in train_inputs
        ] == episode_lengths
        assert [
            len(train_input.discounted_returns) for train_input in train_inputs
        ] == episode_lengths

        return train_inputs

    def add_episode(
        self,
        episode: Any,
        rewards: torch.Tensor,
        action_lls: torch.Tensor,
        terminated: bool,
        action_dists: Optional[ActionDist] = None,
        any_actions_possible: Optional[torch.Tensor] = None,
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
            any_actions_possible (torch.Tensor?): Boolean Tensor indicating if any
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
        if any_actions_possible is None:
            any_actions_possible = torch.ones(num_timesteps, dtype=torch.bool)
        else:
            assert isinstance(any_actions_possible, torch.Tensor)

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
                original_action_dists=action_dists,
                rewards=rewards,
                any_actions_possible=any_actions_possible,
                sample_priority=sample_priority,
                infos=infos,
            )
        )
        self._next_episode_number += 1

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        logging.info(f"Eval primary stat ({primary}): {stats[primary]}")
        log("eval_stats", stats, step=self.total_env_steps, level=stick.RESULTS)
        self.primary_performance = stats[primary]
        self.last_eval_stats = stats
        hparams = self.cfg.to_dict()
        hparams["metric-primary"] = stats[primary]
        for k, v in stats.items():
            hparams[k] = v
        log("hparams", hparams, step=self.total_env_steps)

    def state_dict(self):
        state = {}
        for k in SUB_STATE_DICT_FIELDS:
            state[k] = getattr(self, k).state_dict()
        for k, v in self.__dict__.items():
            if k in IGNORED_FIELDS or k in SUB_STATE_DICT_FIELDS:
                continue
            elif k == "cfg":
                state[k] = v.to_dict()
            elif k == "replay_buffer":
                if self.cfg.checkpoint_replay_buffer:
                    state[k] = v
            else:
                if hasattr(v, "state_dict"):
                    LOGGER.error(
                        f"Field {k} was not expected to have a state_dict method"
                    )
                state[k] = v
        return state

    def load_state_dict(self, state_dict):
        state = state_dict
        for k in SUB_STATE_DICT_FIELDS:
            assert k in state, f"Missing {k!r} from state dict"
        for k, v in state.items():
            if k == "cfg":
                self.cfg = type(self.cfg).from_dict(v)
            elif k in SUB_STATE_DICT_FIELDS:
                getattr(self, k).load_state_dict(v)
            else:
                field_now = getattr(self, k, None)
                if field_now is None:
                    LOGGER.error(f"Attempting to set unknown field {k}")
                else:
                    if hasattr(field_now, "load_state_dict"):
                        LOGGER.error(
                            f"Field {k} was not expected to have a load_state_dict method"
                        )
                setattr(self, k, v)

        # Make sure optimizers are attached to parameters
        assert self.kl_coef_opt.param_groups[0]["params"][0] is self.kl_coef
        assert self.vf_optimizer.param_groups[0]["params"][0] is next(
            self.vf.parameters()
        )
        assert self.actor_optimizer.param_groups[0]["params"][0] is next(
            self.actor.parameters()
        )

    def maybe_checkpoint(self):
        checkpoint_interval = (
            self.cfg.checkpoint_interval >= 0
            and self.train_steps_so_far - self.train_steps_so_far_at_last_checkpoint
            >= self.cfg.checkpoint_interval
        )
        checkpoint_best = (
            self.cfg.checkpoint_best
            and self.primary_performance > self.primary_performance_at_last_checkpoint
        )
        if checkpoint_interval or checkpoint_best:
            state_dict = self.state_dict()
            if checkpoint_interval:
                f_name = os.path.join(
                    self.cfg.log_dir,
                    self.cfg.run_name,
                    f"train_step_{self.train_steps_so_far}.pkl",
                )
                LOGGER.info(f"Checkpointing to {f_name!r}")
                with open(
                    f_name,
                    "wb",
                ) as f:
                    pickle.dump(state_dict, f)
                self.train_steps_so_far_at_last_checkpoint = self.train_steps_so_far
            if checkpoint_best:
                f_name = os.path.join(self.cfg.log_dir, self.cfg.run_name, f"best.pkl")
                LOGGER.info(f"Checkpointing to {f_name!r}")
                with open(f_name, "wb") as f:
                    pickle.dump(state_dict, f)
            return True
        else:
            return False

    def attempt_resume(
        self, prefer_best: bool = False, checkpoint_dir: Optional[str] = None
    ):
        if checkpoint_dir is None:
            checkpoint_dir = os.path.join(self.cfg.log_dir, self.cfg.run_name)
        best_ckpt = os.path.join(checkpoint_dir, "best.pkl")
        if prefer_best and os.path.exists(best_ckpt):
            try:
                with open(best_ckpt, "rb") as f:
                    state = pickle.load(f)
                    self.load_state_dict(state)
                    LOGGER.critical(f"Resuming from checkpoint {best_ckpt}")
                    return True
            except (pickle.UnpicklingError, ValueError) as ex:
                LOGGER.error(f"Could not load {best_ckpt}: {ex}")
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
                    LOGGER.critical(f"Resuming from checkpoint {f_name}")
                    return True
            except (pickle.UnpicklingError, ValueError) as ex:
                LOGGER.error(f"Could not load {f_name}: {ex}")
        return False


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


@torch.jit.script
def v_trace_estimation(
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
        gammas (torch.Tensor): A 2D tensor of per-timestep discount factors.
            Used to avoid discounting across states where no action was
            possible.
        vf_x (torch.Tensor): A 2D tensor of value function estimates with shape
            (N, T + 1), where N is the batch dimension (number of episodes) and
            T is the maximum episode length experienced by the actor. If an
            episode terminates in fewer than T time steps, the remaining
            elements in that episode should be set to 0.
        rewards (torch.Tensor): A 2D tensor of per-step rewards with shape
            (N, T), where N is the batch dimension (number of episodes) and T
            is the maximum episode length experienced by the actor.
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
    advantages = rho * (rewards * gammas * v_s[:, 1:] - vf_x[:, :-1])
    return advantages, v_s
