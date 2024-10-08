"""
All purely reinforcement learning code for PyTorch should be in this file.

TorchTrainer contains the primary interface to ubrl.
"""

import dataclasses
from dataclasses import dataclass
from typing import Any, Optional, Generator, Callable
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

import ubrl.torch_utils as tu
import ubrl.torch_cluster
from ubrl.config import TrainerConfig

_LOGGER = logging.getLogger("ubrl")

EpisodeID = int


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
        See [gym_utils.py](examples/hf_llm_example.py) or
        [hf_llm_example.py](examples/hf_llm_example.py) for example
        implementations.

        More in depth documentation is available in
        [agent_api.qmd](notes/agent_api.qmd).
        """
        del inputs
        raise NotImplementedError()


@dataclass(eq=False)
class AgentInput:
    """Input to Agent forward() method.

    See `AgentOutput` for the return type.
    """

    episodes: list[Any]
    """List of episodes. Will be whatever type was passed in to
    `TorchTrainer.add_episode`.
    """

    need_full: bool = True
    """If True, the agent must return action_lls and state_encodings for the
    entire episode (i.e. `AgentOutput.valid_mask` should be None or all True).

    If False, the agent can optionally only propagate a portion of timesteps.
    """


@dataclass(eq=False)
class AgentOutput:
    """Result from all episodes concatenated together when calling .forward()
    on an agent.

    Differentiating both `state_encodings` and `action_lls` should affect
    earlier layers in the agent.

    See `AgentInput` for the input type.
    """

    state_encodings: torch.Tensor
    """Differentiable representation of the observations. Because there should
    be a trailing final state, this tensor should be longer than action_lls by
    the number of episodes in AgentInput."""

    action_lls: torch.Tensor
    """Differentiable log-likelihood of actions taken, packed into a single
    dimension."""

    n_timesteps: list[int]
    """Length of each episode packed together in `action_lls`.
    Provided as input by `AgentInput.n_timesteps`.
    This is the number of rewards / actions in each episode.
    Each episode should contain one additional `state_encoding`.
    Used for checking shape of other values, and for splitting episodes in CachedAgent.

    If it is necessary to compute fewer timesteps, pad the values as if all
    values were computed and set the valid_mask to indicate which timesteps are
    fully computed.
    Value function training always requires all state encodings and action lls,
    which are requested by AgentInput.need_full.
    """

    terminated: torch.Tensor
    """Boolean Tensor indicating if the episode reached a terminal state.
    Should have length equal to the number of episodes.
    Should contain False for an episode in an infinite horizon MDPs or when an
    episode reaches a timeout.
    """

    original_action_lls: torch.Tensor
    """Original action lls provided when the episode was added. Used for
    v-trace off-policy correction and in the PPO loss (when enabled).
    Packed into a single dimansion.

    Do not require gradients.
    """

    rewards: torch.Tensor
    """Rewards acquired in the episode. Should be packed into a single
    dimension of the same shape as the action_lls (i.e. sum(n_timesteps)).
    """

    action_dists: Optional[list[tu.ActionDist]] = None
    """Distribution used to generate actions.

    Will be used for the KL penalty if not None and `cfg.use_approx_kl` is False.

    Will be used for entropy regularization if not None and `cfg.approx_entropy`
    is False.
    """

    original_action_dists: Optional[tu.ActionDist] = None
    """Original action distributions optionally provided when the episode was
    added."""

    valid_mask: Optional[torch.Tensor] = None
    """Mask of timesteps where `state_encodings` and `action_lls` were computed
    and can be differentiated for.
    If not None, must have same shape as `action_encodings`.

    Mostly used for rescaling losses.
    If None, all timesteps are presumed to be valid.
    """

    inherent_loss: Optional[torch.Tensor] = None
    """Loss computed by the model itself. Can be used for regularization or
    supervised co-training.

    Typically should be near unit scale when used.
    """

    infos: dict[str, list[Any]] = dataclasses.field(default_factory=dict)
    """Not used by ubrl, but can be used if agent.forward() needs to return
    additional values for other use cases."""

    @property
    def n_observations(self):
        return [n + 1 for n in self.n_timesteps]

    def __post_init__(self):
        """Checks types and shapes of output data."""

        assert (
            len(self.state_encodings.shape) == 2
        ), "state_encodings should have exactly two dimensions"
        assert (
            len(self.action_lls.shape) == 1
        ), "action_lls should have exactly one dimension"
        assert self.action_lls.shape[0] == sum(
            self.n_timesteps
        ), "Total number of timesteps does not match number of provided action_lls"
        assert (
            len(self.original_action_lls.shape) == 1
        ), "action_lls should have exactly one dimension"
        assert self.original_action_lls.shape[0] == sum(
            self.n_timesteps
        ), "Total number of timesteps does not match number of provided action_lls"
        assert (
            self.state_encodings.shape[0] != self.action_lls.shape[0]
        ), "There should be one more state_encoding per episode than action_ll. "
        num_expected_states = sum(self.n_timesteps) + len(self.n_timesteps)
        if self.state_encodings.shape[0] != num_expected_states:
            raise AssertionError(
                f"AgentOutput number of state_encodings did not match total number of states in episodes. There should be one more state_encoding per episode than action_lls. Expected: {num_expected_states} Actual: {self.state_encodings.shape[0]}"
            )

        if self.valid_mask is not None:
            assert (
                len(self.valid_mask.shape) == 1
            ), "valid_mask should have exactly one dimension"
            assert (
                self.valid_mask.shape[0] == self.state_encodings.shape[0]
            ), "valid_mask shape does not match state_encodings shape"
            assert (
                self.valid_mask.dtype == torch.bool
            ), "valid_mask does not use torch.bool dtype"

        if self.inherent_loss is not None:
            assert (
                len(self.inherent_loss.shape) == 1
            ), "Inherent loss should have one dimension, the episode / batch dimension"
            assert self.inherent_loss.shape[0] == len(
                self.n_timesteps
            ), "If provided, there should be one inherent_loss value per episode"

        if self.action_dists is not None:
            assert isinstance(self.action_dists, list)
            assert (
                len(self.action_dists) == len(self.n_timesteps)
            ), 'If provided, action_dists should be a list containing one "ActionDist" per episode'

    def full_valid(self):
        """If the AgentOutput has contains all timesteps for the requested
        episodes."""
        return self.valid_mask is None or self.valid_mask.all()

    def unpack(self) -> list["AgentOutput"]:
        """Splits an `AgentOutput` into one `AgentOutput` per episode.
        Used by `CachedAgent` to cache individual episode outputs.
        """
        return tu.unpack_dataclass(
            self,
            {
                "action_lls": self.n_timesteps,
                "rewards": self.n_timesteps,
                "terminated": [1 for _ in self.n_timesteps],
                "inherent_loss": [1 for _ in self.n_timesteps],
                "original_action_lls": self.n_timesteps,
                "state_encodings": self.n_observations,
                "valid_mask": self.n_observations,
            },
        )

    @classmethod
    def pack(cls, agent_outputs: list["AgentOutput"]) -> "AgentOutput":
        return tu.pack_dataclass(cls, agent_outputs)

    def detach(self) -> "AgentOutput":
        detached = dataclasses.replace(
            self,
            state_encodings=self.state_encodings.detach(),
            action_lls=self.action_lls.detach(),
        )
        for field_name, field_val in vars(detached).items():
            if isinstance(field_val, torch.Tensor) and field_val.requires_grad:
                raise ValueError(
                    f"AgentOutput.{field_name} should not require_grad after being detached"
                )
        return detached


@dataclass(eq=False)
class EpisodeData:
    """Wrapper around an episode that maintains metadata and caches computed values."""

    episode: Any
    """The episode (treated as an opaque object). Will be passed back to the
    agent during optimization."""

    episode_id: EpisodeID
    """Records which add_episode() call this episode came from. Used as a key
    by CachedAgent."""

    n_timesteps: int
    """Number of time steps in episode. This value is mostly used for progress bars.
    The `AgentOutput.n_timesteps` is treated as more authoritative in all
    circumstances where it is available.
    """

    memory_size: float
    """The amount of memory required to include this episode into a forward
    pass. This value is used to gather episodes into minibatches of a given
    total size.
    Can be an arbitrary unit, but must match
    `TrainerConfig.minibatch_target_size` (which defaults to 1024) and
    `TrainerConfig.minibatch_max_size`."""

    infos: dict[str, Any]
    """User infos. Will be logged if possible."""

    def __post_init__(self):
        assert self.n_timesteps > 0
        assert self.memory_size > 0


@dataclass(eq=False)
class Targets:
    """Input values to the loss function not present in AgentOutput.

    Analogous to "targets" or "labels" in supervised learning.
    These values are recomputed every `TorchTrainer.train_step()`."""

    advantages: torch.Tensor
    """Advantages for this episode. Used in PPO loss (when enabled)."""

    exp_advantages: torch.Tensor
    """Coefficients used in the AWR loss (when enabled).
    Normalized and clipped, and therefore not exactly equivalent to `advantages.exp()`.
    """

    vf_targets: torch.Tensor
    """Target values for the value function loss."""

    vf_returns: torch.Tensor
    """Estimated returns from the value function. Used in logging."""

    vf_loss_mask: torch.Tensor
    """Mask for which states the VF loss should be applied to.
    Mostly used to exclude final non-terminal states."""

    n_timesteps: list[int]
    """Copied from `AgentOutput.n_timesteps`. Mostly just used in self-asserts
    within this type.
    """

    def __post_init__(self):
        assert not self.vf_returns.requires_grad
        assert not self.vf_targets.requires_grad
        assert not self.exp_advantages.requires_grad
        assert not self.advantages.requires_grad
        total_timesteps = sum(self.n_timesteps)
        n_observations = len(self.n_timesteps) + total_timesteps
        # All of these fields should be packed
        assert self.advantages.shape == (total_timesteps,)
        assert self.exp_advantages.shape == (total_timesteps,)
        assert self.vf_targets.shape == (n_observations,)
        assert self.vf_returns.shape == (n_observations,)
        assert self.vf_loss_mask.shape == (n_observations,)

    @classmethod
    def pack(cls, targets: list["Targets"]) -> "Targets":
        return tu.pack_dataclass(cls, targets)

    def unpack(self) -> list["Targets"]:
        n_observations = [n + 1 for n in self.n_timesteps]
        return tu.unpack_dataclass(
            self,
            {
                "advantages": self.n_timesteps,
                "exp_advantages": self.n_timesteps,
                "vf_targets": n_observations,
                "vf_returns": n_observations,
                "vf_loss_mask": n_observations,
            },
        )


class CachedAgent:
    """Performs checking of the `Agent` API and caches outputs.

    Cached values are used to cheaply fine-tune value function training.
    This class does _not_ implement Transformer key caching and is used with
    all implementations of the `Agent` API.
    """

    def __init__(
        self, agent: Agent, cfg: "TrainerConfig", cluster: "ubrl.cluster.Cluster",
        use_cache: bool = True
    ):
        self.agent: Agent = agent
        self.cfg = cfg
        self.cluster = cluster
        self.agent_out_cache: dict[EpisodeID, AgentOutput] = {}
        self.use_cache = use_cache

    def clear_caches(self):
        self.agent_out_cache = {}

    def fill_caches(self, episode_data: list[EpisodeData]):
        """Ensure that all episodes in `episode_data` have cached values."""
        if not self.use_cache:
            return
        missing_episodes = [
            ep_data
            for ep_data in episode_data
            if ep_data.episode_id not in self.agent_out_cache
        ]
        with torch.no_grad():
            for minibatch in minibatch_episodes(
                self.cluster,
                missing_episodes,
                desc="Caching state encodings",
                minibatch_max_size=self.cfg.minibatch_max_size,
            ):
                self.agent_forward(minibatch, need_full=True)

    def cached_forward(self, episode_data: list[EpisodeData]) -> AgentOutput:
        """Get output "as if" a forward pass was run, but avoid running the
        forward pass if possible.

        The `AgentOutput` returned by this method will not require gradients
        (they are "detached").
        """
        if not self.use_cache:
            return self.agent_forward(episode_data, need_full=True).detach()
        self.fill_caches(episode_data)
        agent_outputs = [
            self.agent_out_cache[ep_data.episode_id] for ep_data in episode_data
        ]
        for output in agent_outputs:
            assert (
                len(output.n_timesteps) == 1
            ), "More than one episode in cached AgentOutput"
        return AgentOutput.pack(agent_outputs)

    def agent_forward(
        self, episode_data: list[EpisodeData], need_full: bool
    ) -> AgentOutput:
        """Wrapper around the Agent forward() method.

        Handles delegating to ubrl_forward() if necessary, checking that
        need_full is respected, and caching.
        """
        agent_input = AgentInput(
            episodes=[ep_data.episode for ep_data in episode_data],
            need_full=need_full,
        )

        if hasattr(self.agent, "ubrl_forward"):
            agent_output = self.agent.ubrl_forward(agent_input)
        else:
            agent_output = self.agent(agent_input)
        if agent_output.full_valid():
            if self.use_cache:
                for i, output in enumerate(agent_output.unpack()):
                    episode_id = episode_data[i].episode_id
                    self.agent_out_cache[episode_id] = output.detach()
        elif need_full:
            total_valid = agent_output.valid_mask.sum().item()
            total_timesteps = sum(agent_output.n_timesteps)
            raise ValueError(
                f"Requested fully valid output but Agent only provided valid output for {total_valid}/{total_timesteps}"
            )
        return agent_output


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
    "advantage_normalizer",
    "awr_coef_normalizer",
]

_PARAM_FIELDS = [
    "kl_coef",
]

_IGNORED_FIELDS = ["_is_full_backward_hook", "cluster"]


class TorchTrainer:
    """An implementation of a bespoke reinforcement learning algorithm.

    The trainer trains an Agent to acquire higher rewards according to the data
    provided to it. To allow repeatedly adding new data to the `TorchTrainer`,
    it is a class instead of a function.

    The typical sequence of methods called on this class are as follows:

        1. Construction using a `TrainerConfig` (the default value is typically
           fine) and `Agent`. Optionally, call `TorchTrainer.attempt_resume()`
           to resume from a prior checkpoint.

        2. Repeated calls to `TorchTrainer.add_episode()` to add new data and
           `TorchTrainer.train_step()` to process the data. Typically multiple
           episodes (on the order of 10-100) should be added between each
           `TorchTrainer.train_step()` call.

        3. Periodic calls to `TorchTrainer.add_eval_stats()` and
           `TorchTrainer.maybe_checkpoint()`.
    """

    def __init__(self, cfg: "TrainerConfig", agent: "Agent"):
        """Constructs a `TorchTrainer`."""

        super().__init__()

        n_params = sum(p.numel() for p in agent.parameters())
        cfg = cfg.choose_device(n_params=n_params)

        self.cfg: "TrainerConfig" = cfg
        """The configuration used to contruct the `TorchTrainer`.

        Modifying this field after constructing the TorchTrainer *may* change the
        TorchTrainer's behavior, but is not guaranteed to and should be avoided
        (except via `load_state_dict()`).
        """
        self.cluster = ubrl.torch_cluster.DefaultCluster(self.cfg)

        # Will be passed through accelerator in _setup_optimizers
        self.agent: "Agent" = agent
        """The agent being optimized. Provides action (log-likelihoods) and
        state encodings."""

        self._state_encoding_size = self.agent.state_encoding_size

        # Will be passed through accelerator in _setup_optimizers
        self.vf: nn.Module = tu.make_mlp(
            input_size=self._state_encoding_size,
            hidden_sizes=self.cfg.vf_hidden_sizes,
            output_size=0,
            use_dropout=True,
        )
        """The value function. Feed-forward networks that predicts future
        rewards from `state_encodings`."""

        # Zero the initial VF output to stabilize training
        vf_output = self.vf.get_submodule("output_linear")
        vf_output.weight.data.copy_(0.01 * vf_output.weight.data)

        self.reward_normalizer: tu.ExponentialWeightedNormalizer = (
            self.cluster.prepare_module(
                tu.ExponentialWeightedNormalizer(
                    use_mean=False, use_var=cfg.normalize_rewards
                )
            )
        )
        """Normalizer used to make rewards have unit variance if
        `cfg.normalize_rewards` is True."""

        self.advantage_normalizer: tu.ExponentialWeightedNormalizer = (
            self.cluster.prepare_module(
                tu.ExponentialWeightedNormalizer(
                    use_mean=cfg.normalize_batch_advantages,
                    use_var=cfg.normalize_batch_advantages,
                )
            )
        )
        """Normalizer used to make advantages have zero mean and unit variance
        if `cfg.normalize_batch_advantages` is True."""

        self.awr_coef_normalizer: tu.ExponentialWeightedNormalizer = (
            self.cluster.prepare_module(
                tu.ExponentialWeightedNormalizer(
                    use_mean=True,
                    use_var=True,
                )
            )
        )
        """Normalizer used to AWR coefficients have zero mean and unit variance."""

        self.total_env_steps: int = 0
        """Total number of environment steps passed to `add_episode()`.
        May not count timesteps only used to compute statistics passed to
        `add_eval_stats()`."""

        self.train_steps_so_far: int = 0
        """Number of times `train_step()` has been called."""

        self.last_eval_stats: dict[str, float] = {}
        """Last stats passd to `add_eval_stats()`."""

        self.primary_performance: float = float("-inf")
        """Value of the `"primary_performance"` stat at the last
        `add_eval_stats()` call.
        """

        self.best_checkpoint_primary_performance: float = float("-inf")
        """Largest value of the `"primary_performance"` stat among all
        `add_eval_stats()` calls.
        """

        self.train_steps_so_far_at_last_checkpoint: int = 0
        """Number of (completed) `train_step()` calls at last periodic
        checkpoint.
        """

        # TODO(krzentner): Figure out if kl_coef should be on cpu or used through cluster
        # Maybe it would be simpler to just make kl_coef a nn.Linear(1, 1, bias=False)
        self.kl_coef: nn.Parameter = nn.Parameter(
            torch.tensor(float(self.cfg.kl_coef_init))
        )
        """Dynamically adjusted parameter used for KL regularization."""

        self.starting_entropy: Optional[float] = None
        """Initial mean entropy of action distributions, measured immediately
        at the start of cfg.entropy_schedule_start_train_step."""

        self.replay_buffer: list[EpisodeData] = []
        self.next_episode_id: EpisodeID = 0

        self.total_agent_grad_steps = 0
        self.agent_grad_steps_at_start_of_train_step = 0
        self.agent_grad_steps_last_train_step = 0

        # This method is also called after loading the state dict to re-attach
        # parameters
        self._setup_optimizers()

    def _setup_optimizers(self):
        """(Re)create all of the optimizers to use the current parameters.

        This method is called in `__init__` and also in `load_state_dict()`.
        """
        self.agent, self.agent_optimizer = self.cluster.prepare_module_opt(
            self.agent,
            torch.optim.AdamW(
                self.agent.parameters(),
                lr=self.cfg.agent_lr_start,
                weight_decay=self.cfg.agent_weight_decay,
            ),
        )
        self.agent_lr_scheduler = tu.make_scheduler(
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
        self.vf_lr_scheduler = tu.make_scheduler(
            self.vf_optimizer,
            self.cfg.vf_lr_schedule,
            self.cfg.vf_lr_start,
            self.cfg.vf_lr_end,
            self.cfg.expected_train_steps,
        )

        self.kl_coef_opt = torch.optim.AdamW([self.kl_coef], lr=self.cfg.kl_coef_lr)

    def combined_loss_function(
        self, agent_output: "AgentOutput", target: Targets
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reinforcement learning combined loss function.

        Combines together losses from many sources to train both the agent and
        value function.

        Returns a loss which is usually near unit scale and a dictionary of
        infos to log for this gradient step.

        Note that the "lagrange multiplier" loss for KL is optimized in kl_loss_of().
        """
        kl_loss, kl_infos = kl_loss_of(self.cfg, agent_output, target,
                                       self.kl_coef, self.kl_coef_opt)
        ppo_loss, ppo_infos = ppo_loss_of(self.cfg, agent_output, target)
        awr_loss, awr_infos = awr_loss_of(self.cfg, agent_output, target)
        vf_loss, vf_infos = vf_loss_of(self.cfg, agent_output, target, self.vf)

        approx_entropy, entropy = self.entropy_of(
            agent_output.action_lls,
            agent_output.action_dists,
        )
        entropy_loss, entropy_infos = entropy_loss_of(
            self.cfg, agent_output, self.entropy_target(), entropy)

        if agent_output.inherent_loss is not None:
            inherent_loss = (
                self.cfg.inherent_loss_coef * agent_output.inherent_loss.mean()
            )
        else:
            inherent_loss = 0.0

        loss = ppo_loss + awr_loss + vf_loss + kl_loss + entropy_loss + inherent_loss

        # *_infos will all get included in locals of this method
        tu.used_for_logging(kl_infos, ppo_infos, awr_infos, vf_infos,
                            entropy_infos, approx_entropy)
        return loss, locals()

    def new_cached_agent(self):
        return CachedAgent(self.agent, self.cfg, self.cluster)

    def start_train_step(self) -> float:
        """Prunes the replay buffer, computes the deadline for the current
        train_step(), and sets the networks to training mode."""
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

        self.agent_grad_steps_at_start_of_train_step = self.total_agent_grad_steps
        if len(self.replay_buffer) > self.cfg.replay_buffer_episodes:
            self.replay_buffer = list(
                self.replay_buffer[-self.cfg.replay_buffer_episodes :]
            )
            assert (
                len(self.replay_buffer) == self.cfg.replay_buffer_episodes
            ), "replay buffer length shorter than expected"


        self.agent.train(mode=True)
        self.vf.train(mode=True)
        return deadline

    def train_step(self):
        """Runs one "policy step" of training.

        This method should be called repeatedly until training is complete.
        Unless training is completely off-policy, new episodes should be added
        between each call to this method using `add_episode()`.

        All options for tuning this method are present in the TrainerConfig
        passed on TorchTrainer initialization.
        """

        deadline = self.start_train_step()
        c_agent = self.new_cached_agent()

        pre_train_epochs = self.cfg.vf_pre_training_epochs
        if self.train_steps_so_far == 0:
            pre_train_epochs = max(self.cfg.vf_warmup_training_epochs, pre_train_epochs)

        timed_out = self.train_vf(
            lambda: c_agent.cached_forward(self.replay_buffer),
            pre_train_epochs,
            desc="Pre Training VF",
            deadline=deadline,
        )

        if self.cfg.precompute_targets:
            cache_out = c_agent.cached_forward(self.replay_buffer)
            self._maybe_record_starting_entropy(lambda: cache_out)
            target = self.targets_of(cache_out)
            targets = {
                ep_data.episode_id: loss_in
                for (ep_data, loss_in) in tu.strict_zip(
                    self.replay_buffer, target.unpack()
                )
            }
            c_agent.clear_caches()
        else:
            targets = {}

        # Run primary training loop.
        timed_out |= self.train_agent(c_agent, targets, deadline)

        # Log dataset stats
        cache_out = c_agent.cached_forward(self.replay_buffer)
        self._maybe_record_starting_entropy(lambda: cache_out)
        target = self.targets_of(cache_out)
        self.log_dataset(cache_out, target)

        # Extra VF tuning after the primary training loop.
        # This achieves similar objectives to PPG by simply not doing updates
        # on the agent network in this phase.
        # Inputs are guaranteed to be cached, since we ran at least one full
        # epoch in the primary loop.
        timed_out |= self.train_vf(
            lambda: c_agent.cached_forward(self.replay_buffer),
            self.cfg.vf_post_training_epochs,
            desc="Post Training VF",
            deadline=deadline,
        )
        self.end_train_step(timed_out)

    def end_train_step(self, timed_out):
        """Update training schedules, set networks back to eval mode."""
        # Update schedules
        self.agent_lr_scheduler.step()
        self.vf_lr_scheduler.step()
        self.train_steps_so_far += 1
        self.agent_grad_steps_last_train_step = (
            self.total_agent_grad_steps - self.agent_grad_steps_at_start_of_train_step
        )
        self.agent_grad_steps_at_start_of_train_step = self.total_agent_grad_steps

        self.agent.train(mode=False)
        self.vf.train(mode=False)

        if timed_out:
            _LOGGER.error("train_step() timed out")

    def train_agent(
        self,
        cached_agent: CachedAgent,
        target_cache: dict[EpisodeID, Targets],
        deadline: float,
    ) -> bool:
        errors = 0
        for batch_i, minibatch in enumerate(
            minibatch_episodes(
                self.cluster,
                self.replay_buffer,
                desc="Training Agent",
                epochs=self.cfg.policy_epochs_per_train_step,
                shuffle=True,
                minibatch_target_size=self.cfg.minibatch_target_size,
                minibatch_max_size=self.cfg.minibatch_max_size,
            )
        ):
            if time.monotonic() > deadline:
                return True

            agent_output = cached_agent.agent_forward(minibatch, need_full=False)

            if self.cfg.precompute_targets:
                targets = Targets.pack(
                    [target_cache[ep_data.episode_id] for ep_data in minibatch]
                )
            else:
                if self.train_vf(
                    lambda: agent_output.detach(),
                    self.cfg.target_vf_mini_epochs,
                    desc=None,
                    deadline=deadline,
                ):
                    # Propagate timeout
                    return True
                targets = self.targets_of(agent_output.detach())
                for ep_data, target in tu.strict_zip(
                    minibatch, targets.unpack()
                ):
                    target_cache[ep_data.episode_id] = target

            loss, loss_infos = self.combined_loss_function(agent_output, targets)
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
        return False

    def train_vf(
        self,
        agent_output_gen: Callable[[], AgentOutput],
        training_epochs: int,
        desc: Optional[str],
        deadline: float,
    ) -> bool:
        """Train just the VF using cached agent outputs.

        This method does not tune the parameters of the agent.

        Because this training is only tuning the memoryless VF tail, it uses
        smaller minibatches of shuffled timesteps from across multiple
        episodes.

        Returns True iff the deadline is reached.
        """

        timed_out = time.monotonic() > deadline
        if training_epochs == 0 or timed_out:
            return timed_out

        agent_output = agent_output_gen()

        n_timesteps = agent_output.n_timesteps
        state_enc_packed = agent_output.state_encodings
        action_lls_now = tu.pad_packed(
            agent_output.action_lls, agent_output.n_timesteps
        )

        assert (
            not state_enc_packed.requires_grad
        ), "state_enc unexpectedly required grad"
        assert not action_lls_now.requires_grad, "action_lls unexpectedly required grad"

        padded_rewards = tu.pad_packed(agent_output.rewards, agent_output.n_timesteps)
        rewards_normed = self.reward_normalizer.normalize_batch(padded_rewards)

        original_action_lls = tu.pad_packed(
            agent_output.original_action_lls, agent_output.n_timesteps
        )

        discount = 1 - self.cfg.discount_inv
        gammas = discount * torch.ones_like(rewards_normed)

        with tqdm(
            desc=desc, total=training_epochs * sum(n_timesteps), disable=desc is None
        ) as pbar:
            for epoch in range(training_epochs):
                if (
                    epoch == 0
                    or epoch == training_epochs
                    or self.cfg.vf_recompute_targets
                ):
                    vf_x_packed = self.vf(state_enc_packed)
                    vf_x = tu.pad_packed(vf_x_packed, agent_output.n_observations)

                    # TODO: Refactor / eliminate this loop
                    for i, episode_length in enumerate(n_timesteps):
                        # zero vf_{t+1} in terminated episodes
                        if agent_output.terminated[i]:
                            vf_x[i, episode_length] = 0.0

                    with torch.no_grad():
                        _, vf_targets = scalar_v_trace_estimation(
                            lmbda=self.cfg.v_trace_lambda,
                            rho_max=self.cfg.v_trace_rho_max,
                            c_max=self.cfg.v_trace_c_max,
                            gammas=gammas,
                            vf_x=vf_x,
                            rewards=rewards_normed,
                            action_lls=action_lls_now,
                            original_action_lls=original_action_lls,
                            terminated=agent_output.terminated,
                            n_timesteps=torch.tensor(n_timesteps),
                        )

                    vf_targets_packed = tu.pack_padded(
                        vf_targets, agent_output.n_observations
                    )

                # pyright doesn't understand that epoch 0 is guaranteed to happen first
                dataset = tu.DictDataset(
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

    def log_dataset(self, agent_output: AgentOutput, target: Targets):
        full_episode_rewards = tu.pad_packed(
            agent_output.rewards, target.n_timesteps
        )
        discounted_returns = tu.pack_padded(
            tu.discount_cumsum(
                full_episode_rewards, discount=1 - self.cfg.discount_inv
            ),
            agent_output.n_timesteps,
        )
        vf_returns_no_final = tu.truncate_packed(
            target.vf_returns, new_lengths=target.n_timesteps, to_cut=1
        )
        dataset_stats = {
            "episode_total_rewards": full_episode_rewards.sum(dim=-1).mean(dim=0),
            "vf_explained_variance": tu.explained_variance(
                vf_returns_no_final,
                discounted_returns,
            ),
            "discounted_rewards": discounted_returns.mean(),
            "vf_mean": target.vf_returns.mean(),
            "vf_target_mean": target.vf_targets.mean(),
            "total_env_steps": self.total_env_steps,
        }
        for k in self.replay_buffer[0].infos.keys():
            dataset_stats[k] = tu.force_concat(
                [data.infos[k] for data in self.replay_buffer]
            )

        noko.log_row(
            "dataset_stats",
            dataset_stats,
            level=noko.INFO,
            step=self.total_env_steps,
        )

    def targets_of(
        self,
        agent_output: AgentOutput,
    ) -> Targets:
        """Compute Targets of AgentOutput using self.vf"""
        # Compute vf_returns
        self.agent.train(mode=False)
        self.vf.train(mode=False)
        n_timesteps = agent_output.n_timesteps
        n_observations = agent_output.n_observations

        state_encodings = agent_output.state_encodings.detach()
        action_lls_now = tu.pad_packed(agent_output.action_lls.detach(), n_timesteps)
        original_action_lls = tu.pad_packed(
            agent_output.original_action_lls, n_timesteps
        )

        with torch.no_grad():
            # TODO(krzentner): split up this call if necessary
            vf_returns_packed = self.vf(state_encodings)
        self.agent.train(mode=True)
        self.vf.train(mode=True)

        vf_returns = tu.pad_packed(vf_returns_packed, n_observations)
        vf_loss_masks = []

        # TODO(krzentner): Refactor / eliminate this loop
        for i, episode_length in enumerate(n_timesteps):
            ep_vf_loss_mask = self.cluster.prepare_tensor(torch.ones(episode_length + 1))
            if agent_output.terminated[i]:
                # zero vf_{t+1} in terminated episodes
                vf_returns[i, episode_length] = 0.0
            else:
                # Don't run VF loss on non-terminal final states.
                ep_vf_loss_mask[episode_length] = 0.0
            vf_loss_masks.append(ep_vf_loss_mask)

        padded_rewards = tu.pad_packed(agent_output.rewards, n_timesteps)
        rewards_normed = self.reward_normalizer(padded_rewards)

        discount = 1 - self.cfg.discount_inv
        gammas = discount * torch.ones_like(rewards_normed)

        padded_advantages, vf_targets = scalar_v_trace_estimation(
            lmbda=self.cfg.v_trace_lambda,
            rho_max=self.cfg.v_trace_rho_max,
            c_max=self.cfg.v_trace_c_max,
            gammas=gammas,
            vf_x=vf_returns,
            rewards=rewards_normed,
            action_lls=action_lls_now,
            original_action_lls=original_action_lls,
            terminated=agent_output.terminated,
            n_timesteps=self.cluster.prepare_tensor(torch.tensor(n_timesteps)),
        )

        adv_packed = tu.pack_padded(padded_advantages, n_timesteps)
        assert not adv_packed.requires_grad, "advantages unexpectedly require grad"

        adv_packed = self.advantage_normalizer(adv_packed)

        # This 1000 is documented in the TrainerConfig.awr_temperature
        # docstring.

        if self.cfg.awr_temperature >= 1000:
            heated_adv = adv_packed / self.cfg.awr_temperature
            max_exp_adv = torch.tensor(self.cfg.advantage_clip).exp()

            adv_exp = heated_adv.exp()
            clip_mask = ~torch.isfinite(adv_exp) | (adv_exp > max_exp_adv)
            clipped_adv_exp = adv_exp.clone()
            clipped_adv_exp[clip_mask] = max_exp_adv

            normed_exp_adv = self.awr_coef_normalizer(clipped_adv_exp)
            awr_clip_ratio = (normed_exp_adv == max_exp_adv).mean(dtype=torch.float32)
        else:
            awr_clip_ratio = 0.0
            normed_exp_adv = torch.ones_like(adv_packed)

        vf_loss_mask = tu.pack_tensors_check(vf_loss_masks, n_observations)
        target = Targets(
            advantages=adv_packed,
            vf_returns=tu.pack_padded(vf_returns, n_observations),
            vf_targets=tu.pack_padded(vf_targets, n_observations),
            vf_loss_mask=vf_loss_mask,
            exp_advantages=normed_exp_adv,
            n_timesteps=n_timesteps,
        )

        tu.used_for_logging(awr_clip_ratio)
        infos = locals()
        del infos["self"]
        del infos["n_timesteps"]
        del infos["n_observations"]
        noko.log_row(
            "preprocess",
            infos,
            level=noko.TRACE,
            step=self.total_env_steps,
        )
        return target

    def add_episode(
        self,
        episode: Any,
        *,
        n_timesteps: int,
        memory_size: Optional[float] = None,
        infos: Optional[dict[str, torch.Tensor]] = None,
    ):
        """Add a new episode to the replay buffer.

        Args:

            episode (Any): The episode. Can be any value that the agent accepts
                (in a list) to its forward method.
            n_timesteps (int): The number of full (observation, action, rewards)
                tuples in the episode.
            memory_size (float): The amount of memory required to include this
                episode into a forward pass. Can be an arbitrary unit, but
                must match `TrainerConfig.minibatch_target_size` (defaults
                to 1024) and `TrainerConfig.minibatch_max_size`.
                Defaults to n_timesteps if not provided.
            infos (Optional[dict[str,torch.Tensor]]): extra information about
                the episode. Will be summarized and logged every training step.

        """
        if infos is None:
            infos = {}
        if memory_size is None:
            memory_size = float(n_timesteps)

        self.replay_buffer.append(
            EpisodeData(
                episode,
                memory_size=memory_size,
                episode_id=self.next_episode_id,
                n_timesteps=n_timesteps,
                infos=infos,
            )
        )
        self.next_episode_id += 1

    def add_eval_stats(self, stats: dict[str, float], primary: str):
        """Add evaluation statistics for the current agent.

        Will be logged to {cfg.runs_dir}/{cfg.run_name}/eval_stats.csv.

        The primary stat should be present in stats and indicates how to choose
        the "best" agent for purposes of checkpointing and hyper-parameter
        tuning.
        """
        assert "primary" not in stats, "'primary' stat already present in stats"
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
        assert (
            self.kl_coef_opt.param_groups[0]["params"][0] is self.kl_coef
        ), "kl_coef_opt not optimizing kl_coef"
        assert self.vf_optimizer.param_groups[0]["params"][0] is next(
            self.vf.parameters()
        ), "vf_optimizer not optimizing vf"
        assert self.agent_optimizer.param_groups[0]["params"][0] is next(
            self.agent.parameters()
        ), "agent optimizer not optimizing ageint"

    def entropy_target(self) -> Optional[float]:
        if self.starting_entropy is None or self.cfg.entropy_schedule is None:
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
        assert step_fraction >= 0, "step_fraction < 0"
        assert step_fraction <= 1, "step_fraction > 1"
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

        assert mix >= 0, "entropy mix < 0"
        assert mix <= 1, "entropy mix > 1"
        target = (1 - mix) * self.starting_entropy + mix * final_entropy
        return target

    def entropy_of(
        self, action_lls: torch.Tensor, action_dists: Optional[tu.ActionDist]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Computes the approximate entropy (and exact entropy, if possible)."""
        approx_entropy = tu.approx_entropy_of(action_lls)
        entropy = None
        if not self.cfg.use_approx_entropy:
            try:
                # Mixed case is warned on add_episode()
                if action_dists is not None:
                    entropy = tu.entropy_of(action_dists)
            except NotImplementedError:
                pass
        if entropy is None:
            entropy = approx_entropy
        return approx_entropy, entropy

    def _maybe_record_starting_entropy(
        self,
        agent_output_gen: Callable[[], AgentOutput],
    ):
        """Record starting_etropy if train_steps_so_far >= cfg.entropy_schedule_start_train_step."""
        if (
            self.starting_entropy is None
            and self.train_steps_so_far >= self.cfg.entropy_schedule_start_train_step
        ):
            agent_output = agent_output_gen()
            
            approx_entropy, entropy = self.entropy_of(
                agent_output.action_lls, agent_output.action_dists
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


## Loss Functions

def ppo_loss_of(
    cfg: TrainerConfig, agent_output: "AgentOutput", target: Targets
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Proximal Policy Optimization clipping loss function."""
    ratio = torch.exp(agent_output.action_lls - agent_output.original_action_lls)
    ratio_clipped = torch.clamp(
        ratio, 1 / (1 + cfg.ppo_clip_epsilon), 1 + cfg.ppo_clip_epsilon
    )

    policy_gradient = ratio * target.advantages
    clip_policy_gradient = ratio_clipped * target.advantages

    ppo_loss = -torch.min(policy_gradient, clip_policy_gradient)
    norm_coef = cfg.ppo_loss_coef / cfg.minibatch_norm_div
    ppo_loss_scaled = norm_coef * ppo_loss.sum()
    infos = locals()
    infos["clip_portion"] = (ratio_clipped != ratio).mean(dtype=torch.float32)
    return (ppo_loss_scaled, infos)


def awr_loss_of(
    cfg: TrainerConfig, agent_output: "AgentOutput", target: Targets
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Advantage Weighted Regression loss function.

    Much of the work for setting up this loss function is in targets_of().
    """
    awr_loss = -agent_output.action_lls * target.exp_advantages

    log_probs_scale = agent_output.action_lls.detach().abs().mean()
    tu.used_for_logging(log_probs_scale)
    norm_coef = cfg.awr_loss_coef / (cfg.minibatch_norm_div)

    awr_loss_scaled = norm_coef * awr_loss.sum()
    infos = locals()
    return (awr_loss_scaled, infos)


def vf_loss_of(
    cfg: TrainerConfig, agent_output: "AgentOutput", target: Targets, vf: torch.nn.Module
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Value function loss function."""
    critic_out = vf(agent_output.state_encodings)
    vf_loss = F.mse_loss(critic_out, target.vf_targets, reduction="none")
    # Excludes non-terminal final states
    vf_loss_no_final = target.vf_loss_mask * vf_loss

    norm_coef = cfg.vf_loss_coef / cfg.minibatch_norm_div
    vf_loss_scaled = norm_coef * vf_loss_no_final.sum()
    infos = locals()
    del infos["vf"]
    return (vf_loss_scaled, infos)


def kl_loss_of(
    cfg: TrainerConfig, agent_output: "AgentOutput", target: Targets,
    kl_coef: nn.Parameter, kl_coef_opt: torch.optim.Optimizer
) -> tuple[torch.Tensor, dict[str, Any]]:
    """KL divergence loss.

    Updates the `kl_coef` to match the target value in the cfg and penalizes
    large KL values.
    """
    log_probs = agent_output.action_lls.clone()
    old_log_probs = agent_output.original_action_lls
    # Normalize to form a proper distribution across actions P(a)
    log_probs -= log_probs.exp().sum().log()
    old_log_probs -= old_log_probs.exp().sum().log()

    # Approximate per-timestep KL by multiplying back in the number of timesteps

    total_timesteps = sum(target.n_timesteps)
    approx_kl = total_timesteps * tu.approx_kl_div_of(old_log_probs, log_probs)
    total_approx_kl = approx_kl.sum()
    tu.used_for_logging(total_approx_kl)
    new_dists = agent_output.action_dists
    old_dists = agent_output.original_action_dists
    assert (new_dists is None) == (
        old_dists is None
    ), "dists do not consistently exist"

    # Compute KL Divergence
    if (
        cfg.use_approx_kl is False
        and new_dists is not None
        and old_dists is not None
    ):
        # TODO: Add options for other KL directions
        try:
            kl = tu.kl_div_of(old_dists, new_dists)
        except NotImplementedError:
            kl = approx_kl
    else:
        kl = approx_kl

    assert kl.shape == (total_timesteps,), "KL shape does not match total timesteps"
    assert approx_kl.shape == (
        total_timesteps,
    ), "KL approx shape does not match total timesteps"
    kl = kl[torch.isfinite(kl)]

    # Update KL loss coefficient
    if cfg.kl_target_stat == "max":
        kl_coef_loss = kl_coef * (cfg.kl_soft_target - kl.detach().max())
    elif cfg.kl_target_stat == "mean":
        kl_coef_loss = kl_coef * (cfg.kl_soft_target - kl.detach().mean())
    else:
        raise ValueError(f"Unknown kl_target_stat {cfg.kl_target_stat}")
    kl_coef_opt.zero_grad()
    kl_coef_loss.backward()
    try:
        clip_grad_norm_(
            [kl_coef], max_norm=cfg.grad_norm_max, error_if_nonfinite=True
        )
        kl_coef_opt.step()
    except RuntimeError:
        # Probably inf gradients, don't apply them
        pass

    if kl_coef < cfg.kl_coef_min:
        with torch.no_grad():
            kl_coef.copy_(cfg.kl_coef_min)
    # If the KL coef has become non-finite, it's probably because of
    # infinite KL, so set to maximum.
    if kl_coef > cfg.kl_coef_max or not torch.isfinite(kl_coef):
        with torch.no_grad():
            kl_coef.copy_(cfg.kl_coef_max)

    kl_mean = kl.mean()
    kl_max = kl.max()
    tu.used_for_logging(kl_mean, kl_max)

    norm_coef = kl_coef.detach() / cfg.minibatch_norm_div

    kl_loss_scaled = (norm_coef * kl).sum()
    infos = locals()
    return kl_loss_scaled, infos


def entropy_loss_of(
    cfg: TrainerConfig, agent_output: "AgentOutput",
    entropy_target: Optional[float], entropy: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Computes "squared mean error" of entropy relative to target.

    Note that this is intentionally not "mean squared error."
    Outliers in entropy (e.g. "chokepoint states") are typically beneficial, as
    long as the overall training process follows the target curve.
    """
    if entropy_target is not None:
        entropy_loss = cfg.entropy_loss_coef * (
            (entropy - entropy_target).sum() ** 2
        )
    else:
        # Set entropy_target to a float for logging purposes
        entropy_target = float('nan')
        entropy_loss = torch.tensor(0.0)
    infos = locals()
    return entropy_loss, infos


# From here to the end of the file there should be only helper / utlity
# functions.


@torch.jit.script
def scalar_v_trace_estimation(
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
    n_timesteps: torch.Tensor,
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
        n_timesteps (torch.Tensor) A 1D tensor indicating the episode length.
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
    gammas[ep_indices, n_timesteps - 1] *= (~terminated).to(dtype=gammas.dtype)
    assert bool(gammas[0, n_timesteps[0] - 1]) == (not terminated[0])

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


def group_episodes_to_minibatches(
    episodes: list["ubrl.EpisodeData"],
    *,
    minibatch_target_size: Optional[float] = None,
    minibatch_max_size: Optional[float] = None,
) -> list[list["ubrl.EpisodeData"]]:
    """Group a list of episodes into a list of list of episodes.
    Each minibatch (list of episodes) will have at least
    minibatch_target_size size unless adding the next episode would
    make it larger than minibatch_max_size, or it is the last minibatch.

    Raises ValueError if any epsidoe is longer than minibatch_max_size.
    """
    all_minibatches = []

    minibatch_now = []
    minibatch_now_size = 0
    for episode in episodes:
        # If we would go over the maximum
        if (
            minibatch_max_size is not None
            and minibatch_now_size + episode.n_timesteps > minibatch_max_size
        ) or (
            minibatch_target_size is not None
            and minibatch_now_size >= minibatch_target_size
        ):
            if len(minibatch_now) == 0:
                raise ValueError(
                    f"Episode length ({episode.n_timesteps}) exceeds max "
                    f"allowed timesteps in a minibatch ({minibatch_max_size})"
                )
            all_minibatches.append(minibatch_now)
            minibatch_now = []
            minibatch_now_size = 0
        minibatch_now.append(episode)
        minibatch_now_size += episode.n_timesteps
    if minibatch_now_size > 0:
        all_minibatches.append(minibatch_now)

    return all_minibatches


def minibatch_episodes(
    cluster: "ubrl.cluster.Cluster",
    episodes: list[EpisodeData],
    minibatch_target_size: Optional[int] = None,
    minibatch_max_size: Optional[int] = None,
    desc: Optional[str] = None,
    epochs: int = 1,
    shuffle: bool = False,
) -> Generator[list[EpisodeData], None, None]:
    """Top-level wrapper for iterating through minibatches from the replay buffer.

    Handles rendering a progress bar, shuffling episodes, and grouping into
    minibatches.
    Mostly this function is here to avoid adding two more indentation levels to
    the main loss loop.
    """
    with tqdm(
        total=epochs * sum([ep_data.n_timesteps for ep_data in episodes]),
        desc=desc,
        disable=(desc is None),
    ) as pbar:
        for _ in range(epochs):
            episodes = cluster.shard_episodes(episodes, shuffle=shuffle)

            minibatches = group_episodes_to_minibatches(
                episodes=episodes,
                minibatch_target_size=minibatch_target_size,
                minibatch_max_size=minibatch_max_size,
            )
            for minibatch in minibatches:
                yield minibatch
                pbar.update(sum(ep_data.n_timesteps for ep_data in minibatch))


__all__ = [
    "TorchTrainer",
    "Agent",
    "AgentInput",
    "AgentOutput",
    "Targets",
]
