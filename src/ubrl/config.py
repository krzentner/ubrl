import dataclasses
import random
from typing import Optional, Literal
import os

import simple_parsing
import noko

from ubrl.cli import tunable, IntListDistribution, default_run_name


_GENERATED_FROM_TIME = "GENERATED_FROM_TIME"
_CUDA_ON_OVER_ONE_MILLION_PARAMS = "CUDA_ON_OVER_ONE_MILLION_PARAMS"


@dataclasses.dataclass
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

    minibatch_max_size: Optional[int] = None
    """Maximum size to include in a forward pass to the agent.
    Used to avoid out-of-memory errors.

    Defaults to no limit. Automatically decreases on (most) memory errors.
    """

    minibatch_target_size: int = tunable(1024, low=1, high=50000, log=True)
    """Attempt to use minibatches near this total size.
    In practice, this acts as a minimum size of most minibatches.

    Will still run whole episodes if they exceed this cap.
    """

    policy_epochs_per_train_step: int = tunable(15, low=1, high=100, log=True)
    """Number of times to iterate over all data in replay buffer each time
    train_step() is called."""

    normalize_rewards: bool = tunable(True, choices=[True, False])
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

    first_train_step_timeout_coef: float = 2.0
    """Multiple of the normal train_step() timeout to use for the
    first train_step. Useful for accomodating additional time
    needed for JIT compilation or VF warmup (see
    `vf_warmup_training_epochs`).

    Has no effect if train_step_timeout_seconds is None.
    """

    minibatch_norm_div: float = tunable(1024.0, low=1.0, high=4096.0, log=True)
    """Divisor applied to all loss coefficients in place of taking .mean() over
    the minibatch.
    """

    ppo_loss_coef: float = tunable(0.0, low=0.0, high=1000.0)
    """Loss coefficient for the PPO loss. Usually unused, since the AWR loss is
    more flexible.
    """

    awr_loss_coef: float = tunable(1.0, low=0.0, high=1000.0)
    """Loss coefficient for the main RL loss. Usually does not need to be
    tuned."""

    inherent_loss_coef: float = tunable(1.0, low=0.0, high=1000.0)
    """Loss coefficient for the loss computed by the agent itself.

    Only used if the agent returns a AgentOutput with the `inherent_loss` field
    set to a torch.Tensor.
    """

    agent_lr_schedule: Literal[None, "linear", "cosine"] = tunable(
        "cosine", choices=[None, "linear", "cosine"]
    )
    """Learning rate schedule for the agent. Typically used to decrease the
    learning rate to near-zero near the end of training."""

    agent_lr_start: float = tunable(2e-4, low=1e-5, high=5e-2, log=True)
    """Initial learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will be used throughout training. """

    agent_lr_end: float = tunable(1e-5, low=1e-8, high=1e-3, log=True)
    """Final learning rate for the agent. If the agent_lr_schedule is None,
    this learning rate will not be used."""

    agent_weight_decay: float = tunable(1e-8, low=1e-10, high=1e-2, log=True)
    """Weight decay for the agent using AdamW."""

    ppo_clip_epsilon: float = tunable(0.2, low=0.05, high=2.0, log=True)
    """PPO loss will be clipped to only apply the loss when the log-likelihood
    is between 1 / (1 + ppo_clip_epsilon) and 1 + ppo_clip_epsilon.

    Because the ppo_loss is disabled by default, this field also has no effect by default.
    Because ubrl uses regularized VF training, VF clipping is not used.
    """

    vf_lr_schedule: Literal[None, "linear", "cosine"] = tunable(
        "cosine", choices=[None, "linear", "cosine"]
    )
    """Learning rate schedule for the value function parameters. Typically used
    to decrease the learning rate to near-zero near the end of training."""

    vf_lr_start: float = tunable(2e-3, low=1e-4, high=0.1, log=True)
    """Initial learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will be used throughout
    training. """

    vf_lr_end: float = tunable(1e-5, low=1e-8, high=1e-4, log=True)
    """Final learning rate for the value function parameters. If the
    vf_lr_schedule is None, this learning rate will not be used. """

    vf_weight_decay: float = tunable(1e-5, low=1e-8, high=1e-2, log=True)
    """Weight decay for the value function parameters using AdamW."""

    vf_minibatch_size: int = tunable(64, low=1, high=2**32, log=True)
    """Number of timesteps used in minibatches when pre-training and
    post-training the value function from frozen state encodings.

    Because the value function uses a memoryless architecture (the VF relies on
    the agent to encode memory, if necessary), this minibatch size is typically
    met precisely (except for one trailing odd-sized minibatch).
    """

    vf_warmup_training_epochs: int = tunable(30, low=0, high=1000)
    high = """Number of epochs of value function training to run fro
    frozen state encodings on the very first train_step() before
    training the agent.

    Because ubrl uses an AWR-style loss, training the VF before
    the policy is expected.
    """

    vf_pre_training_epochs: int = tunable(3, low=0, high=20)
    """Number of epochs of value function training to run from frozen state
    encodings each train_step() before training the agent.

    Because ubrl uses an AWR-style loss, training the VF before the policy is
    expected.
    """

    vf_post_training_epochs: int = tunable(3, low=0, high=20)
    """Number of epochs of value function training to run from frozen state
    encodings each train_step() after training the agent.

    Because the value function is also tuned with the agent, this pass mostly
    serves to allow the value function to "catch up" to changing state
    encodings before the next train_step begins.
    """

    vf_recompute_targets: bool = tunable(True, choices=[True, False])
    """If true, value function targets will be recomputed every epoch of value
    function optimization.

    This allows the value function to make predictions based on "mixing"
    advantages from multiple episodes, as is typical in Q-learning based
    algorithms.

    If your environment is partially observable, disabling this option may
    improve training reliability.
    """

    vf_loss_coef: float = tunable(0.1, low=1e-6, high=1.0, log=True)
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

    discount_inv: float = tunable(0.01, low=1e-5, high=0.1, log=True)
    """Discount, expresseed such that gamma = 1 - discount_inv to allow
    hyper-parameter tuning in log space.

    This hyper parameter can have a very significant effect.
    """

    v_trace_lambda: float = tunable(0.33, low=0.0, high=1.0)
    """Lambda parameter to v-trace advantage estimation.

    Controls the bias-variance tradeoff between pure belmann bootstraps and
    monte-carlo estimates. A value of 1 corresponds to minimal bias, and
    matches v-trace as originally proposed.
    """

    v_trace_rho_max: float = tunable(3.0, low=1.0, high=1e3, log=True)
    """The "value function truncation" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which rewards from increased likelihood
    off-policy actions can contribute to estimated advantages.
    """

    v_trace_c_max: float = tunable(3.0, low=1.0, high=1e3, log=True)
    """The "trace-cutting" importance weight maximum in v-trace.

    Setting this value to very large values disables it, performing maximally
    off-policy advantage estimation.

    Smaller values limit the degree to which future value function estimates
    from increased likelihood off-policy states can contribute to estimated
    advantages.
    """

    kl_coef_init: float = tunable(0.1, low=0.0, high=100.0)
    """Initial loss coefficient for KL penalty / regularization of the agent.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).

    This penalty is applied exactly if action_dists is provided by the agent,
    or applied approximately using action_lls otherwise.
    """

    kl_coef_lr: float = tunable(0.01, low=1e-6, high=0.1, log=True)
    """How quickly to adapt the loss coefficient for KL penalty.

    The KL coefficient will be tuned using a lagrangian style loss to keep the
    size of policy updates to within a maximal value (kl_soft_target).
    """

    kl_coef_min: float = tunable(0.01, low=1e-3, high=1.0, log=True)
    """Minimum value of the KL coefficient. Setting this to a non-zero value
    can help stabilize training very low-entropy continuous action space
    policies using the PPO loss, but is typically unnecessary.
    """

    kl_coef_max: float = tunable(100.0, low=1e-2, high=1e6, log=True)
    """Maximum value of the KL coefficient. Necessary to ensure eventual
    convergence of the KL penalty.

    If you are experiencing crashes due to NaNs when the kl_coef is high,
    decrease this value.
    """

    kl_target_stat: Literal["mean", "max"] = tunable("mean", choices=["mean", "max"])
    """What statistic of the KL divergence to constrain.

    Constraining the mean KL divergence is typical, but constraining the max KL
    can improve stability during long runs with little disadvantage.
    """

    kl_soft_target: float = tunable(0.1, low=1e-3, high=10.0, log=True)
    """Target per-timestep KL divergence per train-step.

    If this value is exceeded, the kl_coef will become non-zero to limit the
    training step size.

    Because this value is enforced using a lagrangian, KL steps are often 2-3x
    this target.
    """

    kl_fixup_coef: float = tunable(3.0, low=1.1, high=20.0, log=True)
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
        "cosine", choices=[None, "linear", "cosine"]
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

    entropy_schedule_end_fraction: float = tunable(0.01, low=1e-6, high=1.0, log=True)
    """Portion of "starting entropy" to attempt to maintain at end of
    training.

    If starting entropy is negative (due to using continuous
    distributions with a density > 1), then instead the schedule
    will attempt to increase the density by a factor of
    -ln(entropy_schedule_end_fraction).

    Only used if entropy_schedule_end_target is None."""

    entropy_schedule_start_train_step: int = 1
    """Train step at which to measure the "starting entropy".

    This indicates at the end of which train step entropy should be measured.
    The default value measures the entropy after one train step.
    """

    entropy_loss_coef: float = tunable(1e-4, low=0.0, high=1.0)
    """Entropy coefficient.

    Coefficient to apply to entropy loss.
    By default the entropy loss is a mean squared error relative to
    the entropy schedule.
    """

    awr_temperature: float = tunable(0.01, low=1e-2, high=1e3, log=True)
    """AWR temperature.

    Very low values result in a sparse loss that only attempts to repeat
    very high-advantage actions.

    High values cause the AWR loss to ignore advantages and just perform
    behavioral cloning.

    If set to >=1000, will literally just perform behavioral cloning.
    """

    normalize_batch_advantages: bool = tunable(True, choices=[False, True])
    """Whether to normalize the advantages across the batch."""

    advantage_clip: float = tunable(8.0, low=0.1, high=12.0)
    """Max exponent value for advantages in AWR loss.

    Large values can lead to NaN errors.

    Can have a significant effect, since many timesteps will have clipped
    coefficients at low temperatures.
    """

    grad_norm_max: float = tunable(5.0, low=1.0, high=1e2, log=True)
    """Grad norm to clip the actor and value function parameters to in the main
    training loop.

    Small values will consistently lower the loss.
    """

    precompute_loss_inputs: bool = tunable(False, choices=[False, True])
    """Compute advantages and VF targets for all minibatches before running
    loss on any minibatch.

    If True, and `vf_pre_training_epochs == 0`, an additional forward pass may
    be required for each timestep.
    If False, minibatches will become off-policy.
    """

    loss_input_vf_mini_epochs: int = tunable(0, low=0, high=1)
    """Number of epochs of vf training to run on each minibatch when
    `precompute_loss_inputs` is False.
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

    device: str = _CUDA_ON_OVER_ONE_MILLION_PARAMS
    """PyTorch device to use for optimization. Defaults to using cpu if the
    number of params in the agent is less than one million, otherwise defaults
    to cuda."""

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

    def choose_device(self, n_params: int) -> "TrainerConfig":
        if self.device == _CUDA_ON_OVER_ONE_MILLION_PARAMS:
            if n_params >= 1e6:
                return dataclasses.replace(self, device="cuda")
            else:
                return dataclasses.replace(self, device="cpu")
        else:
            return self
