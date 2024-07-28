"""Utilities for interacting with config files, logging, and hyper parameter
optimization.

"""

from dataclasses import dataclass
import dataclasses
from typing import Any, Callable, List, Optional, Type, TypeVar, Union
import os
import random
import sys
import datetime
import copy
from textwrap import dedent
import subprocess
import logging
from pprint import pprint

import noko
import argparse
import simple_parsing
from torch import Value
import yaml
from simple_parsing.helpers.serialization import save_yaml
from simple_parsing.helpers.serialization import load as load_yaml

try:
    import optuna
except ImportError:
    optuna = None

import ubrl

T = TypeVar("T")

_LOGGER = logging.getLogger("ubrl")


class CustomOptunaDistribution:
    """A custom distribution for a tunable hyper paraemter."""

    def sample(self, name: str, trial: "optuna.Trial") -> Any:
        del name, trial
        raise NotImplementedError()


@dataclass
class IntListDistribution(CustomOptunaDistribution):
    """Uniform distribution over lists of integer values.

    Mostly used for parameterizing hidden sizes of MLP networks.
    Samples a length uniformly from between len(low) and len(high),
    then uniformly samples a size from the sizes in low and high.
    """

    low: List[int]
    high: List[int]

    def sample(self, name, trial) -> List[int]:
        list_len = trial.suggest_int(
            f"{name}_len", low=len(self.low), high=len(self.high)
        )
        values = []
        for i in range(list_len):
            low_i = min(i, len(self.low) - 1)
            values.append(
                trial.suggest_int(f"{name}_{i}", low=self.low[low_i], high=self.high[i])
            )
        return values


_OPTUNA_DISTRIBUTION = "OPTUNA_DISTRIBUTION"


def tunable(
    default_val: T,
    distribution: Optional[
        Union["optuna.distributions.BaseDistribution", CustomOptunaDistribution]
    ] = None,
    *,
    low: Optional[float | int] = None,
    high: Optional[float | int] = None,
    log: bool = False,
    choices: Optional[list[T]] = None,
    metadata=None,
    **kwargs,
) -> T:
    """Declares that a configuration field is tunable.

    Expects distribution to be an optuna Distribution, or a
    CustomOptunaDistribution.
    Automatically copies lists if provided as the default value.

    Not that despite the type annotation declaring a return value T, this
    method actually returns a dataclass field.
    """
    if metadata is None:
        metadata = {}
    if distribution is None and optuna is not None:
        if low is not None or high is not None:
            if choices is not None:
                raise ValueError("Must not provide both choices and high or low")
            if low is None:
                raise ValueError("Must provide low if high is provided")
            if high is None:
                raise ValueError("Must provide high if low is provided")
            if isinstance(low, int) and isinstance(high, int):
                assert isinstance(default_val, int)
                distribution = optuna.distributions.IntDistribution(
                    low=low, high=high, log=log
                )
            else:
                assert isinstance(default_val, float)
                distribution = optuna.distributions.FloatDistribution(
                    low=low, high=high, log=log
                )
        elif choices is not None:
            distribution = optuna.distributions.CategoricalDistribution(choices=choices)
        else:
            raise ValueError(
                "Must provide either distribution, choices, or (high and low) to tunable"
            )
    metadata[_OPTUNA_DISTRIBUTION] = distribution
    if isinstance(default_val, list):
        return dataclasses.field(
            default_factory=lambda: copy.deepcopy(default_val),
            **kwargs,
            metadata=metadata,
        )
    else:
        return dataclasses.field(default=default_val, **kwargs, metadata=metadata)


def suggest_config(trial: "optuna.Trial", config: Type, overrides: dict[str, Any]):
    """Samples a Config from an optuna Trial.

    config should be a dataclass, with tunable
    fields declared using ubrl.config.tunable().

    overrides is a dictionary containing "raw" (in the simple_parsing sense)
    values that should not be tuned.
    """
    args = dict(overrides)
    missing_k = []
    for f in dataclasses.fields(config):
        if f.name in overrides:
            trial.set_user_attr(f.name, overrides[f.name])
            continue
        if _OPTUNA_DISTRIBUTION in f.metadata:
            _LOGGER.debug(f"Sampling attirbute {f.name!r}")
            dist = f.metadata[_OPTUNA_DISTRIBUTION]
            if isinstance(dist, CustomOptunaDistribution):
                args[f.name] = dist.sample(f.name, trial)
            else:
                args[f.name] = trial._suggest(f.name, dist)
        else:
            missing_k.append(f.name)
    cfg = config.from_dict(args)
    for k in missing_k:
        try:
            trial.set_user_attr(k, getattr(cfg, k))
        except TypeError:
            trial.set_user_attr(k, repr(getattr(cfg, k)))
    return cfg


def default_run_name() -> str:
    """The main module name and current time in ISO 8601."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", "interactive")
    file_trail = os.path.splitext(os.path.basename(main_file))[0]
    now = datetime.datetime.now().isoformat()
    run_name = f"{file_trail}_{now}"
    # Replace colons on windows
    if os.name == "nt":
        run_name = run_name.replace(":", "_")
    return run_name


def prepare_training_directory(cfg: "ubrl.TrainerConfig", log_dir: Optional[str]):
    """Creates a directory for logging and sets up logging."""
    if log_dir is not None:
        while log_dir.endswith("/"):
            log_dir = log_dir[:-1]
        runs_dir, run_name = os.path.split(log_dir)
        assert run_name
        assert runs_dir
        cfg = dataclasses.replace(cfg, runs_dir=runs_dir, run_name=run_name)

    os.makedirs(os.path.join(cfg.runs_dir, cfg.run_name), exist_ok=True)
    config_path = os.path.join(cfg.runs_dir, cfg.run_name, "config.yaml")
    if not os.path.exists(config_path):
        save_yaml(cfg, config_path)
    else:
        _LOGGER.warn(f"Config file {config_path!r} already exists")

    # noko will handle seeding for us
    noko.init_extra(
        runs_dir=cfg.runs_dir,
        run_name=cfg.run_name,
        config=cfg.to_dict(),
        stderr_log_level=cfg.stderr_log_level,
        tb_log_level=cfg.tb_log_level,
    )
    if cfg.pprint_logging:
        from noko.pprint_output import PPrintOutputEngine

        noko.add_output(PPrintOutputEngine("stdout"))
    if cfg.parquet_logging:
        from noko.arrow_output import ArrowOutputEngine

        noko.add_output(ArrowOutputEngine(runs_dir=cfg.runs_dir, run_name=cfg.run_name))

    return cfg


def _run_self(args):
    import __main__

    cmd = ["python3", __main__.__file__] + [str(a) for a in args]
    print(" ".join(cmd))
    subprocess.run(cmd, capture_output=False, check=False)


def _storage_filename_to_storage(study_storage: str):
    if "://" not in study_storage and study_storage.endswith(".db"):
        study_storage = os.path.abspath(study_storage)
        study_storage = f"sqlite:///{study_storage}"
    return study_storage


def cmd_sample_config(
    override_config: Optional[str],
    study_storage: str,
    study_name: str,
    config_type: "type[ubrl.TrainerConfig]",
    out_path: str,
):
    """Low-level command for manually distributing hyper-parameter optimization.

    Creates a trial config file at a given path using hparam optimization."""
    import optuna

    if override_config is not None:
        with open(override_config, "r") as f:
            # Load "raw" values. suggest_config will call .from_dict to
            # decode based on the type annotations.
            overrides = yaml.safe_load(f)
    else:
        overrides = {}

    study_storage = _storage_filename_to_storage(study_storage)
    study = optuna.load_study(storage=study_storage, study_name=study_name)
    trial = study.ask()
    cfg = suggest_config(trial, config_type, overrides)
    cfg["optuna_trial_number"] = trial.number
    cfg["optuna_study_storage"] = study_storage
    cfg["optuna_study_name"] = study_name
    save_yaml(cfg, out_path)


def cmd_report_trial(config_file: str, run_dirs: list[str]):
    """Low-level command for manually distributing hyper-parameter optimization.

    Reports performance of a config file using multiple runs."""
    import optuna

    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    trial_number = config_data["optuna_trial_number"]
    study = optuna.load_study(
        storage=config_data["study_storage"], study_name=config_data["study_name"]
    )
    seed_results = []
    for run_dir in run_dirs:
        try:
            eval_stats = noko.load_log_file(os.path.join(run_dir, "eval_stats.csv"))
            max_primary_stat = max(eval_stats["primary"])
            last_primary_stat = eval_stats["primary"][-1]
            pprint(
                {
                    "trial": trial_number,
                    "run_dir": run_dir,
                    "max_primary_stat": max_primary_stat,
                    "last_primary_stat": last_primary_stat,
                }
            )
            seed_results.append(max_primary_stat)
        except (ValueError, FileNotFoundError):
            pass
    if len(seed_results) == len(run_dirs):
        # Bottom quartile
        trial_result = 0.5 * min(seed_results) + 0.5 * sum(seed_results) / len(
            seed_results
        )
        study.tell(trial_number, trial_result, state=optuna.trial.TrialState.COMPLETE)
    else:
        study.tell(trial_number, state=optuna.trial.TrialState.FAIL)


def cmd_tune(
    runs_dir: str,
    run_name: str,
    override_config: str,
    study_storage: str,
    n_trials: int,
    fixed_seeds: list[int],
    n_seeds_per_trial: int,
    config_type: type,
):
    """Runs hyper parameter tuning, assuming the current script uses the
    standard ExperimentInvocation()."""
    import optuna

    run_dir = os.path.abspath(os.path.join(runs_dir, run_name))
    os.makedirs(run_dir, exist_ok=True)

    # Load override config
    if override_config is not None:
        with open(override_config, "r") as f:
            # Load "raw" values. suggest_config will call .from_dict to
            # decode based on the type annotations.
            overrides = yaml.safe_load(f)
    else:
        overrides = {}

    save_yaml(overrides, os.path.join(run_dir, "overrides.yaml"))

    # Setup basic noko logging
    noko.init_extra(runs_dir=runs_dir, run_name=run_name, stderr_log_level=noko.INFO)
    from noko.pprint_output import PPrintOutputEngine

    noko.add_output(PPrintOutputEngine("stdout"))

    if study_storage:
        storage_uri = _storage_filename_to_storage(study_storage)
    else:
        storage_uri = f"sqlite:///{run_dir}/optuna.db"

    _LOGGER.info(f"Creating study {run_name!r} in storage {storage_uri!r}")

    study = optuna.create_study(
        storage=storage_uri,
        study_name=run_name,
        direction="maximize",
        load_if_exists=True,
    )

    for trial_index in range(n_trials):
        trial = study.ask()
        cfg = suggest_config(trial, config_type, overrides)
        config_path = os.path.join(run_dir, f"trial_{trial_index}.yaml")
        save_yaml(cfg, config_path)

        if fixed_seeds:
            seeds = fixed_seeds
        else:
            seeds = []

        # Choose args.n_seeds_per_trial unique seeds less than 10k
        max_seed = 10000
        while len(seeds) < n_seeds_per_trial:
            s = random.randrange(max_seed)
            while s in seeds:
                s = (s + 1) % max_seed
            seeds.append(s)

        seed_results = []
        # Run a training run for each seed
        for s in seeds:
            sub_run_name = f"{run_name}_trial={trial_index}_seed={s}"
            _run_self(
                [
                    "train",
                    "--config",
                    config_path,
                    "--seed",
                    s,
                    "--run_name",
                    sub_run_name,
                    "--runs_dir",
                    runs_dir,
                ]
            )
            try:
                eval_stats = noko.load_log_file(
                    os.path.join(runs_dir, sub_run_name, "eval_stats.csv")
                )
                max_primary_stat = max(eval_stats["primary"])
                last_primary_stat = eval_stats["primary"][-1]
                noko.log_row(
                    "seed_results",
                    {
                        "trial": trial_index,
                        "seed": s,
                        "max_primary_stat": max_primary_stat,
                        "last_primary_stat": last_primary_stat,
                    },
                )
                seed_results.append(max_primary_stat)
            except (ValueError, FileNotFoundError):
                pass
        if len(seed_results) == len(seeds):
            # Bottom quartile
            trial_result = 0.5 * min(seed_results) + 0.5 * sum(seed_results) / len(
                seed_results
            )
            study.tell(trial, trial_result, state=optuna.trial.TrialState.COMPLETE)
        else:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)


def run(
    *,
    train_fn: "Callable[[config_type], None]",
    config_type: "type[ubrl.TrainerConfig]",
    run_command: bool = True,
) -> Any | argparse.ArgumentParser:
    """Provides a standard command line interface to ubrl launcher scripts.

    Args:
        train_fn: A function to call to train the agent. The "main function" of
            most launcher programs. If run_command is True, return value will
            be returned from this function.
        config_type: The configuration type to use for command line arguments
            to "train" / to load from file if --config is passed.
        run_command: Whether to run the command. If False, the ArgumentParser
            will be returned instead of calling parser.parse().func().

    Commands:

        train: Parses the config file / command line arguments and runs the
            provided train_fn to produce an agent.
            --config path to load a config file
            --log-dir the directory to log to (will be split into cfg.runs_dir
                and cfg.run_name)
            Config file values can also be overridden by passing them here.

        tune: Runs hparam tuning using optuna using the provided train
            function to maximize the primary_performance stat passed to
            ubrl.Trainer.add_eval_stats().
            --runs_dir Directory to keep runs in. (default: runs)
            --run_name
            --n_trials
            --override-config Path to partial config file with override
                values. Used to restrict the search space of the tuning.
            --n-seeds-per-trial Number of seeds to run for each trial / hyper
                pararmeter configuration. The minimum performance across these
                seeds will be used as the overall trial performance. This
                avoids finding hyper parameter configurations that only work
                for one seed. (default: 2)
            --fixed-seeds A fixed set of seeds to use for each trial. Overrides
                --n-seeds-per-trial.
            --study-storage (default: sqlite:///runs/optuna.db)

        create-study: Low-level command for manually distributing hyper
            parameter tuning. Creates a new optuna study.
            --study-storage (default: sqlite:///runs/optuna.db)
            --study-name

        sample-config: Low-level command for manually distributing hyper
            parameter tuning. Creates a new config as part of an optuna study.
            --study-storage (default: sqlite:///runs/optuna.db)
            --study-name
            --override-config
            --out-path Path to write sampled config file to.

        report-trial: Low-level command for manually distributing hyper
            parameter tuning. Reports results of running a trial.
            --config Config generated by sample-config.
            --run-dirs List of directories where experiments were run.
    """
    parser = simple_parsing.ArgumentParser(
        nested_mode=simple_parsing.NestedMode.WITHOUT_ROOT,
    )

    parser.add_argument("--done-token", type=str, default=None)
    subp = parser.add_subparsers(title="command", dest="command")
    subp.required = True

    def _after_func():
        if args.done_token:
            with open(args.done_token, "w") as f:
                f.write("done\n")

    def _train():
        cfg = prepare_training_directory(args.cfg, args.log_dir)
        result = train_fn(cfg)
        _after_func()
        return result

    def _create_study():
        import optuna

        study_storage = _storage_filename_to_storage(args.study_storage)
        optuna.create_study(storage=study_storage, study_name=args.study_name)
        _after_func()

    def _sample_config():
        cmd_sample_config(
            override_config=args.override_config,
            study_storage=args.study_storage,
            study_name=args.study_name,
            config_type=config_type,
            out_path=args.config_path,
        )
        _after_func()

    def _report_trial():
        cmd_report_trial(args.config_file, args.run_dirs)
        _after_func()

    def _tune():
        cmd_tune(
            runs_dir=args.runs_dir,
            run_name=args.run_name,
            override_config=self.args.override_config,
            study_storage=args.study_storage,
            n_trials=args.n_trials,
            fixed_seeds=args.fixed_seeds,
            n_seeds_per_trial=args.n_seeds_per_trial,
            config_type=config_type,
        )
        _after_func()

    train_parser = subp.add_parser(
        "train",
        add_help=False,
        help="Train the actor",
    )
    train_parser.set_defaults(func=_train)
    train_parser.add_argument("--config", default=None, type=str)
    train_parser.add_argument("--log-dir", default=None, type=str)

    # "High-level" hyper parameter tuning command
    tune_parser = subp.add_parser("tune", help="Automatically tune hyper parameters")
    tune_parser.set_defaults(func=_tune)
    tune_parser.add_argument(
        "--runs_dir", type=str, default="runs", help="Directory to keep runs in."
    )
    tune_parser.add_argument(
        "--run_name", type=str, default="tune_" + default_run_name()
    )
    tune_parser.add_argument("--n_trials", type=int, default=1000)
    tune_parser.add_argument(
        "--override-config",
        type=str,
        default=None,
        help=dedent(
            """\
            Path to partial config file with override values.
            Used to restrict the search space of the tuning.
            """
        ),
    )
    tune_parser.add_argument(
        "--n-seeds-per-trial",
        type=int,
        default=2,
        help=dedent(
            """\
            Number of seeds to run for each trial / hyper pararmeter configuration.
            The minimum performance across these seeds will be used as the
            overall trial performance. This avoids finding hyper parameter
            configurations that only work for one seed.
            """
        ),
    )
    tune_parser.add_argument(
        "--fixed-seeds",
        type=int,
        nargs="*",
        help=dedent(
            """\
            A fixed set of seeds to use for each trial.
            Overrides --n-seeds-per-trial.
            """
        ),
    )
    tune_parser.add_argument(
        "--study-storage", type=str, default="sqlite:///runs/optuna.db"
    )

    # "Low level" optuna commands. Useful for distributed hparam tuning.
    create_parser = subp.add_parser("create-study", help="Create an optuna study")
    create_parser.set_defaults(func=_create_study)
    create_parser.add_argument(
        "--study-storage", type=str, default="sqlite:///runs/optuna.db"
    )
    create_parser.add_argument("--study-name", type=str)

    sample_parser = subp.add_parser(
        "sample-config", help="Sample a new config using optuna"
    )
    sample_parser.set_defaults(func=_sample_config)
    sample_parser.add_argument(
        "--study-storage", type=str, default="sqlite:///runs/optuna.db"
    )
    sample_parser.add_argument("--study-name", type=str)
    sample_parser.add_argument(
        "--out-path", type=str, help="Path to write sampled config file to."
    )
    sample_parser.add_argument("--override-config", type=str, default=None)

    report_trial = subp.add_parser(
        "report-trial", help="Report results of a trial to optuna"
    )
    report_trial.set_defaults(func=_report_trial)
    report_trial.add_argument("--config", type=str)
    report_trial.add_argument(
        "--run-dirs",
        type=str,
        nargs="*",
        help=dedent(
            """\
            List of directories where experiments were run.
            """
        ),
    )

    # First parse the known arguments to get config path.
    # For unclear reasons, modifying the parser after using it is not
    # possible, so copy it first.
    parser_copy = copy.deepcopy(parser)
    args, _ = parser_copy.parse_known_args()

    if getattr(args, "config", None):
        # Command line arguments should override config file entries
        loaded_config = load_yaml(config_type, args.config)
        train_parser.add_arguments(config_type, dest="cfg", default=loaded_config)
    else:
        train_parser.add_arguments(config_type, dest="cfg")

    # Re-add the help command manually, now that we've added all the config arguments
    train_parser.add_argument(
        "-h",
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )

    if run_command:
        args = parser.parse_args()
        return args.func()
    else:
        return parser
