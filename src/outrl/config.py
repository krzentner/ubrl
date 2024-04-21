"""This module handles
Declare configs using dataclass, and automatically handle logging and hparam tuning."""
from dataclasses import dataclass, field, fields
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

import stick
import optuna
import argparse
import simple_parsing
import yaml
from simple_parsing.helpers.serialization import save_yaml
from simple_parsing.helpers.serialization import load as load_yaml

T = TypeVar("T")

_LOGGER = logging.getLogger("outrl")


class CustomOptunaDistribution:
    def sample(self, name: str, trial: optuna.Trial) -> Any:
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
    distribution: Union[
        optuna.distributions.BaseDistribution, CustomOptunaDistribution
    ],
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
    metadata[_OPTUNA_DISTRIBUTION] = distribution
    if isinstance(default_val, list):
        return field(
            default_factory=lambda: copy.deepcopy(default_val),
            **kwargs,
            metadata=metadata,
        )
    else:
        return field(default=default_val, **kwargs, metadata=metadata)


def suggest_config(trial: optuna.Trial, config: Type, overrides: dict[str, Any]):
    """Samples a Config from an optuna Trial.

    config should be a dataclass, with tunable
    fields declared using outrl.config.tunable().

    overrides is a dictionary containing "raw" (in the simple_parsing sense)
    values that should not be tuned.
    """
    args = dict(overrides)
    missing_k = []
    for f in fields(config):
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
        trial.set_user_attr(k, getattr(cfg, k))
    return cfg


def default_run_name() -> str:
    """The main module name and current time in ISO 8601."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", "interactive")
    file_trail = os.path.splitext(os.path.basename(main_file))[0]
    now = datetime.datetime.now().isoformat()
    return f"{file_trail}_{now}"


def prepare_training_directory(cfg: "outrl.TrainerConfig"):
    """Creates a directory for logging and sets up logging."""
    os.makedirs(os.path.join(cfg.runs_dir, cfg.run_name), exist_ok=True)
    save_yaml(cfg, os.path.join(cfg.runs_dir, cfg.run_name, "config.yaml"))

    # stick will handle seeding for us
    stick.init_extra(
        log_dir=cfg.runs_dir,
        run_name=cfg.run_name,
        config=cfg.to_dict(),
        stderr_log_level=cfg.stderr_log_level,
        tb_log_level=cfg.tb_log_level,
    )
    if cfg.pprint_logging:
        from stick.pprint_output import PPrintOutputEngine

        stick.add_output(PPrintOutputEngine("stdout"))
    if cfg.parquet_logging:
        from stick.arrow_output import ArrowOutputEngine

        stick.add_output(ArrowOutputEngine(log_dir=cfg.runs_dir, run_name=cfg.run_name))


def _run_self(args):
    import __main__

    cmd = ["python3", __main__.__file__] + [str(a) for a in args]
    print(" ".join(cmd))
    subprocess.run(cmd, capture_output=False, check=False)


def storage_filename_to_storage(study_storage: str):
    if "://" not in study_storage and study_storage.endswith(".db"):
        study_storage = os.path.abspath(study_storage)
        study_storage = f"sqlite:///{study_storage}"
    return study_storage


def cmd_sample_config(
    override_config: Optional[str],
    study_storage: str,
    study_name: str,
    config_type: "type[outrl.TrainerConfig]",
    out_path: str,
):
    """Low-level command for manually distributing hyper-parameter optimization.

    Creates a trial config file at a given path using hparam optimization."""
    if override_config is not None:
        with open(override_config, "r") as f:
            # Load "raw" values. suggest_config will call .from_dict to
            # decode based on the type annotations.
            overrides = yaml.safe_load(f)
    else:
        overrides = {}

    study_storage = storage_filename_to_storage(study_storage)
    study = optuna.load_study(storage=study_storage, study_name=study_name)
    trial = study.ask()
    cfg = suggest_config(trial, config_type, overrides)
    cfg["optuna_trial_number"] = trial.number
    cfg["optuna_study_storage"] = study_storage
    cfg["optuna_study_name"] = study_name
    save_yaml(cfg, out_path)


def cmd_report_trial(config_file: str, run_dirs: list[str]):
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)
    trial_number = config_data["optuna_trial_number"]
    study = optuna.load_study(
        storage=config_data["study_storage"], study_name=config_data["study_name"]
    )
    seed_results = []
    for run_dir in run_dirs:
        try:
            eval_stats = stick.load_log_file(os.path.join(run_dir, "eval_stats.csv"))
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


class ExperimentInvocation:
    """Provides a standard command line interface to outrl launcher scripts.

    After construction, additional arguments can be added to the parser.

    Call .run() to actually run the command specified on command line.

    Commands:

        train: Parses the config file / command line arguments and runs the
            provided train_fn to produce an agent.
        tune: Runs hparam tuning using optuna using the provided train
            function to maximize the primary_performance stat passed to
            outrl.Trainer.add_eval_stats().
        create-study: Low-level command for manually distributing hyper
            parameter tuning. Creates a new optuna study.
        sample-config: Low-level command for manually distributing hyper
            parameter tuning. Creates a new config as part of an optuna study.
        report-trial: Low-level command for manually distributing hyper
            parameter tuning. Reports results of running a trial.
    """

    def __init__(
        self,
        train_fn: "Callable[[config_type], None]",
        config_type: "type[outrl.TrainerConfig]",
    ):
        self.parser = simple_parsing.ArgumentParser(
            nested_mode=simple_parsing.NestedMode.WITHOUT_ROOT,
        )

        self.parser.add_argument("--done-token", type=str, default=None)
        subp = self.parser.add_subparsers(title="command", dest="command")
        subp.required = True

        def _train():
            prepare_training_directory(self.args.cfg)
            train_fn(self.args.cfg)

        def _create_study():
            study_storage = storage_filename_to_storage(self.args.study_storage)
            optuna.create_study(storage=study_storage, study_name=self.args.study_name)

        def _sample_config():
            cmd_sample_config(
                override_config=self.args.override_config,
                study_storage=self.args.study_storage,
                study_name=self.args.study_name,
                config_type=config_type,
                out_path=self.args.config_path,
            )

        def _report_trial():
            cmd_report_trial(self.args.config_file, self.args.run_dirs)

        def _tune():
            runs_dir = self.args.runs_dir
            run_name = self.args.run_name
            run_dir = os.path.abspath(os.path.join(runs_dir, run_name))
            os.makedirs(run_dir, exist_ok=True)

            # Load override config
            if self.args.override_config is not None:
                with open(self.args.override_config, "r") as f:
                    # Load "raw" values. suggest_config will call .from_dict to
                    # decode based on the type annotations.
                    overrides = yaml.safe_load(f)
            else:
                overrides = {}

            save_yaml(overrides, os.path.join(run_dir, "overrides.yaml"))

            # Setup basic stick logging
            stick.init_extra(
                log_dir=runs_dir, run_name=run_name, stderr_log_level=stick.INFO
            )
            from stick.pprint_output import PPrintOutputEngine

            stick.add_output(PPrintOutputEngine("stdout"))

            if self.args.study_storage:
                storage_uri = storage_filename_to_storage(self.args.study_storage)
            else:
                storage_uri = f"sqlite:///{run_dir}/optuna.db"

            _LOGGER.info(f"Creating study {run_name!r} in storage {storage_uri!r}")

            study = optuna.create_study(
                storage=storage_uri,
                study_name=run_name,
                direction="maximize",
                load_if_exists=True,
            )

            for trial_index in range(self.args.n_trials):
                trial = study.ask()
                cfg = suggest_config(trial, config_type, overrides)
                config_path = os.path.join(run_dir, f"trial_{trial_index}.yaml")
                save_yaml(cfg, config_path)

                if self.args.fixed_seeds:
                    seeds = self.args.fixed_seeds
                else:
                    seeds = []

                # Choose args.n_seeds_per_trial unique seeds less than 10k
                max_seed = 10000
                while len(seeds) < self.args.n_seeds_per_trial:
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
                        eval_stats = stick.load_log_file(
                            os.path.join(runs_dir, sub_run_name, "eval_stats.csv")
                        )
                        max_primary_stat = max(eval_stats["primary"])
                        last_primary_stat = eval_stats["primary"][-1]
                        stick.log(
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
                    trial_result = 0.5 * min(seed_results) + 0.5 * sum(
                        seed_results
                    ) / len(seed_results)
                    study.tell(
                        trial, trial_result, state=optuna.trial.TrialState.COMPLETE
                    )
                else:
                    study.tell(trial, state=optuna.trial.TrialState.FAIL)

        train_parser = subp.add_parser(
            "train",
            add_help=False,
            help="Train the actor",
        )
        train_parser.set_defaults(func=_train)
        train_parser.add_argument("--config", default=None, type=str)

        # "High-level" hyper parameter tuning command
        tune_parser = subp.add_parser(
            "tune", help="Automatically tune hyper parameters"
        )
        tune_parser.set_defaults(func=_tune)
        tune_parser.add_argument("--runs_dir", type=str, default="runs")
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
        tune_parser.add_argument("--study-storage", type=str, default=None)

        # "Low level" optuna commands. Useful for distributed hparam tuning.
        create_parser = subp.add_parser("create-study", help="Create an optuna study")
        create_parser.set_defaults(func=_create_study)
        create_parser.add_argument("--study-storage", type=str)
        create_parser.add_argument("--study-name", type=str)

        sample_parser = subp.add_parser(
            "sample-config", help="Sample a new config using optuna"
        )
        sample_parser.set_defaults(func=_sample_config)
        sample_parser.add_argument("--study-storage", type=str)
        sample_parser.add_argument("--study-name", type=str)
        sample_parser.add_argument("--out-path", type=str)
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
        parser_copy = copy.deepcopy(self.parser)
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

        self.args, _ = self.parser.parse_known_args()

    def run(self):
        self.args = self.parser.parse_args()
        self.args.func()
        if self.args.done_token:
            with open(self.args.done_token, "w") as f:
                f.write("done\n")
