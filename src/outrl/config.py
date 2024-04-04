"""This module handles
Declare configs using dataclass, and automatically handle logging and hparam tuning."""
from dataclasses import dataclass, field, fields
from typing import Any, Callable, List, Type, TypeVar, Union
import os
import random
import sys
import datetime
import copy
from textwrap import dedent
import subprocess
import logging

import stick
import optuna
import argparse
import simple_parsing
import yaml
from simple_parsing.helpers.serialization import save_yaml
from simple_parsing.helpers.serialization import load as load_yaml

T = TypeVar("T")

LOGGER = logging.getLogger("outrl")


class Config(simple_parsing.Serializable):
    pass


class CustomDistribution:
    def sample(self, name: str, trial: optuna.Trial) -> Any:
        del name, trial
        raise NotImplementedError()


@dataclass
class IntListDistribution(CustomDistribution):
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


OPTUNA_DISTRIBUTION = "OPTUNA_DISTRIBUTION"


def tunable(
    default_val: T,
    distribution: Union[optuna.distributions.BaseDistribution, CustomDistribution],
    metadata=None,
    **kwargs,
) -> T:
    """Declares that a configuration field is tunable.

    Expects distribution to be an optuna Distribution, or a CustomDistribution.
    Automatically copies lists if provided as the default value.

    Not that despite the type annotation declaring a return value T, this
    method actually returns a dataclass field.
    """
    if metadata is None:
        metadata = {}
    metadata["OPTUNA_DISTRIBUTION"] = distribution
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

    config should be a dataclass subclass of outrl.config.Config, with tunable
    fields declared using outrl.config.tunable().

    overrides is a dictionary containing "raw" (in the simple_parsing sense)
    values that should not be tuned.
    """
    args = dict(overrides)
    for f in fields(config):
        if f.name in overrides:
            trial.set_user_attr(f.name, overrides[f.name])
            continue
        if OPTUNA_DISTRIBUTION in f.metadata:
            LOGGER.debug(f"Sampling attirbute {f.name!r}")
            dist = f.metadata[OPTUNA_DISTRIBUTION]
            if isinstance(dist, CustomDistribution):
                args[f.name] = dist.sample(f.name, trial)
            else:
                args[f.name] = trial._suggest(f.name, dist)
    return config.from_dict(args)


def default_run_name() -> str:
    """The main module name and current time in ISO 8601."""
    main_file = getattr(sys.modules.get("__main__"), "__file__", "interactive")
    file_trail = os.path.splitext(os.path.basename(main_file))[0]
    now = datetime.datetime.now().isoformat()
    return f"{file_trail}_{now}"


def prepare_training_directory(cfg: "outrl.rl.TrainerConfig"):
    """Creates a directory for logging and sets up logging."""
    os.makedirs(os.path.join(cfg.log_dir, cfg.run_name), exist_ok=True)
    save_yaml(cfg, os.path.join(cfg.log_dir, cfg.run_name, "config.yaml"))

    # Set the default log level before creating the logger
    # TODO: Figure out a better API for this (that doesn't involve cluttering
    # stick.init() with a bunch of options).
    import stick.tb_output

    stick.tb_output.DEFAULT_LOG_LEVEL = stick.LOG_LEVELS[cfg.tb_log_level]

    # stick will handle seeding for us
    stick.init_extra(
        log_dir=cfg.log_dir,
        run_name=cfg.run_name,
        config=cfg.to_dict(),
        stderr_log_level=cfg.stderr_log_level,
    )
    if cfg.pprint_logging:
        from stick.pprint_output import PPrintOutputEngine

        stick.add_output(PPrintOutputEngine("stdout"))
    if cfg.parquet_logging:
        from stick.arrow_output import ArrowOutputEngine

        stick.add_output(ArrowOutputEngine(log_dir=cfg.log_dir, run_name=cfg.run_name))


def _run_self(args):
    import __main__

    cmd = ["python3", __main__.__file__] + [str(a) for a in args]
    print(" ".join(cmd))
    subprocess.run(cmd, capture_output=False, check=False)


class ExperimentInvocation:
    def __init__(
        self,
        train_fn: "Callable[[config_type], None]",
        config_type: "type[outrl.config.TrainerConfig]",
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

        def _sample_config():
            if self.args.override_config is not None:
                with open(self.args.override_config, "r") as f:
                    # Load "raw" values. suggest_config will call .from_dict to
                    # decode based on the type annotations.
                    overrides = yaml.safe_load(f)
            else:
                overrides = {}

            study = optuna.load_study(
                storage=self.args.study_storage, study_name=self.args.study_name
            )
            trial = study.ask()
            cfg = suggest_config(trial, config_type, overrides)
            save_yaml(cfg, self.args.out_path)
            base_path = os.path.splitext(self.args.out_path)[0]
            save_yaml(
                {
                    "trial_number": trial.number,
                    "study_storage": self.args.study_storage,
                    "study_name": self.args.study_name,
                    "config": cfg,
                },
                f"{base_path}-optuna.yaml",
            )

        def _create_study():
            optuna.create_study(
                storage=self.args.study_storage, study_name=self.args.study_name
            )

        def _report_trial():
            with open(self.args.trial_file, "r") as f:
                trial_data = yaml.safe_load(f)
            study = optuna.load_study(
                storage=trial_data["study_storage"], study_name=trial_data["study_name"]
            )
            result_key = self.args.config.minimization_objective
            results = stick.load_log_file(self.args.log_file, keys=[result_key])
            study.tell(trial_data["trial_number"], min(results[result_key]))

        def _tune():
            log_dir = self.args.log_dir
            run_name = self.args.run_name
            run_dir = os.path.abspath(os.path.join(log_dir, run_name))
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
                log_dir=log_dir, run_name=run_name, stderr_log_level=stick.INFO
            )
            from stick.pprint_output import PPrintOutputEngine

            stick.add_output(PPrintOutputEngine("stdout"))

            if self.args.study_storage:
                storage_url = self.args.study_storage
            else:
                storage_url = f"sqlite:///{run_dir}/optuna.db"

            LOGGER.info(f"Creating study {run_name!r} in storage {storage_url!r}")

            study = optuna.create_study(
                storage=storage_url,
                study_name=run_name,
                direction="maximize",
                load_if_exists=True,
            )

            for trial_index in range(self.args.n_trials):
                trial = study.ask()
                cfg = suggest_config(trial, config_type, overrides)
                config_path = os.path.join(run_dir, f"trial_{trial_index}.yaml")
                save_yaml(cfg, config_path)

                # Choose args.n_seeds_per_trial unique seeds less than 10k
                max_seed = 10000
                seeds = []
                for _ in range(self.args.n_seeds_per_trial):
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
                            "--log_dir",
                            log_dir,
                        ]
                    )
                    try:
                        eval_stats = stick.load_log_file(
                            os.path.join(log_dir, sub_run_name, "eval_stats.csv")
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
        tune_parser.add_argument("--log_dir", type=str, default="runs")
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
        tune_parser.add_argument("--study-storage", type=str, default=None)

        # "Low level" optuna commands. Useful for distributed hparam tuning.
        create_parser = subp.add_parser("create-study", help="Create an optuna study")
        create_parser.set_defaults(func=_create_study)
        create_parser.add_argument("--study-storage", type=str)
        create_parser.add_argument("--study-name", type=str)
        report_trial = subp.add_parser(
            "report-trial", help="Report results of a run to optuna"
        )
        report_trial.set_defaults(func=_report_trial)
        report_trial.add_argument("--trial-file", type=str)
        report_trial.add_argument("--log-file", type=str)
        sample_parser = subp.add_parser(
            "sample-config", help="Sample a new config using optuna"
        )
        sample_parser.set_defaults(func=_sample_config)
        sample_parser.add_argument("--study-storage", type=str)
        sample_parser.add_argument("--study-name", type=str)
        sample_parser.add_argument("--out-path", type=str)
        sample_parser.add_argument("--override-config", type=str, default=None)

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
