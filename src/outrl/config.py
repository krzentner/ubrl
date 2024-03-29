"""This module handles
Declare configs using dataclass, and automatically handle logging and hparam tuning."""
from dataclasses import dataclass, field, fields, replace
from typing import Any, List, Type, Optional, TypeVar
import os
import random
import sys
import datetime
import copy

import stick
import optuna
import argparse
import simple_parsing


class Config(simple_parsing.Serializable):
    def state_dict(self):
        return self.to_dict()

    @classmethod
    def sample(cls, trial: optuna.Trial, **kwargs):
        return suggest_config(trial, cls, **kwargs)


def to_yaml(obj) -> str:
    return simple_parsing.helpers.serialization.dumps_yaml(obj)


def save_yaml(obj, path):
    simple_parsing.helpers.serialization.save_yaml(obj, path)


def load_yaml(obj_type, path):
    return simple_parsing.helpers.serialization.load(obj_type, path)


class CustomDistribution:
    def sample(self, name: str, trial: optuna.Trial) -> Any:
        del name, trial
        raise NotImplementedError()


@dataclass
class IntListDistribution(CustomDistribution):
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

T = TypeVar("T")


def tunable(default_val: T, distribution, metadata=None, **kwargs) -> T:
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


def suggest_config(trial: optuna.Trial, config: Type, **kwargs):
    sampled = {}
    for f in fields(config):
        if f.name in kwargs:
            continue
        if OPTUNA_DISTRIBUTION in f.metadata:
            dist = f.metadata[OPTUNA_DISTRIBUTION]
            if isinstance(dist, CustomDistribution):
                sampled[f.name] = dist.sample(f.name, trial)
            else:
                sampled[f.name] = trial._suggest(f.name, dist)
    return config(**kwargs, **sampled)


def default_run_name():
    main_file = getattr(sys.modules.get("__main__"), "__file__", "interactive")
    file_trail = os.path.splitext(os.path.basename(main_file))[0]
    now = datetime.datetime.now().isoformat()
    return f"{file_trail}_{now}"


def prepare_training_directory(cfg):
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


class ExperimentInvocation:
    def __init__(self, train_fn, config_type):
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
            study = optuna.load_study(
                storage=self.args.study_storage, study_name=self.args.study_name
            )
            trial = study.ask()
            cfg = suggest_config(trial, config_type)
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
            trial_data = load_yaml(dict, self.args.trial_file)
            study = optuna.load_study(
                storage=trial_data["study_storage"], study_name=trial_data["study_name"]
            )
            result_key = self.args.config.minimization_objective
            results = stick.load_log_file(self.args.log_file, keys=[result_key])
            study.tell(trial_data["trial_number"], min(results[result_key]))

        train_parser = subp.add_parser("train", add_help=False)
        train_parser.set_defaults(func=_train)
        train_parser.add_argument("--config", default=None, type=str)

        create_parser = subp.add_parser("create-study")
        create_parser.set_defaults(func=_create_study)
        create_parser.add_argument("--study-storage", type=str)
        create_parser.add_argument("--study-name", type=str)
        report_trial = subp.add_parser("report-trial")
        report_trial.set_defaults(func=_report_trial)
        report_trial.add_argument("--trial-file", type=str)
        report_trial.add_argument("--log-file", type=str)
        sample_parser = subp.add_parser("sample-config")
        sample_parser.set_defaults(func=_sample_config)
        sample_parser.add_argument("--study-storage", type=str)
        sample_parser.add_argument("--study-name", type=str)
        sample_parser.add_argument("--out-path", type=str)

        # First parse the known arguments to get config path.
        # For unclear reasons, modifying the parser after using it is not
        # possible, so copy it first.
        parser_copy = copy.deepcopy(self.parser)
        args, _ = parser_copy.parse_known_args()

        if getattr(args, "config", None):
            # Command line arguments should override config file entries
            loaded_config = load_yaml(config_type, args.config)
            loaded_config = loaded_config.fill_defaults()
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
        self.args.cfg = self.args.cfg.fill_defaults()

    def run(self):
        self.args = self.parser.parse_args()
        self.args.cfg = self.args.cfg.fill_defaults()
        self.args.func()
        if self.args.done_token:
            with open(self.args.done_token, "w") as f:
                f.write("done\n")
