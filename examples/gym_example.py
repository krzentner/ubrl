#!/usr/bin/env python3
#
# Usage: examples/gym_example.py train --n_train_steps=2 --seed=7
#

from dataclasses import dataclass, replace

try:
    import gymnasium as gym
except ImportError:
    import gym
from tqdm import tqdm

import stick
from outrl.gym_utils import make_gym_actor, collect, episode_stats
from outrl.rl import Trainer, TrainerConfig
from outrl.config import ExperimentInvocation, tunable, IntListDistribution


@dataclass
class GymConfig(TrainerConfig):
    env_name: str = "CartPole-v1"
    n_envs: int = 10
    max_episode_length: int = 200
    episodes_per_train_step: int = 30
    encoder_hidden_sizes: list[int] = tunable(
        [64, 64],
        IntListDistribution(
            [
                16,
            ],
            [256, 256, 256],
        ),
    )
    pi_hidden_sizes: list[int] = tunable(
        [64, 64],
        IntListDistribution(
            [
                16,
            ],
            [256, 256, 256],
        ),
    )

    n_train_steps: int = 100
    train_steps_per_eval: int = 1
    eval_episodes: int = 20

    def fill_defaults(self):
        return replace(
            super().fill_defaults(),
            max_buffer_episodes=self.episodes_per_train_step,
            expected_train_steps=self.n_train_steps,
        )


def train(cfg: GymConfig):
    envs = [gym.make(cfg.env_name) for _ in range(cfg.n_envs)]

    actor = make_gym_actor(
        envs, hidden_sizes=cfg.encoder_hidden_sizes, pi_hidden_sizes=cfg.pi_hidden_sizes
    )
    print("actor:", actor)

    trainer = Trainer(cfg, actor)

    trainer.attempt_resume(prefer_best=False)

    step = 0
    while True:
        if step % cfg.train_steps_per_eval == 0:
            eval_episodes = collect(
                cfg.max_episode_length * cfg.eval_episodes,
                envs,
                actor,
                best_action=True,
                max_episode_length=cfg.max_episode_length,
                full_episodes_only=True,
            )
            eval_stats = episode_stats(eval_episodes)
            trainer.add_eval_stats(eval_stats, "AverageReturn")
            # TODO: Checkpoint, resume
            trainer.maybe_checkpoint()
        if step == cfg.n_train_steps:
            break
        train_episodes = collect(
            cfg.max_episode_length * cfg.episodes_per_train_step,
            envs,
            actor,
            best_action=False,
            max_episode_length=cfg.max_episode_length,
            full_episodes_only=False,
        )
        stick.log(
            "train_stats",
            episode_stats(train_episodes),
            step=trainer.total_env_steps,
            level=stick.RESULTS,
        )
        for episode in train_episodes:
            trainer.add_episode(
                episode,
                rewards=episode["rewards"],
                action_lls=episode["action_lls"],
                terminated=episode["terminated"],
                action_dists=episode["action_dists"],
                any_actions_possible=episode["any_actions_possible"],
                infos=episode["env_infos"] | episode["agent_infos"],
            )
        trainer.train_step()
        step += 1
    if cfg.env_name == "CartPole-v1":
        assert eval_stats["AverageReturn"] >= 200, "Should have trained optimal policy"


if __name__ == "__main__":
    ExperimentInvocation(train, GymConfig).run()
