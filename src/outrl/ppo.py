import math
from typing import Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from dataclasses import dataclass

from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

# import stick

import outrl
from outrl.fragment_buffer import FragmentDataloader
from outrl.nn import compute_advantages, discount_cumsum
from outrl.utils import (
    get_config,
    to_yaml,
    save_yaml,
    Serializable,
    copy_default,
    RunningMeanVar,
)


@dataclass
class AlgoConfig(Serializable):
    seed: int = 0
    exp_name: Optional[str] = None
    log_dir: Optional[str] = None
    max_episode_length: Optional[int] = None


@dataclass
class PPOConfig(AlgoConfig):
    n_envs: int = 10
    steps_per_epoch: int = 4096
    preprocess_hidden_sizes: list[int] = copy_default([128])
    pi_hidden_sizes: list[int] = copy_default([64, 64])
    vf_hidden_sizes: list[int] = copy_default([64, 64])
    minibatch_size: int = 512
    learning_rate: float = 2.5e-4
    discount: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    fragment_length: int = 1
    epochs_per_policy_step: int = 20
    vf_loss_coeff: float = 0.25
    vf_clip_ratio: Optional[float] = None
    norm_rewards: bool = False
    norm_observations: bool = False


class PPO(LightningModule):
    def __init__(self, cfg: PPOConfig, env_cons):
        super().__init__()
        self.cfg = cfg
        self.sampler = outrl.Sampler(env_cons, self.cfg.n_envs)

        self.env_spec = self.sampler.env_spec
        self.max_episode_length = self.env_spec.max_episode_length
        self.max_episodes_per_epoch = int(
            math.ceil(self.cfg.steps_per_epoch / self.env_spec.max_episode_length)
        )

        self.buffer = outrl.FragmentBuffer(
            self.cfg.steps_per_epoch, self.env_spec.max_episode_length
        )
        self.agent = outrl.StochasticMLPAgent(
            observation_shape=self.env_spec.observation_shape,
            action_shape=self.env_spec.action_shape,
            hidden_sizes=self.cfg.preprocess_hidden_sizes,
            vf_hidden_sizes=self.cfg.vf_hidden_sizes,
            pi_hidden_sizes=self.cfg.pi_hidden_sizes,
            action_dist_cons=outrl.dists.DEFAULT_DIST_TYPES[self.env_spec.action_type],
        )

        self.save_hyperparameters(ignore=["buffer", "sampler", "env_cons"])
        self.obs_normalizer = RunningMeanVar(
            mean=torch.zeros(self.env_spec.observation_shape),
            var=torch.ones(self.env_spec.observation_shape),
        )
        self.reward_normalizer = RunningMeanVar()
        self.automatic_optimization = False

    def training_step(self, mb: dict[str, torch.Tensor]):
        advantages = mb["advantages"].squeeze()
        advantages -= advantages.mean()
        advantages /= advantages.std()

        actor_out, critic_out = self.agent.forward_both(
            mb["observations"], mb["actions"]
        )
        log_prob = -actor_out
        minibatch_size = mb["observations"].shape[0]
        assert log_prob.shape == (minibatch_size,)
        old_log_prob = -mb["action_energy"].squeeze()
        assert old_log_prob.shape == (minibatch_size,)
        assert not old_log_prob.grad_fn

        ratio = torch.exp(log_prob - old_log_prob)
        ratio_clipped = torch.clamp(
            ratio, 1 - self.cfg.clip_ratio, 1 + self.cfg.clip_ratio
        )

        assert ratio.shape == (minibatch_size,)
        assert ratio_clipped.shape == (minibatch_size,)

        policy_gradient = ratio * advantages
        clip_policy_gradient = ratio_clipped * advantages
        assert policy_gradient.shape == (minibatch_size,)
        assert clip_policy_gradient.shape == (minibatch_size,)

        pi_loss = -torch.min(policy_gradient, clip_policy_gradient).mean()

        disc_returns = mb["discounted_returns"]
        if self.cfg.vf_clip_ratio:
            # This clipping logic matches the logic from Tianshou
            # But it isn't scale invariant, so it's probably wrong?
            # Maybe it's fine with reward scaling? But the return scale still
            # depends on the discount even with reward norm?
            # TODO(krzentner): Compare to paper: arXiv:1811.02553v3 Sec. 4.1
            diff = (critic_out - mb["vf_returns"]).clamp(
                -self.cfg.vf_clip_ratio, self.cfg.vf_clip_ratio
            )
            v_clip = mb["vf_returns"] + diff
            vf_loss_1 = (disc_returns - critic_out) ** 2
            vf_loss_2 = (disc_returns - v_clip) ** 2
            # Mean *after* taking max over each element
            vf_loss = torch.max(vf_loss_1, vf_loss_2).mean()
        else:
            vf_loss = F.mse_loss(critic_out, disc_returns)

        self.log_training(
            mb,
            dict(
                clip_portion=(ratio_clipped != ratio).mean(dtype=torch.float32),
                mb_ev=outrl.nn.explained_variance(critic_out, mb["discounted_returns"]),
            ) | locals(),
        )

        loss = pi_loss + self.cfg.vf_loss_coeff * vf_loss
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        # stick.log(table='minibatch')
        self.optimizers().step()

    def log_training(
        self, mb: dict[str, torch.Tensor], train_locals: dict[str, torch.Tensor]
    ):
        if not self.logged_dataset:
            self.log_dataset()
            self.logged_dataset = True
        for k in [
            "vf_loss",
            # "pi_loss",
            "mb_ev",
            # "clip_portion"
        ]:
            self.log(k, train_locals[k].item(), prog_bar=True)

        # for k in ["log_prob"]:
        #     self.log(k, train_locals[k].mean().item(), prog_bar=True)

        self.log("average_timestep_reward", mb["rewards"].mean().item(), on_epoch=True)
        # self.log(
        #     "vf_bias", self.agent.vf_layers.output_linear.bias.item(), prog_bar=True)
        self.log("disc_mean", train_locals["disc_returns"].mean(), prog_bar=True)
        self.log("critic_out_mean", train_locals["critic_out"].mean(), prog_bar=True)

    def log_dataset(self):
        full_episode_rewards = self.buffer.get_full_episodes("rewards")
        vf_returns = self.buffer_dataloader.extra_buffers["vf_returns"]
        discounted_returns = self.buffer_dataloader.extra_buffers["discounted_returns"]
        if len(full_episode_rewards):
            self.log(
                "ep_rew",
                full_episode_rewards.sum(dim=-1).mean(dim=0),
                prog_bar=True,
            )
            # Firt timestep
            # self.log(
            #     "ep_disc_rew",
            #     discounted_returns[:, 0].mean(dim=0),
            #     prog_bar=True,
            # )
            self.log(
                "disc_mean",
                discounted_returns[self.buffer.valid_mask()].mean(),
                prog_bar=True,
            )
            self.log(
                "vf_mean",
                vf_returns[self.buffer.valid_mask()].mean(),
                prog_bar=True,
            )
            self.log(
                "ev",
                outrl.nn.explained_variance(
                    vf_returns[self.buffer.valid_mask()],
                    discounted_returns[self.buffer.valid_mask()],
                ),
                prog_bar=True,
            )

    def configure_optimizers(self) -> list[Optimizer]:
        return [torch.optim.Adam(self.agent.parameters(), lr=self.cfg.learning_rate)]

    def preprocess(self):
        self.logged_dataset = False
        self.buffer.clear_all()
        self.sampler.sample(
            self.buffer, self.agent, timestep_target=self.cfg.steps_per_epoch
        )

        all_obs = torch.cat(
            [
                self.buffer.buffers["observations"],
                torch.zeros(
                    (self.buffer.n_episodes, 1) + self.env_spec.observation_shape
                ),
            ],
            dim=1,
        )
        # Fill in final (t + 1) observation
        for i in range(self.buffer.n_episodes):
            eps_len = self.buffer.episode_length_so_far[i]
            try:
                all_obs[i, eps_len] = self.buffer.episode_data[i]["last_observation"]
            except KeyError:
                pass
        self.obs_normalizer.update(all_obs)
        if self.cfg.norm_observations:
            all_obs = self.reward_normalizer.normalize_batch(all_obs)

        # Compute vf_returns, filter based on episode termination
        with torch.no_grad():
            vf_returns = self.agent.vf_forward(all_obs)
        # Can't use valids mask, since vf_returns goes to t + 1
        for i, episode_length in enumerate(self.buffer.episode_length_so_far):
            # Everything after last valid observation should be empty
            vf_returns[i, episode_length + 1 :] = 0.0
            # zero vf_{t+1} in terminated episodes
            if self.buffer.episode_data[i].get("terminated", False):
                vf_returns[i, episode_length] = 0.0

        rewards = self.buffer.buffers["rewards"].clone()
        rewards[~self.buffer.valid_mask()] = 0.0
        self.reward_normalizer.update(rewards[self.buffer.valid_mask()])
        if self.cfg.norm_rewards:
            rewards_normed = self.reward_normalizer.normalize_batch(rewards)
        else:
            rewards_normed = rewards

        advantages = compute_advantages(
            discount=self.cfg.discount,
            gae_lambda=self.cfg.gae_lambda,
            vf_returns=vf_returns,
            rewards=rewards_normed,
        )

        discounted_returns = discount_cumsum(rewards_normed, discount=self.cfg.discount)
        # self.buffer.buffers["observations"][:, :, -1] = discounted_returns

        assert len(discounted_returns.shape) == 2
        self.buffer_dataloader.extra_buffers = {
            "vf_returns": vf_returns[:, :-1],
            "advantages": advantages,
            "discounted_returns": discounted_returns,
            "valid_rewards": rewards,
        }

    def train_dataloader(self) -> DataLoader:
        self.buffer_dataloader = FragmentDataloader(
            self.buffer,
            batch_size=self.cfg.minibatch_size,
            callback=self.preprocess,
            cycles=self.cfg.epochs_per_policy_step,
        )
        return self.buffer_dataloader


@dataclass
class GymConfig(Serializable):
    env_name: str = "CartPole-v1"
    algorithm: str = "ppo"
    algo_cfg: PPOConfig = copy_default(PPOConfig())


def gym_ppo(cfg: GymConfig):
    config_yaml = to_yaml(cfg)
    print("CONFIG START")
    print(config_yaml)
    print("CONFIG END")

    if cfg.algo_cfg.exp_name is None:
        cfg.algo_cfg.exp_name = "gym_ppo"

    if cfg.algo_cfg.log_dir is None:
        import hashlib

        hexdigest = hashlib.sha1(config_yaml.encode()).hexdigest()
        cfg.algo_cfg.log_dir = f"outputs/{cfg.algorithm}/{hexdigest}"

    if cfg.algo_cfg.max_episode_length is None:
        cfg.algo_cfg.max_episode_length = 200

    import os

    os.makedirs(cfg.algo_cfg.log_dir, exist_ok=True)
    save_yaml(cfg, f"{cfg.algo_cfg.log_dir}/cfg.yaml")

    import outrl.gym_env

    seed_everything(cfg.algo_cfg.seed)
    env_cons = outrl.gym_env.GymEnvCons(
        cfg.env_name, max_episode_length=cfg.algo_cfg.max_episode_length
    )
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{cfg.algo_cfg.log_dir}/checkpoints",
        save_top_k=10,
        save_last=True,
        mode="max",
        every_n_epochs=10,
        monitor="avg_return",
        save_on_train_epoch_end=False,  # Actually want to checkpoint on start of epoch
    )
    logger = CSVLogger(cfg.algo_cfg.log_dir)
    model = PPO(cfg.algo_cfg, env_cons)
    trainer = Trainer(max_epochs=-1, callbacks=[checkpoint_callback], logger=logger)
    trainer.fit(model)


if __name__ == "__main__":
    gym_ppo(get_config(GymConfig))
