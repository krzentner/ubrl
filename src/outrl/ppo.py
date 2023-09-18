import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import outrl
from outrl.fragment_buffer import FragmentDataloader
from outrl.nn import compute_advantages, discount_cumsum


class PPO(LightningModule):
    def __init__(self, cfg, env_cons):
        super().__init__()
        self.cfg = cfg
        self.n_envs = cfg.get("n_envs", None)
        self.sampler = outrl.Sampler(env_cons, self.n_envs)

        self.env_spec = self.sampler.env_spec
        self.max_episode_length = self.env_spec.max_episode_length
        self.steps_per_epoch = cfg.get("steps_per_epoch", 4096)
        self.max_episodes_per_epoch = int(
            math.ceil(self.steps_per_epoch / self.env_spec.max_episode_length)
        )
        self.hidden_sizes = cfg.get("hidden_sizes", [128])

        self.buffer = outrl.FragmentBuffer(
            self.steps_per_epoch, self.env_spec.max_episode_length
        )
        self.agent = outrl.StochasticMLPAgent(
            observation_shape=self.env_spec.observation_shape,
            action_shape=self.env_spec.action_shape,
            hidden_sizes=self.hidden_sizes,
            vf_hidden_sizes=cfg.get("vf_hidden_sizes", [32, 32]),
            pi_hidden_sizes=cfg.get("pi_hidden_sizes", [32, 32]),
            action_dist=outrl.nn.DEFAULT_DIST_TYPES[self.env_spec.action_type],
        )

        self.minibatch_size = cfg.get("minibatch_size", 512)
        self.learning_rate = cfg.get("learning_rate", 2.5e-3)
        self.discount = cfg.get("discount", 0.99)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.fragment_length = cfg.get("fragment_length", 1)
        self.epochs_per_policy_step = cfg.get("epochs_per_policy_step", 20)
        self.vf_loss_coeff = cfg.get("vf_loss_coeff", 0.1)

        self.save_hyperparameters(ignore=["buffer", "sampler", "env_cons"])

    def training_step(self, batch: Dict[str, torch.Tensor]):
        advantages = batch["advantages"].squeeze()
        advantages -= advantages.mean()
        advantages /= advantages.std()

        action_energy, predicted_returns = self.agent.forward_both(
            batch["observations"], batch["actions"]
        )
        log_prob = -action_energy
        minibatch_size = batch["observations"].shape[0]
        assert log_prob.shape == (minibatch_size,)
        old_log_prob = -batch["action_energy"].squeeze()
        assert old_log_prob.shape == (minibatch_size,)
        assert not old_log_prob.grad_fn

        ratio = torch.exp(log_prob - old_log_prob)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

        explained_var = outrl.nn.explained_variance(
            predicted_returns, batch["discounted_returns"]
        )

        assert ratio.shape == (minibatch_size,)
        assert clip_ratio.shape == (minibatch_size,)

        policy_gradient = ratio * advantages
        clip_policy_gradient = clip_ratio * advantages
        assert policy_gradient.shape == (minibatch_size,)
        assert clip_policy_gradient.shape == (minibatch_size,)

        # pi_loss = -policy_gradient.mean()
        pi_loss = -torch.min(policy_gradient, clip_policy_gradient).mean()
        clip_portion = (clip_ratio != ratio).mean(dtype=torch.float32)

        actual_returns = batch["discounted_returns"].squeeze()
        vf_loss = F.mse_loss(predicted_returns, actual_returns)

        self.log_training(batch, locals())
        return pi_loss + self.vf_loss_coeff * vf_loss

    def log_training(
        self, batch: Dict[str, torch.Tensor], train_locals: Dict[str, torch.Tensor]
    ):
        if not self.logged_dataset:
            self.log_dataset()
            self.logged_dataset = True
        for k in ["vf_loss", "pi_loss", "explained_var", "clip_portion"]:
            self.log(k, train_locals[k].item(), prog_bar=True)

        for k in ["log_prob"]:
            self.log(k, train_locals[k].mean().item(), prog_bar=True)

        self.log(
            "average_timestep_reward", batch["rewards"].mean().item(), on_epoch=True
        )

    def log_dataset(self):
        full_episode_rewards = self.buffer.get_full_episodes("rewards")
        vf_returns = self.buffer_dataloader.extra_buffers["vf_returns"]
        discounted_returns = self.buffer_dataloader.extra_buffers["discounted_returns"]
        if len(full_episode_rewards):
            self.log(
                "avg_return",
                full_episode_rewards.sum(dim=-1).mean(dim=0),
                prog_bar=True,
            )
            self.log(
                "data_exp_var",
                outrl.nn.explained_variance(
                    vf_returns[self.valid_mask], discounted_returns[self.valid_mask]
                ),
                prog_bar=True,
            )

    def configure_optimizers(self) -> List[Optimizer]:
        return [torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)]

    def preprocess(self):
        self.logged_dataset = False
        self.buffer.clear_all()
        self.sampler.sample(
            self.buffer, self.agent, timestep_target=self.steps_per_epoch
        )

        valids = self.buffer.valid_mask()
        self.valid_mask = valids

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
                all_obs[i, eps_len] = self.buffer.episode_data[i][
                    "last_observation"
                ]
            except KeyError:
                pass

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
        rewards[~valids] = 0.0

        advantages = compute_advantages(
            discount=self.discount,
            gae_lambda=self.gae_lambda,
            vf_returns=vf_returns,
            rewards=rewards,
        )

        self.discounted_returns = discount_cumsum(
            rewards.unsqueeze(-1), discount=self.discount
        )
        assert len(self.discounted_returns.shape) == 2
        self.buffer_dataloader.extra_buffers = {
            "vf_returns": vf_returns[:, :-1],
            "advantages": advantages,
            "discounted_returns": self.discounted_returns,
            "valid_rewards": rewards,
        }

    def train_dataloader(self) -> DataLoader:
        self.buffer_dataloader = FragmentDataloader(
            self.buffer, batch_size=self.minibatch_size, callback=self.preprocess,
            cycles=self.epochs_per_policy_step,
        )
        return self.buffer_dataloader


def gym_ppo(cfg: DictConfig):
    print("CONFIG START")
    print(OmegaConf.to_yaml(cfg).strip())
    print("CONFIG END")

    import outrl.gym_env

    seed_everything(cfg.get("seed", 0))
    env_cons = outrl.gym_env.GymEnvCons(
        cfg.get("env", "CartPole-v0"),
        max_episode_length=cfg.get("max_episode_length", 200),
    )
    checkpoint_callback = ModelCheckpoint(
        save_top_k=10,
        save_last=True,
        mode="max",
        every_n_epochs=100,
        monitor="avg_return",
    )
    model = PPO(cfg, env_cons)
    trainer = Trainer(max_epochs=-1, callbacks=[checkpoint_callback])
    trainer.fit(model)


if __name__ == "__main__":
    cli_cfg = OmegaConf.from_cli()
    if 'cfg' in cli_cfg:
        cfg = OmegaConf.merge(cli_cfg, OmegaConf.load(cli_cfg['cfg']))
    else:
        cfg = cli_cfg
    gym_ppo(cfg)
