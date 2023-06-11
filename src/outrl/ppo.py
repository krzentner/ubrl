import math
from typing import Dict, List

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule, seed_everything, Trainer

import hydra
from omegaconf import DictConfig, OmegaConf
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

import outrl


class PPO(LightningModule):
    def __init__(self, cfg, env_cons):
        super().__init__()
        self.cfg = cfg
        self.n_envs = cfg.get("n_envs", None)
        self.sampler = outrl.Sampler(env_cons, self.n_envs)

        self.env_spec = self.sampler.env_spec
        self.max_episode_length = self.env_spec.max_episode_length
        self.steps_per_epoch = cfg.get("steps_per_epoch", 2048)
        self.max_episodes_per_epoch = int(
            math.ceil(self.steps_per_epoch / self.env_spec.max_episode_length)
        )
        self.hidden_sizes = cfg.get("hidden_sizes", [128, 128])

        self.buffer = outrl.FragmentBuffer(
            self.steps_per_epoch, self.env_spec.max_episode_length
        )
        self.agent = outrl.StochasticMLPAgent(
            observation_shape=self.env_spec.observation_shape,
            action_shape=self.env_spec.action_shape,
            hidden_sizes=self.hidden_sizes,
            action_dist=outrl.nn.DEFAULT_DIST_TYPES[self.env_spec.action_type],
        )

        self.minibatch_size = cfg.get("minibatch_size", 64)
        self.learning_rate = cfg.get("learning_rate", 2.5e-4)
        self.discount = cfg.get("discount", 0.99)
        self.gae_lambda = cfg.get("gae_lambda", 0.95)
        self.clip_ratio = cfg.get("clip_ratio", 0.2)
        self.fragment_length = cfg.get("fragment_length", 1)
        self.epochs_per_policy_step = cfg.get("epochs_per_policy_step", 10)

        self.save_hyperparameters(ignore=["buffer", "sampler", "env_cons"])

    def training_step(self, batch: Dict[str, torch.Tensor]):
        step = self.agent(batch["observations"], batch["hidden_states"], batch["prev_rewards"])
        log_prob = -step.action_energy
        old_log_prop = -batch["action_energy"]
        ratio = torch.exp(log_prob - old_log_prop)
        clip_ratio = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)
        policy_gradient = ratio * batch["advantages"]
        clip_policy_gradient = clip_ratio * batch["advantages"]
        policy_loss = -torch.min(policy_gradient, clip_policy_gradient).mean()

        vf_loss = F.mse_loss(step.predicted_returns,
                             batch["discounted_returns"].squeeze())

        self.log_training(batch, locals())
        return policy_loss + vf_loss

    def log_training(
        self, batch: Dict[str, torch.Tensor], train_locals: Dict[str, torch.Tensor]
    ):
        if not self.logged_dataset:
            self.log_dataset()
            self.logged_dataset = True
        for k in ["vf_loss", "policy_loss"]:
            self.log(k, train_locals[k].item(), prog_bar=True)

        for k in ["log_prob", "ratio"]:
            self.log(k, train_locals[k].mean().item())

        self.log(
            "explained_var",
            outrl.nn.explained_variance(
                batch["predicted_returns"].detach(), batch["discounted_returns"]
            ),
            prog_bar=True,
        )
        self.log(
            "average_timestep_reward", batch["rewards"].mean().item(), on_epoch=True
        )

    def log_dataset(self):
        full_episode_rewards = self.buffer.get_full_episodes("rewards")
        if len(full_episode_rewards):
            self.log(
                "avg_return",
                full_episode_rewards.sum(dim=-1).mean(dim=0),
                prog_bar=True,
            )

    def configure_optimizers(self) -> List[Optimizer]:
        return [torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate)]

    def dataset_this_epoch(self) -> outrl.fragment_buffer.FragmentDataset:
        self.logged_dataset = False
        if self.current_epoch % self.epochs_per_policy_step == 0:
            self.buffer.clear()
            self.sampler.sample(
                self.buffer, self.agent, timestep_target=self.steps_per_epoch
            )
            self.advantages = outrl.nn.compute_advantages(
                discount=self.discount,
                gae_lambda=self.gae_lambda,
                max_episode_length=self.max_episode_length,
                # These might contain invalid timesteps, will be filtered in batching
                expected_returns=self.buffer.buffers["predicted_returns"].squeeze(),
                rewards=self.buffer.buffers["rewards"],
            )
            assert len(self.buffer.buffers["rewards"].shape) == 2
            self.discounted_returns = outrl.nn.discount_cumsum(
                self.buffer.buffers["rewards"].unsqueeze(-1), discount=self.discount
            )
            assert len(self.discounted_returns.shape) == 2
        return iter(outrl.fragment_buffer.FragmentDataset(
            self.buffer,
            extra_buffers={
                "advantages": self.advantages,
                "discounted_returns": self.discounted_returns,
            },
            fragment_length=self.fragment_length,
        ))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            outrl.IterCallback(self.dataset_this_epoch), batch_size=self.minibatch_size
        )


@hydra.main(version_base=None)
def gym_ppo(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    import outrl.gym_env

    seed_everything(cfg.get("seed", 0))
    env_cons = outrl.gym_env.GymEnvCons(
        cfg.get("env", "CartPole-v0"),
        max_episode_length=cfg.get("max_episode_length", 200),
    )
    model = PPO(cfg, env_cons)
    trainer = Trainer()
    trainer.fit(model)


if __name__ == "__main__":
    gym_ppo()
