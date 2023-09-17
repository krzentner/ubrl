from typing import List, Optional
from tqdm import tqdm
import torch
import psutil

import outrl


class Sampler:
    def __init__(self, env_cons: outrl.env.EnvConstructor, vec_count: Optional[int]):
        if not vec_count:
            vec_count = psutil.cpu_count(logical=False)
        self.vec_count = vec_count
        self.env = env_cons(self.vec_count)
        self.prev_env_step = self.env.reset(
            None, mask=torch.ones(self.vec_count, dtype=torch.bool)
        )
        self.agent_hidden_states: Optional[torch.Tensor] = None
        self.episode_indices: torch.Tensor = torch.tensor(0)
        self.env_active: torch.Tensor = torch.zeros(vec_count, dtype=torch.bool)

    @property
    def env_spec(self):
        return self.env.spec

    @property
    def max_episode_length(self):
        return self.env.spec.max_episode_length

    def sample(
        self,
        buffer: outrl.FragmentBuffer,
        agent: outrl.Agent,
        *,
        finish_episodes: bool = False,
        timestep_target: Optional[int] = None,
        episode_target: Optional[int] = None,
        resume_episodes: bool = False,
        reset_hidden_state: bool = True,
    ):
        assert timestep_target or episode_target
        if timestep_target:
            eff_timestep_target = timestep_target
        elif episode_target:
            eff_timestep_target = episode_target * self.env.spec.max_episode_length
        else:
            raise ValueError("Need to specify timestep_target or episode_target")
        agent.train(mode=False)
        if self.agent_hidden_states is None or reset_hidden_state:
            self.agent_hidden_states = agent.initial_hidden_states(self.vec_count)
        if not resume_episodes:
            self.prev_env_step = self.env.reset(
                None, mask=torch.ones(self.vec_count, dtype=torch.bool)
            )
            self.episode_indices = buffer.start_episodes(self.vec_count)
            self.env_active = torch.zeros(self.vec_count, dtype=torch.bool)
            for i, episode_index in enumerate(self.episode_indices):
                self.env_active[i] = True
                assert not buffer.episode_complete[episode_index]
        steps_gathered = 0
        episodes_gathered = 0
        need_more_episodes = True
        done_gathering = False
        with tqdm(desc="Sampling", total=eff_timestep_target, disable=False) as pbar:
            while not done_gathering:
                if need_more_episodes:
                    for i in range(self.vec_count):
                        if not self.env_active[i]:
                            episode_index = buffer.start_episode()
                            if episode_index is not None:
                                self.env_active[i] = True
                                self.episode_indices[i] = episode_index
                    for i in range(self.vec_count):
                        if self.env_active[i]:
                            assert not buffer.episode_complete[self.episode_indices[i]]

                # TODO: Optimize only some environments being active
                with torch.no_grad():
                    agent_step = agent.step(
                        self.prev_env_step.observations,
                        self.agent_hidden_states,
                        self.prev_env_step.rewards,
                    )

                env_step = self.env.step(
                    self.prev_env_step, agent_step.actions, self.env_active
                )
                data = {}
                data["observations"] = self.prev_env_step.observations
                data["prev_rewards"] = self.prev_env_step.rewards

                for k, v in self.prev_env_step.infos.items():
                    data[f"prev_env_info.{k}"] = v

                data["actions"] = agent_step.actions
                if agent_step.predicted_returns is not None:
                    data["predicted_returns"] = agent_step.predicted_returns
                data["action_energy"] = agent_step.action_energy
                data["actions_encoded"] = agent_step.actions_encoded
                # TODO: Find some way of saving the distributions
                # data["action_dists"] = agent_step.action_dists
                data["hidden_states"] = agent_step.hidden_states
                for k, v in agent_step.infos.items():
                    data[f"agent_info.{k}"] = v

                data["rewards"] = env_step.rewards

                for k, v in env_step.infos.items():
                    data[f"env_info.{k}"] = v

                self.prev_env_step = self.env.reset(env_step, env_step.done)
                next_hidden_states = agent_step.hidden_states.clone()

                buffer.store_timesteps_multiepisode(
                    self.episode_indices[self.env_active],
                    {k: v[self.env_active].unsqueeze(1) for (k, v) in data.items()},
                )
                for i, episode_done in enumerate(env_step.done):
                    if episode_done:
                        # If necessary, will re-enable this environment at the top of loop
                        self.env_active[i] = False
                        buffer.end_episode(
                            self.episode_indices[i],
                            {
                                "last_observation": env_step.observations[i],
                                "truncated": env_step.truncated[i],
                                "terminated": env_step.terminated[i],
                            },
                        )
                        assert buffer.episode_complete[self.episode_indices[i]]
                        next_hidden_states[i] = agent.initial_hidden_states(1)[0]
                        episodes_gathered += 1
                self.agent_hidden_states = next_hidden_states

                steps_gathered += self.vec_count
                pbar.n = steps_gathered
                pbar.refresh()
                if timestep_target is not None and steps_gathered > timestep_target:
                    need_more_episodes = False
                    if not finish_episodes or not self.env_active.any():
                        done_gathering = True
                elif episode_target is not None and episodes_gathered > episode_target:
                    done_gathering = True
                if not done_gathering and buffer.episode_complete.all():
                    raise ValueError("Buffer not large enough")
        agent.train(mode=True)


def test_smoke():
    import outrl.gym_env

    B = 15
    cons = outrl.gym_env.GymEnvCons("CartPole-v0", max_episode_length=200)
    sampler = Sampler(cons, B)
    buffer = outrl.FragmentBuffer(10, sampler.max_episode_length)
    agent = outrl.StochasticMLPAgent(
        observation_shape=sampler.env_spec.observation_shape,
        action_shape=sampler.env_spec.action_shape,
        hidden_sizes=[32, 32],
        action_dist=outrl.nn.CategoricalConstructor,
    )
    try:
        sampler.sample(buffer, agent, timestep_target=100)
        sampler.sample(buffer, agent, episode_target=10)
    except ValueError:
        pass
    buffer = outrl.FragmentBuffer(100, sampler.max_episode_length)
    sampler.sample(buffer, agent, timestep_target=100)
    sampler.sample(buffer, agent, episode_target=10)
