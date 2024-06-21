from dataclasses import dataclass
import random

import torch
from torch.distributions import Categorical
import noko
from transformers import AutoTokenizer, AutoModelForCausalLM

import ubrl
from ubrl.torch_trainer import TorchTrainer
from ubrl.cli import run


@dataclass
class LLMAgentConfig(ubrl.TrainerConfig):
    # pretrained_name: str = "microsoft/Phi-3-mini-128k-instruct"
    pretrained_name: str = "google-t5/t5-small"
    trust_remote_code: bool = True

    state_encoding_size: int = 256
    train_steps_per_checkpoint: int = 100
    n_prompts: int = 10


class CausalLMAgent(ubrl.Agent):

    def __init__(self, cfg: LLMAgentConfig):
        super().__init__(cfg.state_encoding_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code)
        self.causal_llm = AutoModelForCausalLM.from_pretrained(
            cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code)

    def forward(self, inputs: ubrl.AgentInput) -> ubrl.AgentOutput:
        # This method could also be called ubrl_forward
        episodes_str = [episode["text"] for episode in inputs.episodes]
        tokenized_episodes = self.tokenizer.encode(
            episodes_str,
            padding='longest',
            return_tensors='pt')
        model_out = self.causal_llm(
            input_ids=tokenized_episodes,
            output_hidden_states=True,
            return_dict=True)

        # First token is assumed to not be an LLM "action"
        action_ll = torch.softmax(model_out.logits, dim=-1)[tokenized_episodes][:, 1:]

        # Take exponential weighted average over hidden states as state
        # encoding, biased towards later layers.
        state_encodings = model_out.hidden_states[0, :, :self.state_encoding_size]
        for hidden_states in model_out.hidden_states[1:]:
            state_encodings = (
                hidden_states[0, :, :self.state_encoding_size] + 0.5 * state_encodings
            )

        return ubrl.AgentOutput(
            state_encodings=state_encodings,
            action_lls=action_ll,
            action_dists=[Categorical(logits=model_out.logits[b, 1:])
                          for b in range(len(inputs.episodes))]
        )

def compute_rewards(tokens: list[str]) -> list[float]:
    return [0.0 if 'e' in token else 1.0
            for token in tokens]


def _collect_episodes(agent: CausalLMAgent, n_prompts: int) -> list[dict[str, str | torch.Tensor]]:
    prompts = []
    for i in range(n_prompts):
        topic = agent.tokenizer.decode([random.randrange(len(agent.tokenizer))])
        prompt = f"Write me a poem about {topic}:"
        prompts.append(prompt)
    prompt_ids = agent.tokenizer.encode(prompts, padding='longest', return_tensors='pt')
    generated_output = agent.causal_llm.generate(prompt_ids,
                                                 return_dict_in_generate=True,
                                                 output_logits=True)
    generated_ids = generated_output.sequences
    generated_tokens = agent.tokenizer.convert_ids_to_tokens(generated_ids)
    generated_text: list[str] = agent.tokenizer.decode(generated_ids)
    rewards = [compute_rewards(ep_tokens)
               for ep_tokens in generated_tokens]
    return [{"prompt": prompts[i],
             "prompt_token_len": len(prompt_ids[i]),
             "generated_text": generated_text[i],
             "logits": generated_output.logits,
             "rewards": torch.tensor(rewards[i])}
            for i in range(len(prompts))]


def _episode_stats(episodes: list[dict[str, str | torch.Tensor]]) -> dict[str, float]:
    all_rewards = torch.cat([ep["rewards"] for ep in episodes])
    return {'average_reward': all_rewards.mean().item()}


def train_llm_agent(cfg: LLMAgentConfig):
    llm_agent = CausalLMAgent(cfg)
    print("agent:", llm_agent)

    trainer = TorchTrainer(cfg, llm_agent)

    trainer.attempt_resume(prefer_best=False)

    train_stats = _episode_stats(_collect_episodes(llm_agent, 10))

    for step in range(cfg.expected_train_steps + 1):
        if step % cfg.train_steps_per_checkpoint == 0 or step == cfg.expected_train_steps:
            trainer.add_eval_stats(train_stats, "average_reward")
            trainer.maybe_checkpoint()
        if step < cfg.expected_train_steps:
            train_episodes = _collect_episodes(llm_agent, 10)
            for episode in train_episodes:
                ep_len = len(episode["rewards"])
                action_mask = torch.ones(ep_len, dtype=torch.bool)
                action_mask[0:episode["prompt_token_len"]] = False
                trainer.add_episode(
                    episode,
                    rewards=episode["rewards"],
                    action_lls=episode["logits"],
                    terminated=False,
                    any_actions_possible=action_mask,
                )
            train_stats = _episode_stats(train_episodes)
            noko.log_row(
                "train_stats",
                train_stats,
                step=trainer.total_env_steps,
                level=noko.RESULTS,
            )
            trainer.train_step()

if __name__ == '__main__':
    run(train_fn=train_llm_agent, config_type=LLMAgentConfig)
