from dataclasses import dataclass
import random

import torch
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, PreTrainedTokenizer

import noko

import ubrl
from ubrl.torch_trainer import AgentInput, TorchTrainer
from ubrl.cli import run


@dataclass
class LLMAgentConfig(ubrl.TrainerConfig):
    # pretrained_name: str = "microsoft/Phi-3-mini-128k-instruct"
    pretrained_name: str = "google/flan-t5-small"
    trust_remote_code: bool = True

    state_encoding_size: int = 256
    train_steps_per_checkpoint: int = 100

    n_prompts: int = 10
    episode_length: int = 128


class CausalLMAgent(ubrl.Agent):

    def __init__(self, cfg: LLMAgentConfig):
        super().__init__(cfg.state_encoding_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code)
        try:
            self.causal_llm = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code)
        except ValueError:
            self.causal_llm = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code)

    def forward(self, inputs: ubrl.AgentInput) -> ubrl.AgentOutput:
        # This method could also be called ubrl_forward

        # This method needs to match how _collect_episodes runs the agent.
        # In this example, that is checked in check_padding.

        episodes_str = [episode['prompt_text'] + episode["generated_text"]
                        for episode in inputs.episodes]
        prompt_lengths = torch.tensor(
            [len(episode["prompt_ids"]) for episode in inputs.episodes], dtype=torch.long)
        tokenized_episodes = self.tokenizer(episodes_str,
                                            padding='longest',
                                            return_tensors='pt')
        pad_mask = tokenized_episodes.input_ids != self.tokenizer.pad_token_id
        pad_lengths = pad_mask.sum(dim=1)
        assert (pad_mask[:pad_lengths] == self.tokenizer.pad_token_id).all()

        prompt_lengths = torch.tensor(
            [len(episode["prompt_ids"]) for episode in inputs.episodes], dtype=torch.long)
        first_timestep_idx = pad_lengths + prompt_lengths

        # Last token of prompt becomes first observation.
        valid_state_encoding_mask = pad_mask.clone()
        valid_state_encoding_mask[:, :first_timestep_idx] = False

        valid_action_ll_mask = pad_mask.clone()
        valid_action_ll_mask[:, :first_timestep_idx + 1] = False

        model_out = self.causal_llm(
            input_ids=tokenized_episodes.input_ids,
            attention_mask=tokenized_episodes.attention_mask,
            output_hidden_states=True,
            return_dict=True)

        sampled_token_lls = torch.gather(
            model_out.logits, 2, model_out.sequences[:, 1:, None]).squeeze(-1)
        hidden_states = model_out.hidden_states[-1, :, :self.state_encoding_size]

        state_encodings = hidden_states[valid_state_encoding_mask]
        action_lls = sampled_token_lls[valid_action_ll_mask]

        return ubrl.AgentOutput(
            state_encodings=state_encodings,
            action_lls=action_lls,
            # No logit for first token
            action_dists=[Categorical(logits=model_out.logits[b, valid_action_ll_mask[b]])
                          for b in range(len(inputs.episodes))],
            # infos is optional, but we use it in this example to implement
            # check_padding()
            infos={"logits": model_out.logits[:, 1:]}
        )

    def check_padding(self, episodes: list[dict[str, str | torch.Tensor]]):
        """Used to check that padding is consistently applied between
        _collect_episodes() and llm_agent.forward(). This method is not part of
        the agent API, and is not used by ubrl.

        This is especially important for the gap directly between the prompt
        and generated text.

        It only makes sense to call this method immediately after calling
        _collect_episodes() before training.
        """
        agent_out = self(AgentInput(episodes=episodes, need_full=True))
        logits = torch.cat([ep["logits"] for ep in episodes])
        assert torch.allclose(agent_out.infos["logits"], logits)


def compute_rewards(prompt: str, tokens: list[str], full_text: str) -> list[float]:
    return [0.0 if 'e' in token else 1.0
            for token in tokens]


def generate_prompt(agent: CausalLMAgent) -> str:
    topic = agent.tokenizer.decode([random.randrange(len(agent.tokenizer))])
    return f"Write me a poem about {topic}:"


def _collect_episodes(agent: CausalLMAgent, n_prompts: int, episode_length: int) -> list[dict[str, str | torch.Tensor]]:
    prompts = [generate_prompt(agent) for _ in range(n_prompts)]
    encoded_prompts = agent.tokenizer(prompts,
                                 padding='longest',
                                 return_tensors='pt')
    with torch.no_grad():
        generated_output = agent.causal_llm.generate(
            input_ids=encoded_prompts.input_ids,
            attention_mask=encoded_prompts.attention_mask,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            max_new_tokens=episode_length)
    # Move batch index first
    logits = torch.stack(generated_output.logits).transpose(0, 1)
    # If this assert fails, the below line of code is probably wrong for your model
    assert (generated_output.sequences[:, 0] == agent.tokenizer.pad_token_id).all()
    # logits are (B, T, X) here, where X is the token dim
    token_lls = torch.log_softmax(logits, dim=2)
    # First token is a <pad>, so skip it.
    action_lls = torch.gather(token_lls, 2,
                              generated_output.sequences[:, 1:, None]).squeeze(-1)
    valid_tokens_mask = generated_output.sequences != agent.tokenizer.pad_token_id
    # Note that we still include eos_token_id, since that's an "action" the LLM
    # can take

    episodes = []
    for i in range(n_prompts):
        # Unpad the generated output and create an "episode" out of it.
        encoded_prompt = encoded_prompts.input_ids[i][encoded_prompts.attention_mask[i]]
        assert (generated_output.sequences[i, 0] == agent.tokenizer.pad_token_id).item()
        generated_ids = generated_output.sequences[i]
        valid_mask = valid_tokens_mask[i]
        valid_token_ids = generated_ids[valid_mask]
        gen_tokens = agent.tokenizer.convert_ids_to_tokens(valid_token_ids)
        gen_text = agent.tokenizer.decode(valid_token_ids)
        assert "<pad>" not in gen_text
        rewards = compute_rewards(prompts[i], gen_tokens, gen_text)
        act_lls = action_lls[i, valid_mask[1:]]
        assert len(act_lls) == len(gen_tokens)
        assert len(act_lls) == len(rewards)
        episodes.append({
            "prompt_text": prompts[i],
            "prompt_ids": encoded_prompt,
            "generated_ids": valid_token_ids,
            "generated_text": gen_text,
            "action_lls": act_lls, # Log-likelihood of all generated tokens
            "logits": logits[i],
            "rewards": torch.tensor(rewards)
        })
    return episodes


def _episode_stats(episodes: list[dict[str, str | torch.Tensor]]) -> dict[str, float]:
    all_rewards = torch.cat([ep["rewards"] for ep in episodes])
    return {'average_reward': all_rewards.mean().item()}


def train_llm_agent(cfg: LLMAgentConfig):
    llm_agent = CausalLMAgent(cfg)
    print("agent:", llm_agent)

    trainer = TorchTrainer(cfg, llm_agent)

    trainer.attempt_resume(prefer_best=False)

    llm_agent.eval()
    start_episodes = _collect_episodes(llm_agent, cfg.n_prompts, cfg.episode_length)
    llm_agent.check_padding(start_episodes[0])
    train_stats = _episode_stats(start_episodes)

    for step in range(cfg.expected_train_steps + 1):
        if step % cfg.train_steps_per_checkpoint == 0 or step == cfg.expected_train_steps:
            trainer.add_eval_stats(train_stats, "average_reward")
            trainer.maybe_checkpoint()
        if step < cfg.expected_train_steps:
            train_episodes = _collect_episodes(llm_agent, cfg.n_prompts, cfg.episode_length)
            for episode in train_episodes:
                ep_len = len(episode["rewards"])
                action_mask = torch.ones(ep_len, dtype=torch.bool)
                action_mask[0:episode["prompt_token_len"]] = False
                trainer.add_episode(
                    episode,
                    rewards=episode["rewards"],
                    action_lls=episode["action_lls"],
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
