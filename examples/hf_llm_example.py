from dataclasses import dataclass
import random

import torch
from torch.distributions import Categorical
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

import noko

import ubrl
from ubrl.torch_trainer import AgentInput, TorchTrainer
from ubrl.cli import run

import hot_restart


@dataclass
class LLMAgentConfig(ubrl.TrainerConfig):
    # pretrained_name: str = "microsoft/Phi-3-mini-128k-instruct"
    pretrained_name: str = "google/flan-t5-small"
    trust_remote_code: bool = True

    state_encoding_size: int = 256
    train_steps_per_checkpoint: int = 100

    n_prompts: int = 10
    max_generation_len: int = 44
    max_prompt_len: int = 20


@dataclass
class LLMEpisode:
    prompt_text: str
    prompt_ids: torch.Tensor
    generated_text: str
    generated_ids: torch.Tensor
    action_lls: torch.Tensor
    logits: torch.Tensor
    rewards: torch.Tensor
    # action_dist: Categorical


class CausalLMAgent(ubrl.Agent):
    def __init__(self, cfg: LLMAgentConfig):
        super().__init__(cfg.state_encoding_size)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code
        )
        try:
            self.causal_llm = AutoModelForCausalLM.from_pretrained(
                cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code
            )
        except ValueError:
            self.causal_llm = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.pretrained_name, trust_remote_code=cfg.trust_remote_code
            )
        self.cfg = cfg
        self.encoder_task_description = (
            "Please answer the following question by reasoning step-by-step."
        )

    def forward(self, inputs: ubrl.AgentInput) -> ubrl.AgentOutput:
        # This method could also be called ubrl_forward

        # This method needs to match how _collect_episodes runs the agent.
        # In this example, that is checked in agent.check_padding.

        in_episodes: list[LLMEpisode] = inputs.episodes

        n_episodes = len(in_episodes)
        prompt_ids = [ep.prompt_ids for ep in in_episodes]
        gen_ids = [ep.generated_ids for ep in in_episodes]

        episodes_encoded = [
            # Don't re-encode a string here as one sequence.
            # We need to match the tokenization used at generation time, which
            # can theoretically be different from the tokenization sequence the
            # tokenizer would generate from concatenating the strings.
            torch.cat([prompt_ids[i], gen_ids[i]])
            for i in range(n_episodes)
        ]

        max_len = max([len(ids) for ids in episodes_encoded])
        ep_len = max(16, max_len)

        padded_episodes = torch.full(
            (n_episodes, ep_len), self.tokenizer.pad_token_id, dtype=torch.long
        )
        # Also used for selecting state_encodings
        state_encodings_mask = torch.zeros((n_episodes, ep_len), dtype=torch.bool)
        # Mask for tokens we're telling the ubrl count as "actions"
        action_lls_mask = torch.zeros((n_episodes, ep_len), dtype=torch.bool)
        # Mask for actually generated tokens.
        real_logits_mask = torch.zeros((n_episodes, ep_len), dtype=torch.bool)
        for i, encoded in enumerate(episodes_encoded):
            attn_pad_size = ep_len - len(encoded)
            prompt_size = len(prompt_ids[i]) - 1
            # Build masks for the state encodings, action_lls, and real logits
            #  - state_encodings: This should have every observable token,
            #  including the starting <pad> and encoding from of the final
            #  terminal token.
            #  - action_lls: This should have one fewer than the state encodings.
            #  Importantly, this does not include the lls of the "post
            #  terminal" token, which are computed along with the
            #  state_encodings for the terminal token.
            #  - real_logits_mask: The same as the action_lls mask, but starts
            #  at the end of the prompt. Is used to make sure that padding is
            #  correct.

            if self.tokenizer.padding_side == "left":
                padded_episodes[i, attn_pad_size:] = encoded
                state_encodings_mask[i, attn_pad_size:] = True
                action_lls_mask[i, attn_pad_size : ep_len - 1] = True
                real_logits_mask[i, attn_pad_size + prompt_size : ep_len - 1] = True
            else:
                padded_episodes[i, : ep_len - attn_pad_size] = encoded
                state_encodings_mask[i, : ep_len - attn_pad_size] = True
                action_lls_mask[i, : ep_len - attn_pad_size - 1] = True
                real_logits_mask[i, prompt_size : ep_len - attn_pad_size - 1] = True

        # Sometimes there may be pad tokens embedded in the sequence. Count them.
        original_pad_count = sum(
            (encoded == self.tokenizer.pad_token_id).sum()
            for encoded in episodes_encoded
        )
        masked_pad_count = (
            padded_episodes[state_encodings_mask] == self.tokenizer.pad_token_id
        ).sum()
        # Those same pad tokens (and only those) should be visible "through" the mask
        assert original_pad_count == masked_pad_count
        rewards = torch.cat([ep.rewards for ep in in_episodes])

        assert action_lls_mask.sum() == len(rewards)
        assert state_encodings_mask.sum() == n_episodes + len(rewards)

        task_desc_enc = self.tokenizer(
            self.encoder_task_description, return_tensors="pt"
        ).input_ids.repeat(n_episodes, 1)

        model_out = self.causal_llm(
            input_ids=task_desc_enc,
            decoder_input_ids=padded_episodes,
            decoder_attention_mask=state_encodings_mask,
            output_hidden_states=True,
            return_dict=True,
        )

        sampled_token_lls = torch.gather(
            model_out.logits, 2, padded_episodes[:, :, None]
        ).squeeze(-1)
        hidden_states = model_out.decoder_hidden_states[-1][
            :, :, : self.state_encoding_size
        ]

        state_encodings = hidden_states[state_encodings_mask]
        action_lls = sampled_token_lls[action_lls_mask]
        all_logits = [
            model_out.logits[i, real_logits_mask[i]] for i in range(n_episodes)
        ]
        original_action_lls = torch.cat([ep.action_lls for ep in in_episodes])
        terminated = torch.zeros(len(in_episodes), dtype=torch.bool)

        return ubrl.AgentOutput(
            state_encodings=state_encodings,
            action_lls=action_lls,
            n_timesteps=[ep.action_lls.shape[0] for ep in in_episodes],
            rewards=rewards,
            original_action_lls=original_action_lls,
            terminated=terminated,
            # Only construct action dict on action tokens
            # action_dists=[Categorical(logits=logits)
            #               for logits in all_logits],
            # infos is optional, but we use it in this example to implement
            # check_padding()
            infos={"logits": list(all_logits)},
        )


def check_padding(agent, episodes: list[LLMEpisode]):
    """Used to check that padding is consistently applied between
    _collect_episodes() and llm_agent.forward(). This method is not part of
    the agent API, and is not used by ubrl.

    This is especially important for the gap directly between the prompt
    and generated text.

    It only makes sense to call this method immediately after calling
    _collect_episodes() before training.
    """
    assert isinstance(episodes, list)
    logits = torch.cat([ep.logits for ep in episodes])
    agent_out = agent(
        AgentInput(
            episodes=episodes,
            need_full=True,
        )
    )
    # Check that logits during generation match logits during forward pass
    # These tolerances may be too strict for 16 bit (or smaller)
    assert torch.allclose(
        torch.cat(agent_out.infos["logits"]), logits, atol=1e-4, rtol=1e-4
    )


def compute_rewards(
    prompt: str, prompt_tokens: list[str], tokens: list[str], full_text: str
) -> list[float]:
    # We're treating every step between tokens on the input prompt as a
    # timestep, so give rewards for those.
    prompt_rewards = [0.0 for t in prompt_tokens[1:]]

    # Output 0 for each "state transition" in the prompt
    token_rewards = []
    for token in tokens:
        if token == "<pad>":
            token_rewards.append(0)
        elif "e" not in token:
            token_rewards.append(1.0)
        else:
            token_rewards.append(-1.0)

    return prompt_rewards + token_rewards


def generate_prompt(agent: CausalLMAgent) -> str:
    topic = agent.tokenizer.decode([random.randrange(len(agent.tokenizer))])
    # topic = "sisters"
    return f"Here's a poem about {topic}: "


def _collect_episodes(
    agent: CausalLMAgent, n_prompts: int, max_prompt_len: int, max_generation_len: int
) -> list[LLMEpisode]:
    prompts = [generate_prompt(agent) for _ in range(n_prompts)]
    encoded_prompts_unpadded = [
        agent.tokenizer(prompt, return_tensors="pt").input_ids[0] for prompt in prompts
    ]
    # If the following assert fails, remove, you should probably change :-1] to :] in the code below.
    assert all(
        [enc[-1] == agent.tokenizer.eos_token_id for enc in encoded_prompts_unpadded]
    )
    # Truncate to have at most max_prompt_len (including a single starting
    # beginning-of-sequence token, which might be the same as the <pad> token),
    # and remove the trailing end-of-sequence token.
    decoder_inputs_unpadded = [
        # Remove end-of-sequence token
        enc_prompt[max(0, len(enc_prompt) - max_prompt_len - 1) : -1]
        for enc_prompt in encoded_prompts_unpadded
    ]

    n_episodes = len(decoder_inputs_unpadded)
    prompt_padded_len = max_prompt_len
    task_desc_enc = agent.tokenizer(
        agent.encoder_task_description, return_tensors="pt"
    ).input_ids.repeat(n_episodes, 1)
    padded_dec_input = torch.full(
        (n_episodes, prompt_padded_len), agent.tokenizer.pad_token_id, dtype=torch.long
    )
    dec_attn_mask = torch.zeros((n_episodes, prompt_padded_len), dtype=torch.bool)
    for i, dec_in in enumerate(decoder_inputs_unpadded):
        padded_dec_input[i, prompt_padded_len - len(dec_in) :] = dec_in
        # Need to attend to the last pad token, since it serves as the
        # beginning of sequence token.
        dec_attn_mask[i, prompt_padded_len - len(dec_in) - 1 :] = True
    llm_inputs = {
        "input_ids": task_desc_enc,
        "decoder_input_ids": padded_dec_input,
        "decoder_attention_mask": dec_attn_mask,
    }

    with torch.no_grad():
        generated_output = agent.causal_llm.generate(
            **llm_inputs,
            return_dict_in_generate=True,
            output_logits=True,
            do_sample=True,
            max_new_tokens=max_generation_len,
        )

    # We need to account for the <bos> start token fed into the decoder
    gen_seqs = generated_output.sequences[:, prompt_padded_len:]

    # Move batch index first
    logits = torch.stack(generated_output.logits).transpose(0, 1)

    # logits are (B, T, X) here, where X is the token dim
    token_lls = torch.log_softmax(logits, dim=2)
    action_lls = torch.gather(token_lls, 2, gen_seqs[:, :, None]).squeeze(-1)

    valid_tokens_mask = gen_seqs != agent.tokenizer.pad_token_id
    # Note that we still include eos_token_id, since that's an "action" the LLM
    # can take
    # Assume the first generated token is always valid
    valid_tokens_mask[:, 0] = True

    episodes = []
    for i in range(n_prompts):
        # Unpad the generated output and create an "episode" out of it.

        # First, get back our prompt. Trust the attention mask to tell us which
        # tokens we'll need to feed back in.
        # encoded_prompt = encoded_prompts.input_ids[i][
        #     encoded_prompts.attention_mask[i].bool()
        # ]
        encoded_prompt = torch.cat(
            [
                torch.full((1,), agent.tokenizer.pad_token_id, dtype=torch.long),
                decoder_inputs_unpadded[i],
            ],
            dim=0,
        )
        # The tokenizer might have added a sequence end token to it if our
        # model is seq2seq.
        prompt_tokens = agent.tokenizer.convert_ids_to_tokens(encoded_prompt)

        generated_ids = gen_seqs[i]
        valid_mask = valid_tokens_mask[i]
        valid_token_ids = generated_ids[valid_mask]
        gen_tokens = agent.tokenizer.convert_ids_to_tokens(valid_token_ids)

        # Might start with a pad token, but we should have removed all the other padding
        assert "<pad>" not in gen_tokens[1:]
        gen_text = agent.tokenizer.decode(valid_token_ids)
        rewards = compute_rewards(prompts[i], prompt_tokens, gen_tokens, gen_text)
        assert len(rewards) == len(prompt_tokens) + len(gen_tokens) - 1

        # Treat each token in the input as a state with a corresponding "null action"
        act_lls = torch.cat(
            [torch.zeros(len(encoded_prompt) - 1), action_lls[i, valid_mask]]
        )
        episode_logits = logits[i, valid_mask]
        assert len(act_lls) == len(encoded_prompt) + len(gen_tokens) - 1
        assert len(act_lls) == len(rewards)
        assert len(gen_tokens) == len(episode_logits)
        episodes.append(
            LLMEpisode(
                prompt_text=prompts[i],
                prompt_ids=encoded_prompt,
                generated_text=gen_text,
                generated_ids=valid_token_ids,
                action_lls=act_lls,
                logits=episode_logits,
                rewards=torch.tensor(rewards),
                # action_dist=Categorical(logits=episode_logits),
            )
        )
    return episodes


def _episode_stats(episodes: list[LLMEpisode]) -> dict[str, float]:
    all_rewards = torch.cat([ep.rewards for ep in episodes])
    return {"average_reward": all_rewards.mean().item()}


def train_llm_agent(cfg: LLMAgentConfig):
    llm_agent = CausalLMAgent(cfg)
    print("agent:", llm_agent)

    trainer = TorchTrainer(cfg, llm_agent)

    trainer.attempt_resume(prefer_best=False)

    llm_agent.eval()
    start_episodes = _collect_episodes(
        llm_agent, cfg.n_prompts, cfg.max_prompt_len, cfg.max_generation_len
    )
    check_padding(llm_agent, start_episodes)
    train_stats = _episode_stats(start_episodes)

    for step in range(cfg.expected_train_steps + 1):
        if (
            step % cfg.train_steps_per_checkpoint == 0
            or step == cfg.expected_train_steps
        ):
            trainer.add_eval_stats(train_stats, "average_reward")
            trainer.maybe_checkpoint()
        if step < cfg.expected_train_steps:
            train_episodes = _collect_episodes(
                llm_agent, cfg.n_prompts, cfg.max_prompt_len, cfg.max_generation_len
            )
            for episode in train_episodes:
                ep_len = len(episode.action_lls)
                trainer.add_episode(
                    episode,
                    n_timesteps=ep_len,
                    memory_size=ep_len**2,
                    # Transformer models tend to use T^2 memory
                    # It may make more sense to use a different function of
                    # ep_len here, depending on the model architecture.
                )
            train_stats = _episode_stats(train_episodes)
            noko.log_row(
                "train_stats",
                train_stats,
                step=trainer.total_env_steps,
                level=noko.RESULTS,
            )
            trainer.train_step()


hot_restart.wrap_module()
if __name__ == "__main__" and not hot_restart.is_restarting_module():
    run(train_fn=train_llm_agent, config_type=LLMAgentConfig)
