"""vLLM-backed response generation.

The module is import-safe even when ``vllm`` is missing so that training
and analysis commands (which don't need vLLM) continue to work on GPUs
that can't run vLLM. The actual ``vllm`` import happens inside the
functions that need it.
"""

from __future__ import annotations

from typing import Dict, List


def _build_stop_token_ids(tokenizer) -> list[int]:
    stop_ids: set[int] = set()
    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        stop_ids.add(eos_id)

    # End-of-turn markers common across chat-tuned models. We only add an
    # id if the tokenizer actually recognizes the token.
    common_end_tokens = [
        "<|eot_id|>",
        "<|end_of_text|>",
        "<|im_end|>",
        "</s>",
        "<eos>",
    ]
    for tok in common_end_tokens:
        try:
            tok_id = tokenizer.convert_tokens_to_ids(tok)
            if tok_id is not None and tok_id != tokenizer.unk_token_id:
                stop_ids.add(tok_id)
        except Exception:
            pass
    return list(stop_ids)


def _format_chat_prompt(tokenizer, user_text: str) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_text},
    ]
    result = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    if isinstance(result, list):
        # Some tokenizers ignore ``tokenize=False`` and return tokens/strings.
        if result and isinstance(result[0], int):
            return tokenizer.decode(result)
        return "".join(result)
    return result


def generate_responses(
    requirements: List[Dict],
    llm,
    seed: int,
    temperature: float,
    num_per_prompt: int | None = None,
    round: int = 0,
    max_tokens: int = 512,
    top_p: float = 0.95,
) -> List[Dict]:
    """Generate responses for each requirement using an existing ``llm``.

    ``round`` is folded into the sampler's seed so repeated generation
    passes (e.g. rejection-sampling retries) explore different samples.
    """
    from vllm import SamplingParams  # local import keeps module import-safe

    effective_seed = seed + round * 1_000_000
    tokenizer = llm.get_tokenizer()
    stop_token_ids = _build_stop_token_ids(tokenizer)

    generated: List[Dict] = []
    for req in requirements:
        n_req = req["num_responses"]
        if n_req <= 0:
            continue

        prompt_text = _format_chat_prompt(tokenizer, req["prompt"])
        params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=effective_seed,
            n=n_req if num_per_prompt is None else num_per_prompt,
            stop_token_ids=stop_token_ids,
        )
        outputs = llm.generate(prompt_text, params)
        responses = [o.text for o in outputs[0].outputs]
        generated.append({
            "prompt": req["prompt"],
            "responses": responses,
            "original_idx": req["original_idx"],
        })
    return generated


def split_requirements(requirements: List[Dict], num_chunks: int, chunk_idx: int) -> List[Dict]:
    """Round-robin split so each chunk sees a representative slice of prompts."""
    return requirements[chunk_idx::num_chunks]
