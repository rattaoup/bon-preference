"""Reward-model scoring of prompt/response pairs."""

from __future__ import annotations

from typing import Dict

import torch

REWARD_MODELS: Dict[str, str] = {
    "skywork-v2": "Skywork/Skywork-Reward-V2-Llama-3.1-8B",
}


def resolve_reward_model(name_or_id: str) -> str:
    """Accept either a short alias (``skywork-v2``) or a full HF id."""
    if name_or_id in REWARD_MODELS:
        return REWARD_MODELS[name_or_id]
    if "/" in name_or_id:
        return name_or_id
    raise ValueError(
        f"Unknown reward model alias {name_or_id!r}. "
        f"Known aliases: {list(REWARD_MODELS)}."
    )


def load_reward_model(model_id: str, device: str):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32,
        device_map=device if device.startswith("cuda") else None,
        num_labels=1,
        attn_implementation="sdpa",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer


@torch.inference_mode()
def score_pair(model, tokenizer, prompt: str, response: str, device: str) -> float:
    conv = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(conv, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return model(**inputs).logits[0, 0].item()


@torch.inference_mode()
def score_messages(model, tokenizer, messages: list, device: str) -> float:
    """Score a pre-formatted messages list (used by the test-set builder)."""
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    if tokenizer.bos_token and text.startswith(tokenizer.bos_token):
        text = text[len(tokenizer.bos_token):]
    inputs = tokenizer(text, return_tensors="pt").to(device)
    return model(**inputs).logits[0, 0].item()
