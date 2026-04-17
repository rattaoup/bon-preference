"""Backbone loading + last-token pairwise embeddings for preference pairs."""

from __future__ import annotations

from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def load_backbone_and_tokenizer(base_model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    backbone = AutoModel.from_pretrained(
        base_model_name, dtype=torch.bfloat16, device_map="auto",
    )
    backbone.eval()
    return backbone, tokenizer


def _tokenize(tokenizer, text: str, max_length: int = 1024):
    return tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)


@torch.inference_mode()
def _forward_backbone(backbone, tokens):
    device = next(backbone.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}
    return backbone(**tokens).last_hidden_state  # (B, T, C)


def _embed_messages(backbone, tokenizer, messages: list) -> torch.Tensor:
    text = tokenizer.apply_chat_template(messages, tokenize=False)
    tokens = _tokenize(tokenizer, text)
    hidden = _forward_backbone(backbone, tokens)
    return hidden[0, -1, :]  # last-token vector, shape (C,)


def get_pairwise_embeddings(backbone, tokenizer, dataset: List[dict]) -> torch.Tensor:
    """Return a ``(N, 2, H)`` tensor of (chosen, rejected) last-token embeddings."""
    pairs = []
    for item in tqdm(dataset, desc="Embedding pairs"):
        chosen = _embed_messages(backbone, tokenizer, item["chosen"])
        rejected = _embed_messages(backbone, tokenizer, item["rejected"])
        pairs.append(torch.stack([chosen, rejected], dim=0))
    return torch.stack(pairs, dim=0)
