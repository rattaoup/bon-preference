"""Test-set construction for UltraFeedback, PKU-SafeRLHF, and GSM8K.

Replaces the old ``create_test_dataset.ipynb`` notebook. Each test set
is a gzipped JSON list of dicts with the fields consumed by
``bon train`` when evaluating: ``prompt``, ``chosen``, ``rejected``,
``score_chosen``, ``score_rejected``, ``score_diff``.
"""

from __future__ import annotations

import re
from typing import List

import numpy as np
from datasets import load_dataset
from tqdm import tqdm

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL, PATHS
from bon.io import load_gzip, save_gzip
from bon.prompts import load_pku_saferlhf
from bon.scoring import load_reward_model, score_messages
from bon.seeding import set_seed


def _to_messages(prompt: str, chosen: str, rejected: str) -> dict:
    user = [{"role": "user", "content": prompt}]
    return {
        "prompt": prompt,
        "chosen": user + [{"role": "assistant", "content": chosen}],
        "rejected": user + [{"role": "assistant", "content": rejected}],
    }


def _score_pair_rm(rm, tokenizer, ex: dict, device: str) -> tuple[float, float]:
    return (
        score_messages(rm, tokenizer, ex["chosen"], device),
        score_messages(rm, tokenizer, ex["rejected"], device),
    )


def build_binary_test_set(dataset: List[dict], rm, tokenizer, device: str) -> List[dict]:
    """Score chosen/rejected; swap if the reward model disagrees with the original pair.

    Returns a new list of examples with RM-aligned (chosen, rejected) plus
    scores and an ``agree_with_original_pair`` flag.
    """
    out: List[dict] = []
    n_agree = 0
    for ex in tqdm(dataset, desc="Scoring test pairs"):
        sc_chosen, sc_rejected = _score_pair_rm(rm, tokenizer, ex, device)
        if sc_chosen >= sc_rejected:
            out.append({
                "prompt": ex["prompt"],
                "chosen": ex["chosen"],
                "rejected": ex["rejected"],
                "score_chosen": sc_chosen,
                "score_rejected": sc_rejected,
                "score_diff": sc_chosen - sc_rejected,
                "agree_with_original_pair": True,
            })
            n_agree += 1
        else:
            out.append({
                "prompt": ex["prompt"],
                "chosen": ex["rejected"],
                "rejected": ex["chosen"],
                "score_chosen": sc_rejected,
                "score_rejected": sc_chosen,
                "score_diff": sc_rejected - sc_chosen,
                "agree_with_original_pair": False,
            })
    print(f"RM agreement rate: {n_agree / len(dataset):.3f}")
    return out


# ---------------------------------------------------------------- UltraFeedback

def build_ultrafeedback(rm, tokenizer, device: str, limit: int | None = None) -> List[dict]:
    ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
    data = list(ds["test_prefs"])
    if limit is not None:
        data = data[:limit]
    return build_binary_test_set(data, rm, tokenizer, device)


# ---------------------------------------------------------------- PKU-SafeRLHF

def build_pku_saferlhf(rm, tokenizer, device: str, limit: int | None = None) -> List[dict]:
    test = load_pku_saferlhf(split="test")
    data = list(test)
    if limit is not None:
        data = data[:limit]
    return build_binary_test_set(data, rm, tokenizer, device)


# ---------------------------------------------------------------- GSM8K

_GSM_NUMBER = re.compile(r"-?\d+\.?\d*")
_GSM_FINAL = re.compile(r"####\s*(\-?[\d\.\,]+)")


def _extract_answer_gsm8k(text: str) -> str | None:
    m = _GSM_FINAL.search(text)
    if m:
        return m.group(1).replace(",", "")
    numbers = _GSM_NUMBER.findall(text.replace(",", ""))
    return numbers[-1] if numbers else None


def _matches_gold(pred_text: str, gold_text: str) -> bool:
    pred, gold = _extract_answer_gsm8k(pred_text), _extract_answer_gsm8k(gold_text)
    if pred is None or gold is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-5
    except ValueError:
        return False


def _gsm8k_load_scored_responses(
    name: str = "gsm8k_test_set", seed: int = 0, temperature: float = 1.0,
    num_chunks: int = 8, reward_model_name: str = "skywork-v2",
    base_model: str = DEFAULT_GENERATOR_MODEL,
) -> List[dict]:
    ds = load_dataset("openai/gsm8k", "main", split="test")
    merged: List[dict] = []
    for chunk_idx in range(num_chunks):
        resp_path = PATHS.responses_chunk(
            name, seed, temperature, chunk_idx, num_chunks, base_model,
        )
        score_path = PATHS.scores_chunk(
            name, seed, temperature, reward_model_name,
            chunk_idx, num_chunks, base_model,
        )
        merged.extend(zip(load_gzip(resp_path), load_gzip(score_path)))
    if len(merged) != len(ds):
        raise ValueError(
            f"Response count ({len(merged)}) != GSM8K test size ({len(ds)}). "
            "Did you generate/score the gsm8k_test_set responses?"
        )

    items = []
    for (resp, score), question, answer in zip(merged, ds["question"], ds["answer"]):
        # Double-check alignment via original_idx.
        assert resp["original_idx"] == score["original_idx"]
        assert ds["question"][resp["original_idx"]] == resp["prompt"]
        marks = [_matches_gold(r, ds["answer"][resp["original_idx"]])
                 for r in resp["responses"]]
        items.append({
            "prompt": resp["prompt"],
            "responses": resp["responses"],
            "scores": score["scores"],
            "mark": marks,
            "avg_mark": float(np.mean(marks)) if marks else 0.0,
            "correct_answer": ds["answer"][resp["original_idx"]],
            "original_idx": resp["original_idx"],
        })
    return items


def _create_preference_pair_gsm8k(item: dict) -> dict | None:
    """Pair a random correct response against the highest-scoring incorrect one.

    Only pairs where the correct response outscores the best incorrect response
    by at least ``+2`` in reward are kept - this produces the
    ``medium difficulty'' GSM8K test set used in the paper.
    """
    scores = item["scores"]
    marks = item["mark"]
    correct = [i for i, m in enumerate(marks) if m]
    incorrect = [i for i, m in enumerate(marks) if not m]
    if not correct or not incorrect:
        return None

    worst = max(incorrect, key=lambda i: scores[i])
    worst_score = scores[worst]
    valid_correct = [i for i in correct if scores[i] > worst_score + 2]
    if not valid_correct:
        return None

    import random
    chosen_idx = random.choice(valid_correct)
    msg = _to_messages(item["prompt"], item["responses"][chosen_idx], item["responses"][worst])
    msg.update({
        "score_chosen": scores[chosen_idx],
        "score_rejected": worst_score,
        "score_diff": scores[chosen_idx] - worst_score,
        "case": "near_correct_vs_correct",
        "original_idx": item["original_idx"],
        "answer": item["correct_answer"],
        "chosen_correct": marks[chosen_idx],
        "rejected_correct": marks[worst],
    })
    return msg


def build_gsm8k(
    *,
    seed: int = 0,
    response_name: str = "gsm8k_test_set",
    temperature: float = 1.0,
    num_chunks: int = 8,
    reward_model_name: str = "skywork-v2",
    base_model: str = DEFAULT_GENERATOR_MODEL,
) -> List[dict]:
    """Build the medium-difficulty GSM8K test set from pre-generated responses.

    Assumes ``bon generate`` + ``bon score`` have already been run against the
    GSM8K test prompts under ``--name gsm8k_test_set`` with the given
    ``base_model`` / ``reward_model_name``.
    """
    set_seed(seed)
    items = _gsm8k_load_scored_responses(
        name=response_name, seed=seed, temperature=temperature,
        num_chunks=num_chunks, reward_model_name=reward_model_name,
        base_model=base_model,
    )
    pairs: List[dict] = []
    skipped = 0
    for item in items:
        pair = _create_preference_pair_gsm8k(item)
        if pair is None:
            skipped += 1
        else:
            pairs.append(pair)
    print(f"GSM8K: built {len(pairs)} pairs; skipped {skipped} prompts "
          f"(no usable correct/incorrect pair).")
    return pairs


# ---------------------------------------------------------------- Entry point

def build_test_set(
    dataset: str,
    *,
    reward_model: str = DEFAULT_REWARD_MODEL,
    base_model: str = DEFAULT_GENERATOR_MODEL,
    seed: int = 0,
    response_name: str = "gsm8k_test_set",
    temperature: float = 1.0,
    num_chunks: int = 8,
    reward_model_name: str = "skywork-v2",
    limit: int | None = None,
) -> List[dict]:
    """Build a test set for ``dataset``.

    ``reward_model`` is required for every dataset. ``base_model`` is
    used only for GSM8K (which relies on pre-generated model responses).
    """
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if dataset in ("ultrafeedback", "pku-saferlhf"):
        rm, tokenizer = load_reward_model(reward_model, device)
        if dataset == "ultrafeedback":
            return build_ultrafeedback(rm, tokenizer, device, limit=limit)
        return build_pku_saferlhf(rm, tokenizer, device, limit=limit)
    if dataset == "gsm8k":
        pairs = build_gsm8k(
            seed=seed, response_name=response_name, temperature=temperature,
            num_chunks=num_chunks, reward_model_name=reward_model_name,
            base_model=base_model,
        )
        return pairs[:limit] if limit is not None else pairs
    raise ValueError(
        f"Unknown dataset {dataset!r}. "
        "Choose from: ultrafeedback, pku-saferlhf, gsm8k."
    )
