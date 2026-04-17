"""Up/downsampling rejection-sampling dataset construction.

For each prompt we generate responses, score them with the reward model,
and reject or accept based on where each score falls in the calibration
CDF. ``c`` controls the target percentile band (negative values invert
acceptance to downsample), ``w`` its width, and ``p`` the strength of
reshaping (``p = 0`` degenerates to vanilla sampling).
"""

from __future__ import annotations

import gc
import hashlib
import math
import random
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from bon.generation import generate_responses
from bon.scoring import load_reward_model, score_pair


def valid_percentile(c: float, w: float) -> Tuple[float, float]:
    a = max(0.0, abs(c) - w / 2)
    b = min(1.0, abs(c) + w / 2)
    return a, b


def estimate_num_to_generate(c: float, w: float, p: float, safety_factor: float = 2.0) -> int:
    """How many raw samples to generate per target sample (in expectation)."""
    a, b = valid_percentile(c, w)
    if c >= 0:  # upsampling inside the window
        q = (1 - p) + p * (b - a)
    else:       # downsampling outside the window
        q = (1 - p) + p * (1 - (b - a))
    return int(math.ceil(safety_factor / q))


def rejection_seed(N: int, c: float, w: float, p: float, seed: int = 0) -> int:
    # The hash key uses the legacy ``k_`` spelling so that re-running on top
    # of pre-existing rejection datasets produces the same seed as before.
    digest = hashlib.sha256(f"{N}_{c}_{w}_{p}".encode()).hexdigest()
    return int(digest, 16) % 1_000_000 + seed


def _prompt_rng(base_seed: int, original_idx: int, round_: int) -> random.Random:
    return random.Random(base_seed + original_idx * 1000 + round_ * 1_000_000)


def empirical_percentile(score: float, sorted_cal_scores: List[float]) -> float:
    """Fraction of calibration scores <= ``score``."""
    n = len(sorted_cal_scores)
    return sum(1 for s in sorted_cal_scores if s <= score) / n


def verify_if_dataset_is_complete(
    rejection_dataset: Dict[int, dict],
    requirements: List[Dict],
) -> List[Dict]:
    """Return the requirements still needing additional responses."""
    remaining = []
    for req in requirements:
        target = req["num_responses"]
        if target <= 0:
            continue
        have = len(rejection_dataset.get(req["original_idx"], {}).get("responses", []))
        if have < target:
            missing = req.copy()
            missing["num_responses"] = target - have
            remaining.append(missing)
    return remaining


def update_rejection_dataset(
    rejection_dataset: Dict[int, dict],
    accepted_samples: Dict[int, dict],
) -> Dict[int, dict]:
    for og_idx, sample in accepted_samples.items():
        if og_idx not in rejection_dataset:
            rejection_dataset[og_idx] = sample
        else:
            rejection_dataset[og_idx]["responses"].extend(sample["responses"])
            rejection_dataset[og_idx]["scores"].extend(sample["scores"])
    return rejection_dataset


def scale_requirements(requirements: List[Dict], N: int) -> List[Dict]:
    """Multiply every ``num_responses`` by ``N`` to match the target triplet size."""
    out = []
    for req in requirements:
        r = req.copy()
        r["num_responses"] = r["num_responses"] * N
        out.append(r)
    return out


def generate_and_score(
    requirements: List[Dict],
    *,
    model: str,
    reward_model: str,
    temperature: float,
    base_seed: int,
    sample_factor: float,
    round_: int,
) -> List[Dict]:
    """Generate ``sample_factor * num_responses`` candidates and reward-score them."""
    from vllm import LLM

    llm = LLM(model)
    responses: List[Dict] = []
    for req in requirements:
        if req["num_responses"] <= 0:
            continue
        out = generate_responses(
            [req], llm, seed=base_seed, temperature=temperature,
            num_per_prompt=int(sample_factor * req["num_responses"]),
            round=round_,
        )
        responses.extend(out)
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rm, tokenizer = load_reward_model(reward_model, device)
    print(f"Scoring {len(responses)} prompts")
    for item in tqdm(responses):
        item["scores"] = [
            score_pair(rm, tokenizer, item["prompt"], r, device)
            for r in item["responses"]
        ]
    del rm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def rejection_sampling(
    responses: List[Dict],
    calibration: Dict[int, List[float]],
    *,
    base_seed: int,
    c: float,
    w: float,
    p: float,
    round_: int = 0,
) -> Dict[int, dict]:
    accepted: Dict[int, dict] = {}
    lo, hi = valid_percentile(c=c, w=w)

    for item in responses:
        og_idx = item["original_idx"]
        cal = calibration[og_idx]
        assert len(cal) >= 100, (
            f"Calibration for prompt {og_idx} has only {len(cal)} samples")
        rng = _prompt_rng(base_seed, og_idx, round_)

        keep_responses, keep_scores = [], []
        for response, score in zip(item["responses"], item["scores"]):
            if rng.random() < 1 - p:
                keep_responses.append(response)
                keep_scores.append(score)
                continue
            percentile = empirical_percentile(score, cal)
            inside = lo <= percentile <= hi
            accept = inside if c >= 0 else not inside
            if accept:
                keep_responses.append(response)
                keep_scores.append(score)

        accepted[og_idx] = {
            "prompt": item["prompt"],
            "responses": keep_responses,
            "scores": keep_scores,
        }
    return accepted


def trim_to_requirements(
    rejection_dataset: Dict[int, dict],
    requirements: List[Dict],
) -> Dict[int, dict]:
    final: Dict[int, dict] = {}
    for req in requirements:
        target = req["num_responses"]
        if target <= 0:
            continue
        og_idx = req["original_idx"]
        cur = rejection_dataset.get(og_idx, {})
        responses = cur.get("responses", [])
        scores = cur.get("scores", [])
        assert len(responses) >= target, (
            f"Prompt {og_idx} is short of responses ({len(responses)} < {target})")
        final[og_idx] = {
            "prompt": cur.get("prompt", ""),
            "responses": responses[:target],
            "scores": scores[:target],
        }
    return final
