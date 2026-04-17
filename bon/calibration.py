"""Per-prompt reward calibration used by distribution reshaping.

For each prompt in a requirements file, we sample ``num_samples`` fresh
responses with the generator and score them with the reward model; the
sorted scores serve as an empirical CDF that ``rejection.py`` uses to
map raw rewards to percentiles.
"""

from __future__ import annotations

import gc
import os
from typing import Dict, List

import torch
from tqdm import tqdm

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL, PATHS
from bon.generation import generate_responses
from bon.io import load_gzip, load_json, save_gzip
from bon.scoring import load_reward_model, score_pair


def create_calibration(
    requirements: List[Dict],
    *,
    model: str,
    reward_model: str,
    temperature: float = 1.0,
    calibration_seed: int = 0,
    num_samples: int = 100,
) -> Dict[int, List[float]]:
    """Sample ``num_samples`` responses per prompt and return sorted rewards."""
    from vllm import LLM

    llm = LLM(model)
    responses = generate_responses(
        requirements, llm, seed=calibration_seed, temperature=temperature,
        num_per_prompt=num_samples, round=0,
    )
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rm, tokenizer = load_reward_model(reward_model, device)

    calibration: Dict[int, List[float]] = {}
    print(f"Calibrating {len(responses)} prompts")
    for item in tqdm(responses):
        scores = [score_pair(rm, tokenizer, item["prompt"], r, device)
                  for r in item["responses"]]
        calibration[item["original_idx"]] = sorted(scores)

    del rm
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return calibration


def load_or_build_calibration(
    name: str,
    seed: int,
    *,
    model: str = DEFAULT_GENERATOR_MODEL,
    reward_model: str = DEFAULT_REWARD_MODEL,
    num_samples: int = 100,
) -> Dict[int, List[float]]:
    """Return the calibration for ``name``, extending it with missing prompts.

    The calibration file is keyed by ``(name, base_model, reward_model)`` on
    disk, so it is reusable across every ``(c, w, p)`` sweep for the same
    ``(name, seed, base_model, reward_model)``.
    """
    calibration_path = PATHS.calibration_file(name, model, reward_model)
    os.makedirs(os.path.dirname(calibration_path), exist_ok=True)
    print(f"Calibration cache: {calibration_path}")

    if os.path.exists(calibration_path):
        existing = load_gzip(calibration_path)
        calibration = {int(k): v for k, v in existing.items()}
        print(f"  cache hit ({len(calibration)} prompts already calibrated)")
    else:
        calibration = {}
        print("  cache miss, calibrating from scratch")

    requirements = load_json(PATHS.requirements(name, seed))
    needed = [
        req for req in requirements
        if req["num_responses"] > 0 and req["original_idx"] not in calibration
    ]
    if not needed:
        print(f"  calibration is complete for {name!r}.")
        return calibration

    print(f"  calibrating {len(needed)} missing prompt(s) with base={model}, "
          f"reward={reward_model}")
    extra = create_calibration(
        needed,
        model=model,
        reward_model=reward_model,
        temperature=1.0,
        calibration_seed=0,
        num_samples=num_samples,
    )
    calibration.update(extra)
    save_gzip(calibration, calibration_path)
    return calibration
