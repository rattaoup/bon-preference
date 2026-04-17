"""Requirements planner + prompt-to-response mapping utilities.

The planner turns a set of ``(N, train_size, seed)`` configurations into
two files:

* A ``requirements`` list: for each original prompt index, how many
  responses to generate.
* A ``mappings`` dict keyed by ``k_{N}_n_{train_size}_seed_{seed}``: for
  each sampled triplet, which slice of that prompt's response pool to use.

The on-disk mapping keys keep the legacy ``k_{}_n_{}_seed_{}`` spelling
so that pre-generated artifacts continue to load. In Python, ``N``
replaces ``k`` (Best-of-N triplet size, per the paper) and
``train_size`` replaces ``n`` (number of sampled prompts).

``map_dataset`` / ``map_rejection_dataset`` consume the ``mappings``
dict together with the (response, score) pool to build the final
preference pairs used for training.
"""

from __future__ import annotations

import os
import random
from itertools import product
from typing import Dict, Iterable, List, Tuple

import torch

from bon.config import PATHS
from bon.io import load_gzip, load_json
from bon.prompts import ExperimentConfig


def plan_requirements(
    configs: Iterable[ExperimentConfig],
    train_prompts: List[str],
) -> Tuple[List[dict], Dict[str, List[dict]]]:
    """Compute per-prompt response counts and per-config slice mappings."""
    requirements = [
        {"original_idx": i, "prompt": train_prompts[i], "num_responses": 0}
        for i in range(len(train_prompts))
    ]
    mappings: Dict[str, List[dict]] = {}
    for config in configs:
        g = torch.Generator().manual_seed(config.seed)
        prompt_ids = torch.randint(
            0, len(train_prompts), (config.train_size,), generator=g,
        ).tolist()
        current_mapping = []
        for prompt_id in prompt_ids:
            requirements[prompt_id]["num_responses"] += config.N
            current_mapping.append({
                "original_idx": prompt_id,
                "response_start": requirements[prompt_id]["num_responses"] - config.N,
                "response_end": requirements[prompt_id]["num_responses"],
            })
        # Key spelling is legacy (k_/n_) to stay compatible with
        # pre-generated artifacts; in Python these are N / train_size.
        key = f"k_{config.N}_n_{config.train_size}_seed_{config.seed}"
        mappings[key] = current_mapping
    return requirements, mappings


def build_sweep_configs(
    seeds: Iterable[int],
    Ns: Iterable[int],
    train_sizes: Iterable[int],
) -> List[ExperimentConfig]:
    return [
        ExperimentConfig(N=N, train_size=train_size, seed=seed)
        for seed, N, train_size in product(seeds, Ns, train_sizes)
    ]


def load_mapping(name: str, seed: int, mapping_name: str | None = None) -> Dict[str, List[dict]]:
    """Load a mapping file by name.

    Pass ``mapping_name`` explicitly when a dataset was generated with the
    default (Llama) backbone but re-scored against another model, so that
    the responses live under a different ``name`` than the mapping.
    """
    lookup = mapping_name if mapping_name is not None else name
    path = PATHS.mapping(lookup, seed)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Mapping file not found: {path}")
    return load_json(path)


def merge_responses_and_scores(
    name: str,
    reward_model_name: str,
    seed: int,
    temperature: float,
    num_chunks: int,
    generator_model: str | None = None,
) -> Dict[int, dict]:
    """Load the chunked responses + scores for one (name, seed) pair."""
    from bon.config import DEFAULT_GENERATOR_MODEL
    generator_model = generator_model or DEFAULT_GENERATOR_MODEL

    data: Dict[int, dict] = {}
    for chunk_idx in range(num_chunks):
        resp_path = PATHS.responses_chunk(
            name, seed, temperature, chunk_idx, num_chunks, generator_model)
        score_path = PATHS.scores_chunk(
            name, seed, temperature, reward_model_name,
            chunk_idx, num_chunks, generator_model)
        if not os.path.exists(resp_path):
            raise FileNotFoundError(f"Missing responses chunk: {resp_path}")
        if not os.path.exists(score_path):
            raise FileNotFoundError(f"Missing scores chunk: {score_path}")

        responses = load_gzip(resp_path)
        scores = load_gzip(score_path)
        for resp, score in zip(responses, scores):
            assert resp["original_idx"] == score["original_idx"], (
                f"responses/scores desync at chunk {chunk_idx}")
            og_idx = resp["original_idx"]
            if og_idx in data:
                raise ValueError(f"Duplicate original_idx {og_idx} across chunks")
            data[og_idx] = {
                "prompt": resp["prompt"],
                "response": resp["responses"],
                "score": score["scores"],
            }
    return data


def _sample_negative_index(rng: random.Random, num_responses: int, positive_idx: int) -> int:
    neg = rng.randint(0, num_responses - 1)
    while neg == positive_idx:
        neg = rng.randint(0, num_responses - 1)
    return neg


def map_dataset(
    data: Dict[int, dict],
    mapping: Dict[str, List[dict]],
    train_size: int,
    seed: int,
    N: int,
    west_of_n: bool = False,
) -> List[dict]:
    """Build training pairs for one ``(N, train_size, seed)`` configuration.

    When ``west_of_n`` is True, the negative is the lowest-scoring
    response in the slice; otherwise it is sampled uniformly at random
    (Best-vs-Random).
    """
    key = f"k_{N}_n_{train_size}_seed_{seed}"
    rng = random.Random(seed)
    pairs: List[dict] = []
    for entry in mapping[key]:
        og_idx = entry["original_idx"]
        start, end = entry["response_start"], entry["response_end"]
        cur = data[og_idx]
        prompt = cur["prompt"]
        responses = cur["response"][start:end]
        scores = cur["score"][start:end]

        positive_idx = scores.index(max(scores))
        positive = responses[positive_idx]
        if west_of_n:
            negative_idx = scores.index(min(scores))
        else:
            negative_idx = _sample_negative_index(rng, len(responses), positive_idx)
        negative = responses[negative_idx]

        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "user", "content": prompt},
                       {"role": "assistant", "content": positive}],
            "rejected": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": negative}],
        })
    return pairs


def map_rejection_dataset(
    data: Dict[int, dict],
    mapping: Dict[str, List[dict]],
    seed: int,
    train_size: int | None = None,
    west_of_n: bool = False,
) -> List[dict]:
    """Build training pairs from a rejection-sampling dataset.

    Rejection-sampling datasets have exactly one key in the mapping (since
    they fix N upfront) and the responses already live under ``data``.
    """
    assert len(mapping) == 1, "rejection-sample mapping must have a single config"
    key = next(iter(mapping.keys()))
    entries = mapping[key]
    if train_size is not None:
        assert train_size <= len(entries), (
            f"train_size={train_size} exceeds number of entries in mapping ({len(entries)})"
        )
        entries = entries[:train_size]

    rng = random.Random(seed)
    pairs: List[dict] = []
    for entry in entries:
        og_idx = entry["original_idx"]
        cur = data[og_idx]
        prompt = cur["prompt"]
        responses = cur["responses"]
        scores = cur["scores"]

        positive_idx = scores.index(max(scores))
        positive = responses[positive_idx]
        if west_of_n:
            negative_idx = scores.index(min(scores))
        else:
            negative_idx = _sample_negative_index(rng, len(responses), positive_idx)
        negative = responses[negative_idx]

        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "user", "content": prompt},
                       {"role": "assistant", "content": positive}],
            "rejected": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": negative}],
            "chosen_score": scores[positive_idx],
            "rejected_score": scores[negative_idx],
        })
    return pairs
