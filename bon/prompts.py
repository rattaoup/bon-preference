"""Prompt loaders for the supported datasets.

The three datasets we use for the real-world experiments are:

* ``ultrafeedback`` - ``HuggingFaceH4/ultrafeedback_binarized`` train prompts
* ``gsm8k`` - ``openai/gsm8k`` train split, ``question`` column
* ``pku-saferlhf`` - ``PKU-Alignment/PKU-SafeRLHF``, filtered to examples
  with a non-ambiguous safer-response label
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
from datasets import Dataset, load_dataset


SUPPORTED_DATASETS = ("ultrafeedback", "gsm8k", "pku-saferlhf")


def _pku_to_messages(example: dict) -> dict:
    prompt_messages = [{"role": "user", "content": example["prompt"]}]
    return {
        "prompt": example["prompt"],
        "chosen": prompt_messages + [{"role": "assistant", "content": example["chosen"]}],
        "rejected": prompt_messages + [{"role": "assistant", "content": example["rejected"]}],
    }


def load_pku_saferlhf(split: str = "train") -> Dataset:
    """Return the PKU-SafeRLHF split as a messages-format Dataset.

    Examples with ``safer_response_id`` outside {0, 1} are dropped.
    """
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split=split)
    converted = []
    for example in ds:
        safer_id = example["safer_response_id"]
        if safer_id not in (0, 1):
            continue
        if safer_id == 0:
            chosen, rejected = example["response_0"], example["response_1"]
        else:
            chosen, rejected = example["response_1"], example["response_0"]
        converted.append(_pku_to_messages({
            "prompt": example["prompt"],
            "chosen": chosen,
            "rejected": rejected,
        }))
    return Dataset.from_list(converted)


def get_train_prompts(dataset_name: str) -> List[str]:
    """Return the list of training prompts for ``dataset_name``."""
    if dataset_name == "ultrafeedback":
        ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized")
        return ds["train_prefs"]["prompt"]
    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        return ds["question"]
    if dataset_name == "pku-saferlhf":
        ds = load_pku_saferlhf(split="train")
        return ds["prompt"]
    raise ValueError(
        f"Dataset {dataset_name!r} not supported. Choose from {SUPPORTED_DATASETS}."
    )


@dataclass(frozen=True)
class ExperimentConfig:
    """A single (N, train_size, seed) configuration within a sweep.

    ``N`` is the Best-of-N per-prompt triplet size (called ``k`` in earlier
    revisions) and ``train_size`` is the number of prompts drawn (called
    ``n`` in earlier revisions). The old names are still used in on-disk
    mapping keys / cached filenames for backward compat with pre-generated
    artifacts.
    """

    N: int
    train_size: int
    seed: int


def sample_prompts(config: ExperimentConfig, train_prompts: List[str]) -> List[dict]:
    """Draw ``config.train_size`` prompt ids with replacement using ``config.seed``."""
    g = torch.Generator().manual_seed(config.seed)
    prompt_ids = torch.randint(
        0, len(train_prompts), (config.train_size,), generator=g,
    ).tolist()
    return [{"prompt_id": i, "prompt": train_prompts[i]} for i in prompt_ids]
