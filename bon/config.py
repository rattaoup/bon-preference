"""Centralized filesystem layout.

Every path used by the pipeline is built here so the layout under
``data/`` and ``llm_results/`` is described in exactly one place. The
layout intentionally matches the pre-refactor repository so existing
caches and pre-generated artifacts continue to work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_GENERATOR_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_REWARD_MODEL = "Skywork/Skywork-Reward-V2-Llama-3.1-8B"
DEFAULT_REWARD_MODEL_NAME = "skywork-v2"


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def _dash_slug(model_id: str) -> str:
    return model_id.replace("/", "-")


@dataclass(frozen=True)
class Paths:
    """Paths are expressed relative to ``root`` (the current working dir by default)."""

    root: Path = Path(".")

    # Inputs planned by `bon plan`.
    def requirements(self, name: str, seed: int) -> Path:
        return self.root / f"data/requirements/{name}_seed_{seed}.json"

    def mapping(self, name: str, seed: int) -> Path:
        return self.root / f"data/mappings/{name}_seed_{seed}.json"

    # Generated responses and reward-model scores.
    def responses_dir(self, name: str, seed: int, temperature: float,
                      generator_model: str = DEFAULT_GENERATOR_MODEL) -> Path:
        base = self.root / f"data/llm_responses/{name}/seed_{seed}/temperature_{temperature}"
        if generator_model == DEFAULT_GENERATOR_MODEL:
            return base
        return base / f"model_{_slug(generator_model)}"

    def responses_chunk(self, name: str, seed: int, temperature: float,
                        chunk_idx: int, num_chunks: int,
                        generator_model: str = DEFAULT_GENERATOR_MODEL) -> Path:
        return (self.responses_dir(name, seed, temperature, generator_model)
                / f"responses_chunk_{chunk_idx}_of_{num_chunks}.gz")

    def scores_dir(self, name: str, seed: int, temperature: float,
                   generator_model: str = DEFAULT_GENERATOR_MODEL) -> Path:
        return self.responses_dir(name, seed, temperature, generator_model) / "scores"

    def scores_chunk(self, name: str, seed: int, temperature: float,
                     reward_model_name: str, chunk_idx: int, num_chunks: int,
                     generator_model: str = DEFAULT_GENERATOR_MODEL) -> Path:
        return (self.scores_dir(name, seed, temperature, generator_model)
                / f"model_{reward_model_name}_chunk_{chunk_idx}_of_{num_chunks}.gz")

    # Calibration & rejection-sampling artifacts.
    def calibration_file(self, name: str,
                         generator_model: str = DEFAULT_GENERATOR_MODEL,
                         reward_model: str = DEFAULT_REWARD_MODEL) -> Path:
        gen = generator_model.replace("/", "-")
        rew = reward_model.replace("/", "-")
        return (self.root / f"data/calibrations/{name}/model_{gen}_reward_{rew}"
                / "calibration_file.gzip")

    def rejection_dataset(self, name: str, seed: int, k: int, c: float, w: float, p: float,
                          generator_model: str = DEFAULT_GENERATOR_MODEL,
                          reward_model: str = DEFAULT_REWARD_MODEL) -> Path:
        gen = _slug(generator_model)
        rew = _slug(reward_model)
        return (self.root / f"data/llm_responses/{name}/seed_{seed}"
                / f"model_{gen}_reward_{rew}"
                / f"rejection_sampling_k_{k}_c_{c}_w_{w}_p_{p}.gz")

    def rejection_dir_tag(self, generator_model: str = DEFAULT_GENERATOR_MODEL,
                          reward_model: str = DEFAULT_REWARD_MODEL) -> str:
        return f"model_{_slug(generator_model)}_reward_{_slug(reward_model)}"

    # Embedding caches (training-time artifacts).
    def embeddings_dir(self, dataset_name: str, seed: int, temperature: float,
                       base_model: str, data_type: str) -> Path:
        tag = _dash_slug(base_model)
        suffix = "_west_of_n" if data_type == "west-of-n" else ""
        return (self.root / f"data/llm_responses/{dataset_name}{suffix}"
                / f"seed_{seed}/temperature_{temperature}/embeddings_cache/{tag}")

    def train_embeddings_cache(self, dataset_name: str, seed: int, temperature: float,
                               base_model: str, data_type: str, n: int, k: int,
                               c: float | None = None, w: float | None = None,
                               p: float | None = None) -> Path:
        directory = self.embeddings_dir(dataset_name, seed, temperature, base_model, data_type)
        if data_type in ("standard", "west-of-n"):
            filename = f"pairs_n{n}_k{k}_seed{seed}.pt"
        elif data_type == "rejection_sample":
            filename = f"n_{n}_k{k}_c{c}_w{w}_p{p}_seed{seed}.pt"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        return directory / filename

    def test_set(self, test_dataset_name: str) -> Path:
        return self.root / f"data/test_set/{test_dataset_name}_test.gz"

    def test_embeddings(self, test_dataset_name: str, base_model: str) -> Path:
        tag = _dash_slug(base_model)
        return self.root / f"data/test_set/{test_dataset_name}/{tag}/embeddings.pt"

    # Training sweep caches and result JSONs.
    def sweep_config_cache(self, dataset_name: str, base_model: str, data_type: str,
                           seed: int, n: int, k: int,
                           c: float | None = None, w: float | None = None,
                           p: float | None = None) -> Path:
        tag = _dash_slug(base_model)
        if data_type == "standard":
            directory = self.root / f"llm_results/sweep_configs/dataset_{dataset_name}/backbone_{tag}"
            filename = f"n{n}_k{k}_seed{seed}.json"
        elif data_type == "west-of-n":
            directory = self.root / f"llm_results/sweep_configs/dataset_{dataset_name}_west_of_n/backbone_{tag}"
            filename = f"n{n}_k{k}_seed{seed}.json"
        elif data_type == "rejection_sample":
            directory = self.root / f"llm_results/sweep_configs/dataset_{dataset_name}_rejection_sample/backbone_{tag}"
            filename = f"n{n}_k{k}_c{c}_w{w}_p{p}_seed{seed}.json"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        return directory / filename

    def results_dir(self, dataset_name: str, test_dataset_name: str,
                    base_model: str, data_type: str) -> Path:
        tag = _dash_slug(base_model)
        if data_type == "standard":
            suffix = ""
        elif data_type == "west-of-n":
            suffix = "_west_of_n"
        elif data_type == "rejection_sample":
            suffix = "_rejection_sample"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        return (self.root / f"llm_results/dataset_{dataset_name}{suffix}"
                / f"test_dataset_{test_dataset_name}/backbone_{tag}")

    def result_json(self, dataset_name: str, test_dataset_name: str,
                    base_model: str, data_type: str, seed: int, n: int, k: int,
                    c: float | None = None, w: float | None = None,
                    p: float | None = None) -> Path:
        directory = self.results_dir(dataset_name, test_dataset_name, base_model, data_type)
        if data_type in ("standard", "west-of-n"):
            filename = f"n{n}_k{k}_seed{seed}.json"
        elif data_type == "rejection_sample":
            filename = f"n{n}_k{k}_c{c}_w{w}_p{p}_seed{seed}.json"
        else:
            raise ValueError(f"Invalid data type: {data_type}")
        return directory / filename

    # Misc.
    @property
    def logs(self) -> Path:
        return self.root / "logs"


PATHS = Paths(root=Path(os.environ.get("BON_ROOT", ".")).resolve())
