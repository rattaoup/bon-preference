"""``bon connectivity`` - compute the connectivity degree from cached embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL_NAME


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-name", "--dataset-name",
                        dest="train_name",
                        type=str, required=True,
                        help="Experiment tag of the training data. "
                             "`--dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--test-name", "--test-dataset-name",
                        dest="test_name",
                        type=str, required=True,
                        help="File stem at `data/test_set/<test-name>_test.gz`. "
                             "`--test-dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--reward-model-name", type=str, default=DEFAULT_REWARD_MODEL_NAME)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--N", "--k",
                        dest="N",
                        type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Best-of-N triplet size(s) to evaluate. "
                             "`--k` is accepted as a legacy alias.")
    parser.add_argument("--train-size", "--n",
                        dest="train_size",
                        type=int, required=True,
                        help="Training-set size (single value). "
                             "`--n` is accepted as a legacy alias.")
    parser.add_argument("--eps", type=float, default=1e-5)
    parser.add_argument("--output-path", type=Path, default=None,
                        help="Optional JSON file to save per-seed connectivity lists.")


def _connectivity_for_config(args, seed: int, N: int) -> float:
    import torch
    from bon.commands.train import _load_or_build_test_embeddings, _load_or_build_train_embeddings
    from bon.connectivity import compute_sigma, smallest_generalized_eigenvalue

    train_args = SimpleNamespace(
        train_name=args.train_name,
        test_name=args.test_name,
        reward_model_name=args.reward_model_name,
        seed=seed,
        temperature=args.temperature,
        num_chunks=args.num_chunks,
        train_size=args.train_size,
        N=N,
        base_model=args.base_model,
        data_type="standard",
        mapping_name=None,
        c=0.5, w=1.0, p=0.9,
        generator_model=DEFAULT_GENERATOR_MODEL,
        reward_model_path=None,  # unused for standard
        force_recache=False,
    )
    with torch.no_grad():
        train_emb = _load_or_build_train_embeddings(train_args)
        test_emb, _ = _load_or_build_test_embeddings(train_args)
        sigma_train = compute_sigma(train_emb)
        sigma_test = compute_sigma(test_emb)
        return smallest_generalized_eigenvalue(sigma_test, sigma_train, eps=args.eps).item()


def run(args: argparse.Namespace) -> int:
    connectivity: dict[int, list[float]] = {}
    for seed in args.seeds:
        values: list[float] = []
        for N in args.N:
            val = _connectivity_for_config(args, seed=seed, N=N)
            print(
                f"[seed={seed}, N={N}, train_size={args.train_size}] "
                f"connectivity = {val:.6e}"
            )
            values.append(val)
        connectivity[seed] = values

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        args.output_path.write_text(json.dumps({
            "train_name": args.train_name,
            "test_name": args.test_name,
            "base_model": args.base_model,
            "train_size": args.train_size,
            "N": args.N,
            "seeds": args.seeds,
            "connectivity": {str(s): connectivity[s] for s in args.seeds},
        }, indent=2))
        print(f"Saved connectivity to {args.output_path}")
    return 0
