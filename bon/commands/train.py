"""``bon train`` - build / load train embeddings, run the CV sweep, evaluate on test."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bon.config import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_REWARD_MODEL,
    DEFAULT_REWARD_MODEL_NAME,
    PATHS,
)
from bon.io import load_gzip
from bon.seeding import set_seed


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-name", "--dataset-name",
                        dest="train_name",
                        type=str, default="ultrafeedback-base",
                        help="Experiment tag of the training data "
                             "(matches `--name` passed to `bon plan/generate/score`). "
                             "`--dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--test-name", "--test-dataset-name",
                        dest="test_name",
                        type=str, default="ultrafeedback",
                        help="File stem at `data/test_set/<test-name>_test.gz`. "
                             "`--test-dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--reward-model-name", type=str, default=DEFAULT_REWARD_MODEL_NAME,
                        help="Reward-model alias; must match the value used with bon score.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--train-size", "--n",
                        dest="train_size",
                        type=int, default=2048,
                        help="Training-set size (number of prompts). "
                             "`--n` is accepted as a legacy alias.")
    parser.add_argument("--N", "--k",
                        dest="N",
                        type=int, default=8,
                        help="Best-of-N triplet size. "
                             "`--k` is accepted as a legacy alias.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="Backbone used to embed train/test pairs. "
                             "Must match the base model used with bon generate.")
    parser.add_argument("--data-type", type=str, default="standard",
                        choices=["standard", "west-of-n", "rejection_sample"])
    parser.add_argument("--mapping-name", type=str, default=None,
                        help="Override train-name when looking up the mapping "
                             "(useful when the same mapping is reused across backbones).")

    parser.add_argument("--c", type=float, default=0.5,
                        help="rejection_sample only: percentile centre "
                             "(positive upsamples the band, negative downsamples it).")
    parser.add_argument("--w", type=float, default=1.0,
                        help="rejection_sample only: percentile-band width (>=0).")
    parser.add_argument("--p", type=float, default=0.9,
                        help="rejection_sample only: reshaping strength.")
    parser.add_argument("--generator-model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="rejection_sample only: generator used to build the dataset.")
    parser.add_argument("--reward-model-path", type=str, default=DEFAULT_REWARD_MODEL,
                        help="rejection_sample only: reward model used to score the dataset.")

    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="reward-sweeps")
    parser.add_argument("--run-name", type=str, default=None)

    parser.add_argument("--force-recache", action="store_true",
                        help="Recompute cached embeddings.")
    parser.add_argument("--force-sweep", action="store_true",
                        help="Re-run the CV sweep even if a cached config exists.")


def _serializable_args(args: argparse.Namespace) -> dict:
    """Return ``vars(args)`` with non-JSON-safe entries stripped.

    ``bon.cli`` stashes the subcommand's ``run`` function on the namespace
    under ``_run``; dropping dunder/underscore keys and callables keeps the
    result JSON-friendly while preserving every user-facing flag.
    """
    out = {}
    for k, v in vars(args).items():
        if k.startswith("_") or callable(v):
            continue
        out[k] = v
    return out


def _load_rejection_sample(
    train_name: str,
    seed: int,
    N: int,
    c: float,
    w: float,
    p: float,
    generator_model: str = DEFAULT_GENERATOR_MODEL,
    reward_model: str = DEFAULT_REWARD_MODEL,
):
    # Route through ``PATHS`` so BON_ROOT is honored; ``.gzip`` is an
    # accepted legacy extension for pre-generated rejection datasets.
    primary = PATHS.rejection_dataset(
        train_name, seed, N, c, w, p,
        generator_model=generator_model, reward_model=reward_model,
    )
    candidates = [primary, primary.with_suffix(".gzip")]
    for path in candidates:
        if path.exists():
            return load_gzip(path)
    raise FileNotFoundError(
        f"Rejection-sample dataset not found: {primary}"
    )


def _build_train_dataset(args) -> list:
    from bon.mapping import (
        load_mapping,
        map_dataset,
        map_rejection_dataset,
        merge_responses_and_scores,
    )

    mapping_name = args.mapping_name or args.train_name
    mapping = load_mapping(args.train_name, args.seed, mapping_name=mapping_name)

    if args.data_type in ("standard", "west-of-n"):
        data = merge_responses_and_scores(
            name=args.train_name,
            reward_model_name=args.reward_model_name,
            seed=args.seed,
            temperature=args.temperature,
            num_chunks=args.num_chunks,
            generator_model=args.base_model,
        )
        return map_dataset(
            data, mapping,
            train_size=args.train_size, seed=args.seed, N=args.N,
            west_of_n=(args.data_type == "west-of-n"),
        )

    if args.data_type == "rejection_sample":
        data = _load_rejection_sample(
            args.train_name, args.seed,
            args.N, args.c, args.w, args.p,
            generator_model=args.generator_model,
            reward_model=args.reward_model_path,
        )
        data = {int(k): v for k, v in data.items()}
        return map_rejection_dataset(
            data, mapping, seed=args.seed, train_size=args.train_size,
        )

    raise ValueError(f"Invalid data type: {args.data_type}")


def _load_or_build_train_embeddings(args):
    import torch
    from bon.embeddings import get_pairwise_embeddings, load_backbone_and_tokenizer

    # ``config.py`` still uses legacy keyword names (n, k) so that on-disk
    # filenames keep the pre-rename layout for cache reuse.
    cache_path = PATHS.train_embeddings_cache(
        args.train_name, args.seed, args.temperature, args.base_model,
        args.data_type,
        n=args.train_size, k=args.N,
        c=args.c, w=args.w, p=args.p,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Embeddings cache: {cache_path}")

    if args.force_recache and cache_path.exists():
        print(f"Force recache: removing {cache_path}")
        cache_path.unlink()

    if cache_path.exists():
        print(f"Cache hit, loading embeddings from {cache_path}")
        return torch.load(cache_path)

    print("Cache miss, building train dataset + embeddings")
    dataset = _build_train_dataset(args)
    backbone, tokenizer = load_backbone_and_tokenizer(args.base_model)
    embeddings = get_pairwise_embeddings(backbone, tokenizer, dataset)
    torch.save(embeddings, cache_path)
    return embeddings


def _load_or_build_test_embeddings(args):
    import torch
    from bon.embeddings import get_pairwise_embeddings, load_backbone_and_tokenizer

    test_path = PATHS.test_set(args.test_name)
    try:
        test_set = load_gzip(test_path)
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Test set not found at {test_path}. "
            "Run `bon build-test-set` first."
        ) from e

    score_diffs = [ex["score_chosen"] - ex["score_rejected"] for ex in test_set]
    cache_path = PATHS.test_embeddings(args.test_name, args.base_model)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cached = torch.load(cache_path)
        if len(cached) == len(test_set):
            print("Test embeddings cache hit")
            return cached, score_diffs
        print(
            f"Test embeddings cache stale "
            f"(cached={len(cached)} pairs, current test set={len(test_set)} pairs); "
            f"rebuilding."
        )

    print("Test embeddings cache miss, embedding")
    backbone, tokenizer = load_backbone_and_tokenizer(args.base_model)
    embeddings = get_pairwise_embeddings(backbone, tokenizer, test_set)
    torch.save(embeddings, cache_path)
    return embeddings, score_diffs


def run(args: argparse.Namespace) -> int:
    from bon.training import (
        DEFAULT_SWEEP_SPACE,
        accuracy_on_hard_pairs,
        eval_accuracy,
        resolve_device,
        sweep_pick_and_train_final,
        train_one_fold,
    )

    set_seed(args.seed)

    wandb_run = None
    if args.use_wandb:
        import wandb  # imported lazily so wandb is optional
        wandb_run = wandb.init(project=args.wandb_project, name=args.run_name)
        wandb.config.update(_serializable_args(args), allow_val_change=True)

    train_embeddings = _load_or_build_train_embeddings(args)

    sweep_space = {**DEFAULT_SWEEP_SPACE, "device": resolve_device()}
    sweep_cfg_path: Path = PATHS.sweep_config_cache(
        args.train_name, args.base_model, args.data_type,
        args.seed,
        n=args.train_size, k=args.N,
        c=args.c, w=args.w, p=args.p,
    )
    sweep_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    if not args.force_sweep and sweep_cfg_path.exists():
        print(f"Loading cached sweep config from {sweep_cfg_path}")
        cached = json.loads(sweep_cfg_path.read_text())
        best_cfg = {**cached["best_cfg"], "device": resolve_device(),
                    "max_epochs": cached["final_epochs"]}
        print(f"Training final model with cached config: {best_cfg}")
        final = train_one_fold(train_embeddings, None, best_cfg)
        sweep_result = {
            "model": final["model"],
            "best_cfg": cached["best_cfg"],
            "final_epochs": cached["final_epochs"],
            "best_cv": cached["best_cv"],
            "all_cv": None,
        }
    else:
        if not args.force_sweep:
            print(f"No cached config at {sweep_cfg_path}, running full sweep")
        sweep_result = sweep_pick_and_train_final(
            pairs=train_embeddings, sweep_space=sweep_space, k_folds=5, seed=args.seed,
        )
        sweep_cfg_path.write_text(json.dumps({
            "best_cfg": sweep_result["best_cfg"],
            "final_epochs": sweep_result["final_epochs"],
            "best_cv": {
                "mean_val_loss": sweep_result["best_cv"]["mean_val_loss"],
                "std_val_loss": sweep_result["best_cv"]["std_val_loss"],
            },
        }, indent=2))
        print(f"Sweep config cached to {sweep_cfg_path}")

    model = sweep_result["model"]

    test_embeddings, score_diffs = _load_or_build_test_embeddings(args)
    test_accuracy, correct = eval_accuracy(model, test_embeddings)
    hard_acc = accuracy_on_hard_pairs(correct, score_diffs, 0.3)
    very_hard_acc = accuracy_on_hard_pairs(correct, score_diffs, 0.1)
    print(f"Test accuracy:            {100 * test_accuracy:.3f}")
    print(f"Hard pairs accuracy:      {100 * hard_acc:.3f}")
    print(f"Very hard pairs accuracy: {100 * very_hard_acc:.3f}")

    result_path: Path = PATHS.result_json(
        args.train_name, args.test_name, args.base_model,
        args.data_type, args.seed,
        n=args.train_size, k=args.N,
        c=args.c, w=args.w, p=args.p,
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_payload = {
        "args": _serializable_args(args),
        "test_accuracy": float(test_accuracy),
        "hard_accuracy": float(hard_acc),
        "very_hard_accuracy": float(very_hard_acc),
        "best_cfg": sweep_result["best_cfg"],
        "best_cv": {
            "mean_val_loss": sweep_result["best_cv"]["mean_val_loss"],
            "std_val_loss": sweep_result["best_cv"]["std_val_loss"],
        },
    }
    result_path.write_text(json.dumps(result_payload, indent=2))
    print(f"Results saved to {result_path}")

    if wandb_run is not None:
        wandb_run.log({
            "test_accuracy": float(test_accuracy),
            "hard_accuracy": float(hard_acc),
            "very_hard_accuracy": float(very_hard_acc),
        })
        wandb_run.finish()
    return 0
