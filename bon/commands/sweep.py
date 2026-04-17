"""``bon sweep`` - sweep ``bon train`` across (n, k, seed) or (c, w, seed)."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import re
import sys
from pathlib import Path

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL_NAME, PATHS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-name", "--dataset-name",
                        dest="train_name",
                        type=str, default="ultrafeedback-base",
                        help="Experiment tag of the training data "
                             "(matches `--name` from `bon plan/generate/score`). "
                             "`--dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--test-name", "--test-dataset-name",
                        dest="test_name",
                        type=str, default="ultrafeedback",
                        help="File stem at `data/test_set/<test-name>_test.gz`. "
                             "`--test-dataset-name` is accepted as a legacy alias.")
    parser.add_argument("--base-model", type=str, default=DEFAULT_GENERATOR_MODEL)
    parser.add_argument("--reward-model-name", type=str, default=DEFAULT_REWARD_MODEL_NAME)
    parser.add_argument("--data-type", type=str, default="standard",
                        choices=["standard", "west-of-n", "rejection_sample"])
    parser.add_argument("--mapping-name", type=str, default=None)

    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--gpus", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6, 7])
    parser.add_argument("--poll", type=float, default=5.0)

    parser.add_argument("--train-size", "--n",
                        dest="train_size",
                        type=int, nargs="+", default=None,
                        help="train_size values to sweep (default: auto-detect from "
                             "mapping for standard/west-of-n; required for "
                             "rejection_sample). `--n` is accepted as a legacy alias.")
    parser.add_argument("--N", "--k",
                        dest="N",
                        type=int, nargs="+", default=None,
                        help="N (Best-of-N triplet size) values to sweep (default: "
                             "auto-detect for standard/west-of-n; required for "
                             "rejection_sample). `--k` is accepted as a legacy alias.")

    parser.add_argument("--c", type=float, nargs="+", default=[0.15, 0.3, 0.5, 0.7, 0.85],
                        help="rejection_sample only: c values to sweep.")
    parser.add_argument("--w", type=float, nargs="+", default=[0.1, 0.2, 0.3],
                        help="rejection_sample only: w values to sweep.")
    parser.add_argument("--p", type=float, default=0.6,
                        help="rejection_sample only: single p value.")
    parser.add_argument("--temperature", type=float, default=1.0)


def _detect_train_sizes_and_Ns(train_name: str, seed: int) -> tuple[list[int], list[int]]:
    """Parse available ``(train_size, N)`` pairs from the mapping file.

    The regex still matches the legacy ``k_{N}_n_{train_size}_seed_{seed}``
    key format that we keep on disk for backward compat.
    """
    mapping_path = PATHS.mapping(train_name, seed)
    if not mapping_path.exists():
        print(f"Warning: mapping file not found: {mapping_path}")
        return [], []
    with open(mapping_path) as f:
        mapping = json.load(f)
    pattern = re.compile(r"k_(\d+)_n_(\d+)_seed_(\d+)")
    train_sizes, Ns = set(), set()
    for key in mapping:
        m = pattern.match(key)
        if m:
            Ns.add(int(m.group(1)))
            train_sizes.add(int(m.group(2)))
    return sorted(train_sizes), sorted(Ns)


def _train_cmd_standard(args, seed: int, train_size: int, N: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "bon", "train",
        "--train-name", args.train_name,
        "--test-name", args.test_name,
        "--base-model", args.base_model,
        "--reward-model-name", args.reward_model_name,
        "--data-type", args.data_type,
        "--seed", str(seed),
        "--train-size", str(train_size),
        "--N", str(N),
        "--temperature", str(args.temperature),
    ]
    if args.mapping_name:
        cmd += ["--mapping-name", args.mapping_name]
    return cmd


def _train_cmd_rejection(args, seed: int, c: float, w: float,
                          train_size: int, N: int) -> list[str]:
    cmd = [
        sys.executable, "-m", "bon", "train",
        "--train-name", args.train_name,
        "--test-name", args.test_name,
        "--base-model", args.base_model,
        "--reward-model-name", args.reward_model_name,
        "--data-type", "rejection_sample",
        "--seed", str(seed),
        "--train-size", str(train_size),
        "--N", str(N),
        "--c", str(c),
        "--w", str(w),
        "--p", str(args.p),
        "--temperature", str(args.temperature),
    ]
    if args.mapping_name:
        cmd += ["--mapping-name", args.mapping_name]
    return cmd


def _aggregate_standard(args) -> None:
    results_dir = PATHS.results_dir(
        args.train_name, args.test_name, args.base_model, args.data_type)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return
    rows = []
    for f in results_dir.glob("n*_k*_seed*.json"):
        data = json.loads(f.read_text())
        # Result JSONs store ``train_size`` / ``N`` under ``args`` for runs
        # produced after the rename; older runs used ``n`` / ``k``. The CSV
        # columns keep the legacy spelling so downstream analysis scripts
        # don't need to change.
        args_data = data["args"]
        train_size = args_data.get("train_size", args_data.get("n"))
        N = args_data.get("N", args_data.get("k"))
        rows.append({
            "n": train_size,
            "k": N,
            "seed": args_data["seed"],
            "test_accuracy": data["test_accuracy"],
            "mean_val_loss": data["best_cv"]["mean_val_loss"],
            "std_val_loss": data["best_cv"]["std_val_loss"],
            "learning_rate": data["best_cfg"]["learning_rate"],
            "weight_decay": data["best_cfg"]["weight_decay"],
        })
    if not rows:
        print("No results to aggregate yet.")
        return
    rows.sort(key=lambda r: (r["n"], r["k"], r["seed"]))
    out_path = results_dir / f"aggregated_final_accuracy_results_{args.train_name}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Aggregated results saved to {out_path}")


def run(args: argparse.Namespace) -> int:
    from bon.parallel import Job, run_parallel

    if args.data_type in ("standard", "west-of-n"):
        if args.train_size is None or args.N is None:
            avail_train_sizes, avail_Ns = _detect_train_sizes_and_Ns(
                args.train_name, args.seeds[0],
            )
            args.train_size = args.train_size or avail_train_sizes
            args.N = args.N or avail_Ns
        if not args.train_size or not args.N:
            raise SystemExit(
                "Could not determine --train-size / --N. "
                "Pass them explicitly or run `bon plan` first."
            )

        jobs: list[Job] = []
        for seed, train_size, N in itertools.product(
            args.seeds, args.train_size, args.N,
        ):
            jobs.append(Job(
                job_id=f"train_s{seed}_n{train_size}_k{N}",
                payload={"seed": seed, "train_size": train_size, "N": N},
                command=_train_cmd_standard(args, seed, train_size, N),
            ))
    elif args.data_type == "rejection_sample":
        if args.train_size is None or args.N is None:
            raise SystemExit(
                "--train-size and --N are required for rejection_sample sweeps."
            )
        train_size = args.train_size[0]
        N = args.N[0]
        jobs = []
        for seed, c, w in itertools.product(args.seeds, args.c, args.w):
            jobs.append(Job(
                job_id=f"train_s{seed}_c{c}_w{w}",
                payload={"seed": seed, "c": c, "w": w},
                command=_train_cmd_rejection(args, seed, c, w, train_size, N),
            ))
    else:
        raise ValueError(f"Invalid data type: {args.data_type}")

    print(f"Running {len(jobs)} training jobs across {len(args.gpus)} GPUs ({args.gpus})")
    rc = run_parallel(jobs, gpus=args.gpus, poll=args.poll, log_dir=PATHS.logs)

    if args.data_type in ("standard", "west-of-n"):
        _aggregate_standard(args)
    return rc
