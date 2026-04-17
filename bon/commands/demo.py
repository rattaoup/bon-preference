"""``bon demo`` - tiny end-to-end smoke test.

Runs the full plan -> generate -> score -> build-test-set -> train
pipeline with a pinned tiny configuration isolated under ``data/demo/``
(controlled via the ``BON_ROOT`` env var that :mod:`bon.config` reads).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


NAME = "demo-ultrafeedback"
DATASET = "ultrafeedback"
SEED = 0
N_VAL = 2           # Best-of-N triplet size (was K).
TRAIN_SIZE = 32     # Number of sampled prompts (was N).
NUM_CHUNKS = 1
TEMPERATURE = 1.0
TEST_LIMIT = 64


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=Path("data/demo"),
                        help="Demo output root (BON_ROOT for subcommands).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the command plan without executing.")
    parser.add_argument("--clean", action="store_true",
                        help="Remove --root before running.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip steps whose main output already exists.")


def _run(cmd: Sequence[str], env: dict) -> None:
    print(f"\n$ BON_ROOT={env['BON_ROOT']} {' '.join(cmd)}")
    result = subprocess.run(list(cmd), env=env)
    if result.returncode != 0:
        raise SystemExit(f"Step failed (rc={result.returncode}): {' '.join(cmd)}")


def _cmd(subcmd: str, *args: str) -> list[str]:
    return [sys.executable, "-m", "bon", subcmd, *args]


def run(args: argparse.Namespace) -> int:
    root: Path = args.root
    if args.clean and root.exists():
        print(f"Removing {root}")
        shutil.rmtree(root)
    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["BON_ROOT"] = str(root.resolve())

    steps: list[tuple[str, Path, list[str]]] = [
        ("plan",
         root / f"data/requirements/{NAME}_seed_{SEED}.json",
         _cmd("plan",
              "--source-dataset", DATASET,
              "--name", NAME,
              "--seeds", str(SEED),
              "--N", str(N_VAL),
              "--train-size", str(TRAIN_SIZE))),
        ("generate",
         root / f"data/llm_responses/{NAME}/seed_{SEED}/temperature_{TEMPERATURE}"
              / f"responses_chunk_0_of_{NUM_CHUNKS}.gz",
         _cmd("generate",
              "--name", NAME,
              "--seed", str(SEED),
              "--chunk-idx", "0",
              "--num-chunks", str(NUM_CHUNKS),
              "--temperature", str(TEMPERATURE))),
        ("score",
         root / f"data/llm_responses/{NAME}/seed_{SEED}/temperature_{TEMPERATURE}"
              / f"scores/model_skywork-v2_chunk_0_of_{NUM_CHUNKS}.gz",
         _cmd("score",
              "--name", NAME,
              "--seed", str(SEED),
              "--chunk-idx", "0",
              "--num-chunks", str(NUM_CHUNKS),
              "--temperature", str(TEMPERATURE))),
        ("build-test-set",
         root / "data/test_set/ultrafeedback_test.gz",
         _cmd("build-test-set",
              "--source-dataset", "ultrafeedback",
              "--limit", str(TEST_LIMIT))),
        ("train",
         None,  # result path depends on backbone slug; always re-run for demo
         _cmd("train",
              "--train-name", NAME,
              "--test-name", "ultrafeedback",
              "--seed", str(SEED),
              "--train-size", str(TRAIN_SIZE),
              "--N", str(N_VAL),
              "--num-chunks", str(NUM_CHUNKS),
              "--temperature", str(TEMPERATURE))),
    ]

    if args.dry_run:
        print(f"[dry-run] BON_ROOT={env['BON_ROOT']}")
        for name, out, cmd in steps:
            suffix = f"  -> {out}" if out else ""
            print(f"[dry-run] {name}: {' '.join(cmd)}{suffix}")
        return 0

    for name, out, cmd in steps:
        if args.skip_existing and out is not None and out.exists():
            print(f"\n[skip] {name}: {out} already exists")
            continue
        _run(cmd, env)

    latest = sorted((root / "llm_results").rglob("n*_k*_seed*.json"),
                    key=lambda p: p.stat().st_mtime, reverse=True)
    if not latest:
        print("\nDemo finished, but no result JSON was found under llm_results/.")
        return 0
    result_path = latest[0]
    payload = json.loads(result_path.read_text())
    print("\n================ Demo complete ================")
    print(f"Result JSON:              {result_path}")
    print(f"Test accuracy:            {100 * payload['test_accuracy']:.3f}")
    print(f"Hard pairs accuracy:      {100 * payload['hard_accuracy']:.3f}")
    print(f"Very hard pairs accuracy: {100 * payload['very_hard_accuracy']:.3f}")
    print("===============================================")
    return 0
