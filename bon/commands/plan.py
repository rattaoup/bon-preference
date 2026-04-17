"""``bon plan`` - plan the requirements and mappings for a sweep."""

from __future__ import annotations

import argparse

from bon.config import PATHS
from bon.io import save_json

SUPPORTED_DATASETS = ("ultrafeedback", "gsm8k", "pku-saferlhf")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dataset", "--dataset",
                        dest="source_dataset",
                        type=str, required=True,
                        choices=SUPPORTED_DATASETS,
                        help="Source HF dataset that supplies prompts "
                             "(see bon.prompts.get_train_prompts). "
                             "`--dataset` is accepted as a legacy alias.")
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name; used as the tag in output filenames. "
                             "Reused across plan/generate/score/calibrate/rejection/train - "
                             "pick one value and stick with it.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--N", "--k",
                        dest="N",
                        type=int, nargs="+", default=[2, 4, 8, 16],
                        help="Best-of-N triplet size(s). "
                             "`--k` is accepted as a legacy alias.")
    parser.add_argument("--train-size", "--n",
                        dest="train_size",
                        type=int, nargs="+", default=[32, 128, 512, 2048, 8192],
                        help="Training-set size(s) (number of prompts sampled per config). "
                             "`--n` is accepted as a legacy alias.")


def run(args: argparse.Namespace) -> int:
    from bon.mapping import build_sweep_configs, plan_requirements
    from bon.prompts import get_train_prompts

    train_prompts = get_train_prompts(args.source_dataset)
    print(f"Loaded {len(train_prompts)} prompts from {args.source_dataset}")

    for seed in args.seeds:
        configs = build_sweep_configs(
            seeds=[seed], Ns=args.N, train_sizes=args.train_size,
        )
        requirements, mappings = plan_requirements(configs, train_prompts)

        req_path = PATHS.requirements(args.name, seed)
        map_path = PATHS.mapping(args.name, seed)
        save_json(requirements, str(req_path))
        save_json(mappings, str(map_path))
        total = sum(r["num_responses"] for r in requirements)
        print(f"[seed {seed}] wrote {total} total responses across "
              f"{len(mappings)} configs to {req_path.name}")
    return 0
