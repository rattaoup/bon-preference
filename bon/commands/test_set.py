"""``bon build-test-set`` - build a binarized reward-scored test set."""

from __future__ import annotations

import argparse

from bon.config import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_REWARD_MODEL,
    DEFAULT_REWARD_MODEL_NAME,
    PATHS,
)
from bon.io import save_gzip


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--source-dataset", "--dataset",
                        dest="source_dataset",
                        type=str, required=True,
                        choices=["ultrafeedback", "pku-saferlhf", "gsm8k"],
                        help="Which test set to build. "
                             "`--dataset` is accepted as a legacy alias.")
    parser.add_argument("--reward-model", type=str, default=DEFAULT_REWARD_MODEL,
                        help="Full HF id of the reward model used to score test pairs. "
                             "The resulting `data/test_set/<test-name>_test.gz` is specific "
                             "to this reward model - switch reward models, rebuild.")
    parser.add_argument("--reward-model-name", type=str, default=DEFAULT_REWARD_MODEL_NAME,
                        help="Short alias of the reward model (GSM8K only - used to "
                             "locate the pre-generated scores chunk files).")
    parser.add_argument("--base-model", "--generator-model",
                        dest="base_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="GSM8K only: base/generator model the test-set responses were "
                             "produced with (must match the --base-model passed to "
                             "bon generate --name <response-name>). Ignored for "
                             "ultrafeedback / pku-saferlhf. `--generator-model` is accepted "
                             "as a legacy alias.")
    parser.add_argument("--test-name", "--output-name",
                        dest="test_name",
                        type=str, default=None,
                        help="Stem for the output file under data/test_set/<test-name>_test.gz. "
                             "Defaults to a dataset-specific value (see the README). "
                             "`--output-name` is accepted as a legacy alias.")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for GSM8K pair selection.")
    parser.add_argument("--response-name", type=str, default="gsm8k_test_set",
                        help="GSM8K only: `name` under which responses/scores are stored.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap the number of examples scored (useful for the demo).")


def run(args: argparse.Namespace) -> int:
    from bon.test_set import build_test_set

    pairs = build_test_set(
        args.source_dataset,
        reward_model=args.reward_model,
        base_model=args.base_model,
        seed=args.seed,
        response_name=args.response_name,
        temperature=args.temperature,
        num_chunks=args.num_chunks,
        reward_model_name=args.reward_model_name,
        limit=args.limit,
    )

    test_name = args.test_name or {
        "ultrafeedback": "ultrafeedback",
        "pku-saferlhf": "pku_saferlhf",
        "gsm8k": "gsm8k_medium_difficulty_final",
    }[args.source_dataset]
    out_path = PATHS.test_set(test_name)
    save_gzip(pairs, out_path)
    suffix = (
        f" (base_model={args.base_model})" if args.source_dataset == "gsm8k" else ""
    )
    print(
        f"Wrote {len(pairs)} pairs to {out_path} "
        f"(reward_model={args.reward_model}){suffix}"
    )
    return 0
