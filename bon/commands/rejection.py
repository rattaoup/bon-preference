"""``bon rejection`` - build the up/downsampled rejection-sampling training dataset."""

from __future__ import annotations

import argparse
import os

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL, PATHS
from bon.io import load_json, save_gzip


MAX_ROUNDS = 20


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, default="ultrafeedback-pminus",
                        help="Experiment name; must match the --name used with "
                             "bon plan/generate/score/calibrate.")
    parser.add_argument("--base-model", "--generator-model",
                        dest="base_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="Base/generator model used to sample responses. Must match "
                             "the base model used for the calibration at the same "
                             "(name, seed). `--generator-model` is accepted as a legacy alias.")
    parser.add_argument("--reward-model", type=str, default=DEFAULT_REWARD_MODEL,
                        help="Full HF id of the reward model used to score responses. "
                             "Must match the reward model used for the calibration.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--c", type=float, default=0.5,
                        help="Percentile centre (positive upsamples the band, negative "
                             "downsamples it).")
    parser.add_argument("--w", type=float, default=1.0,
                        help="Percentile band width (always >= 0).")
    parser.add_argument("--N", "--k",
                        dest="N", type=int, default=4,
                        help="Best-of-N triplet size for the reshaped dataset. "
                             "`--k` is accepted as a legacy alias.")
    parser.add_argument("--p", type=float, default=0.9,
                        help="Reshaping strength (fraction of samples passed through "
                             "the filter; 0 = no reshaping).")

    single = parser.add_argument_group("single-job (default)")
    single.add_argument("--seed", type=int, default=0)

    fanout = parser.add_argument_group("multi-GPU fan-out")
    fanout.add_argument("--parallel", action="store_true")
    fanout.add_argument("--seeds", type=int, nargs="+", default=None)
    fanout.add_argument("--names", type=str, nargs="+", default=None)
    fanout.add_argument("--cs", type=float, nargs="+", default=None,
                        help="c values to sweep.")
    fanout.add_argument("--ws", type=float, nargs="+", default=None,
                        help="w values to sweep.")
    fanout.add_argument("--gpus", type=int, nargs="+", default=None)
    fanout.add_argument("--poll", type=float, default=5.0)


def _run_single(args: argparse.Namespace) -> int:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
    from bon.calibration import load_or_build_calibration
    from bon.rejection import (
        generate_and_score,
        estimate_num_to_generate,
        rejection_sampling,
        rejection_seed,
        scale_requirements,
        trim_to_requirements,
        update_rejection_dataset,
        verify_if_dataset_is_complete,
    )

    filename = PATHS.rejection_dataset(
        args.name, args.seed, args.N, args.c, args.w, args.p,
        args.base_model, args.reward_model,
    )
    if filename.exists():
        print(f"Dataset already exists at {filename}, skipping.")
        return 0

    requirements = load_json(PATHS.requirements(args.name, args.seed))
    calibration = load_or_build_calibration(
        name=args.name, seed=args.seed,
        model=args.base_model, reward_model=args.reward_model,
    )
    print("Calibration loaded.")

    base_seed = rejection_seed(args.N, args.c, args.w, args.p, args.seed)
    sample_factor = estimate_num_to_generate(args.c, args.w, args.p)
    requirements = scale_requirements(requirements, N=args.N)

    rejection_dataset: dict[int, dict] = {}
    round_ = 0
    remaining = verify_if_dataset_is_complete(rejection_dataset, requirements)
    while remaining:
        print(f"--- round {round_}: {len(remaining)} prompts remaining ---")
        responses = generate_and_score(
            remaining,
            model=args.base_model,
            reward_model=args.reward_model,
            temperature=args.temperature,
            base_seed=base_seed,
            # Double the sample budget every round (capped).
            sample_factor=min(sample_factor * (2 ** round_), 1024),
            round_=round_,
        )
        accepted = rejection_sampling(
            responses, calibration,
            base_seed=base_seed, c=args.c, w=args.w, p=args.p, round_=round_,
        )
        rejection_dataset = update_rejection_dataset(rejection_dataset, accepted)
        remaining = verify_if_dataset_is_complete(rejection_dataset, requirements)
        round_ += 1
        if round_ > MAX_ROUNDS:
            raise RuntimeError(f"Rejection sampling failed after {MAX_ROUNDS} rounds")

    print(f"Finished rejection sampling after {round_} rounds.")
    final = trim_to_requirements(rejection_dataset, requirements)
    save_gzip(final, filename)
    print(f"Wrote rejection-sampling dataset to {filename}")
    return 0


def _run_parallel(args: argparse.Namespace) -> int:
    import itertools
    import sys
    from bon.parallel import Job, run_parallel

    if not args.gpus:
        raise SystemExit("--parallel requires --gpus.")
    names = args.names or [args.name]
    seeds = args.seeds or [args.seed]
    cs = args.cs or [args.c]
    ws = args.ws or [args.w]

    jobs = []
    for name, seed, c, w in itertools.product(names, seeds, cs, ws):
        jobs.append(Job(
            job_id=f"rej_{name}_s{seed}_c{c}_w{w}",
            payload={"name": name, "seed": seed, "c": c, "w": w},
            command=[
                sys.executable, "-m", "bon", "rejection",
                "--name", name,
                "--seed", str(seed),
                "--c", str(c),
                "--w", str(w),
                "--N", str(args.N),
                "--p", str(args.p),
                "--base-model", args.base_model,
                "--reward-model", args.reward_model,
                "--temperature", str(args.temperature),
            ],
        ))
    return run_parallel(
        jobs, gpus=args.gpus, poll=args.poll, log_dir=PATHS.logs,
    )


def run(args: argparse.Namespace) -> int:
    if args.parallel:
        return _run_parallel(args)
    return _run_single(args)
