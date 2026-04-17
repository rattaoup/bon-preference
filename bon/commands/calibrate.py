"""``bon calibrate`` - build per-prompt reward calibrations for distribution reshaping."""

from __future__ import annotations

import argparse

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL, PATHS


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, default="ultrafeedback-pminus",
                        help="Experiment name; must match the --name used with "
                             "bon plan/generate/score.")
    parser.add_argument("--base-model", "--generator-model",
                        dest="base_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="Base/generator model used to sample calibration responses. "
                             "Calibration is keyed by (base_model, reward_model) on disk "
                             "and is reusable across every (c, w, p) sweep for the same "
                             "(name, seed, base_model, reward_model). "
                             "`--generator-model` is accepted as a legacy alias.")
    parser.add_argument("--reward-model", type=str, default=DEFAULT_REWARD_MODEL,
                        help="Full HF id of the reward model used to score calibration "
                             "responses.")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Samples per prompt for the empirical reward CDF.")

    single = parser.add_argument_group("single-job (default)")
    single.add_argument("--seed", type=int, default=0)

    fanout = parser.add_argument_group("multi-GPU fan-out")
    fanout.add_argument("--parallel", action="store_true")
    fanout.add_argument("--seeds", type=int, nargs="+", default=None)
    fanout.add_argument("--names", type=str, nargs="+", default=None,
                        help="If given, sweeps over (name, seed) across GPUs.")
    fanout.add_argument("--gpus", type=int, nargs="+", default=None)
    fanout.add_argument("--poll", type=float, default=5.0)


def _run_single(args: argparse.Namespace) -> int:
    from bon.calibration import load_or_build_calibration

    calibration = load_or_build_calibration(
        name=args.name,
        seed=args.seed,
        model=args.base_model,
        reward_model=args.reward_model,
        num_samples=args.num_samples,
    )
    print(f"Calibration for {args.name!r}: {len(calibration)} prompts.")
    return 0


def _run_parallel(args: argparse.Namespace) -> int:
    import sys
    from bon.parallel import Job, run_parallel

    names = args.names or [args.name]
    seeds = args.seeds or [args.seed]
    if not args.gpus:
        raise SystemExit("--parallel requires --gpus.")

    jobs = []
    for name in names:
        for seed in seeds:
            jobs.append(Job(
                job_id=f"cal_{name}_s{seed}",
                payload={"name": name, "seed": seed},
                command=[
                    sys.executable, "-m", "bon", "calibrate",
                    "--name", name,
                    "--seed", str(seed),
                    "--base-model", args.base_model,
                    "--reward-model", args.reward_model,
                    "--num-samples", str(args.num_samples),
                ],
            ))
    return run_parallel(
        jobs, gpus=args.gpus, poll=args.poll, log_dir=PATHS.logs,
    )


def run(args: argparse.Namespace) -> int:
    if args.parallel:
        return _run_parallel(args)
    return _run_single(args)
