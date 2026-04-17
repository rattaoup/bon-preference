"""``bon prepare`` - one-shot data-preparation front-door.

Runs ``plan -> generate -> score`` end-to-end (and optionally
``calibrate -> rejection`` when ``--reshape`` is passed) by invoking the
individual subcommands' ``run()`` functions directly. Using the same
code path as the dedicated subcommands keeps flags in sync and lets
failures surface cleanly without dragging extra subprocesses through
``sys.executable``.
"""

from __future__ import annotations

import argparse
from argparse import Namespace

from bon.config import (
    DEFAULT_GENERATOR_MODEL,
    DEFAULT_REWARD_MODEL,
    DEFAULT_REWARD_MODEL_NAME,
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    required = parser.add_argument_group("required")
    required.add_argument("--name", type=str, required=True,
                          help="Experiment name; used as the tag in all "
                               "output filenames.")
    required.add_argument("--source-dataset", "--dataset",
                          dest="source_dataset",
                          type=str, required=True,
                          choices=["ultrafeedback", "gsm8k", "pku-saferlhf"],
                          help="Source HF dataset that supplies prompts.")
    required.add_argument("--gpus", type=int, nargs="+", required=True,
                          help="GPU ids to fan out over (e.g. --gpus 0 1 2 3).")

    plan = parser.add_argument_group("plan (what to generate)")
    plan.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    plan.add_argument("--N", "--k",
                      dest="N",
                      type=int, nargs="+", default=[2, 4, 8, 16],
                      help="Best-of-N triplet size(s). "
                           "`--k` is accepted as a legacy alias.")
    plan.add_argument("--train-size", "--n",
                      dest="train_size",
                      type=int, nargs="+", default=[32, 128, 512, 2048, 8192],
                      help="Training-set size(s) (number of prompts sampled per config). "
                           "`--n` is accepted as a legacy alias.")

    models = parser.add_argument_group("models")
    models.add_argument("--base-model", "--generator-model",
                        dest="base_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="Generator used by plan/generate (and calibrate/rejection "
                             "when --reshape is set). The training dataset is identified "
                             "by (name, seed, N, train_size, base_model, reward_model); "
                             "changing this invalidates cached responses/scores for the "
                             "same --name. `--generator-model` is accepted as a legacy alias.")
    models.add_argument("--reward-model", type=str, default=DEFAULT_REWARD_MODEL,
                        help="Full HF id of the reward model used by score (and by "
                             "calibrate/rejection when --reshape is set).")
    models.add_argument("--reward-model-name", type=str, default=DEFAULT_REWARD_MODEL_NAME,
                        help="Short alias of the reward model used in cached filenames "
                             "(see data/.../scores/model_<reward-model-name>_chunk_*.gz).")

    gen = parser.add_argument_group("generate / score")
    gen.add_argument("--num-chunks", type=int, default=8)
    gen.add_argument("--temperature", type=float, default=1.0)
    gen.add_argument("--max-model-len", type=int, default=40960)
    gen.add_argument("--max-tokens", type=int, default=512)
    gen.add_argument("--top-p", type=float, default=0.95)

    flow = parser.add_argument_group("flow control")
    flow.add_argument("--poll", type=float, default=5.0)
    flow.add_argument("--skip-existing", action="store_true",
                      help="Skip individual (seed, chunk) jobs whose output already exists.")
    flow.add_argument("--skip-plan", action="store_true",
                      help="Skip `bon plan` and reuse existing requirements/mappings.")
    flow.add_argument("--stop-after", type=str, default=None,
                      choices=["plan", "generate", "score", "calibrate"],
                      help="Stop after the named stage (useful for splitting a long "
                           "run across machines).")

    reshape = parser.add_argument_group("distribution reshaping (optional)")
    reshape.add_argument("--reshape", action="store_true",
                         help="Also run calibrate -> rejection to build the "
                              "reshaping dataset (see Phase 1 of the README).")
    reshape.add_argument("--c", type=float, nargs="+", default=[0.5],
                         help="[--reshape] c values (percentile centre; "
                              "positive upsamples the band, negative downsamples it).")
    reshape.add_argument("--w", type=float, nargs="+", default=[1.0],
                         help="[--reshape] w values (percentile band width; always >= 0).")
    reshape.add_argument("--p", type=float, default=0.8,
                         help="[--reshape] reshaping strength (fraction of samples "
                              "passed through the filter; 0 = no reshaping).")
    reshape.add_argument("--N-reshape", "--k-reshape",
                         dest="N_reshape", type=int, default=4,
                         help="[--reshape] Best-of-N triplet size for the reshaped "
                              "dataset. `--k-reshape` is accepted as a legacy alias.")
    reshape.add_argument("--num-calibration-samples", type=int, default=100,
                         help="[--reshape] samples per prompt for the calibration CDF.")


def _banner(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n== [bon prepare] {title}\n{bar}")


def run(args: argparse.Namespace) -> int:
    # Lazy imports so ``python -m bon --help`` stays fast and vLLM-less environments
    # can still use the other subcommands.
    from bon.commands import (
        calibrate as calibrate_cmd,
        generate as generate_cmd,
        plan as plan_cmd,
        rejection as rejection_cmd,
        score as score_cmd,
    )

    def _should_stop(after: str) -> bool:
        return args.stop_after == after

    if not args.skip_plan:
        _banner("1/3 plan")
        rc = plan_cmd.run(Namespace(
            source_dataset=args.source_dataset,
            name=args.name,
            seeds=args.seeds,
            N=args.N,
            train_size=args.train_size,
        ))
        if rc:
            return rc
    if _should_stop("plan"):
        return 0

    _banner("2/3 generate")
    rc = generate_cmd.run(Namespace(
        name=args.name,
        model=args.base_model,
        temperature=args.temperature,
        num_chunks=args.num_chunks,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        seed=args.seeds[0],
        chunk_idx=0,
        parallel=True,
        seeds=args.seeds,
        gpus=args.gpus,
        poll=args.poll,
        skip_existing=args.skip_existing,
        status=None,
    ))
    if rc:
        return rc
    if _should_stop("generate"):
        return 0

    _banner("3/3 score")
    rc = score_cmd.run(Namespace(
        name=args.name,
        model=args.reward_model_name,
        base_model=args.base_model,
        temperature=args.temperature,
        num_chunks=args.num_chunks,
        seed=args.seeds[0],
        chunk_idx=0,
        parallel=True,
        seeds=args.seeds,
        gpus=args.gpus,
        poll=args.poll,
        skip_existing=args.skip_existing,
    ))
    if rc:
        return rc
    if _should_stop("score"):
        return 0

    if args.reshape:
        _banner("reshape / 1 calibrate")
        rc = calibrate_cmd.run(Namespace(
            name=args.name,
            base_model=args.base_model,
            reward_model=args.reward_model,
            num_samples=args.num_calibration_samples,
            seed=args.seeds[0],
            parallel=True,
            seeds=args.seeds,
            names=[args.name],
            gpus=args.gpus,
            poll=args.poll,
        ))
        if rc:
            return rc
        if _should_stop("calibrate"):
            return 0

        _banner("reshape / 2 rejection")
        rc = rejection_cmd.run(Namespace(
            name=args.name,
            base_model=args.base_model,
            reward_model=args.reward_model,
            temperature=args.temperature,
            c=args.c[0],
            w=args.w[0],
            N=args.N_reshape,
            p=args.p,
            seed=args.seeds[0],
            parallel=True,
            seeds=args.seeds,
            names=[args.name],
            cs=args.c,
            ws=args.w,
            gpus=args.gpus,
            poll=args.poll,
        ))
        if rc:
            return rc

    _banner("done")
    return 0
