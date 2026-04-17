"""``bon score`` - reward-model-score a chunk of generated responses."""

from __future__ import annotations

import argparse

from bon.config import DEFAULT_GENERATOR_MODEL, DEFAULT_REWARD_MODEL_NAME, PATHS
from bon.io import load_gzip, save_gzip


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--model", type=str, default=DEFAULT_REWARD_MODEL_NAME,
                        help="Reward-model alias or HF id (default: skywork-v2).")
    parser.add_argument("--base-model", "--generator-model",
                        dest="base_model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="Base/generator model the responses were produced with "
                             "(affects path). `--generator-model` is accepted as a legacy alias.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=8)

    single = parser.add_argument_group("single-job (default)")
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--chunk-idx", type=int, default=0)

    fanout = parser.add_argument_group("multi-GPU fan-out")
    fanout.add_argument("--parallel", action="store_true")
    fanout.add_argument("--seeds", type=int, nargs="+", default=None)
    fanout.add_argument("--gpus", type=int, nargs="+", default=None)
    fanout.add_argument("--poll", type=float, default=5.0)
    fanout.add_argument("--skip-existing", action="store_true")


def _run_single(args: argparse.Namespace) -> int:
    import torch
    from tqdm import tqdm
    from bon.scoring import load_reward_model, resolve_reward_model, score_pair

    torch.set_float32_matmul_precision("high")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    resp_path = PATHS.responses_chunk(
        args.name, args.seed, args.temperature,
        args.chunk_idx, args.num_chunks, args.base_model,
    )
    data = load_gzip(resp_path)

    model_id = resolve_reward_model(args.model)
    rm, tokenizer = load_reward_model(model_id, device)
    rm = torch.compile(rm)

    scored = []
    for item in tqdm(data, desc="Scoring"):
        scores = [score_pair(rm, tokenizer, item["prompt"], r, device)
                  for r in item["responses"]]
        scored.append({"original_idx": item["original_idx"], "scores": scores})

    out_path = PATHS.scores_chunk(
        args.name, args.seed, args.temperature,
        args.model, args.chunk_idx, args.num_chunks, args.base_model,
    )
    save_gzip(scored, out_path)
    print(f"Wrote scores for {len(scored)} prompts to {out_path}")
    return 0


def _run_parallel(args: argparse.Namespace) -> int:
    import itertools
    import sys
    from bon.parallel import Job, run_parallel

    if not args.seeds:
        raise SystemExit("--parallel requires --seeds.")
    if not args.gpus:
        raise SystemExit("--parallel requires --gpus.")

    def output_path(j: dict):
        return PATHS.scores_chunk(
            args.name, j["seed"], args.temperature,
            args.model, j["chunk_idx"], args.num_chunks, args.base_model,
        )

    jobs = []
    for seed, chunk_idx in itertools.product(args.seeds, range(args.num_chunks)):
        jobs.append(Job(
            job_id=f"score_s{seed}_c{chunk_idx:04d}",
            payload={"seed": seed, "chunk_idx": chunk_idx},
            command=[
                sys.executable, "-m", "bon", "score",
                "--name", args.name,
                "--model", args.model,
                "--base-model", args.base_model,
                "--temperature", str(args.temperature),
                "--num-chunks", str(args.num_chunks),
                "--seed", str(seed),
                "--chunk-idx", str(chunk_idx),
            ],
            output_path=output_path,
        ))

    return run_parallel(
        jobs, gpus=args.gpus, poll=args.poll,
        skip_existing=args.skip_existing,
        log_dir=PATHS.logs,
    )


def run(args: argparse.Namespace) -> int:
    if args.parallel:
        return _run_parallel(args)
    return _run_single(args)
