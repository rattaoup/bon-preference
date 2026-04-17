"""``bon generate`` - generate responses with vLLM for one chunk, or fan out across GPUs."""

from __future__ import annotations

import argparse

from bon.config import DEFAULT_GENERATOR_MODEL, PATHS
from bon.io import load_json, save_gzip


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", type=str, required=True,
                        help="Experiment name (e.g. ultrafeedback-base).")
    parser.add_argument("--model", type=str, default=DEFAULT_GENERATOR_MODEL,
                        help="HF generator model id.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num-chunks", type=int, default=8,
                        help="Total number of chunks the work is split into.")
    parser.add_argument("--max-model-len", type=int, default=40960)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--top-p", type=float, default=0.95)

    single = parser.add_argument_group("single-job (default)")
    single.add_argument("--seed", type=int, default=0)
    single.add_argument("--chunk-idx", type=int, default=0)

    fanout = parser.add_argument_group("multi-GPU fan-out")
    fanout.add_argument("--parallel", action="store_true",
                        help="Run over (seed, chunk_idx) across --gpus.")
    fanout.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Seeds to fan out over (requires --parallel).")
    fanout.add_argument("--gpus", type=int, nargs="+", default=None,
                        help="GPU ids to use (requires --parallel).")
    fanout.add_argument("--poll", type=float, default=5.0)
    fanout.add_argument("--skip-existing", action="store_true",
                        help="Skip jobs whose output chunk already exists.")
    fanout.add_argument("--status", type=str, default=None,
                        help="Optional CSV to persist per-job status across resumes.")


def _run_single(args: argparse.Namespace) -> int:
    from vllm import LLM
    from bon.generation import generate_responses, split_requirements

    req_path = PATHS.requirements(args.name, args.seed)
    if not req_path.exists():
        raise FileNotFoundError(f"Requirements file not found: {req_path}. "
                                "Run `bon plan` first.")
    requirements = load_json(req_path)
    chunk = split_requirements(requirements, args.num_chunks, args.chunk_idx)

    llm = LLM(model=args.model, max_model_len=args.max_model_len)
    generated = generate_responses(
        chunk, llm, seed=args.seed, temperature=args.temperature,
        max_tokens=args.max_tokens, top_p=args.top_p,
    )

    out_path = PATHS.responses_chunk(
        args.name, args.seed, args.temperature,
        args.chunk_idx, args.num_chunks, args.model,
    )
    save_gzip(generated, out_path)
    print(f"Wrote {len(generated)} prompts' responses to {out_path}")
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
        return PATHS.responses_chunk(
            args.name, j["seed"], args.temperature,
            j["chunk_idx"], args.num_chunks, args.model,
        )

    jobs = []
    for seed, chunk_idx in itertools.product(args.seeds, range(args.num_chunks)):
        jobs.append(Job(
            job_id=f"gen_s{seed}_c{chunk_idx:04d}",
            payload={"seed": seed, "chunk_idx": chunk_idx},
            command=[
                sys.executable, "-m", "bon", "generate",
                "--name", args.name,
                "--model", args.model,
                "--temperature", str(args.temperature),
                "--num-chunks", str(args.num_chunks),
                "--seed", str(seed),
                "--chunk-idx", str(chunk_idx),
                "--max-model-len", str(args.max_model_len),
                "--max-tokens", str(args.max_tokens),
                "--top-p", str(args.top_p),
            ],
            output_path=output_path,
        ))
    return run_parallel(
        jobs, gpus=args.gpus, poll=args.poll,
        skip_existing=args.skip_existing,
        status_path=args.status,
        log_dir=PATHS.logs,
    )


def run(args: argparse.Namespace) -> int:
    if args.parallel:
        return _run_parallel(args)
    return _run_single(args)
