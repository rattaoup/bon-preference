"""Top-level command-line interface.

Each subcommand lives in :mod:`bon.commands.<name>` and exposes:

* ``add_arguments(parser)`` - register CLI flags on a subparser
* ``run(args)`` - execute the command (heavy imports go here, not at
  module top level, so ``python -m bon --help`` stays fast)
"""

from __future__ import annotations

import argparse
import importlib

SUBCOMMANDS: dict[str, tuple[str, str]] = {
    # --- Phase 1: generate training data ----------------------------------
    "prepare": ("bon.commands.prepare",
                "One-shot data prep: plan -> generate -> score (+ optional reshape)."),
    "plan": ("bon.commands.plan",
             "[advanced] Write the requirements/mapping files for a dataset."),
    "generate": ("bon.commands.generate",
                 "[advanced] Generate responses with vLLM."),
    "score": ("bon.commands.score",
              "[advanced] Score responses with a reward model."),
    "calibrate": ("bon.commands.calibrate",
                  "[advanced, --reshape] Build per-prompt reward calibrations."),
    "rejection": ("bon.commands.rejection",
                  "[advanced, --reshape] Build the up/downsampled rejection dataset."),
    "build-test-set": ("bon.commands.test_set",
                       "Build the binarized reward-scored test set for a dataset."),
    # --- Phase 2: train ---------------------------------------------------
    "train": ("bon.commands.train",
              "Train a linear reward head on a (train, test) pair."),
    "sweep": ("bon.commands.sweep",
              "Sweep `python -m bon train` across (train_size, N, seed) or (c, w, seed)."),
    # --- Phase 3: analysis ------------------------------------------------
    "connectivity": ("bon.commands.connectivity",
                     "Compute the connectivity degree for a (train, test) pair."),
    # --- Misc -------------------------------------------------------------
    "demo": ("bon.commands.demo",
             "Run a tiny end-to-end pipeline as a smoke test."),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bon",
        description="Learning reward models from Best-of-N preference data.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="<command>")

    for name, (module_path, help_text) in SUBCOMMANDS.items():
        subparser = sub.add_parser(name, help=help_text, description=help_text)
        module = importlib.import_module(module_path)
        module.add_arguments(subparser)
        subparser.set_defaults(_run=module.run)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args._run(args) or 0
