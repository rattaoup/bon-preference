"""Deterministic seeding helpers.

``torch`` is imported lazily so modules that only need to parse CLI flags
can import ``set_seed`` without incurring the ``torch`` import cost.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and (CUDA) PyTorch for best-effort determinism."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
