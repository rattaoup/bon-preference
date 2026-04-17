"""Linear reward-head training and evaluation on top of frozen embeddings."""

from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import KFold, ParameterGrid
from torch.utils.data import DataLoader
from tqdm import tqdm

from bon.seeding import set_seed


class RewardModel(nn.Module):
    """Single linear head on top of frozen backbone embeddings."""

    def __init__(self, d: int):
        super().__init__()
        self.linear = nn.Linear(d, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def bt_loss(model: nn.Module, embeddings: torch.Tensor) -> torch.Tensor:
    """Bradley-Terry pairwise loss for ``(N, 2, H)`` embeddings."""
    rewards = model(embeddings).squeeze(-1)
    return -F.logsigmoid(rewards[:, 0] - rewards[:, 1]).mean()


@torch.no_grad()
def eval_bt_loss(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        total += bt_loss(model, batch).item() * batch.size(0)
        count += batch.size(0)
    return total / max(count, 1)


@torch.no_grad()
def eval_accuracy(model: nn.Module, embeddings: torch.Tensor):
    """Return ``(accuracy, per_example_correct)`` on ``embeddings``."""
    device = next(model.parameters()).device
    embeddings = embeddings.to(device=device, dtype=torch.float32)
    model.eval()
    rewards = model(embeddings).squeeze(-1)
    correct = (rewards[:, 0] > rewards[:, 1]).float()
    return correct.mean().cpu(), correct.cpu()


def accuracy_on_hard_pairs(correct: torch.Tensor, score_diffs: Iterable[float],
                           percentile: float) -> float:
    threshold = np.percentile(list(score_diffs), percentile * 100)
    mask = np.asarray(list(score_diffs)) <= threshold
    return correct[mask].float().mean().item()


def train_one_fold(train_pairs: torch.Tensor, val_pairs: torch.Tensor | None, cfg: dict) -> dict:
    """Train a single fold with optional early stopping."""
    device = cfg["device"]
    set_seed(int(cfg.get("seed", 0)))

    model = RewardModel(d=train_pairs.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"],
    )

    train_pairs = train_pairs.to(dtype=torch.float32)
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 0)))
    train_loader = DataLoader(
        train_pairs, batch_size=cfg["batch_size"], shuffle=True, generator=g,
    )

    early_stopping = val_pairs is not None
    if early_stopping:
        val_pairs = val_pairs.to(dtype=torch.float32)
        val_loader = DataLoader(val_pairs, batch_size=cfg["batch_size"], shuffle=False)
        patience = cfg.get("patience", 3)
        min_delta = cfg.get("min_delta", 0.0)
        best_val = float("inf")
        best_epoch = 0
        best_state = None
        bad_epochs = 0

    for epoch in range(cfg["max_epochs"]):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            loss = bt_loss(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if early_stopping:
            val_loss = eval_bt_loss(model, val_loader, device)
            if val_loss < best_val - min_delta:
                best_val = val_loss
                best_epoch = epoch + 1
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    break

    if early_stopping:
        if best_state is not None:
            model.load_state_dict(best_state)
        return {"model": model, "best_val_loss": best_val, "best_epoch": best_epoch}
    return {"model": model, "best_val_loss": None, "best_epoch": cfg["max_epochs"]}


def run_cv(pairs: torch.Tensor, cfg: dict, k: int = 5, seed: int = 0) -> dict:
    """K-fold cross-validation for a single config."""
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    losses, best_epochs = [], []
    for fold, (tr_idx, va_idx) in enumerate(kf.split(np.arange(pairs.shape[0]))):
        cfg_fold = copy.deepcopy(cfg)
        cfg_fold["seed"] = seed * 1000 + fold
        out = train_one_fold(pairs[tr_idx], pairs[va_idx], cfg_fold)
        losses.append(out["best_val_loss"])
        best_epochs.append(out["best_epoch"])
    return {
        "cfg": cfg,
        "mean_val_loss": float(np.mean(losses)),
        "std_val_loss": float(np.std(losses)),
        "mean_best_epoch": float(np.mean(best_epochs)),
    }


def _as_param_grid(space: dict) -> dict:
    return {k: v if isinstance(v, (list, tuple)) else [v] for k, v in space.items()}


def sweep_pick_and_train_final(
    pairs: torch.Tensor,
    sweep_space: dict,
    *,
    k_folds: int,
    seed: int,
) -> dict:
    """CV over ``sweep_space`` then retrain on all data with the best config."""
    grid = ParameterGrid(_as_param_grid(sweep_space))
    all_cv, best_cv = [], None

    print(f"Sweeping {len(grid)} configurations")
    for cfg in tqdm(grid):
        cv = run_cv(pairs, cfg, k=k_folds, seed=seed)
        all_cv.append(cv)
        if best_cv is None or cv["mean_val_loss"] < best_cv["mean_val_loss"]:
            best_cv = cv

    final_epochs = max(1, int(round(best_cv["mean_best_epoch"])))
    best_cfg = copy.deepcopy(best_cv["cfg"])
    best_cfg["max_epochs"] = final_epochs

    print("Training the final model with the best configuration")
    final = train_one_fold(pairs, None, best_cfg)
    return {
        "model": final["model"],
        "best_cfg": best_cv["cfg"],
        "final_epochs": final_epochs,
        "best_cv": best_cv,
        "all_cv": all_cv,
    }


DEFAULT_SWEEP_SPACE = {
    "learning_rate": [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5],
    "weight_decay": [0.0, 1e-4, 1e-5, 1e-3, 1e-2],
    "batch_size": 1024,
    "max_epochs": 1000,
    "patience": 5,
    "min_delta": 0.0,
}


def resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
