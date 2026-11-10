"""
Training loop -- binary classification variant.

Differences from the regression template:
  * loss is BCEWithLogitsLoss, never BCELoss.  The sigmoid is fused into the
    loss for numerical stability, so THE MODEL MUST NOT APPLY A SIGMOID -- its
    final Linear emits a raw logit.
  * `pos_weight` (= n_neg / n_pos on the TRAINING rows) corrects the class
    imbalance; without it the model drifts to the majority class.
  * optional magnitude weighting: each day is weighted by |arcsinh(y/scale_c)|,
    so a 3-trillion-won day counts more than a 1-billion-won day.
  * `predict_at` returns PROBABILITIES (sigmoid of the logit), not logits --
    AUC and the Brier scores need probabilities, not hard 0/1 calls.

Model contract is unchanged: forward(X, y_hist) -> logit, or (logit, *extras).
`y_hist` here carries the 0/1 direction history.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from .dataset import WindowDataset


@dataclass
class TrainCfg:
    T: int = 30
    batch_size: int = 64
    max_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    early_stopping_patience: int = 10
    min_delta: float = 1e-6
    loss: str = "bce"
    optimizer: str = "adam"
    use_scheduler: bool = False
    lr_scheduler_patience: int = 6
    attn_entropy_lambda: float = 0.0
    min_epochs: int = 1
    seed: int = 42
    pos_weight: float | None = None
    magnitude_weighting: bool = False


def _forward(model, X, yh):
    out = model(X, yh)
    if isinstance(out, (tuple, list)):
        return out[0], tuple(out[1:])
    return out, ()


def _make_loss(cfg: TrainCfg, device):
    if cfg.loss != "bce":
        raise ValueError("this folder is the classification variant; loss must be 'bce'")
    pw = (torch.tensor([cfg.pos_weight], dtype=torch.float32, device=device)
          if cfg.pos_weight else None)
    reduction = "none" if cfg.magnitude_weighting else "mean"
    return torch.nn.BCEWithLogitsLoss(pos_weight=pw, reduction=reduction)


def _make_opt(model, cfg: TrainCfg):
    cls = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[cfg.optimizer]
    return cls(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)


def _seed(cfg: TrainCfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.deterministic = True   # exact reproducibility per seed
        torch.backends.cudnn.benchmark = False


def _reduce(raw, w, cfg: TrainCfg):
    """raw is per-sample when magnitude weighting is on, already reduced otherwise."""
    if not cfg.magnitude_weighting:
        return raw
    w = w.view_as(raw)
    return (raw * w).sum() / w.sum().clamp_min(1e-12)


@torch.no_grad()
def _evaluate(model, dl, lossfn, cfg, device):
    model.eval()
    tot = n = 0.0
    for X, yh, yt, w in dl:
        logit, _ = _forward(model, X.to(device), yh.to(device))
        loss = _reduce(lossfn(logit.view(-1).cpu(), yt.view(-1)), w, cfg)
        tot += loss.detach().item() * len(yt)
        n += len(yt)
    return tot / max(n, 1)


def _train_one_epoch(model, dl, opt, lossfn, cfg: TrainCfg, device) -> float:
    model.train()
    tot = n = 0.0
    for X, yh, yt, w in dl:
        X, yh, yt, w = X.to(device), yh.to(device), yt.to(device), w.to(device)
        opt.zero_grad()
        logit, extras = _forward(model, X, yh)
        loss = _reduce(lossfn(logit.view(-1), yt.view(-1)), w, cfg)
        if cfg.attn_entropy_lambda and extras and hasattr(model, "attention_entropy"):
            loss = loss + cfg.attn_entropy_lambda * model.attention_entropy(extras[0])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        tot += loss.detach().item() * len(yt)
        n += len(yt)
    return tot / max(n, 1)


def _train_loader(ds, tpos, n_train, cfg):
    idx = np.where(tpos < n_train)[0]
    g = torch.Generator().manual_seed(cfg.seed)
    return DataLoader(ds, batch_size=cfg.batch_size, sampler=SubsetRandomSampler(idx, generator=g))


def train_fold(Xs, ys, weights, n_train, n_val, build_model, cfg: TrainCfg, device, verbose=False,
               y_hist=None):
    """`ys` is the 0/1 direction; `weights` the per-day sample weight;
    `y_hist` the continuous history handed to the model (see WindowDataset)."""
    _seed(cfg)
    ds = WindowDataset(Xs, ys, cfg.T, weights, y_hist)
    tpos = ds.target_positions()
    train_dl = _train_loader(ds, tpos, n_train, cfg)
    val_idx = np.where((tpos >= n_train) & (tpos < n_train + n_val))[0]
    val_dl = DataLoader(ds, batch_size=256, sampler=val_idx.tolist())

    model = build_model(Xs.shape[1], cfg.T).to(device)
    opt = _make_opt(model, cfg)
    sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5,
                                                        patience=cfg.lr_scheduler_patience)
             if cfg.use_scheduler else None)
    lossfn = _make_loss(cfg, device)
    lossfn_cpu = _make_loss(cfg, torch.device("cpu"))

    best_val, best_state, best_epoch, since, history = float("inf"), None, -1, 0, []
    for epoch in range(cfg.max_epochs):
        tr = _train_one_epoch(model, train_dl, opt, lossfn, cfg, device)
        val = _evaluate(model, val_dl, lossfn_cpu, cfg, device)
        if sched is not None:
            sched.step(val)
        history.append({"epoch": epoch + 1, "train_loss": tr, "val_loss": val})
        if verbose:
            print(f"    [SELECT] epoch {epoch + 1:03d}  train={tr:.6f}  val={val:.6f}")
        if val < best_val - cfg.min_delta:
            best_val, best_state, best_epoch, since = val, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            since += 1
            if epoch + 1 >= cfg.min_epochs and since >= cfg.early_stopping_patience:
                break
    model.load_state_dict(best_state)
    return model, {"best_val_loss": best_val, "best_epoch": best_epoch + 1,
                   "epochs_run": len(history), "history": history,
                   "val_curve": [h["val_loss"] for h in history]}


def train_fixed_epochs(Xs, ys, weights, n_train, epochs, build_model, cfg: TrainCfg,
                       device, verbose=False, y_hist=None):
    _seed(cfg)
    ds = WindowDataset(Xs, ys, cfg.T, weights, y_hist)
    dl = _train_loader(ds, ds.target_positions(), n_train, cfg)
    model = build_model(Xs.shape[1], cfg.T).to(device)
    opt = _make_opt(model, cfg)
    lossfn = _make_loss(cfg, device)
    history = []
    for epoch in range(epochs):
        tr = _train_one_epoch(model, dl, opt, lossfn, cfg, device)
        history.append({"epoch": epoch + 1, "train_loss": tr})
        if verbose:
            print(f"    [REFIT ] epoch {epoch + 1:03d}/{epochs:03d}  train={tr:.6f}")
    return model, history


@torch.no_grad()
def predict_at(model, Xs, ys, weights, positions, T, device, batch: int = 256, y_hist=None):
    """Returns P(net buy) -- sigmoid of the logit -- plus any model extras."""
    model.eval()
    ds = WindowDataset(Xs, ys, T, weights, y_hist)
    idx = [int(p) - T for p in positions]
    if min(idx) < 0:
        raise ValueError("a requested position has fewer than T rows of history")
    dl = DataLoader(ds, batch_size=batch, sampler=idx)
    probs, extras = [], None
    for X, yh, _, _ in dl:
        logit, ex = _forward(model, X.to(device), yh.to(device))
        probs.append(torch.sigmoid(logit).view(-1).cpu().numpy())
        if ex:
            extras = extras or [[] for _ in ex]
            for k, e in enumerate(ex):
                extras[k].append(e.cpu().numpy())
    names = getattr(model, "extra_names", None) or [f"extra{k}" for k in range(len(extras or []))]
    return np.concatenate(probs), {n: np.concatenate(e) for n, e in zip(names, extras or [])}
