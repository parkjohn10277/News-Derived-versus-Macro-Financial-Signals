"""
Per-model XAI driver -- run ONE model per call (each codebase has its own
`src` package, so models are never mixed in one Python process).

    python run_xai.py --list                       # which models have saved E7 artifacts
    python run_xai.py --model bilstm               # all steps for one model
    python run_xai.py --model xgboost --steps group native
    python run_xai.py --model lstm --max-days 300  # quick trial on a subsample

Steps (all read the SAVED checkpoint -- nothing is retrained):

  verify     always.  Rebuild the inputs, re-predict the test period and compare
             with the saved predictions.csv.  Stops if they disagree.
  group      EXACT grouped Shapley values (4 information groups, all 16
             coalitions, interventional background masking):
               * local group SHAP of every day's P(net buy)   -> temporal + regime panels
               * performance game on the TEST days: AUC and log-loss
                 decomposed into group contributions + pairwise
                 interactions, with block-bootstrap CIs      -> interaction panel
  native     the shap library, feature x lag resolution:
               * sequence models: shap.GradientExplainer (expected gradients)
               * tree models:     shap.TreeExplainer (interventional, probability)
             aggregated to group x lag-bucket                -> "where in the window" panel
  attention  DA-RNN / iTransformer only: native attention weights by group
             (to contrast attention with SHAP)

Outputs:  XAI/outputs/<model>/<exp>/
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import xai_config as C        # noqa: E402
import xai_core as core       # noqa: E402


def log(msg: str) -> None:
    print(msg, flush=True)


def slug(g: str) -> str:
    return C.GROUP_SLUG.get(g, g.lower().replace(" ", "_"))


# =================================================================== list
def list_models(exp: str, seed: int) -> None:
    log(f"{'model':<15}{'family':<16}{'kind':<10}artifacts for {exp}/seed_{seed}")
    for key, spec in C.MODELS.items():
        folder = Path(spec["folder"])
        if spec["kind"] == "jun":
            run_dir = folder / spec["results"] / exp / f"seed_{seed}"
        else:
            name = {"bilstm": "bilstm", "darnn": "darnn", "itransformer": "itransformer",
                    "random_forest": "random_forest", "xgboost": "xgboost",
                    "lightgbm": "lightgbm"}[key]
            run_dir = folder / "results" / name / exp / f"seed_{seed}"
        ok = (run_dir / "metrics.json").exists()
        done = (C.OUT_ROOT / key / exp / "meta.json").exists()
        log(f"{key:<15}{spec['family']:<16}{spec['kind']:<10}"
            f"{'found' if ok else 'MISSING'}   xai outputs: {'yes' if done else 'no'}")


# =================================================================== verify
def verify(b, prob: np.ndarray) -> dict:
    saved = b.saved_test_prob
    test_dates = b.dates[b.is_test]
    common = test_dates.intersection(saved.index)
    rebuilt = pd.Series(prob[b.is_test], index=test_dates).reindex(common)
    diff = float(np.max(np.abs(rebuilt.to_numpy() - saved.reindex(common).to_numpy()))) if len(common) else np.inf
    y_test = b.y[b.is_test]
    return {"n_test_rebuilt": int(b.is_test.sum()), "n_test_saved": int(len(saved)),
            "n_matched": int(len(common)), "max_abs_diff": diff,
            "auc_test_rebuilt": core.fast_auc(y_test, prob[b.is_test])}


# =================================================================== selection
def select_days(b, max_days: int | None) -> np.ndarray:
    """All days, or (for a quick trial) every test day + evenly spaced train days."""
    n = len(b.dates)
    if not max_days or max_days >= n:
        return np.arange(n)
    test = np.flatnonzero(b.is_test)
    other = np.flatnonzero(~b.is_test)
    k = max(0, max_days - len(test))
    pick = other[np.linspace(0, len(other) - 1, k).round().astype(int)] if k else np.array([], int)
    return np.sort(np.concatenate([pick, test]))


# =================================================================== group step
def coalition_values(b, players: list[str], rows: np.ndarray, bg_idx: np.ndarray,
                     prob: np.ndarray, batch: int) -> np.ndarray:
    """V[day, S] = E_b f(x_S, b_notS) for every coalition S of the players."""
    n = len(players)
    full = (1 << n) - 1
    col_player = np.array([players.index(g) for g in b.col_group])
    yh_player = players.index(C.KRX_GROUP) if b.uses_yhist else None

    X = b.inputs[rows]
    YH = b.yhist[rows] if b.uses_yhist else None
    bgX = b.inputs[bg_idx]
    bgYH = b.yhist[bg_idx] if b.uses_yhist else None
    nb = len(bg_idx)

    V = np.empty((len(rows), 1 << n), dtype=np.float64)
    V[:, full] = prob[rows]
    V[:, 0] = float(np.mean(b.predict(bgX, bgYH)))

    chunk = max(1, batch // nb)
    t0 = time.time()
    for S in range(1, full):
        keep = ((S >> col_player) & 1).astype(bool)
        keep_yh = True if yh_player is None else bool((S >> yh_player) & 1)
        for i in range(0, len(X), chunk):
            xc = X[i:i + chunk]
            c = len(xc)
            if b.kind == "sequence":
                m = np.where(keep[None, None, None, :], xc[:, None], bgX[None]).reshape(c * nb, *xc.shape[1:])
            else:
                m = np.where(keep[None, None, :], xc[:, None], bgX[None]).reshape(c * nb, xc.shape[1])
            yh = None
            if b.uses_yhist:
                yc = YH[i:i + c]
                yh = (np.repeat(yc[:, None], nb, axis=1) if keep_yh
                      else np.broadcast_to(bgYH[None], (c, nb, bgYH.shape[1]))).reshape(c * nb, -1)
            V[i:i + c, S] = b.predict(m, yh).reshape(c, nb).mean(axis=1)
        present = [players[i] for i in range(n) if S >> i & 1]
        log(f"    coalition {S:>2}/{full - 1}  {'+'.join(present):<45} {time.time() - t0:7.1f}s")
    return V


def run_group_step(b, players, rows, prob, regimes, regimes_test, rng, args, out: Path,
                   meta: dict) -> None:
    n = len(players)
    train_pos = np.flatnonzero(b.is_train)
    bg_idx = np.sort(rng.choice(train_pos, size=min(args.n_bg, len(train_pos)), replace=False))
    batch = args.batch or (C.GROUP_BATCH_SEQ if b.kind == "sequence" else C.GROUP_BATCH_TAB)
    log(f"[group] {n} players {players}  |  {1 << n} coalitions  |  {len(rows)} days  |  "
        f"{len(bg_idx)} background days  |  ~{len(rows) * ((1 << n) - 2) * len(bg_idx):,} forward rows")
    V = coalition_values(b, players, rows, bg_idx, prob, batch)

    phi = core.shapley_values(V, n)
    I_loc = core.interaction_values(V, n)
    eff = float(np.max(np.abs(phi.sum(1) - (V[:, -1] - V[:, 0]))))
    log(f"[group] efficiency check  max|sum(phi) - (f(x) - E f)| = {eff:.2e}  (must be ~0)")
    meta["group"] = {"players": players, "n_background": int(len(bg_idx)),
                     "efficiency_max_error": eff, "base_value": float(V[0, 0])}

    d = b.dates[rows]
    split = np.where(b.is_test[rows], "test", np.where(b.is_train[rows], "train", "gap"))
    local = pd.DataFrame({"date": d, "split": split, "y": b.y[rows], "prob": prob[rows],
                          "base_value": V[:, 0], "regime_var": b.regime_var[rows],
                          "regime": regimes[rows], "regime_test": regimes_test[rows]})
    for i, g in enumerate(players):
        local[f"phi_{slug(g)}"] = phi[:, i]
    local.to_csv(out / "local_group_shap.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(out / "coalition_values.npz", V=V, players=np.array(players, dtype=str),
                        dates=np.array(d.strftime("%Y-%m-%d"), dtype=str), y=b.y[rows],
                        is_test=b.is_test[rows], regime=np.array(regimes[rows], dtype=str),
                        background_rows=bg_idx)

    # local interactions (prediction level) -- summary per scope
    scopes = {"all": np.ones(len(rows), bool), "test": b.is_test[rows],
              "low": regimes[rows] == "low", "high": regimes[rows] == "high"}
    li = []
    for sc, msk in scopes.items():
        for i in range(n):
            for j in range(i + 1, n):
                v = I_loc[msk, i, j]
                li.append({"scope": sc, "a": players[i], "b": players[j], "n_days": int(msk.sum()),
                           "mean_signed": float(v.mean()) if v.size else np.nan,
                           "mean_abs": float(np.abs(v).mean()) if v.size else np.nan})
    pd.DataFrame(li).to_csv(out / "local_interactions_summary.csv", index=False)

    # performance game on OUT-OF-SAMPLE days: overall and by volatility regime
    # (regimes_test: terciles of the test period's own VKOSPI -- see xai_config)
    test_rows = b.is_test[rows]
    perf_scopes = {"test": test_rows, "test_low": test_rows & (regimes_test[rows] == "low"),
                   "test_high": test_rows & (regimes_test[rows] == "high")}
    recs = []
    for sc, msk in perf_scopes.items():
        yv = b.y[rows][msk]
        if msk.sum() < 30 or yv.min() == yv.max():
            log(f"[group] performance scope '{sc}': only {int(msk.sum())} days -- skipped")
            continue
        for metric in ("auc", "logloss"):
            res = core.performance_decomposition(V[msk], yv, n, metric, args.boot, args.block, rng)
            lo, hi = core.ci(res["boot_total"])
            recs.append({"scope": sc, "metric": metric, "kind": "total", "a": "ALL", "b": "",
                         "value": res["total"], "ci_lo": lo, "ci_hi": hi, "n_days": res["n_days"]})
            plo, phi_ = core.ci(res["boot_phi"], axis=0)
            for i, g in enumerate(players):
                recs.append({"scope": sc, "metric": metric, "kind": "main", "a": g, "b": "",
                             "value": res["phi"][i], "ci_lo": plo[i], "ci_hi": phi_[i],
                             "n_days": res["n_days"]})
            ilo, ihi = core.ci(res["boot_I"], axis=0)
            for i in range(n):
                for j in range(i + 1, n):
                    recs.append({"scope": sc, "metric": metric, "kind": "interaction",
                                 "a": players[i], "b": players[j], "value": res["I"][i, j],
                                 "ci_lo": ilo[i, j], "ci_hi": ihi[i, j], "n_days": res["n_days"]})
            if metric == "auc":
                log(f"[group] {sc:<9} AUC-0.5 = {res['total']:+.4f}  -> "
                    + "  ".join(f"{g}: {res['phi'][i]:+.4f}" for i, g in enumerate(players)))
    pd.DataFrame(recs).to_csv(out / "perf_decomposition.csv", index=False)


# =================================================================== native step
def _normalize_grad(sv, shapes):
    """shap 0.51 returns (n, *input_shape, 1) for a single-output model, and a
    list of such arrays for several inputs.  Strip the output axis."""
    if isinstance(sv, tuple):
        sv = sv[0]
    parts = list(sv) if isinstance(sv, list) else [sv]
    if len(shapes) == 1 and len(parts) != 1:
        parts = parts[:1]
    out = []
    for arr, shape in zip(parts, shapes):
        arr = np.asarray(arr)
        if arr.ndim == len(shape) + 1 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.shape != tuple(shape):
            raise ValueError(f"unexpected GradientExplainer output shape {arr.shape}, wanted {shape}")
        out.append(arr.astype(np.float32))
    return out


def _normalize_tree(sv) -> np.ndarray:
    """Positive class (net buy): last list entry / last output axis."""
    if isinstance(sv, list):
        sv = sv[-1]
    arr = np.asarray(sv, dtype=float)
    return arr[..., -1] if arr.ndim == 3 else arr


def native_sequence(b, rows, rng, args, log_prefix="[native]"):
    import shap
    import torch

    torch.backends.cudnn.enabled = False     # cuDNN refuses LSTM/GRU backward in eval mode
    head = b.grad_module.to(b.device).eval()
    train_pos = np.flatnonzero(b.is_train)
    gbg = np.sort(rng.choice(train_pos, size=min(args.grad_bg, len(train_pos)), replace=False))

    def to_t(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=b.device)

    bg = [to_t(b.inputs[gbg])] + ([to_t(b.yhist[gbg])] if b.uses_yhist else [])
    explainer = shap.GradientExplainer(head, bg if b.uses_yhist else bg[0], batch_size=args.grad_batch)
    with torch.no_grad():
        expected = float(head(*bg).mean())

    N = len(rows)
    T, F = b.inputs.shape[1:]
    phiX = np.zeros((N, T, F), np.float32)
    phiY = np.zeros((N, T), np.float32) if b.uses_yhist else None
    t0 = time.time()
    for ci_, i in enumerate(range(0, N, args.grad_chunk)):
        r = rows[i:i + args.grad_chunk]
        xs = [to_t(b.inputs[r])] + ([to_t(b.yhist[r])] if b.uses_yhist else [])
        sv = explainer.shap_values(xs if b.uses_yhist else xs[0], nsamples=args.nsamples,
                                   rseed=args.seed + ci_)
        shapes = [(len(r), T, F)] + ([(len(r), T)] if b.uses_yhist else [])
        parts = _normalize_grad(sv, shapes)
        phiX[i:i + len(r)] = parts[0]
        if b.uses_yhist:
            phiY[i:i + len(r)] = parts[1]
        done = i + len(r)
        el = time.time() - t0
        log(f"{log_prefix} GradientExplainer {done:>5}/{N} days  {el:7.1f}s  "
            f"ETA {el / done * (N - done):7.1f}s")
    return phiX, phiY, expected, gbg, "shap.GradientExplainer (expected gradients)"


def native_tree(b, rows, rng, args, log_prefix="[native]"):
    import shap

    train_pos = np.flatnonzero(b.is_train)
    bg_idx = np.sort(rng.choice(train_pos, size=min(args.tree_bg, len(train_pos)), replace=False))
    method = "shap.TreeExplainer (interventional, probability output)"
    explainer = None
    try:
        explainer = shap.TreeExplainer(b.tree_model, data=b.inputs[bg_idx],
                                       feature_perturbation="interventional",
                                       model_output="probability")
        explainer.shap_values(b.inputs[rows[:2]], check_additivity=False)   # fail fast
    except Exception as e:                                                 # noqa: BLE001
        log(f"{log_prefix} [WARN] interventional/probability TreeExplainer failed ({e}); "
            f"falling back to tree_path_dependent on the raw output. Group SHARES stay "
            f"comparable; absolute values are then on the model's raw scale.")
        explainer = shap.TreeExplainer(b.tree_model, feature_perturbation="tree_path_dependent")
        method = "shap.TreeExplainer (tree_path_dependent, raw output) -- fallback"

    N, D = len(rows), b.inputs.shape[1]
    phi = np.zeros((N, D))
    t0 = time.time()
    for i in range(0, N, C.TREE_CHUNK):
        r = rows[i:i + C.TREE_CHUNK]
        phi[i:i + len(r)] = _normalize_tree(explainer.shap_values(b.inputs[r], check_additivity=False))
        done = i + len(r)
        el = time.time() - t0
        log(f"{log_prefix} TreeExplainer {done:>5}/{N} days  {el:7.1f}s  ETA {el / done * (N - done):7.1f}s")
    ev = explainer.expected_value           # read AFTER shap_values: set lazily in some modes
    expected = float(np.ravel(ev)[-1]) if ev is not None else float("nan")
    return phi, expected, bg_idx, method


def run_native_step(b, players, rows, prob, regimes, rng, args, out: Path, meta: dict) -> None:
    n = len(players)
    buckets = C.LAG_BUCKETS
    col_player = np.array([players.index(g) for g in b.col_group])
    nB = len(buckets)
    N = len(rows)
    A = np.zeros((N, n, nB), np.float32)                  # signed group x lag-bucket sums

    if b.kind == "sequence":
        phiX, phiY, expected, bg_idx, method = native_sequence(b, rows, rng, args)
        T = b.inputs.shape[1]
        bucket_t = core.lag_bucket_index(T - np.arange(T), buckets)   # window row k = lag T-k
        yh_player = players.index(C.KRX_GROUP) if b.uses_yhist else None
        for g in range(n):
            per_t = phiX[:, :, col_player == g].sum(-1)
            if b.uses_yhist and g == yh_player:
                per_t = per_t + phiY
            for k in range(nB):
                A[:, g, k] = per_t[:, bucket_t == k].sum(1)
        feat_names = list(b.columns) + ([C.YHIST_NAME] if b.uses_yhist else [])
        feat_phi = phiX.sum(1)
        if b.uses_yhist:
            feat_phi = np.hstack([feat_phi, phiY.sum(1, keepdims=True)])
        total = phiX.sum((1, 2)) + (phiY.sum(1) if b.uses_yhist else 0)
        np.savez_compressed(out / "native_shap_full.npz", phi=phiX,
                            phi_yhist=phiY if phiY is not None else np.zeros(0))
    else:
        phi, expected, bg_idx, method = native_tree(b, rows, rng, args)
        bucket_c = core.lag_bucket_index(b.col_lag_days, buckets)
        for g in range(n):
            for k in range(nB):
                A[:, g, k] = phi[:, (col_player == g) & (bucket_c == k)].sum(1)
        bases = [C.base_feature(c) for c in b.columns]
        feat_names = list(dict.fromkeys(bases))
        feat_phi = np.stack([phi[:, [j for j, s in enumerate(bases) if s == f]].sum(1)
                             for f in feat_names], axis=1)
        total = phi.sum(1)

    # completeness: SHAP values should add up to f(x) - E[f(background)]
    target = prob[rows] - expected
    gap = np.abs(total - target)
    comp = {"expected_value": expected,
            "corr_sum_phi_vs_fx_minus_E": float(np.corrcoef(total, target)[0, 1]) if N > 2 else np.nan,
            "median_abs_gap": float(np.median(gap)),
            "median_abs_target": float(np.median(np.abs(target)))}
    log(f"[native] completeness: corr(sum phi, f(x)-E f) = {comp['corr_sum_phi_vs_fx_minus_E']:.3f}, "
        f"median |gap| = {comp['median_abs_gap']:.4f} vs median |f(x)-E f| = {comp['median_abs_target']:.4f}")

    feat_groups = [C.group_of(f) for f in feat_names]
    np.savez_compressed(out / "native_shap.npz", A=A, feat_phi=feat_phi.astype(np.float32),
                        feat_names=np.array(feat_names, dtype=str),
                        feat_groups=np.array(feat_groups, dtype=str),
                        players=np.array(players, dtype=str),
                        buckets=np.array([lab for lab, _, _ in buckets], dtype=str),
                        dates=np.array(b.dates[rows].strftime("%Y-%m-%d"), dtype=str),
                        regime=np.array(regimes[rows], dtype=str), is_test=b.is_test[rows],
                        background_rows=bg_idx)
    meta["native"] = {"method": method, "n_background": int(len(bg_idx)),
                      "nsamples": args.nsamples if b.kind == "sequence" else None, **comp}

    # agreement with the exact group SHAP of the group step (if present)
    lp = out / "local_group_shap.csv"
    if lp.exists():
        from scipy.stats import spearmanr
        loc = pd.read_csv(lp, parse_dates=["date"], encoding="utf-8-sig").set_index("date")
        dd = pd.DatetimeIndex(b.dates[rows])
        agree = {}
        for gi, g in enumerate(players):
            col = f"phi_{slug(g)}"
            if col in loc.columns:
                ex = loc[col].reindex(dd).to_numpy()
                ok = np.isfinite(ex)
                if ok.sum() > 10:
                    agree[g] = float(spearmanr(np.abs(ex[ok]), np.abs(A[ok, gi, :].sum(1))).correlation)
        meta["native"]["spearman_abs_exact_vs_native"] = agree
        log(f"[native] agreement with exact group SHAP (Spearman of |phi| per day): {agree}")


# =================================================================== attention step
def run_attention_step(b, players, rows, regimes, out: Path, meta: dict) -> None:
    if b.attention_fn is None:
        log("[attention] model has no native attention -- skipped")
        return
    att = b.attention_fn(b.inputs[rows], b.yhist[rows] if b.uses_yhist else None)
    att = att / np.clip(att.sum(1, keepdims=True), 1e-12, None)
    col_player = np.array([players.index(g) for g in b.col_group])
    df = pd.DataFrame({"date": b.dates[rows], "regime": regimes[rows],
                       "split": np.where(b.is_test[rows], "test", "train")})
    for i, g in enumerate(players):
        df[f"att_{slug(g)}"] = att[:, col_player == i].sum(1)
        df[f"uniform_{slug(g)}"] = float((col_player == i).mean())
    df.to_csv(out / "attention_by_group.csv", index=False, encoding="utf-8-sig")
    meta["attention"] = {"max_over_uniform": float((att * att.shape[1]).max()),
                         "mean_entropy_over_max": float(np.mean(-(att * np.log(np.clip(att, 1e-12, None))).sum(1))
                                                        / np.log(att.shape[1]))}
    log(f"[attention] saved; mean entropy / max entropy = {meta['attention']['mean_entropy_over_max']:.3f} "
        f"(1.0 = perfectly uniform)")


# =================================================================== main
def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                              # noqa: BLE001
        pass

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(C.MODELS))
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--exp", default=C.EXPERIMENT)
    ap.add_argument("--seed", type=int, default=C.SEED)
    ap.add_argument("--steps", nargs="+", default=["group", "native", "attention"],
                    choices=["group", "native", "attention"])
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--n-bg", type=int, default=C.N_BG_GROUP)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--grad-bg", type=int, default=C.N_BG_GRAD)
    ap.add_argument("--nsamples", type=int, default=C.GRAD_NSAMPLES)
    ap.add_argument("--grad-batch", type=int, default=C.GRAD_BATCH)
    ap.add_argument("--grad-chunk", type=int, default=C.GRAD_CHUNK)
    ap.add_argument("--tree-bg", type=int, default=C.N_BG_TREE)
    ap.add_argument("--boot", type=int, default=C.BOOT_B)
    ap.add_argument("--block", type=int, default=C.BOOT_BLOCK)
    ap.add_argument("--max-days", type=int, default=None,
                    help="quick trial: all test days + evenly spaced train days, N in total")
    ap.add_argument("--no-verify", action="store_true",
                    help="continue even if the rebuilt predictions differ from predictions.csv")
    args = ap.parse_args()

    if args.list or not args.model:
        list_models(args.exp, args.seed)
        if not args.model:
            return

    import torch
    from adapters import load_bundle

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available())
                          else ("cpu" if args.device == "auto" else args.device))
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out = C.OUT_ROOT / args.model / args.exp
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    log(f"=== XAI  model={args.model}  exp={args.exp}  seed={args.seed}  device={device} ===")

    b = load_bundle(args.model, args.exp, args.seed, device)
    log(f"loaded {b.label} ({b.family}, {b.kind}): {len(b.dates)} days "
        f"{b.dates[0].date()} -> {b.dates[-1].date()}, {len(b.columns)} input columns, "
        f"{int(b.is_test.sum())} test days")

    # ---- 1. verify
    prob = b.predict(b.inputs, b.yhist)
    vr = verify(b, prob)
    log(f"[verify] {vr['n_matched']}/{vr['n_test_saved']} saved test days matched, "
        f"max |rebuilt - saved| = {vr['max_abs_diff']:.2e}, rebuilt test AUC = {vr['auc_test_rebuilt']:.4f}")
    if (vr["max_abs_diff"] > C.VERIFY_TOL or vr["n_matched"] != vr["n_test_saved"]) and not args.no_verify:
        raise SystemExit("[verify] FAILED -- the rebuilt model does not reproduce its saved test "
                         "predictions, so its explanations would describe a different model. "
                         "Check that data/ and the checkpoint are from the same run.")

    players = [g for g in C.GROUPS if g in set(b.col_group)]
    unknown = sorted(set(b.col_group) - set(C.GROUPS))
    if unknown:
        raise SystemExit(f"groups {unknown} are not listed in xai_config.GROUPS")
    counts = {g: int(sum(1 for x in b.col_group if x == g)) for g in players}
    log(f"[groups] {counts}")

    regimes, thr = core.assign_regimes(b.regime_var, b.is_train, C.REGIME_METHOD)
    rows = select_days(b, args.max_days)
    reg_counts = {r: int((regimes[rows] == r).sum()) for r in ("low", "mid", "high", "na")}
    test_reg = {r: int(((regimes[rows] == r) & b.is_test[rows]).sum()) for r in ("low", "mid", "high")}
    log(f"[regime] {C.REGIME_VARIABLE} {C.REGIME_METHOD} (train thresholds): low <= {thr['low_max']:.2f}, "
        f"high >= {thr['high_min']:.2f} | all days {reg_counts} | test days {test_reg}")
    ref_mask = b.is_test if C.REGIME_TEST_REFERENCE == "test" else b.is_train
    regimes_test, thr_test = core.assign_regimes(b.regime_var, ref_mask, C.REGIME_METHOD)
    test_reg2 = {r: int(((regimes_test[rows] == r) & b.is_test[rows]).sum()) for r in ("low", "mid", "high")}
    log(f"[regime] out-of-sample split ({C.REGIME_TEST_REFERENCE} thresholds): "
        f"low <= {thr_test['low_max']:.2f}, high >= {thr_test['high_min']:.2f} | test days {test_reg2}")

    meta = {"model": args.model, "label": b.label, "family": b.family, "kind": b.kind,
            "exp": args.exp, "seed": args.seed, "device": str(device),
            "columns": b.columns, "col_group": b.col_group, "players": players,
            "group_sizes": counts, "n_days": int(len(rows)), "max_days": args.max_days,
            "verify": vr, "regime": {**thr, "variable": C.REGIME_VARIABLE,
                                     "counts_all": reg_counts, "counts_test": test_reg},
            "regime_test": {**thr_test, "reference": C.REGIME_TEST_REFERENCE, "counts_test": test_reg2},
            "loader": b.meta, "steps": args.steps}

    def save_meta():
        meta["runtime_s"] = round(time.time() - t_start, 1)
        (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    save_meta()
    if "group" in args.steps:
        run_group_step(b, players, rows, prob, regimes, regimes_test, rng, args, out, meta)
        save_meta()
    if "native" in args.steps:
        run_native_step(b, players, rows, prob, regimes, rng, args, out, meta)
        save_meta()
    if "attention" in args.steps:
        run_attention_step(b, players, rows, regimes, out, meta)
        save_meta()
    log(f"=== done in {time.time() - t_start:.0f}s -> {out}")


if __name__ == "__main__":
    main()
