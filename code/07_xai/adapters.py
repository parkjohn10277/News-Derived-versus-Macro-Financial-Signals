"""
Rebuild every saved classifier WITHOUT retraining.

Each loader reproduces the phase-2 inputs exactly as the training run built
them (same data, same missing-value policy, same arcsinh scale constant, the
SAVED x_scaler, the same AUX clipping, the same window construction), loads
the saved checkpoint, and returns a `Bundle` with one uniform interface:

    bundle.inputs      sequence models: (N, T, F) windows    trees: (N, D) design rows
    bundle.predict()   inputs -> P(net buy)
    bundle.dates       the target day of every row
    bundle.col_group   information group of every input column

Three codebases are covered:
    jun       Desktop/Jun/<model>/<model>_classifier.py   (features built from raw files)
    template  BiLSTM_Classifier / Att_LSTM_Classifier / iTransformer_Classifier
    tree      TreeModels_Classifier/<RandomForest|XGBoost|LightGBM>

Correctness is checked, not assumed: run_xai.py re-predicts the test period
and compares it with the saved predictions.csv before explaining anything.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import pickle
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn

import xai_config as C


# =================================================================== bundle
@dataclass
class Bundle:
    key: str
    label: str
    family: str
    kind: str                      # "sequence" | "tabular"
    exp: str
    seed: int
    dates: pd.DatetimeIndex        # target day of each explained row
    y: np.ndarray                  # 1 = net buy, 0 = net sell
    is_train: np.ndarray           # target day <= official train end
    is_test: np.ndarray            # target day >= test start
    inputs: np.ndarray             # (N, T, F) windows or (N, D) rows
    yhist: np.ndarray | None       # (N, T) continuous flow history (DA-RNN decoder)
    uses_yhist: bool
    columns: list[str]
    col_group: list[str]
    col_lag_days: np.ndarray | None  # trees: days before target of each column
    T: int | None
    regime_var: np.ndarray         # VKOSPI on the day BEFORE each target
    saved_test_prob: pd.Series     # predictions.csv, indexed by date
    predict: Callable[..., np.ndarray]
    device: Any = None
    grad_module: nn.Module | None = None
    tree_model: Any = None
    attention_fn: Callable[..., np.ndarray] | None = None
    meta: dict = field(default_factory=dict)


# =================================================================== helpers
def _require(run_dir: Path, names) -> None:
    missing = [n for n in names if not (run_dir / n).exists()]
    if missing:
        raise FileNotFoundError(f"{run_dir} is missing {missing} -- run the E0-E7 "
                                f"classifier for this model first.")


def _check_same(name: str, rebuilt: float, saved) -> None:
    """The rebuilt scale constant must equal the one saved at training time;
    otherwise the input data changed since the model was trained."""
    if saved is None:
        return
    if not np.isclose(float(rebuilt), float(saved), rtol=1e-9, atol=0):
        raise ValueError(f"{name}: rebuilt {rebuilt!r} != saved {saved!r}. The input "
                         f"data changed after training -- the checkpoint no longer "
                         f"matches these features.")


def _saved_test_prob(run_dir: Path) -> pd.Series:
    p = pd.read_csv(run_dir / "predictions.csv", encoding="utf-8-sig", parse_dates=["date"])
    return p.set_index("date")["prob_net_buy"].astype(float)


def _prev_day(series: pd.Series, target_dates) -> np.ndarray:
    """Value on the previous trading day (the level known at forecast time)."""
    s = pd.to_numeric(series, errors="coerce").ffill().shift(1)
    return s.reindex(pd.DatetimeIndex(target_dates)).to_numpy(float)


def _purge_modules(names=("src", "run")) -> None:
    for k in list(sys.modules):
        if k in names or any(k.startswith(n + ".") for n in names):
            del sys.modules[k]


def _import_folder(root: Path):
    """Import a template/tree folder's run.py and src package (one model per process)."""
    _purge_modules()
    sys.path.insert(0, str(root))
    return importlib.import_module("run")


def _import_file(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)          # the __main__ guard keeps main() from running
    return mod


def _torch_load(path: Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:                     # very old torch without weights_only
        return torch.load(path, map_location=device)


# =================================================================== torch cores
class TemplateCore(nn.Module):
    """forward(X, y_hist) -> logit (template models may return (logit, extras...))."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, X, YH=None):
        out = self.model(X, YH)
        return out[0] if isinstance(out, (tuple, list)) else out


class JunCore(nn.Module):
    """The Jun models take only the window."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, X, YH=None):
        return self.model(X)


class ITransformerGradCore(nn.Module):
    """Numerically identical to ITransformer.forward, WITHOUT the .detach() the
    upstream code puts on the window mean.  The detach does not change any
    output value, but it cuts the gradient through the normalisation, which
    would make gradient-based SHAP attributions wrong.  Equality with the real
    forward is asserted when the model is loaded."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, X, YH=None):
        m = self.model
        if m.cfg.use_norm:
            means = X.mean(dim=1, keepdim=True)
            centered = X - means
            stdev = torch.sqrt(torch.var(centered, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_model = centered / stdev
        else:
            means = stdev = None
            x_model = X
        enc_out, _ = m.encoder(m.enc_embedding(x_model))
        dec_out = m.projector(enc_out).permute(0, 2, 1)
        if m.cfg.use_norm:
            dec_out = dec_out * stdev[:, 0, :].unsqueeze(1) + means[:, 0, :].unsqueeze(1)
        return dec_out[:, -1, m.target_index]


class ProbabilityHead(nn.Module):
    """(B, 1) probability -- the output shape shap.GradientExplainer requires."""

    def __init__(self, core: nn.Module):
        super().__init__()
        self.core = core

    def forward(self, X, YH=None):
        return torch.sigmoid(self.core(X, YH).reshape(-1, 1))


def _seq_predictor(core: nn.Module, device, uses_yhist: bool, default_batch: int = 4096):
    @torch.no_grad()
    def predict(X, YH=None, batch_size: int = default_batch) -> np.ndarray:
        core.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch_size]),
                                 dtype=torch.float32, device=device)
            yb = None
            if uses_yhist:
                yb = torch.as_tensor(np.ascontiguousarray(YH[i:i + batch_size]),
                                     dtype=torch.float32, device=device)
            out.append(torch.sigmoid(core(xb, yb).reshape(-1)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0, np.float32)
    return predict


def _attention_fn(model: nn.Module, device, name: str):
    """Native attention over input columns: DA-RNN input attention averaged over
    the window; iTransformer target-token attention over variates."""
    @torch.no_grad()
    def fn(X, YH=None, batch_size: int = 1024) -> np.ndarray:
        model.eval()
        out = []
        for i in range(0, len(X), batch_size):
            xb = torch.as_tensor(np.ascontiguousarray(X[i:i + batch_size]),
                                 dtype=torch.float32, device=device)
            if name == "darnn":
                yb = torch.as_tensor(np.ascontiguousarray(YH[i:i + batch_size]),
                                     dtype=torch.float32, device=device)
                _, alphas, _ = model(xb, yb)          # (B, T, F)
                a = alphas.mean(dim=1)
            else:
                _, a = model(xb)                      # (B, F)
            out.append(a.float().cpu().numpy())
        return np.concatenate(out)
    return fn


# =================================================================== template
def load_template(key: str, spec: dict, exp: str, seed: int, device) -> Bundle:
    import yaml

    root = Path(spec["folder"])
    trun = _import_folder(root)
    feats_mod = importlib.import_module("src.features")
    models_mod = importlib.import_module("src.models")
    proto = importlib.import_module("src.protocol")

    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    model_name = cfg["model"]["name"]
    run_dir = root / cfg["results"] / model_name / exp / f"seed_{seed}"
    _require(run_dir, ["best_model.pt", "x_scaler.pkl", "metrics.json", "predictions.csv"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # ---- identical to run.py + protocol.run_single_split (phase 2)
    raw = trun.load_input(cfg)
    groups = feats_mod.feature_groups(raw)
    feats = feats_mod.experiment_features(groups, exp)
    df = trun.prepare(raw, groups, feats)
    dates = pd.DatetimeIndex(df.index)
    split, T = cfg["split"], int(cfg["model"]["T"])
    train_end, test_start = pd.Timestamp(split["train_end"]), pd.Timestamp(split["test_start"])
    off_end = int((dates <= train_end).sum())

    y_raw = df[groups["KRX"][0]].to_numpy(np.float64)
    y_dir = (y_raw >= 0).astype(np.float32)
    scale_c = proto.scale_constant(y_raw, off_end)
    _check_same("target_scale_c", scale_c, metrics.get("target_scale_c"))

    X_t = proto.transform_features(df, feats, groups, scale_c)
    with open(run_dir / "x_scaler.pkl", "rb") as f:
        xs2 = pickle.load(f)
    X2, _ = proto.clip_aux(xs2.transform(X_t), feats, groups["AUX"],
                           cfg["preprocess"]["aux_z_clip"])
    X2 = np.asarray(X2, dtype=np.float32)

    y_as = np.arcsinh(y_raw / scale_c).reshape(-1, 1)
    y_cont = StandardScaler().fit(y_as[:off_end]).transform(y_as).ravel().astype(np.float32)

    model = models_mod.build_model(model_name, len(feats), T, cfg["model"].get("params", {}) or {})
    model.load_state_dict(_torch_load(run_dir / "best_model.pt", device))
    model.to(device).eval()

    # ---- windows: target p uses rows [p-T, p)
    positions = np.arange(T, len(df))
    W = np.stack([X2[p - T:p] for p in positions]).astype(np.float32)
    YH = np.stack([y_cont[p - T:p] for p in positions]).astype(np.float32)
    tdates = dates[positions]

    uses_yhist = model_name == "darnn"
    core = TemplateCore(model)
    grad_core: nn.Module = core
    if model_name == "itransformer":
        grad_core = ITransformerGradCore(model)
        with torch.no_grad():
            xb = torch.as_tensor(W[:64], device=device)
            gap = float((core(xb) - grad_core(xb)).abs().max())
        if gap > 1e-5:
            raise RuntimeError(f"iTransformer gradient wrapper differs from forward by {gap:.2e}")

    attn = _attention_fn(model, device, model_name) if model_name in ("darnn", "itransformer") else None

    return Bundle(
        key=key, label=spec["label"], family=spec["family"], kind="sequence", exp=exp, seed=seed,
        dates=tdates, y=y_dir[positions].astype(int),
        is_train=np.asarray(tdates <= train_end), is_test=np.asarray(tdates >= test_start),
        inputs=W, yhist=YH, uses_yhist=uses_yhist,
        columns=list(feats), col_group=[C.group_of(c) for c in feats], col_lag_days=None, T=T,
        regime_var=_prev_day(raw[C.REGIME_VARIABLE], tdates),
        saved_test_prob=_saved_test_prob(run_dir),
        predict=_seq_predictor(core, device, uses_yhist),
        device=device, grad_module=ProbabilityHead(grad_core), attention_fn=attn,
        meta={"run_dir": str(run_dir), "model_name": model_name, "T": T,
              "train_end": str(train_end.date()), "test_start": str(test_start.date()),
              "target_scale_c": scale_c, "n_features": len(feats)},
    )


# =================================================================== jun
def _build_jun_model(mod, script_path: Path, feats: list[str]) -> nn.Module:
    """Construct the model with the EXACT arguments main() used, read from the
    script's own `final_model = Class(...).to(device)` block."""
    src = Path(script_path).read_text(encoding="utf-8")
    m = re.search(r"final_model\s*=\s*(\w+)\((.*?)\)\.to\(device\)", src, re.S)
    if not m:
        raise RuntimeError(f"could not find the final_model constructor in {script_path}")
    return eval(f"{m.group(1)}({m.group(2)})", vars(mod), {"feature_cols": feats})  # noqa: S307


def load_jun(key: str, spec: dict, exp: str, seed: int, device) -> Bundle:
    folder = Path(spec["folder"])
    script = folder / spec["script"]
    mod = _import_file(script, f"xai_jun_{key}")
    run_dir = folder / spec["results"] / exp / f"seed_{seed}"
    _require(run_dir, ["best_model.pt", "x_scaler.pkl", "metrics.json", "predictions.csv"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    DATE, TARGET = mod.DATE_COL, mod.TARGET_COL

    # ---- identical to main() of the classifier script
    if metrics.get("input_mode") == "aligned":
        mod.AUX_FEATURES = mod.ALIGNED_AUX_FEATURES          # as main() does without --raw-pipeline
        df, krx_cols, lda_en_cols, lda_ko_cols, llm_cols = mod.load_aligned_input(metrics["input_csv"])
    else:
        df, krx_cols, lda_en_cols, lda_ko_cols, llm_cols = mod.merge_all_data(
            exp=exp, krx_path=mod.DEFAULT_KRX_PATH, lda_en_path=mod.DEFAULT_LDA_EN_PATH,
            lda_ko_path=mod.DEFAULT_LDA_KO_PATH, aux_path=mod.DEFAULT_AUX_PATH,
            sent_gdelt_path=mod.DEFAULT_SENT_GDELT_PATH, sent_mcst_path=mod.DEFAULT_SENT_MCST_PATH,
            sent_naver_path=mod.DEFAULT_SENT_NAVER_PATH, sent_nyt_path=mod.DEFAULT_SENT_NYT_PATH,
        )
    df = df[df[DATE] >= mod.OFFICIAL_TRAIN_START].copy()
    df = df.sort_values(DATE).reset_index(drop=True)
    lda_cols = lda_en_cols + lda_ko_cols
    groups = mod.get_feature_groups(df, krx_cols=krx_cols, lda_cols=lda_cols, llm_cols=llm_cols)
    feats = mod.get_experiment_features(groups, exp)

    df[lda_cols] = df[lda_cols].fillna(0.0)
    aux_cols = groups["AUX"]
    for c in aux_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if aux_cols:
        df[aux_cols] = df[aux_cols].ffill()
    llm = groups["LLM"]
    if llm:
        for c in llm:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df[llm] = df[llm].fillna(0.0)
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    df = df.dropna(subset=feats).reset_index(drop=True)

    official_train_mask = (df[DATE] >= mod.OFFICIAL_TRAIN_START) & (df[DATE] <= mod.OFFICIAL_TRAIN_END)
    official_train_end = int(np.flatnonzero(official_train_mask.to_numpy())[-1] + 1)

    X_raw = df[feats].to_numpy(dtype=np.float64)
    y_raw = df[TARGET].to_numpy(dtype=np.float64)
    scale_c = float(np.median(np.abs(y_raw[:official_train_end])))
    _check_same("target_scale_c", scale_c, metrics.get("target_scale_c"))
    krx_idx = [i for i, c in enumerate(feats) if c in groups["KRX"]]
    X_t = X_raw.copy()
    if krx_idx:
        X_t[:, krx_idx] = np.arcsinh(X_t[:, krx_idx] / scale_c)

    with open(run_dir / "x_scaler.pkl", "rb") as f:
        x_scaler = pickle.load(f)
    X_final = x_scaler.transform(X_t).astype(np.float32)
    X_final, _ = mod.clip_standardized_auxiliary(
        X_scaled=X_final, feature_cols=feats, aux_cols=groups["AUX"],
        clip_value=float(metrics.get("aux_z_clip", 3.0)), stage_name="XAI rebuild")
    X_final = np.asarray(X_final, dtype=np.float32)
    y_dir = (y_raw >= 0).astype(np.float32)

    lookback = int(metrics["lookback"])
    horizon = int(metrics.get("prediction_horizon", 1))
    positions = np.arange(lookback + horizon - 1, len(df))
    ends = positions - horizon + 1                      # SequenceDataset: X_all[end-lookback:end]
    W = np.stack([X_final[e - lookback:e] for e in ends]).astype(np.float32)
    tdates = pd.DatetimeIndex(df[DATE].to_numpy()[positions])

    model = _build_jun_model(mod, script, feats)
    model.load_state_dict(_torch_load(run_dir / "best_model.pt", device))
    model.to(device).eval()
    core = JunCore(model)

    # regime variable (VKOSPI is an AUX column in E1/E5/E6/E7; merged otherwise)
    reg_series = None
    if C.REGIME_VARIABLE in df.columns:
        reg_series = pd.Series(df[C.REGIME_VARIABLE].to_numpy(float), index=pd.DatetimeIndex(df[DATE]))
    else:
        try:
            aux = mod.merge_auxiliary(df[[DATE]].copy(), mod.DEFAULT_AUX_PATH)
            reg_series = pd.Series(pd.to_numeric(aux[C.REGIME_VARIABLE], errors="coerce").to_numpy(),
                                   index=pd.DatetimeIndex(aux[DATE]))
        except Exception as e:                                              # noqa: BLE001
            print(f"[WARN] could not load {C.REGIME_VARIABLE} for regimes: {e}")
    regime_var = (_prev_day(reg_series, tdates) if reg_series is not None
                  else np.full(len(tdates), np.nan))

    train_end, test_start = pd.Timestamp(mod.OFFICIAL_TRAIN_END), pd.Timestamp(mod.TEST_START)
    return Bundle(
        key=key, label=spec["label"], family=spec["family"], kind="sequence", exp=exp, seed=seed,
        dates=tdates, y=y_dir[positions].astype(int),
        is_train=np.asarray(tdates <= train_end), is_test=np.asarray(tdates >= test_start),
        inputs=W, yhist=None, uses_yhist=False,
        columns=list(feats), col_group=[C.group_of(c) for c in feats], col_lag_days=None, T=lookback,
        regime_var=regime_var, saved_test_prob=_saved_test_prob(run_dir),
        predict=_seq_predictor(core, device, False),
        device=device, grad_module=ProbabilityHead(core), attention_fn=None,
        meta={"run_dir": str(run_dir), "model_class": type(model).__name__, "T": lookback,
              "horizon": horizon, "train_end": str(train_end.date()),
              "test_start": str(test_start.date()), "target_scale_c": scale_c,
              "n_features": len(feats)},
    )


# =================================================================== tree
def load_tree(key: str, spec: dict, exp: str, seed: int, device) -> Bundle:
    import joblib
    import yaml

    root = Path(spec["folder"])
    trun = _import_folder(root)
    feats_mod = importlib.import_module("src.features")
    tab = importlib.import_module("src.tabular")

    cfg = yaml.safe_load((root / "config.yaml").read_text(encoding="utf-8"))
    model_name = cfg["model"]["name"]
    run_dir = root / cfg["results"] / model_name / exp / f"seed_{seed}"
    _require(run_dir, ["best_model.joblib", "x_scaler.pkl", "metrics.json", "predictions.csv"])
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))

    # ---- identical to run.py + protocol.run_single_split (phase 2)
    raw = trun.load_input(cfg)
    groups = feats_mod.feature_groups(raw)
    feats = feats_mod.experiment_features(groups, exp)
    df = trun.prepare(raw, groups, feats)
    lags = cfg["features"]["lags"]
    split = cfg["split"]
    train_end, test_start = pd.Timestamp(split["train_end"]), pd.Timestamp(split["test_start"])

    _, y_raw, dates = tab.build_design(df, feats, groups, lags)
    off_end = int((dates <= train_end).sum())
    scale_c = tab.scale_constant(y_raw, off_end)
    _check_same("target_scale_c", scale_c, metrics.get("target_scale_c"))
    X, y_raw, dates = tab.build_design(df, feats, groups, lags, scale_c)
    names = list(X.columns)
    with open(run_dir / "x_scaler.pkl", "rb") as f:
        xt = pickle.load(f)
    X2, _ = tab.clip_aux(xt.transform(X), names, groups, cfg["preprocess"]["aux_z_clip"])
    X2 = np.asarray(X2, dtype=np.float64)

    model = joblib.load(run_dir / "best_model.joblib")
    try:
        model.set_params(n_jobs=-1)       # prediction only; training used n_jobs=1
    except Exception:                      # noqa: BLE001
        pass

    def predict(Xb, YH=None, batch_size: int = 200_000) -> np.ndarray:
        out = [model.predict_proba(Xb[i:i + batch_size])[:, 1] for i in range(0, len(Xb), batch_size)]
        return np.concatenate(out) if out else np.zeros(0)

    dates = pd.DatetimeIndex(dates)
    return Bundle(
        key=key, label=spec["label"], family=spec["family"], kind="tabular", exp=exp, seed=seed,
        dates=dates, y=(y_raw >= 0).astype(int),
        is_train=np.asarray(dates <= train_end), is_test=np.asarray(dates >= test_start),
        inputs=X2, yhist=None, uses_yhist=False,
        columns=names, col_group=[C.group_of(n) for n in names],
        col_lag_days=np.array([C.tree_lag_days(n) for n in names]), T=None,
        regime_var=_prev_day(raw[C.REGIME_VARIABLE], dates),
        saved_test_prob=_saved_test_prob(run_dir), predict=predict,
        device=None, tree_model=model,
        meta={"run_dir": str(run_dir), "model_name": model_name, "lags": list(lags),
              "train_end": str(train_end.date()), "test_start": str(test_start.date()),
              "target_scale_c": scale_c, "n_design_features": len(names),
              "best_params": metrics.get("best_params")},
    )


# =================================================================== dispatch
def load_bundle(key: str, exp: str, seed: int, device) -> Bundle:
    spec = C.MODELS[key]
    loader = {"template": load_template, "jun": load_jun, "tree": load_tree}[spec["kind"]]
    return loader(key, spec, exp, seed, device)
