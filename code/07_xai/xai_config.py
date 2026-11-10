"""
XAI configuration -- which classifiers are explained, where their saved
artifacts live, and how every input column maps onto the four information
groups used in all figures.

Only the direction CLASSIFIERS are analysed.  Nothing here retrains a model:
every script loads the phase-2 checkpoint the E0-E7 run already saved
(best_model.pt / best_model.joblib + x_scaler.pkl) and rebuilds its inputs.

Edit this file (not the scripts) to change groups, regimes or sample sizes.
"""
from __future__ import annotations

from pathlib import Path

# ----------------------------------------------------------------- locations
DESKTOP = Path(r".")
XAI_ROOT = Path(__file__).resolve().parent
OUT_ROOT = XAI_ROOT / "outputs"        # outputs/<model>/<exp>/...
FIG_ROOT = XAI_ROOT / "figures"        # figures/<exp>/*.pdf  + tables/

EXPERIMENT = "E7"      # full information set: every group is present
SEED = 42

# kind: "jun"       single-file scripts in Desktop/Jun (features built from raw files)
#       "template"  BiLSTM / DA-RNN / iTransformer shared-template folders
#       "tree"      TreeModels_Classifier (lagged design matrix)
MODELS: dict[str, dict] = {
    "lstm": dict(kind="jun", family="Hybrid LSTM", label="LSTM",
                 folder=DESKTOP / "Jun" / "LSTM", script="lstm_classifier.py",
                 results="results_lstm_classifier_aligned"),
    "cnn_lstm": dict(kind="jun", family="Hybrid LSTM", label="CNN-LSTM",
                     folder=DESKTOP / "Jun" / "CNN-LSTM", script="cnn_lstm_classifier.py",
                     results="results_cnn_lstm_classifier_aligned"),
    "gru_lstm": dict(kind="jun", family="Hybrid LSTM", label="GRU-LSTM",
                     folder=DESKTOP / "Jun" / "GRU-LSTM", script="gru_lstm_classifier.py",
                     results="results_gru_lstm_classifier_aligned"),
    "bigru_lstm": dict(kind="jun", family="Hybrid LSTM", label="BiGRU-LSTM",
                       folder=DESKTOP / "Jun" / "BiGRU-LSTM", script="bigru_lstm_classifier.py",
                       results="results_bigru_lstm_classifier_aligned"),
    "bilstm": dict(kind="template", family="Hybrid LSTM", label="BiLSTM",
                   folder=DESKTOP / "BiLSTM_Classifier"),
    "darnn": dict(kind="template", family="Hybrid LSTM", label="DA-RNN",
                  folder=DESKTOP / "Att_LSTM_Classifier"),
    "itransformer": dict(kind="template", family="Transformer", label="iTransformer",
                         folder=DESKTOP / "iTransformer_Classifier"),
    "random_forest": dict(kind="tree", family="Tree ensemble", label="Random Forest",
                          folder=DESKTOP / "TreeModels_Classifier" / "RandomForest"),
    "xgboost": dict(kind="tree", family="Tree ensemble", label="XGBoost",
                    folder=DESKTOP / "TreeModels_Classifier" / "XGBoost"),
    "lightgbm": dict(kind="tree", family="Tree ensemble", label="LightGBM",
                     folder=DESKTOP / "TreeModels_Classifier" / "LightGBM"),
}
FAMILIES = ["Hybrid LSTM", "Transformer", "Tree ensemble"]

# ----------------------------------------------------------------- groups
# The four information sources of the XAI figures.  Order = row order everywhere.
GROUPS = ["Domestic news", "Foreign news", "Market", "Macro"]
GROUP_SLUG = {"Domestic news": "domestic_news", "Foreign news": "foreign_news",
              "Market": "market", "Macro": "macro"}
GROUP_COLORS = {"Domestic news": "#2a78d6", "Foreign news": "#eb6834",
                "Market": "#1baf7a", "Macro": "#4a3aa7"}

# The target's own history (foreign net-buy flow) is market trading data, so it
# sits in "Market".  To make it a fifth player instead, set KRX_GROUP to a new
# name AND add that name to GROUPS / GROUP_SLUG / GROUP_COLORS above.
KRX_COL = "foreign_net_buy"
KRX_GROUP = "Market"
YHIST_NAME = "foreign_net_buy[y_hist]"   # DA-RNN's decoder input: same series, own input

# Domestic = Korean LDA topics + NAVER.  Foreign = English LDA topics, GDELT,
# NYT, and MCST (foreign-press coverage of Korea, so a foreign source).
DOMESTIC_NEWS_PREFIXES = ("lda_ko", "naver")
FOREIGN_NEWS_PREFIXES = ("lda_en", "gdelt", "nyt", "newyorktimes", "mcst")
MARKET_COLS = {
    "foreign_net_buy", "sp500_logret", "global_logret", "sox_logret",
    "krw_usd", "krw_usd_logret", "vkospi",
}
MACRO_COLS = {
    "ktb3y_minus_base", "ktb3y_minus_msb91", "ktb10y_minus_ktb3y",
    "cpi_total", "cpi_yoy", "gdp_yoy", "gdp_qoq",
    "semi_prod_sa", "semi_prod_yoy", "unemp_rate",
    "korea_epu", "global_epu", "gpr_global", "gpr_korea",
}


def base_feature(col: str) -> str:
    """Strip the tree-design lag suffix: 'vkospi_lag5' -> 'vkospi'."""
    head, sep, tail = col.rpartition("_lag")
    return head if sep and tail.isdigit() else col


def tree_lag_days(col: str) -> int:
    """Days before the target a tree design column refers to: lag0 = t-1."""
    _, sep, tail = col.rpartition("_lag")
    return int(tail) + 1 if sep and tail.isdigit() else 1


def group_of(col: str) -> str:
    base = base_feature(col)
    if base in (KRX_COL, YHIST_NAME):
        return KRX_GROUP
    low = base.lower()
    if low.startswith(DOMESTIC_NEWS_PREFIXES):
        return "Domestic news"
    if low.startswith(FOREIGN_NEWS_PREFIXES):
        return "Foreign news"
    if low in MARKET_COLS:
        return "Market"
    if low in MACRO_COLS:
        return "Macro"
    raise KeyError(f"column '{col}' has no information group -- add '{low}' to "
                   f"MARKET_COLS or MACRO_COLS in xai_config.py")


def canonical_feature(col: str) -> str:
    """One name per underlying series across the two codebases, for the
    feature-level supplementary figure.  The Jun scripts number LDA topics by
    csv index (topic_16); the template uses paper numbering (lda_en_17 = +1)."""
    base = base_feature(col)
    if base == YHIST_NAME:
        return KRX_COL
    low = base.lower()
    for lang in ("en", "ko"):
        tag = f"lda_{lang}_topic_"
        if low.startswith(tag) and low[len(tag):].isdigit():
            return f"lda_{lang}_{int(low[len(tag):]) + 1:02d}"
    for src, name in (("gdelt", "gdelt_sentiment"), ("mcst", "mcst_sentiment"),
                      ("naver", "naver_sentiment"), ("newyorktimes", "nyt_sentiment"),
                      ("nyt", "nyt_sentiment")):
        if low.startswith(src) and "sentiment" in low:
            return name
    return low


# ----------------------------------------------------------------- lag buckets
# Columns of the "where in the lookback window" heatmaps.  Tree lags map in
# naturally: lag0 -> 1, lag1 -> 2, lag2 -> 3, lag5 -> 6, lag10 -> 11, lag20 -> 21.
LAG_BUCKETS = [("1", 1, 1), ("2", 2, 2), ("3", 3, 3),
               ("4–6", 4, 6), ("7–11", 7, 11), ("12+", 12, 10_000)]

# ----------------------------------------------------------------- regimes
# Volatility regime of each target day = VKOSPI on the PREVIOUS trading day
# (the level known when the forecast is made).  Thresholds come from the
# official-train days only, so the split uses no test-period information.
REGIME_VARIABLE = "vkospi"
REGIME_METHOD = "tercile"   # "tercile": low = bottom third, high = top third
                            # "median" : two halves
REGIME_REFERENCE = "train"
# The OUT-OF-SAMPLE period (2025-05 .. 2026-06) is itself a high-volatility
# episode: its minimum VKOSPI sits at the training 67th percentile, so with
# training thresholds it has no "low" days at all.  The performance game by
# regime (panel 3d) therefore splits the TEST days by their own terciles
# ("relatively calm" vs "relatively turbulent" test days).  Descriptive
# stratification only -- nothing here feeds the model.
REGIME_TEST_REFERENCE = "test"

# ----------------------------------------------------------------- sample sizes
N_BG_GROUP = 50        # background windows for the exact group Shapley values
GROUP_BATCH_SEQ = 4096 # masked windows per forward pass (sequence models)
GROUP_BATCH_TAB = 32768
N_BG_GRAD = 100        # background windows for shap.GradientExplainer
GRAD_NSAMPLES = 100    # expected-gradient samples per explained day
GRAD_BATCH = 50        # GradientExplainer internal batch
GRAD_CHUNK = 128       # days per shap_values call (progress granularity)
N_BG_TREE = 100        # background rows for shap.TreeExplainer (interventional)
TREE_CHUNK = 250
BOOT_B = 1000          # bootstrap replicates for the performance decomposition
BOOT_BLOCK = 10        # circular block length (trading days) -- keeps autocorrelation
VERIFY_TOL = 1e-4      # max |reconstructed - saved| test probability allowed
