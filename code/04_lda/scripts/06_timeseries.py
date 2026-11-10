"""
Stage 6 -- topic time series  TR_{t,k}, per language.

Soft aggregation (Larsen & Thorsrud 2019), NOT the hard argmax count of
Han & Kim (2021):

    TR_{t,k} = (1 / N_t) * sum_{d published on day t} theta_{d,k}

where theta_{d,k} is the full inferred topic proportion for document d
(minimum_probability=0 so every one of the K topics is present).

Transforms (config-driven):
  * optional rolling-mean smoothing of the raw series (TR_ROLLING_WINDOW)
  * optional resample to W / M (TR_RESAMPLE)
  * optional YoY diff/growth (TR_YOY_MODE) -- default OFF for daily/weekly targets
  * z-score standardisation per topic column (TR_STANDARDIZE)

Outputs (per lang):
  output/06_topic_timeseries_{lang}_raw.csv     (date, N_t, topic_00..topic_{K-1})
  output/06_topic_timeseries_{lang}_zscore.csv
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel, LdaMulticore

import config


def _load_model(path: str):
    try:
        return LdaMulticore.load(path)
    except Exception:
        return LdaModel.load(path)


def _doc_topic_matrix(model, corpus) -> np.ndarray:
    k = model.num_topics
    out = np.zeros((len(corpus), k), dtype=np.float64)
    for i, bow in enumerate(corpus):
        for tid, prob in model.get_document_topics(bow, minimum_probability=0.0):
            out[i, tid] = prob
    # renormalise defensively (get_document_topics can round)
    rs = out.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return out / rs


def build_for_lang(lang: str) -> None:
    paths = config.corpus_paths(lang)
    model = _load_model(str(paths["model"]))
    corpus = MmCorpus(str(paths["corpus"]))
    meta = pd.read_parquet(paths["docmeta"]).reset_index(drop=True)
    k = model.num_topics

    theta = _doc_topic_matrix(model, corpus)
    assert len(theta) == len(meta), (len(theta), len(meta))

    cols = [f"topic_{i:02d}" for i in range(k)]
    doc_df = pd.DataFrame(theta, columns=cols)
    doc_df["date"] = pd.to_datetime(meta["id_date"]).dt.normalize().values

    # N_t and TR_{t,k} = daily mean of theta
    grp = doc_df.groupby("date")
    tr = grp[cols].mean()
    tr.insert(0, "N_t", grp.size())

    # full calendar, zero volume on gap days
    full = pd.date_range(tr.index.min(), tr.index.max(), freq="D")
    tr = tr.reindex(full)
    tr["N_t"] = tr["N_t"].fillna(0).astype(int)
    tr[cols] = tr[cols].fillna(0.0)
    tr.index.name = "date"

    # optional smoothing of the raw topic series (volume-agnostic)
    if config.TR_ROLLING_WINDOW:
        tr[cols] = tr[cols].rolling(config.TR_ROLLING_WINDOW, min_periods=1).mean()

    # optional resample (volume-weighted mean so busy days count more)
    if config.TR_RESAMPLE and config.TR_RESAMPLE != "D":
        w = tr[cols].mul(tr["N_t"], axis=0).resample(config.TR_RESAMPLE).sum()
        n = tr["N_t"].resample(config.TR_RESAMPLE).sum().replace(0, np.nan)
        tr = w.div(n, axis=0)
        tr.insert(0, "N_t", n.fillna(0).astype(int))
        tr[cols] = tr[cols].fillna(0.0)

    tr.to_csv(paths["tr_raw"], encoding="utf-8-sig")
    print(f"[{lang}] raw TR  {tr.shape}  -> {paths['tr_raw'].name}")

    # ---- downstream transform ---------------------------------------- #
    ser = tr.copy()
    if config.TR_YOY_MODE == "diff":
        lag = config.TR_YOY_PERIODS if config.TR_RESAMPLE in (None, "D") else 1 if config.TR_RESAMPLE == "M" else 52
        ser[cols] = ser[cols].diff(periods=lag)
    elif config.TR_YOY_MODE == "growth":
        lag = config.TR_YOY_PERIODS if config.TR_RESAMPLE in (None, "D") else 1 if config.TR_RESAMPLE == "M" else 52
        ser[cols] = ser[cols].pct_change(periods=lag)

    ser = ser.dropna(subset=cols, how="all")
    if config.TR_STANDARDIZE:
        mu = ser[cols].mean()
        sd = ser[cols].std(ddof=0).replace(0, np.nan)
        ser[cols] = (ser[cols] - mu) / sd
        ser[cols] = ser[cols].fillna(0.0)

    ser.to_csv(paths["tr_z"], encoding="utf-8-sig")
    print(f"[{lang}] transformed TR  {ser.shape}  "
          f"(yoy={config.TR_YOY_MODE}, z={config.TR_STANDARDIZE})  -> {paths['tr_z'].name}")


def main() -> None:
    for lang in ("ko", "en"):
        build_for_lang(lang)
    print("done. topic_XX columns match gensim topic ids (pyLDAvis shows id+1).")


if __name__ == "__main__":
    main()
