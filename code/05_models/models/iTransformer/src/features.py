"""
Feature groups and the E0-E7 experiment grid.

    E0  KRX                       baseline
    E1  KRX + AUX
    E2  KRX + LDA
    E3  KRX + LLM
    E4  KRX + LDA + LLM
    E5  KRX + LDA + AUX
    E6  KRX + LLM + AUX
    E7  KRX + LDA + LLM + AUX     full model

Groups are resolved against the columns of data/input.csv:
    KRX  the target's own history        foreign_net_buy
    LDA  lda_*  topic z-scores
    LLM  *_sentiment  daily mean EXAONE scores (one per news source)
    AUX  every remaining column except AUX_EXCLUDE
"""
from __future__ import annotations

import pandas as pd

TARGET_COL = "foreign_net_buy"
KRX_COLS = [TARGET_COL]
AUX_EXCLUDE = ["korea_epu"]          # not published for 2026 -> dropped for the 2026 test

EXPERIMENTS = {
    "E0": ["KRX"],
    "E1": ["KRX", "AUX"],
    "E2": ["KRX", "LDA"],
    "E3": ["KRX", "LLM"],
    "E4": ["KRX", "LDA", "LLM"],
    "E5": ["KRX", "LDA", "AUX"],
    "E6": ["KRX", "LLM", "AUX"],
    "E7": ["KRX", "LDA", "LLM", "AUX"],
}


def feature_groups(df: pd.DataFrame) -> dict[str, list[str]]:
    cols = list(df.columns)
    if TARGET_COL not in cols:
        raise ValueError(f"input is missing the target column {TARGET_COL}")
    lda = [c for c in cols if c.startswith("lda_")]
    llm = [c for c in cols if c.endswith("_sentiment")]
    used = set(KRX_COLS) | set(lda) | set(llm) | set(AUX_EXCLUDE) | {"date"}
    aux = [c for c in cols if c not in used]
    return {"KRX": KRX_COLS, "LDA": lda, "LLM": llm, "AUX": aux}


def experiment_features(groups: dict[str, list[str]], exp: str) -> list[str]:
    if exp not in EXPERIMENTS:
        raise ValueError(f"unknown experiment {exp}; choose from {list(EXPERIMENTS)}")
    out: list[str] = []
    for g in EXPERIMENTS[exp]:
        if not groups[g]:
            raise ValueError(f"{exp} needs group {g} but no such columns are in the input")
        out += [c for c in groups[g] if c not in out]
    return out
