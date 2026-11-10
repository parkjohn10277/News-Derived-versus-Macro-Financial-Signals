"""
GRU-LSTM  |  DIRECTION of KRX foreign net buying  (E0~E7)  --  CLASSIFIER

This is the classification variant of gru_lstm_krx_lda_llm_aux_v2.py.
The research question is not "how many won", it is "will foreign investors be
net buyers or net sellers tomorrow", so the target is a 0/1 label and the model
emits a logit.

What changed versus the regression script
-----------------------------------------
1. Loss      nn.SmoothL1Loss()  ->  nn.BCEWithLogitsLoss(pos_weight=...)
             BCEWithLogitsLoss, never BCELoss: the sigmoid is fused into the
             loss for numerical stability, so the model must NOT apply one.
             The final layer is unchanged -- nn.Linear(hidden, 1) now emits a
             raw logit.
2. Target    y = (y_raw >= 0)  ->  1 = net buy, 0 = net sell.
             No y_scaler, no arcsinh on the target, no sinh inversion, no
             target clipping -- none of that applies to a probability.
             SCALE_C still rescales the KRX *input* feature, because the flow's
             own history is a predictor and still needs its tail tamed.
3. pos_weight  n_neg / n_pos on the TRAINING rows only.  The classes are about
             44/56, so this is a mild correction, but without it the model
             drifts to the majority class and MCC suffers.
4. Predict   torch.sigmoid(logit) at predict time only -> P(net buy).
             AUC uses the PROBABILITY, never the hard 0/1 call.
5. Metrics   MAE / RMSE / R2 / R2_oos are gone -- they no longer exist.
             Brier_Skill replaces R2_oos as the skill score:

                 Brier_Skill = 1 - Brier(model) / Brier(train base rate)

             0.00 = no better than always predicting the historical base rate,
             positive = genuine skill, negative = worse than the base rate.
             LogLoss_Skill is the same idea under log loss.

Also new
--------
* --horizon h : predict the direction h trading days ahead.  The input window
  ends h days before the predicted day, so h=1 reproduces the original setup
  and h>1 adds no leakage.
* All default data paths now resolve to  <parent folder>/data.

Split, feature groups, LDA topics and the AUX z-clip are untouched, so the
classifier and the regression run are directly comparable on the same days.
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ============================================================
# 0. 기본 경로
# ============================================================

DATA_DIR = (
    Path(__file__).resolve().parent.parent
    / "data"
)

DEFAULT_KRX_PATH = str(
    DATA_DIR / "KRX_150_companies.csv"
)

DEFAULT_LDA_EN_PATH = str(
    DATA_DIR / "06_topic_timeseries_en_zscore.csv"
)

DEFAULT_LDA_KO_PATH = str(
    DATA_DIR / "06_topic_timeseries_ko_zscore.csv"
)

DEFAULT_AUX_PATH = str(
    DATA_DIR / "panel_daily_ffill.csv"
)

DEFAULT_SENT_GDELT_PATH = str(
    DATA_DIR / "GDELT_EXAONE_7.8B_sentiment_score.csv"
)

DEFAULT_SENT_MCST_PATH = str(
    DATA_DIR / "MCST_EXAONE_7.8B_sentiment_score.csv"
)

DEFAULT_SENT_NAVER_PATH = str(
    DATA_DIR / "NAVER_EXAONE_7.8B_sentiment_score.csv"
)

DEFAULT_SENT_NYT_PATH = str(
    DATA_DIR / "NewYorkTimes_EXAONE_7.8B_sentiment_score.csv"
)


# ============================================================
# 1. 연구 설정
# ============================================================

DATE_COL = "date"
TARGET_COL = "foreign_net_buy"

# ------------------------------------------------------------
# [v2 변경 1] 80 / 10 / 10 분할
# ------------------------------------------------------------
# 전체 사용 가능 행 수 = 2,780
#   Train      2015-03-02 ~ 2024-03-12   2,224 rows (80%)
#   Validation 2024-03-13 ~ 2025-05-07     278 rows (10%)
#   Test       2025-05-08 ~ 2026-06-26     278 rows (10%)
OFFICIAL_TRAIN_START = pd.Timestamp("2015-02-28")
OFFICIAL_TRAIN_END = pd.Timestamp("2025-05-07")   # train + validation (PHASE 2 재학습 구간)
TEST_START = pd.Timestamp("2025-05-08")

# Test를 건드리지 않고 best epoch를 선택하기 위한 내부 validation
VAL_START = pd.Timestamp("2024-03-13")
VAL_END = pd.Timestamp("2025-05-07")

# ------------------------------------------------------------
# 최종 선정 LDA Topics
# ------------------------------------------------------------
# English: Topic 16, 17, 18
SELECTED_LDA_EN_TOPICS = [16, 17, 18]

# Korean: Topic 3, 4, 5, 6, 7, 11, 12, 19
SELECTED_LDA_KO_TOPICS = [3, 4, 5, 6, 7, 11, 12, 19]

# 총 LDA 입력변수 수 = 11개 (English 3 + Korean 8)

# 업로드된 KRX 파일의 실제 컬럼
KRX_DATE_CANDIDATES = ["일자", "date", "Date"]
KRX_TARGET_CANDIDATES = [
    "외국인_순매수금액_총합",
    "foreign_net_buy",
    "foreign_net_buying",
]

# 현재 Auxiliary 파일에서 실제 대응 가능한 컬럼
AUX_FEATURES = [
    "korea_epu",
    "global_epu",
    "cpi_yoy",
    "gdp_yoy",
    "krw_usd",
    "vkospi",
    "ktb3y_minus_base",
    "gpr_global",
    "semi_prod_yoy",
    "unemp_rate",
    "sp500_logret",
    "global_logret",
]

# ESG가 이후 추가될 경우 자동으로 포함 시도
OPTIONAL_AUX_FEATURES = [
    "ESG_Rating",
    "ESG_Rating_Score",
    "ESG",
]

# 반도체 특화 보조변수로 추가하고 싶다면 아래 주석 해제
OPTIONAL_SEMICONDUCTOR_FEATURES = [
    # "sox_logret",
]

# LLM 일별 감성 컬럼은 추후 감성 파일 병합 후 자동 탐지
# 이름에 sentiment_score가 있으면 LLM 그룹으로 인식
LLM_KEYWORDS = [
    "sentiment_score",
    "exaone",
]


# ============================================================
# 2. 유틸
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def canonical(s: str) -> str:
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())


def read_csv_flexible(path: str | Path) -> pd.DataFrame:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    last_error = None

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as e:
            last_error = e

    if last_error is not None:
        raise last_error

    return pd.read_csv(path)


def resolve_one_column(
    df: pd.DataFrame,
    candidates: list[str],
    required: bool = False,
) -> str | None:

    cmap = {canonical(c): c for c in df.columns}

    for candidate in candidates:
        key = canonical(candidate)

        if key in cmap:
            return cmap[key]

    if required:
        raise ValueError(
            f"컬럼을 찾을 수 없습니다.\n"
            f"후보: {candidates}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    return None


def resolve_columns(
    df: pd.DataFrame,
    requested: list[str],
) -> list[str]:

    cmap = {canonical(c): c for c in df.columns}
    out = []

    for name in requested:
        key = canonical(name)

        if key in cmap:
            out.append(cmap[key])

    return out


# ============================================================
# 3. 파일 로딩 / 병합
# ============================================================

def load_krx(path: str) -> tuple[pd.DataFrame, list[str]]:
    """
    KRX 파일 구조:
    일자
    외국인_순매수금액_총합
    001_삼성전자 ... 150_아이씨티케이

    E0의 KRX 입력은 '외국인_순매수금액_총합'의 과거 시계열 1개.
    target은 다음 거래일의 150개 기업 외국인 순매수 총합.

    개별기업 149개 유효 컬럼은 데이터 무결성 검증용으로만 사용하고,
    CNN-LSTM 입력 feature로는 사용하지 않습니다.
    """

    df = read_csv_flexible(path)

    date_col = resolve_one_column(
        df,
        KRX_DATE_CANDIDATES,
        required=True,
    )

    target_col = resolve_one_column(
        df,
        KRX_TARGET_CANDIDATES,
        required=True,
    )

    rename_map = {
        date_col: DATE_COL,
        target_col: TARGET_COL,
    }

    df = df.rename(columns=rename_map)

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=[DATE_COL])
        .sort_values(DATE_COL)
        .drop_duplicates(DATE_COL)
        .reset_index(drop=True)
    )

    # 001_기업명 형태의 150개 기업 컬럼 자동 탐지
    company_cols = [
        c for c in df.columns
        if re.match(r"^\d{3}_", str(c))
    ]

    if not company_cols:
        raise ValueError(
            "KRX 개별기업 컬럼(예: 001_삼성전자)을 찾지 못했습니다."
        )

    # 전체기간이 모두 NaN인 placeholder는 제거
    all_nan_cols = [
        c for c in company_cols
        if df[c].isna().all()
    ]

    if all_nan_cols:
        print(
            "[INFO] 전체 기간이 NaN이라 제거하는 KRX 컬럼:",
            all_nan_cols,
        )

    company_cols = [
        c for c in company_cols
        if c not in all_nan_cols
    ]

    # target과 기업별 값 숫자 변환
    numeric_cols = [TARGET_COL] + company_cols

    for c in numeric_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    if df[TARGET_COL].isna().any():
        raise ValueError("KRX target에 결측치가 있습니다.")

    # KRX 총합은 원 파일에서 NaN 기업을 제외하고 합산됨.
    # 따라서 상장 전/데이터 없음 NaN을 0으로 두는 것이 총합 정의와 일치.
    df[company_cols] = df[company_cols].fillna(0.0)

    # 데이터 무결성 확인: 개별기업 합 == 제공된 총합
    reconstructed = df[company_cols].sum(axis=1)

    max_abs_diff = float(
        np.max(
            np.abs(
                reconstructed.to_numpy()
                - df[TARGET_COL].to_numpy()
            )
        )
    )

    print(f"[INFO] KRX rows: {len(df):,}")
    print(f"[INFO] KRX company features: {len(company_cols)}")
    print(
        "[INFO] max |provided target - company sum| = "
        f"{max_abs_diff:,.6f}"
    )

    # 연구 설계에 맞춰 KRX feature는 총합 시계열 1개만 사용
    krx_feature_cols = [TARGET_COL]

    print(
        "[INFO] GRU-LSTM KRX input feature: "
        f"{TARGET_COL} (1 aggregate series)"
    )

    return df, krx_feature_cols


def load_lda(
    path: str,
    prefix: str,
    selected_topics: list[int],
) -> tuple[pd.DataFrame, list[str]]:
    """
    06_topic_timeseries_*_zscore.csv:
    date, N_t, topic_00 ... topic_19

    topic들만 LDA feature로 사용.
    N_t는 기본 실험에서는 제외.
    """

    df = read_csv_flexible(path)

    date_col = resolve_one_column(
        df,
        ["date", "Date"],
        required=True,
    )

    if date_col != DATE_COL:
        df = df.rename(
            columns={date_col: DATE_COL}
        )

    df[DATE_COL] = pd.to_datetime(
        df[DATE_COL],
        errors="coerce",
    )

    df = (
        df
        .dropna(subset=[DATE_COL])
        .sort_values(DATE_COL)
        .drop_duplicates(DATE_COL)
        .reset_index(drop=True)
    )

    # 선정된 topic만 사용
    # 예: Topic 4 -> topic_04
    raw_topic_cols = [
        f"topic_{topic_id:02d}"
        for topic_id in selected_topics
    ]

    missing_topics = [
        c for c in raw_topic_cols
        if c not in df.columns
    ]

    if missing_topics:
        raise ValueError(
            f"선택한 LDA topic 컬럼이 파일에 없습니다: {missing_topics}\n"
            f"파일: {path}\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    rename_map = {
        c: f"{prefix}_{c}"
        for c in raw_topic_cols
    }

    df = df.rename(columns=rename_map)

    topic_cols = [
        rename_map[c]
        for c in raw_topic_cols
    ]

    print(
        f"[INFO] {prefix} selected LDA topics: "
        + ", ".join(raw_topic_cols)
    )

    keep_cols = [DATE_COL] + topic_cols

    for c in topic_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    return df[keep_cols], topic_cols


def merge_auxiliary(
    base_df: pd.DataFrame,
    aux_path: str | None,
) -> pd.DataFrame:

    if not aux_path:
        return base_df

    aux_file = Path(aux_path)

    if not aux_file.exists():
        print(
            "[WARNING] Auxiliary 파일이 없어 병합을 건너뜁니다:",
            aux_path,
        )
        return base_df

    aux = read_csv_flexible(aux_file)

    aux_date_col = resolve_one_column(
        aux,
        ["date", "Date", "일자", "trading_date", "day"],
        required=True,
    )

    if aux_date_col != DATE_COL:
        aux = aux.rename(
            columns={aux_date_col: DATE_COL}
        )

    aux[DATE_COL] = pd.to_datetime(
        aux[DATE_COL],
        errors="coerce",
    )

    aux = (
        aux
        .dropna(subset=[DATE_COL])
        .sort_values(DATE_COL)
        .drop_duplicates(DATE_COL)
        .reset_index(drop=True)
    )

    # 같은 이름이 이미 있으면 base 우선
    merge_cols = [
        c for c in aux.columns
        if c == DATE_COL or c not in base_df.columns
    ]

    merged = base_df.merge(
        aux[merge_cols],
        on=DATE_COL,
        how="left",
        validate="one_to_one",
    )

    print("[INFO] Auxiliary merged")

    return merged


def load_sentiment_daily(
    path: str,
    source_name: str,
) -> tuple[pd.DataFrame, str]:
    """
    기사 단위/일 단위 감성 CSV를 일별 평균 감성점수로 변환합니다.

    출력 컬럼 예:
      GDELT_sentiment_score
      MCST_sentiment_score
      NAVER_sentiment_score
      NewYorkTimes_sentiment_score
    """

    df = read_csv_flexible(path)

    date_col = resolve_one_column(
        df,
        [
            "date",
            "Date",
            "일자",
            "날짜",
            "published_date",
            "publish_date",
            "publication_date",
            "pub_date",
            "datetime",
            "published_at",
            "time",
        ],
        required=False,
    )

    if date_col is None:
        # 이름에 date/time이 들어간 컬럼을 추가 탐색
        candidates = [
            c for c in df.columns
            if "date" in str(c).lower()
            or "time" in str(c).lower()
            or "날짜" in str(c)
            or "일자" in str(c)
        ]
        if not candidates:
            raise ValueError(
                f"[{source_name}] 날짜 컬럼을 찾지 못했습니다.\n"
                f"현재 컬럼: {list(df.columns)}"
            )
        date_col = candidates[0]

    # 감성점수 컬럼 탐색
    # 예: sentiment_score, EXAONE_Score
    score_candidates = [
        c for c in df.columns
        if (
            ("sentiment" in str(c).lower() and "score" in str(c).lower())
            or canonical(c) in {
                "sentimentscore",
                "score",
                "exaonescore",
                "exaonesentimentscore",
            }
        )
    ]

    if not score_candidates:
        raise ValueError(
            f"[{source_name}] sentiment score 컬럼을 찾지 못했습니다.\n"
            f"현재 컬럼: {list(df.columns)}"
        )

    # 가장 명확한 sentiment_score 컬럼 우선
    score_col = sorted(
        score_candidates,
        key=lambda c: (
            0 if canonical(c) == "sentimentscore" else 1,
            len(str(c)),
        ),
    )[0]

    temp = df[[date_col, score_col]].copy()

    temp[DATE_COL] = pd.to_datetime(
        temp[date_col],
        errors="coerce",
    ).dt.normalize()

    temp["_score"] = pd.to_numeric(
        temp[score_col],
        errors="coerce",
    )

    temp = temp.dropna(
        subset=[DATE_COL, "_score"]
    )

    output_col = f"{source_name}_sentiment_score"

    # 기사 단위 파일이면 동일 날짜 기사들의 평균 감성점수 사용
    daily = (
        temp.groupby(DATE_COL, as_index=False)["_score"]
        .mean()
        .rename(columns={"_score": output_col})
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    print(
        f"[INFO] {source_name} sentiment: "
        f"{len(df):,} raw rows -> {len(daily):,} daily rows | "
        f"score column='{score_col}'"
    )

    return daily, output_col


def merge_sentiments(
    base_df: pd.DataFrame,
    sentiment_paths: dict[str, str | None],
) -> tuple[pd.DataFrame, list[str]]:

    df = base_df.copy()
    llm_cols: list[str] = []

    for source_name, path in sentiment_paths.items():

        if not path:
            print(
                f"[WARNING] {source_name} sentiment path가 비어 있어 건너뜁니다."
            )
            continue

        file_path = Path(path)

        if not file_path.exists():
            print(
                f"[WARNING] {source_name} sentiment 파일을 찾을 수 없어 "
                f"건너뜁니다: {path}"
            )
            continue

        daily, out_col = load_sentiment_daily(
            path=path,
            source_name=source_name,
        )

        df = df.merge(
            daily,
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )

        llm_cols.append(out_col)

    if llm_cols:
        print(
            "[INFO] LLM sentiment merged:",
            ", ".join(llm_cols),
        )

    return df, llm_cols



def experiment_requirements(exp: str) -> dict[str, bool]:
    """
    E0~E7에 필요한 데이터 소스만 로드합니다.

    E0: KRX
    E1: KRX + AUX
    E2: KRX + LDA
    E3: KRX + LLM
    E4: KRX + LDA + LLM
    E5: KRX + LDA + AUX
    E6: KRX + LLM + AUX
    E7: KRX + LDA + LLM + AUX
    """
    table = {
        "E0": {"lda": False, "llm": False, "aux": False},
        "E1": {"lda": False, "llm": False, "aux": True},
        "E2": {"lda": True,  "llm": False, "aux": False},
        "E3": {"lda": False, "llm": True,  "aux": False},
        "E4": {"lda": True,  "llm": True,  "aux": False},
        "E5": {"lda": True,  "llm": False, "aux": True},
        "E6": {"lda": False, "llm": True,  "aux": True},
        "E7": {"lda": True,  "llm": True,  "aux": True},
    }
    return table[exp]


# ------------------------------------------------------------
# [ALIGNED INPUTS] identical inputs to the template / tree models
# ------------------------------------------------------------
# By default this script now reads data/input_aligned.csv -- a copy of the
# exact input.csv the BiLSTM / DA-RNN / iTransformer / tree models use.  Same
# values, same AUX set (19 variables incl. SOX; korea_epu excluded because it
# is not published for 2026), same news timing (news dated d enters on the
# next trading day, weekend news rolled forward).  Only the architecture
# differs between the two codebases.  --raw-pipeline restores the original
# raw-file merge.
ALIGNED_INPUT_CSV = str(DATA_DIR / "input_aligned.csv")
ALIGNED_AUX_FEATURES = [
    "sp500_logret", "global_logret", "sox_logret", "krw_usd", "vkospi",
    "ktb3y_minus_base", "cpi_yoy", "gdp_yoy", "semi_prod_yoy", "unemp_rate",
    "global_epu", "gpr_global", "gpr_korea",
]


def load_aligned_input(path: str):
    df = read_csv_flexible(path)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    lda_en = [c for c in df.columns if c.startswith("lda_en_")]
    lda_ko = [c for c in df.columns if c.startswith("lda_ko_")]
    llm = [c for c in df.columns if c.endswith("_sentiment")]
    print(f"[INFO] aligned input {path}: {len(df):,} rows | LDA EN {len(lda_en)} + KO "
          f"{len(lda_ko)} | LLM {len(llm)} | AUX {len(ALIGNED_AUX_FEATURES)}")
    return df, [TARGET_COL], lda_en, lda_ko, llm


def merge_all_data(
    exp: str,
    krx_path: str,
    lda_en_path: str,
    lda_ko_path: str,
    aux_path: str | None,
    sent_gdelt_path: str | None,
    sent_mcst_path: str | None,
    sent_naver_path: str | None,
    sent_nyt_path: str | None,
) -> tuple[
    pd.DataFrame,
    list[str],
    list[str],
    list[str],
    list[str],
]:
    """
    실험에 필요한 파일만 읽습니다.
    E0는 KRX만 읽으므로 LDA/LLM/AUX 파일 오류의 영향을 받지 않습니다.
    """

    req = experiment_requirements(exp)

    # 항상 KRX는 필요
    krx, krx_cols = load_krx(krx_path)
    df = krx.copy()

    lda_en_cols: list[str] = []
    lda_ko_cols: list[str] = []
    llm_cols: list[str] = []

    # LDA가 필요한 실험만 로드
    if req["lda"]:
        lda_en, lda_en_cols = load_lda(
            lda_en_path,
            prefix="lda_en",
            selected_topics=SELECTED_LDA_EN_TOPICS,
        )
        lda_ko, lda_ko_cols = load_lda(
            lda_ko_path,
            prefix="lda_ko",
            selected_topics=SELECTED_LDA_KO_TOPICS,
        )

        df = df.merge(
            lda_en,
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )
        df = df.merge(
            lda_ko,
            on=DATE_COL,
            how="left",
            validate="one_to_one",
        )

        lda_cols = lda_en_cols + lda_ko_cols

        lda_missing = df[lda_cols].isna().all(axis=1).sum()

        print(
            f"[INFO] LDA features: "
            f"EN {len(lda_en_cols)} + "
            f"KO {len(lda_ko_cols)} = "
            f"{len(lda_cols)}"
        )
        print(
            "[INFO] Trading days with all LDA missing:",
            int(lda_missing),
        )

        # 뉴스가 없는 거래일은 topic z-score를 0으로 처리
        df[lda_cols] = df[lda_cols].fillna(0.0)

    # 보조변수가 필요한 실험만 로드
    if req["aux"]:
        df = merge_auxiliary(
            df,
            aux_path,
        )

    # LLM이 필요한 실험만 로드
    if req["llm"]:
        sentiment_paths = {
            "GDELT": sent_gdelt_path,
            "MCST": sent_mcst_path,
            "NAVER": sent_naver_path,
            "NewYorkTimes": sent_nyt_path,
        }

        df, llm_cols = merge_sentiments(
            df,
            sentiment_paths,
        )

    return (
        df,
        krx_cols,
        lda_en_cols,
        lda_ko_cols,
        llm_cols,
    )


# ============================================================
# 4. Feature 그룹
# ============================================================

def detect_llm_columns(df: pd.DataFrame) -> list[str]:
    """
    추후 일별 LLM 감성 컬럼이 병합됐을 때 자동 탐지.
    """
    out = []

    for c in df.columns:
        cc = str(c).lower()

        if (
            "sentiment_score" in cc
            or (
                "exaone" in cc
                and "sentiment" in cc
            )
        ):
            out.append(c)

    return sorted(set(out))


def get_feature_groups(
    df: pd.DataFrame,
    krx_cols: list[str],
    lda_cols: list[str],
    llm_cols: list[str] | None = None,
) -> dict[str, list[str]]:

    aux_cols = resolve_columns(
        df,
        AUX_FEATURES,
    )

    for c in resolve_columns(
        df,
        OPTIONAL_AUX_FEATURES
        + OPTIONAL_SEMICONDUCTOR_FEATURES,
    ):
        if c not in aux_cols:
            aux_cols.append(c)

    if llm_cols is None:
        llm_cols = detect_llm_columns(df)

    return {
        "KRX": krx_cols,
        "LDA": lda_cols,
        "LLM": llm_cols,
        "AUX": aux_cols,
    }


def get_experiment_features(
    groups: dict[str, list[str]],
    exp: str,
) -> list[str]:

    KRX = groups["KRX"]
    LDA = groups["LDA"]
    LLM = groups["LLM"]
    AUX = groups["AUX"]

    experiments = {
        "E0": KRX,
        "E1": KRX + AUX,
        "E2": KRX + LDA,
        "E3": KRX + LLM,
        "E4": KRX + LDA + LLM,
        "E5": KRX + LDA + AUX,
        "E6": KRX + LLM + AUX,
        "E7": KRX + LDA + LLM + AUX,
    }

    # LLM 실험은 4개 뉴스 소스를 모두 사용하는 것을 원칙으로 함
    if exp in {"E3", "E4", "E6", "E7"} and len(LLM) != 4:
        raise ValueError(
            f"{exp}는 4개 LLM sentiment feature가 필요하지만 "
            f"현재 {len(LLM)}개만 병합되었습니다: {LLM}\n"
            "필요 소스: GDELT / MCST / NAVER / NewYorkTimes"
        )

    features = []

    for c in experiments[exp]:
        if c not in features:
            features.append(c)

    if not features:
        raise ValueError(
            f"{exp} 입력 feature가 없습니다."
        )

    return features



def clip_standardized_auxiliary(
    X_scaled: np.ndarray,
    feature_cols: list[str],
    aux_cols: list[str],
    clip_value: float,
    stage_name: str,
) -> tuple[np.ndarray, int]:
    """
    StandardScaler 적용 후 Auxiliary 변수에만 z-score clipping을 적용합니다.

    중요:
    - StandardScaler 자체는 Train 구간에만 fit됩니다.
    - clip 기준은 고정값(기본 ±3)이라 Test 정보를 사용하지 않습니다.
    - KRX/LDA/LLM 변수는 이 함수에서 clipping하지 않습니다.
    """
    if clip_value is None or clip_value <= 0:
        print(f"[INFO] {stage_name}: AUX z-score clipping disabled")
        return X_scaled, 0

    aux_indices = [
        i for i, c in enumerate(feature_cols)
        if c in aux_cols
    ]

    if not aux_indices:
        print(f"[INFO] {stage_name}: no AUX features to clip")
        return X_scaled, 0

    X_out = X_scaled.copy()

    before = X_out[:, aux_indices]
    clipped_mask = np.abs(before) > clip_value
    n_clipped = int(clipped_mask.sum())

    max_abs_before = float(
        np.nanmax(np.abs(before))
    ) if before.size else 0.0

    X_out[:, aux_indices] = np.clip(
        before,
        -clip_value,
        clip_value,
    )

    print(
        f"[INFO] {stage_name}: AUX z-score clip "
        f"[-{clip_value}, +{clip_value}] | "
        f"clipped values={n_clipped:,} | "
        f"max |z| before={max_abs_before:.3f}"
    )

    return X_out, n_clipped


# ============================================================
# 5. Sequence Dataset
# ============================================================

class SequenceDataset(Dataset):
    """
    target index = t

    X:
      t-lookback ... t-1

    y:
      t

    즉 과거 30거래일 -> 다음 거래일 외국인 순매수 총합.
    """

    def __init__(
        self,
        X_all: np.ndarray,
        y_all: np.ndarray,
        target_indices: np.ndarray,
        lookback: int,
        horizon: int = 1,
    ):
        self.X_all = X_all
        self.y_all = y_all
        self.target_indices = target_indices
        self.lookback = lookback
        self.horizon = int(horizon)

    def __len__(self):
        return len(self.target_indices)

    def __getitem__(self, idx):

        t = int(
            self.target_indices[idx]
        )

        # The window ENDS `horizon` days before the predicted day, so nothing
        # inside it is dated on or after day t.  horizon=1 -> [t-lookback, t).
        end = t - self.horizon + 1

        X = self.X_all[
            end - self.lookback:end
        ]

        y = self.y_all[t]

        return (
            torch.tensor(
                X,
                dtype=torch.float32,
            ),
            torch.tensor(
                [y],
                dtype=torch.float32,
            ),
            t,
        )


# ============================================================
# 6. GRU-LSTM
# ============================================================

class GRULSTMModel(nn.Module):

    def __init__(
        self,
        n_features: int,
        gru_hidden: int = 64,
        lstm_hidden: int = 64,
        dense_hidden: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.lstm = nn.LSTM(
            input_size=gru_hidden,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(dropout)

        self.dense = nn.Linear(lstm_hidden, dense_hidden)
        self.relu = nn.ReLU()
        self.out = nn.Linear(dense_hidden, 1)

    def forward(self, x):
        x, _ = self.gru(x)
        x = self.dropout1(x)

        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.dropout2(x)

        x = self.relu(self.dense(x))
        return self.out(x)

# ============================================================
# 7. Training
# ============================================================

@torch.no_grad()
def evaluate_loss(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    losses = []

    for xb, yb, _ in loader:

        xb = xb.to(device)
        yb = yb.to(device)

        pred = model(xb)

        loss = criterion(
            pred,
            yb,
        )

        losses.append(
            loss.item()
        )

    return (
        float(np.mean(losses))
        if losses
        else math.nan
    )


def fit_with_validation(
    model,
    train_loader,
    val_loader,
    device,
    lr,
    max_epochs,
    patience,
    pos_weight=None,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # BCEWithLogitsLoss, NOT BCELoss: the sigmoid is fused in for numerical
    # stability, so the model must not apply one itself.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    best_val = float("inf")
    best_epoch = 1
    best_state = None
    wait = 0
    history = []

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        batch_losses = []

        for xb, yb, _ in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            batch_losses.append(
                loss.item()
            )

        train_loss = float(
            np.mean(batch_losses)
        )

        val_loss = evaluate_loss(
            model,
            val_loader,
            criterion,
            device,
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(
            f"[SELECT] Epoch {epoch:03d} | "
            f"train={train_loss:.6f} | "
            f"val={val_loss:.6f}"
        )

        if val_loss < best_val - 1e-6:

            best_val = val_loss
            best_epoch = epoch

            best_state = {
                k: v.detach().cpu().clone()
                for k, v
                in model.state_dict().items()
            }

            wait = 0

        else:

            wait += 1

            if wait >= patience:
                print(
                    "[SELECT] Early stopping | "
                    f"best epoch={best_epoch}"
                )
                break

    if best_state is None:
        raise RuntimeError(
            "best_state 생성 실패"
        )

    model.load_state_dict(
        best_state
    )

    return (
        pd.DataFrame(history),
        best_epoch,
        best_val,
    )


def fit_fixed_epochs(
    model,
    train_loader,
    device,
    lr,
    epochs,
    pos_weight=None,
):

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    # BCEWithLogitsLoss, NOT BCELoss: the sigmoid is fused in for numerical
    # stability, so the model must not apply one itself.
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    history = []

    for epoch in range(
        1,
        epochs + 1,
    ):

        model.train()

        losses = []

        for xb, yb, _ in train_loader:

            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            pred = model(xb)

            loss = criterion(
                pred,
                yb,
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )

            optimizer.step()

            losses.append(
                loss.item()
            )

        epoch_loss = float(
            np.mean(losses)
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss,
            }
        )

        print(
            f"[REFIT ] Epoch "
            f"{epoch:03d}/{epochs:03d} | "
            f"train={epoch_loss:.6f}"
        )

    return pd.DataFrame(
        history
    )


@torch.no_grad()
def predict(
    model,
    loader,
    device,
):

    model.eval()

    predictions = []
    actuals = []
    indices = []

    for xb, yb, ib in loader:

        xb = xb.to(device)

        # sigmoid HERE, at predict time only -- it is fused into the loss
        # during training, so the model itself emits a raw logit.
        pred = (
            torch.sigmoid(model(xb))
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        predictions.extend(
            pred.tolist()
        )

        actuals.extend(
            yb.numpy()
            .reshape(-1)
            .tolist()
        )

        indices.extend(
            ib.numpy()
            .reshape(-1)
            .tolist()
        )

    return (
        np.asarray(predictions),
        np.asarray(actuals),
        np.asarray(
            indices,
            dtype=int,
        ),
    )


# ============================================================
# 8. main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--krx",
        default=DEFAULT_KRX_PATH,
    )

    parser.add_argument(
        "--lda-en",
        default=DEFAULT_LDA_EN_PATH,
    )

    parser.add_argument(
        "--lda-ko",
        default=DEFAULT_LDA_KO_PATH,
    )

    parser.add_argument(
        "--aux",
        default=DEFAULT_AUX_PATH,
    )


    parser.add_argument(
        "--sent-gdelt",
        default=DEFAULT_SENT_GDELT_PATH,
    )

    parser.add_argument(
        "--sent-mcst",
        default=DEFAULT_SENT_MCST_PATH,
    )

    parser.add_argument(
        "--sent-naver",
        default=DEFAULT_SENT_NAVER_PATH,
    )

    parser.add_argument(
        "--sent-nyt",
        default=DEFAULT_SENT_NYT_PATH,
    )

    parser.add_argument(
        "--exp",
        default="E0",
        choices=[f"E{i}" for i in range(8)],
    )

    parser.add_argument(
        "--lookback",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help=(
            "Predict the direction h trading days ahead. "
            "The input window ends h days before the predicted day."
        ),
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="P(net buy) cutoff for the hard 0/1 call.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
    )


    parser.add_argument(
        "--aux-z-clip",
        type=float,
        default=3.0,
        help=(
            "StandardScaler 이후 Auxiliary 변수 z-score clipping 범위. "
            "기본값 3.0이면 [-3, +3]. 0 이하이면 비활성화."
        ),
    )

    parser.add_argument(
        "--outdir",
        default="results_gru_lstm_classifier_aligned",
    )

    parser.add_argument(
        "--raw-pipeline",
        action="store_true",
        help="use the original raw-file merge instead of the aligned input.csv",
    )

    args = parser.parse_args()

    if not args.raw_pipeline:
        globals()["AUX_FEATURES"] = ALIGNED_AUX_FEATURES

    set_seed(
        args.seed
    )

    # 상대경로는 현재 PowerShell 위치가 아니라 이 .py 파일 위치를 기준으로 저장
    script_dir = Path(__file__).resolve().parent
    outdir_base = Path(args.outdir)

    if not outdir_base.is_absolute():
        outdir_base = script_dir / outdir_base

    output_dir = (
        outdir_base
        / args.exp
        / f"seed_{args.seed}"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 병합
    # --------------------------------------------------------

    df, krx_cols, lda_en_cols, lda_ko_cols, llm_cols = (
        load_aligned_input(ALIGNED_INPUT_CSV) if not args.raw_pipeline else
        merge_all_data(
            exp=args.exp,
            krx_path=args.krx,
            lda_en_path=args.lda_en,
            lda_ko_path=args.lda_ko,
            aux_path=args.aux,
            sent_gdelt_path=args.sent_gdelt,
            sent_mcst_path=args.sent_mcst,
            sent_naver_path=args.sent_naver,
            sent_nyt_path=args.sent_nyt,
        )
    )

    # 지정 시작일 이후만 사용
    df = df[
        df[DATE_COL]
        >= OFFICIAL_TRAIN_START
    ].copy()

    df = (
        df
        .sort_values(DATE_COL)
        .reset_index(drop=True)
    )

    lda_cols = (
        lda_en_cols
        + lda_ko_cols
    )

    groups = get_feature_groups(
        df,
        krx_cols=krx_cols,
        lda_cols=lda_cols,
        llm_cols=llm_cols,
    )

    feature_cols = get_experiment_features(
        groups,
        args.exp,
    )

    # --------------------------------------------------------
    # Feature 결측 처리
    # --------------------------------------------------------

    # KRX 입력은 외국인 순매수 총합 1개 시계열.
    # target 결측은 load_krx 단계에서 이미 검증했으므로 임의 보간하지 않습니다.

    # LDA z-score:
    df[lda_cols] = (
        df[lda_cols]
        .fillna(0.0)
    )

    # Auxiliary:
    # 미래값 backward-fill 금지, 과거값 forward-fill만 허용
    aux_cols = groups["AUX"]

    for c in aux_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    if aux_cols:
        df[aux_cols] = (
            df[aux_cols]
            .ffill()
        )

    # LLM은 추후 연결 시 없는 날 0 처리
    llm_cols = groups["LLM"]

    if llm_cols:
        for c in llm_cols:
            df[c] = pd.to_numeric(
                df[c],
                errors="coerce",
            )

        df[llm_cols] = (
            df[llm_cols]
            .fillna(0.0)
        )

    # 모든 사용 feature 숫자화
    for c in feature_cols:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    # target
    df[TARGET_COL] = pd.to_numeric(
        df[TARGET_COL],
        errors="coerce",
    )

    df = df.dropna(
        subset=[TARGET_COL]
    ).reset_index(drop=True)

    # 현재 선택 실험에서 필요한 feature에 남은 NaN 제거
    before_drop = len(df)

    df = df.dropna(
        subset=feature_cols
    ).reset_index(drop=True)

    dropped = before_drop - len(df)

    if dropped:
        print(
            f"[WARNING] feature 결측으로 {dropped} rows 제거"
        )

    # --------------------------------------------------------
    # 날짜 split
    # --------------------------------------------------------

    official_train_mask = (
        (df[DATE_COL] >= OFFICIAL_TRAIN_START)
        & (df[DATE_COL] <= OFFICIAL_TRAIN_END)
    )

    selection_train_mask = (
        (df[DATE_COL] >= OFFICIAL_TRAIN_START)
        & (df[DATE_COL] < VAL_START)
    )

    validation_mask = (
        (df[DATE_COL] >= VAL_START)
        & (df[DATE_COL] <= VAL_END)
    )

    test_mask = (
        df[DATE_COL] >= TEST_START
    )

    if selection_train_mask.sum() == 0:
        raise ValueError(
            "Train 데이터가 없습니다."
        )

    if validation_mask.sum() == 0:
        raise ValueError(
            "Validation 데이터가 없습니다."
        )

    if test_mask.sum() == 0:
        raise ValueError(
            "Test 데이터가 없습니다."
        )

    selection_train_end = int(
        np.flatnonzero(
            selection_train_mask.to_numpy()
        )[-1] + 1
    )

    official_train_end = int(
        np.flatnonzero(
            official_train_mask.to_numpy()
        )[-1] + 1
    )

    test_start_idx = int(
        np.flatnonzero(
            test_mask.to_numpy()
        )[0]
    )

    # --------------------------------------------------------
    # 실험 정보 출력
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("GRU-LSTM EXPERIMENT  [CLASSIFIER]")
    print("=" * 80)

    print(f"Experiment : {args.exp}")
    print(f"Lookback   : {args.lookback}")
    print(f"Seed       : {args.seed}")
    print(f"AUX z-clip : ±{args.aux_z_clip}" if args.aux_z_clip > 0 else "AUX z-clip : OFF")
    print(f"Horizon    : {args.horizon} trading day(s) ahead")
    print(f"Threshold  : {args.threshold}")
    print("Target     : direction, 1 = net buy / 0 = net sell")

    print("\nFeature groups")
    print(
        f"KRX : {len(groups['KRX'])} "
        f"(aggregate foreign net buy history)"
    )
    print(
        f"LDA : {len(groups['LDA'])} "
        f"(selected EN {len(SELECTED_LDA_EN_TOPICS)} "
        f"+ KO {len(SELECTED_LDA_KO_TOPICS)})"
    )
    print(
        f"LLM : {len(groups['LLM'])}"
    )
    print(
        f"AUX : {len(groups['AUX'])}"
    )
    print(
        f"USED: {len(feature_cols)}"
    )

    print("\nDate split (80 / 10 / 10)")

    n_total = len(df)

    print(
        "Epoch-selection Train : "
        f"{df.loc[selection_train_mask, DATE_COL].min().date()} "
        "~ "
        f"{df.loc[selection_train_mask, DATE_COL].max().date()} "
        f"({selection_train_mask.sum():,} rows, "
        f"{selection_train_mask.sum() / n_total:.1%})"
    )

    print(
        "Validation            : "
        f"{df.loc[validation_mask, DATE_COL].min().date()} "
        "~ "
        f"{df.loc[validation_mask, DATE_COL].max().date()} "
        f"({validation_mask.sum():,} rows, "
        f"{validation_mask.sum() / n_total:.1%})"
    )

    print(
        "Official Train        : "
        f"{df.loc[official_train_mask, DATE_COL].min().date()} "
        "~ "
        f"{df.loc[official_train_mask, DATE_COL].max().date()} "
        f"({official_train_mask.sum():,} rows, "
        f"{official_train_mask.sum() / n_total:.1%})"
    )

    print(
        "Final Test            : "
        f"{df.loc[test_mask, DATE_COL].min().date()} "
        "~ "
        f"{df.loc[test_mask, DATE_COL].max().date()} "
        f"({test_mask.sum():,} rows, "
        f"{test_mask.sum() / n_total:.1%})"
    )

    print("=" * 80)

    # --------------------------------------------------------
    # 원자료
    # --------------------------------------------------------
    # Raw arrays
    # --------------------------------------------------------

    X_raw = df[
        feature_cols
    ].to_numpy(
        dtype=np.float64
    )

    y_raw = df[
        TARGET_COL
    ].to_numpy(
        dtype=np.float64
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\nDevice:", device)

    # --------------------------------------------------------
    # arcsinh scale constant for the KRX INPUT feature
    # --------------------------------------------------------
    # The target is now a 0/1 label, so arcsinh no longer touches it.  The
    # foreign-net-buy HISTORY is still a predictor though, and it is still
    # heavy tailed (~1e11 won), so it still gets arcsinh(x / SCALE_C) with
    # SCALE_C = median |y| over the TRAIN rows only.  No test row is used.
    SCALE_C = float(
        np.median(
            np.abs(
                y_raw[:official_train_end]
            )
        )
    )

    print(
        f"[INFO] arcsinh scale constant c = {SCALE_C:.6e} "
        "(train median |foreign_net_buy|, applied to the KRX input feature)"
    )

    krx_feature_indices = [
        i
        for i, c in enumerate(feature_cols)
        if c in groups["KRX"]
    ]

    X_transformed = X_raw.copy()

    if krx_feature_indices:
        X_transformed[
            :,
            krx_feature_indices,
        ] = np.arcsinh(
            X_transformed[
                :,
                krx_feature_indices,
            ] / SCALE_C
        )

    # --------------------------------------------------------
    # [CLASSIFIER] target = direction
    # --------------------------------------------------------
    # 1 = net buy (y >= 0), 0 = net sell.  float32, not long:
    # BCEWithLogitsLoss expects float targets.
    y_dir = (
        y_raw >= 0
    ).astype(np.float32)

    n_pos = float(
        y_dir[:official_train_end].sum()
    )

    n_neg = float(
        official_train_end - n_pos
    )

    base_rate_train = float(
        y_dir[:official_train_end].mean()
    )

    pos_weight_value = (
        n_neg / n_pos
        if n_pos > 0
        else 1.0
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=device,
    )

    print(
        f"[INFO] train base rate = {base_rate_train:.4f} "
        f"(net buy days) | pos_weight = {pos_weight_value:.4f}"
    )

    # First day that has a full lookback window ending `horizon` days earlier.
    first_target = args.lookback + args.horizon - 1

    if first_target >= selection_train_end:
        raise ValueError(
            "lookback + horizon is longer than the training block"
        )

    # ========================================================
    # PHASE 1: Train / Validation  -> best epoch
    # ========================================================

    x_scaler_select = StandardScaler()

    x_scaler_select.fit(
        X_transformed[
            :selection_train_end
        ]
    )

    X_select = (
        x_scaler_select
        .transform(
            X_transformed
        )
        .astype(np.float32)
    )

    X_select, aux_clip_count_select = clip_standardized_auxiliary(
        X_scaled=X_select,
        feature_cols=feature_cols,
        aux_cols=groups["AUX"],
        clip_value=args.aux_z_clip,
        stage_name="PHASE 1",
    )

    train_select_indices = np.arange(
        first_target,
        selection_train_end,
    )

    val_indices = np.arange(
        max(
            selection_train_end,
            first_target,
        ),
        official_train_end,
    )

    train_select_ds = SequenceDataset(
        X_select,
        y_dir,
        train_select_indices,
        args.lookback,
        args.horizon,
    )

    val_ds = SequenceDataset(
        X_select,
        y_dir,
        val_indices,
        args.lookback,
        args.horizon,
    )

    train_select_loader = DataLoader(
        train_select_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    selection_model = GRULSTMModel(
        n_features=len(feature_cols),
        dropout=0.2,
    ).to(device)

    (
        selection_history,
        best_epoch,
        best_val_loss,
    ) = fit_with_validation(
        model=selection_model,
        train_loader=train_select_loader,
        val_loader=val_loader,
        device=device,
        lr=args.lr,
        max_epochs=args.epochs,
        patience=args.patience,
        pos_weight=pos_weight,
    )

    print(
        f"\n[INFO] Best epoch = {best_epoch}"
    )

    # ========================================================
    # PHASE 2: refit on Train + Validation, then predict Test
    # ========================================================

    set_seed(
        args.seed
    )

    x_scaler_final = StandardScaler()

    x_scaler_final.fit(
        X_transformed[
            :official_train_end
        ]
    )

    X_final = (
        x_scaler_final
        .transform(
            X_transformed
        )
        .astype(np.float32)
    )

    X_final, aux_clip_count_final = clip_standardized_auxiliary(
        X_scaled=X_final,
        feature_cols=feature_cols,
        aux_cols=groups["AUX"],
        clip_value=args.aux_z_clip,
        stage_name="PHASE 2",
    )

    full_train_indices = np.arange(
        first_target,
        official_train_end,
    )

    test_indices = np.arange(
        max(
            test_start_idx,
            first_target,
        ),
        len(df),
    )

    full_train_ds = SequenceDataset(
        X_final,
        y_dir,
        full_train_indices,
        args.lookback,
        args.horizon,
    )

    test_ds = SequenceDataset(
        X_final,
        y_dir,
        test_indices,
        args.lookback,
        args.horizon,
    )

    full_train_loader = DataLoader(
        full_train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    final_model = GRULSTMModel(
        n_features=len(feature_cols),
        dropout=0.2,
    ).to(device)

    refit_history = fit_fixed_epochs(
        model=final_model,
        train_loader=full_train_loader,
        device=device,
        lr=args.lr,
        epochs=best_epoch,
        pos_weight=pos_weight,
    )

    # ========================================================
    # FINAL TEST
    # ========================================================

    # `prob` is P(net buy) -- sigmoid was applied inside predict().
    prob, true_direction, pred_indices = predict(
        final_model,
        test_loader,
        device,
    )

    true_direction = true_direction.astype(int)

    pred_direction = (
        prob >= args.threshold
    ).astype(int)

    # --------------------------------------------------------
    # Classification metrics
    # --------------------------------------------------------
    # Brier_Skill plays exactly the role R2_oos played in the regression
    # script: it benchmarks the model against a forecaster that always
    # predicts the TRAINING base rate.
    #     0.00  no better than the historical base rate
    #     > 0   genuine skill
    #     < 0   worse than the base rate
    brier = float(
        brier_score_loss(
            true_direction,
            prob,
        )
    )

    brier_base = float(
        np.mean(
            (base_rate_train - true_direction) ** 2
        )
    )

    brier_skill = (
        float(1.0 - brier / brier_base)
        if brier_base > 0
        else float("nan")
    )

    eps = 1e-12

    p_clipped = np.clip(
        prob,
        eps,
        1 - eps,
    )

    b_clipped = float(
        np.clip(
            base_rate_train,
            eps,
            1 - eps,
        )
    )

    logloss = float(
        -np.mean(
            true_direction * np.log(p_clipped)
            + (1 - true_direction) * np.log(1 - p_clipped)
        )
    )

    logloss_base = float(
        -np.mean(
            true_direction * np.log(b_clipped)
            + (1 - true_direction) * np.log(1 - b_clipped)
        )
    )

    logloss_skill = (
        float(1.0 - logloss / logloss_base)
        if logloss_base > 0
        else float("nan")
    )

    # threshold-free: AUC must use the PROBABILITY, not the hard 0/1 call
    direction_auc = (
        float(
            roc_auc_score(
                true_direction,
                prob,
            )
        )
        if len(np.unique(true_direction)) > 1
        else float("nan")
    )

    directional_accuracy = float(
        accuracy_score(
            true_direction,
            pred_direction,
        )
    )

    balanced_accuracy = float(
        balanced_accuracy_score(
            true_direction,
            pred_direction,
        )
    )

    direction_f1 = float(
        f1_score(
            true_direction,
            pred_direction,
            zero_division=0,
        )
    )

    direction_mcc = float(
        matthews_corrcoef(
            true_direction,
            pred_direction,
        )
    )

    direction_precision = float(
        precision_score(
            true_direction,
            pred_direction,
            zero_division=0,
        )
    )

    direction_recall = float(
        recall_score(
            true_direction,
            pred_direction,
            zero_division=0,
        )
    )

    metrics = {
        "model": "GRU-LSTM",
        "task": "classification",
        "input_mode": "raw" if args.raw_pipeline else "aligned",
        "input_csv": None if args.raw_pipeline else ALIGNED_INPUT_CSV,
        "experiment": args.exp,
        "seed": args.seed,
        "lookback": args.lookback,
        "prediction_horizon": args.horizon,
        "threshold": float(args.threshold),

        "krx_features": len(
            groups["KRX"]
        ),
        "lda_features": len(
            groups["LDA"]
        ),
        "llm_features": len(
            groups["LLM"]
        ),
        "aux_features": len(
            groups["AUX"]
        ),
        "n_features_used": len(
            feature_cols
        ),

        "official_train_start_requested":
            str(OFFICIAL_TRAIN_START.date()),

        "official_train_start_actual":
            str(
                df.loc[
                    official_train_mask,
                    DATE_COL,
                ].min().date()
            ),

        "official_train_end_requested":
            str(OFFICIAL_TRAIN_END.date()),

        "official_train_end_actual":
            str(
                df.loc[
                    official_train_mask,
                    DATE_COL,
                ].max().date()
            ),

        "val_start_requested":
            str(VAL_START.date()),

        "val_end_requested":
            str(VAL_END.date()),

        "test_start_requested":
            str(TEST_START.date()),

        "test_start_actual":
            str(
                df.loc[
                    test_mask,
                    DATE_COL,
                ].min().date()
            ),

        "test_end_actual":
            str(
                df.loc[
                    test_mask,
                    DATE_COL,
                ].max().date()
            ),

        "n_rows_total":
            int(len(df)),

        "n_rows_train":
            int(selection_train_mask.sum()),

        "n_rows_validation":
            int(validation_mask.sum()),

        "n_rows_test":
            int(len(pred_indices)),

        "best_epoch":
            int(best_epoch),

        "best_validation_loss":
            float(best_val_loss),

        # ---- skill scores (0 = no better than the train base rate) ----
        "Brier_Skill":
            float(brier_skill),

        "LogLoss_Skill":
            float(logloss_skill),

        # ---- threshold-free ----
        "AUC":
            float(direction_auc),

        # ---- threshold-dependent ----
        "MCC":
            float(direction_mcc),

        "F1_Score":
            float(direction_f1),

        "Balanced_Accuracy":
            float(balanced_accuracy),

        "Directional_Accuracy":
            float(directional_accuracy),

        "Precision":
            float(direction_precision),

        "Recall":
            float(direction_recall),

        # ---- raw probabilistic losses ----
        "Brier":
            float(brier),

        "LogLoss":
            float(logloss),

        # ---- context ----
        "base_rate_train":
            float(base_rate_train),

        "base_rate_test":
            float(true_direction.mean()),

        "pred_positive_rate":
            float(pred_direction.mean()),

        "pos_weight":
            float(pos_weight_value),

        "target_scale_c":
            float(SCALE_C),

        "aux_z_clip":
            float(args.aux_z_clip),

        "aux_values_clipped_phase1":
            int(aux_clip_count_select),

        "aux_values_clipped_phase2":
            int(aux_clip_count_final),
    }

    print("\n" + "=" * 80)
    print("FINAL TEST RESULT (classification)")
    print("=" * 80)

    print(
        json.dumps(
            metrics,
            indent=2,
            ensure_ascii=False,
        )
    )

    print(
        "\nChance references: "
        "Brier_Skill 0 | MCC 0 | AUC 0.5 | Balanced_Accuracy 0.5"
    )

    # ========================================================
    # Save
    # ========================================================

    torch.save(
        final_model.state_dict(),
        output_dir
        / "best_model.pt",
    )

    selection_history.to_csv(
        output_dir
        / "epoch_selection_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    refit_history.to_csv(
        output_dir
        / "full_train_refit_history.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prediction_df = pd.DataFrame(
        {
            "date":
                df.iloc[
                    pred_indices
                ][DATE_COL]
                .to_numpy(),

            "actual_flow_krw":
                y_raw[pred_indices],

            "actual_direction":
                true_direction,

            "prob_net_buy":
                prob,

            "predicted_direction":
                pred_direction,
        }
    )

    prediction_df.to_csv(
        output_dir
        / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_dir / "metrics.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2,
            ensure_ascii=False,
        )

    pd.DataFrame(
        {
            "feature": feature_cols
        }
    ).to_csv(
        output_dir
        / "features_used.csv",
        index=False,
        encoding="utf-8-sig",
    )

    with open(
        output_dir
        / "x_scaler.pkl",
        "wb",
    ) as f:

        pickle.dump(
            x_scaler_final,
            f,
        )

    # No y_scaler: the target is a 0/1 label, nothing to invert.

    with open(
        output_dir
        / "target_scale_c.json",
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            {
                "target_scale_c": float(SCALE_C),
                "applies_to": "KRX input feature only",
            },
            f,
            indent=2,
        )

    print(
        "\nSaved to:",
        output_dir.resolve(),
    )


if __name__ == "__main__":
    main()
