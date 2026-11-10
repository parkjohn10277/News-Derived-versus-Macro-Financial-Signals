"""
4개 뉴스 데이터셋(MCST / GDELT / NAVER / NYT) → 단일 통합 테이블

요구사항
  - pandas >= 2.0   (pd.to_datetime(..., format="mixed"))
  - pyarrow         (parquet 저장;  pip install pyarrow)

출력
  - merged/news_merged.parquet
  - merged/news_merged.csv   (SAVE_CSV=True 일 때)
"""

from pathlib import Path
import hashlib
import numpy as np
import pandas as pd


# ============================================================
# 0. 경로 / 옵션
# ============================================================

BASE = Path(r".\1_News")

FILES = {
    "MCST":  BASE / "MCST_EcoTech..csv",
    "GDELT": BASE / "GDELT_API" / "GDELT_preprocessed.csv",
    "NAVER": BASE / "NAVER.csv",
    "NYT":   BASE / "NewYorkTimes.csv",
}

OUT_DIR = BASE / "merged"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SAVE_CSV = True          # parquet 외에 csv도 저장 (용량 큼, 느림)


# ============================================================
# 1. 통합 스키마
# ============================================================

COMMON_COLS = [
    "article_id",   # 통합 ID = dataset 접두어 + md5(url|title|date)
    "dataset",      # MCST / GDELT / NAVER / NYT
    "date",         # 보도일 (datetime64, 00:00 정규화)
    "title",
    "summary",
    "body",         # 전문 (MCST, GDELT만 있음, 나머지 NaN)
    "url",
    "media",        # 매체명 / source
    "country",      # MCST 국가
]

# 원본 컬럼명 → 통합 컬럼명
RENAME = {
    "MCST": {
        "제목(국문)": "title",
        "매체명":     "media",
        "국가":       "country",
        "보도일시":   "date",
        "URL":        "url",
        "내용":       "body",
        "요약내용":   "summary",
    },
    "GDELT": {
        "title_proxy": "title",
        "body":        "body",
        "source":      "media",
        "datetime":    "date",
        "url":         "url",
    },
    "NAVER": {
        "제목":       "title",
        "요약내용":   "summary",
        "보도일자":   "date",
        "URL":        "url",
    },
    "NYT": {
        "제목":       "title",
        "요약내용":   "summary",
        "보도일자":   "date",
        "URL":        "url",
        "source":     "media",
    },
}


# ============================================================
# 2. 공통 유틸
# ============================================================

def to_dt(s):
    """문자열 → naive datetime64. 형식 혼재/타임존 모두 허용, 실패는 NaT"""
    dt = pd.to_datetime(s, errors="coerce", format="mixed", utc=True)
    return dt.dt.tz_localize(None)


def load_raw(name):
    path = FILES[name]
    print(f"\n[{name}] 읽는 중: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, low_memory=False)
    print(f"[{name}] 원본 {len(df):,}건 / {len(df.columns)}컬럼")
    return df


# ============================================================
# 3. 데이터셋별 전처리
# ============================================================

def prep_naver(df):
    df["URL"] = df["URL"].fillna(df["네이버링크"])   # 원문 URL 없으면 네이버 링크로 대체
    return df


PREP = {"NAVER": prep_naver}


# ============================================================
# 4. 공통 마무리 (스키마 정렬 + ID + 중복 제거)
# ============================================================

def finalize(df, name):

    df = df.rename(columns=RENAME[name])
    df["dataset"] = name
    df["date"]    = to_dt(df["date"]).dt.normalize()

    # 스키마 통일: 없는 컬럼은 NaN, 여분 컬럼은 버림
    df = df.reindex(columns=COMMON_COLS)

    # 공백/빈문자열 → NaN
    for c in ["title", "summary", "body", "url", "media", "country"]:
        df[c] = df[c].astype(object).str.strip().replace("", np.nan)

    # 통합 ID
    key = (
        df["url"].fillna("") + "|"
        + df["title"].fillna("") + "|"
        + df["date"].dt.strftime("%Y-%m-%d").fillna("")
    )
    df["article_id"] = name + "_" + key.map(
        lambda k: hashlib.md5(k.encode("utf-8")).hexdigest()[:16]
    )

    # 데이터셋 내부 중복 제거 (NAVER는 종목별 중복 등록 기사가 여기서 1건으로 합쳐짐)
    before = len(df)
    df = df.drop_duplicates(subset="article_id", keep="first")
    print(f"[{name}] 중복 제거 {before - len(df):,}건 → {len(df):,}건")
    print(f"[{name}] date 파싱 실패 {df['date'].isna().sum():,}건")

    return df


# ============================================================
# 5. 실행
# ============================================================

parts = []
for name in FILES:
    raw = load_raw(name)
    raw = PREP.get(name, lambda d: d)(raw)
    parts.append(finalize(raw, name))
    del raw

merged = (
    pd.concat(parts, ignore_index=True)
    .sort_values(["date", "dataset"], kind="stable")
    .reset_index(drop=True)
)
del parts


# ============================================================
# 6. 점검
# ============================================================

print("\n" + "=" * 80)
print("통합 결과")
print("=" * 80)
print(f"총 {len(merged):,}건 / {len(merged.columns)}컬럼")

print("\n[데이터셋별 건수 / 기간]")
print(merged.groupby("dataset")["date"].agg(["size", "min", "max"]).to_string())

print("\n[컬럼별 결측률 (%)]")
print(merged.isna().mean().mul(100).round(2).to_string())

cross = merged.dropna(subset=["url"]).groupby("url")["dataset"].nunique()
print(f"\n[데이터셋 간 URL 겹침] {int((cross > 1).sum()):,}개 URL")


# ============================================================
# 7. 저장
# ============================================================

pq = OUT_DIR / "news_merged.parquets"
merged.to_parquet(pq, index=False)
print(f"\n저장: {pq}")

if SAVE_CSV:
    csv = OUT_DIR / "news_merged.csv"
    merged.to_csv(csv, index=False, encoding="utf-8-sig")
    print(f"저장: {csv}")

print("\n완료")