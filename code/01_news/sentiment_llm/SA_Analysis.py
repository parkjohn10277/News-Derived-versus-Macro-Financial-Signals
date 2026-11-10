from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# 0. 파일 경로
# ============================================================

FILES = {

    "MCST": Path(
        r".\1_News\Sentimental Analysis\MCST_EXAONE_7.8B_sentiment_score.csv"
    ),

    "GDELT": Path(
        r".\1_News\Sentimental Analysis\GDELT_EXAONE_7.8B_sentiment_score.csv"
    ),

    "NAVER": Path(
        r".\1_News\Sentimental Analysis\NAVER_EXAONE_7.8B_sentiment_score.csv"
    ),

    "NYT": Path(
        r".\1_News\Sentimental Analysis\NewYorkTimes_EXAONE_7.8B_sentiment_score.csv"
    ),
}


# ============================================================
# 1. 데이터 로딩
# ============================================================

all_results = []


for source, path in FILES.items():

    if not path.exists():

        print(
            f"[WARNING] 파일 없음: "
            f"{source}\n{path}"
        )

        continue


    df = pd.read_csv(
        path,
        encoding="utf-8-sig",
        low_memory=False,
    )


    # 필요한 컬럼 확인
    required = [
        "EXAONE_Sentiment",
        "EXAONE_Score",
    ]


    missing = [
        col
        for col in required
        if col not in df.columns
    ]


    if missing:

        print(
            f"[WARNING] {source} "
            f"필수 컬럼 없음: {missing}"
        )

        continue


    df["Source"] = (
        source
    )


    df["EXAONE_Score"] = pd.to_numeric(
        df["EXAONE_Score"],
        errors="coerce",
    )


    all_results.append(
        df
    )


# ============================================================
# 2. 통합
# ============================================================

if not all_results:

    raise RuntimeError(
        "분석 가능한 파일이 없습니다."
    )


combined = pd.concat(
    all_results,
    ignore_index=True,
)


# ============================================================
# 3. Dataset Size
# ============================================================

print("\n" + "=" * 90)
print("1. DATASET SIZE")
print("=" * 90)


for source in FILES.keys():

    temp = combined[
        combined["Source"] == source
    ]

    if temp.empty:

        continue


    print(
        f"{source:10s}: "
        f"{len(temp):,} articles"
    )


# ============================================================
# 4. ERROR / EMPTY
# ============================================================

print("\n" + "=" * 90)
print("2. ERROR / EMPTY CHECK")
print("=" * 90)


for source in FILES.keys():

    temp = combined[
        combined["Source"] == source
    ]


    if temp.empty:

        continue


    counts = (
        temp["EXAONE_Sentiment"]
        .value_counts(
            dropna=False
        )
    )


    error_n = int(
        counts.get(
            "ERROR",
            0
        )
    )


    empty_n = int(
        counts.get(
            "EMPTY",
            0
        )
    )


    print(
        f"{source:10s}"
        f" | ERROR = {error_n:,}"
        f" | EMPTY = {empty_n:,}"
    )


# ============================================================
# 5. 정상 Label만
# ============================================================

VALID_LABELS = [
    "Positive",
    "Neutral",
    "Negative",
]


valid = combined[
    combined[
        "EXAONE_Sentiment"
    ].isin(
        VALID_LABELS
    )
].copy()


# ============================================================
# 6. Sentiment Distribution
# ============================================================

print("\n" + "=" * 90)
print("3. SENTIMENT DISTRIBUTION")
print("=" * 90)


for source in FILES.keys():

    temp = valid[
        valid["Source"] == source
    ]


    if temp.empty:

        continue


    counts = (
        temp[
            "EXAONE_Sentiment"
        ]
        .value_counts()
        .reindex(
            VALID_LABELS,
            fill_value=0,
        )
    )


    rates = (
        counts
        /
        len(temp)
        *
        100
    )


    print(
        f"\n[{source}]"
    )


    for label in VALID_LABELS:

        print(
            f"{label:8s}: "
            f"{counts[label]:,} "
            f"({rates[label]:.2f}%)"
        )


# ============================================================
# 7. Score 기술통계
# ============================================================

print("\n" + "=" * 90)
print("4. SCORE DESCRIPTIVE STATISTICS")
print("=" * 90)


summary_rows = []


for source in FILES.keys():

    scores = valid.loc[
        valid["Source"] == source,
        "EXAONE_Score",
    ].dropna()


    if scores.empty:

        continue


    summary_rows.append({

        "Source":
            source,

        "N":
            len(scores),

        "Mean":
            scores.mean(),

        "Median":
            scores.median(),

        "Std":
            scores.std(),

        "Min":
            scores.min(),

        "Q1":
            scores.quantile(
                0.25
            ),

        "Q3":
            scores.quantile(
                0.75
            ),

        "Max":
            scores.max(),
    })


summary_df = pd.DataFrame(
    summary_rows
)


print(
    summary_df.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}",
    )
)


# ============================================================
# 8. Label별 Score
# ============================================================

print("\n" + "=" * 90)
print("5. SCORE BY SENTIMENT LABEL")
print("=" * 90)


label_stats = (

    valid
    .groupby(
        [
            "Source",
            "EXAONE_Sentiment",
        ]
    )[
        "EXAONE_Score"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    )
)


print(
    label_stats.to_string(
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# 9. Neutral Score 분석
# ============================================================

print("\n" + "=" * 90)
print("6. NEUTRAL SCORE ANALYSIS")
print("=" * 90)


for source in FILES.keys():

    scores = valid.loc[
        (
            valid["Source"]
            == source
        )
        &
        (
            valid[
                "EXAONE_Sentiment"
            ]
            ==
            "Neutral"
        ),
        "EXAONE_Score",
    ].dropna()


    if scores.empty:

        continue


    print(
        f"\n[{source}]"
    )


    print(
        f"N                : "
        f"{len(scores):,}"
    )


    print(
        f"Mean             : "
        f"{scores.mean():.4f}"
    )


    print(
        f"Median           : "
        f"{scores.median():.4f}"
    )


    print(
        f"|Score| <= 0.05  : "
        f"{(scores.abs() <= 0.05).mean() * 100:.2f}%"
    )


    print(
        f"|Score| <= 0.10  : "
        f"{(scores.abs() <= 0.10).mean() * 100:.2f}%"
    )


    print(
        f"|Score| <= 0.20  : "
        f"{(scores.abs() <= 0.20).mean() * 100:.2f}%"
    )


# ============================================================
# 10. Label-Score 일관성
# ============================================================

print("\n" + "=" * 90)
print("7. LABEL-SCORE CONSISTENCY")
print("=" * 90)


for source in FILES.keys():

    temp = valid[
        valid["Source"] == source
    ]


    if temp.empty:

        continue


    positive_wrong = temp[
        (
            temp[
                "EXAONE_Sentiment"
            ]
            ==
            "Positive"
        )
        &
        (
            temp[
                "EXAONE_Score"
            ]
            <
            0
        )
    ]


    negative_wrong = temp[
        (
            temp[
                "EXAONE_Sentiment"
            ]
            ==
            "Negative"
        )
        &
        (
            temp[
                "EXAONE_Score"
            ]
            >
            0
        )
    ]


    print(
        f"\n[{source}]"
    )


    print(
        f"Positive but Score < 0 : "
        f"{len(positive_wrong):,}"
    )


    print(
        f"Negative but Score > 0 : "
        f"{len(negative_wrong):,}"
    )


# ============================================================
# 11. Sentiment Intensity
# ============================================================

print("\n" + "=" * 90)
print("8. SENTIMENT INTENSITY")
print("=" * 90)


def intensity(score):

    x = abs(
        score
    )


    if x < 0.10:

        return "Very Weak"


    elif x < 0.30:

        return "Weak"


    elif x < 0.60:

        return "Moderate"


    elif x < 0.80:

        return "Strong"


    else:

        return "Very Strong"


valid[
    "Intensity"
] = (

    valid[
        "EXAONE_Score"
    ]

    .apply(
        intensity
    )
)


intensity_order = [
    "Very Weak",
    "Weak",
    "Moderate",
    "Strong",
    "Very Strong",
]


for source in FILES.keys():

    temp = valid[
        valid["Source"] == source
    ]


    if temp.empty:

        continue


    counts = (
        temp[
            "Intensity"
        ]
        .value_counts()
        .reindex(
            intensity_order,
            fill_value=0,
        )
    )


    rates = (
        counts
        /
        len(temp)
        *
        100
    )


    print(
        f"\n[{source}]"
    )


    for level in intensity_order:

        print(
            f"{level:11s}: "
            f"{counts[level]:,} "
            f"({rates[level]:.2f}%)"
        )


# ============================================================
# 12. Source 간 비교
# ============================================================

print("\n" + "=" * 90)
print("9. BETWEEN-SOURCE COMPARISON")
print("=" * 90)


comparison = (

    valid
    .groupby(
        "Source"
    )[
        "EXAONE_Score"
    ]
    .agg(
        [
            "count",
            "mean",
            "median",
            "std",
        ]
    )
)


print(
    comparison.to_string(
        float_format=lambda x:
            f"{x:.4f}"
    )
)


# ============================================================
# 13. 극단값
# ============================================================

print("\n" + "=" * 90)
print("10. EXTREME SCORE COUNTS")
print("=" * 90)


for source in FILES.keys():

    temp = valid[
        valid["Source"] == source
    ]


    if temp.empty:

        continue


    strong_positive = (
        temp[
            "EXAONE_Score"
        ]
        >=
        0.8
    ).sum()


    strong_negative = (
        temp[
            "EXAONE_Score"
        ]
        <=
        -0.8
    ).sum()


    print(
        f"{source:10s}"
        f" | Score >= +0.8: "
        f"{strong_positive:,}"
        f" | Score <= -0.8: "
        f"{strong_negative:,}"
    )


print("\n" + "=" * 90)
print("ANALYSIS COMPLETE")
print("=" * 90)