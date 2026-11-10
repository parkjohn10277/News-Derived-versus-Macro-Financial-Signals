from pathlib import Path
import pandas as pd


# ============================================================
# 1. 현재 Python 파일이 있는 폴더
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 2. 통합할 CSV 파일
# ============================================================

FILES = [
    BASE_DIR / "nyt_2015_2026.csv",
    BASE_DIR / "nyt_2015_2026_korea.csv",
    BASE_DIR / "nyt_2015_2026_semiconductor.csv"
]


# 최종 결과 파일
OUTPUT_FILE = BASE_DIR / "NewYorkTimes.csv"


# ============================================================
# 3. CSV 읽기 함수
# ============================================================

def read_csv_auto(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]

    for enc in encodings:

        try:

            df = pd.read_csv(
                file_path,
                encoding=enc,
                low_memory=False
            )

            print(
                f"[읽기 완료] {file_path.name} "
                f"→ {len(df):,}건 "
                f"({enc})"
            )

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"파일 인코딩 확인 실패: {file_path}"
    )


# ============================================================
# 4. 세 CSV 읽기
# ============================================================

dataframes = []

for file_path in FILES:

    if not file_path.exists():

        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다:\n{file_path}"
        )

    df = read_csv_auto(file_path)

    dataframes.append(df)


# ============================================================
# 5. 세 파일 세로 통합
# ============================================================

merged_df = pd.concat(
    dataframes,
    ignore_index=True,
    sort=False
)


print("\n" + "=" * 70)

print(
    f"단순 통합 후 기사 수: "
    f"{len(merged_df):,}건"
)


# ============================================================
# 6. article_id 기준 중복 제거
# ============================================================

before = len(merged_df)


if "article_id" in merged_df.columns:

    merged_df = merged_df.drop_duplicates(
        subset=["article_id"],
        keep="first"
    )


removed_article_id = (
    before - len(merged_df)
)


print(
    f"article_id 중복 삭제: "
    f"{removed_article_id:,}건"
)


# ============================================================
# 7. URL 기준 중복 제거
#
# article_id가 달라도 같은 URL이면
# 같은 기사일 가능성이 높으므로 제거
# ============================================================

before = len(merged_df)


if "URL" in merged_df.columns:

    # URL이 존재하는 데이터만 중복 판단
    has_url = (
        merged_df["URL"]
        .notna()
        &
        merged_df["URL"]
        .astype(str)
        .str.strip()
        .ne("")
    )

    with_url = merged_df[
        has_url
    ].drop_duplicates(
        subset=["URL"],
        keep="first"
    )

    without_url = merged_df[
        ~has_url
    ]

    merged_df = pd.concat(
        [
            with_url,
            without_url
        ],
        ignore_index=True
    )


removed_url = (
    before - len(merged_df)
)


print(
    f"URL 중복 삭제: "
    f"{removed_url:,}건"
)


# ============================================================
# 8. 날짜 처리
# ============================================================

if "보도일자" in merged_df.columns:

    merged_df["보도일자"] = pd.to_datetime(
        merged_df["보도일자"],
        errors="coerce"
    )


    # 날짜순 정렬
    merged_df = (
        merged_df
        .sort_values(
            "보도일자"
        )
        .reset_index(
            drop=True
        )
    )


    # YYYY-MM-DD 형식
    merged_df["보도일자"] = (
        merged_df["보도일자"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )


# ============================================================
# 9. 저장
# ============================================================

merged_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 10. 결과 출력
# ============================================================

print("\n" + "=" * 70)

print("NYT 뉴스 통합 완료")

print("=" * 70)

print(
    f"최종 기사 수: "
    f"{len(merged_df):,}건"
)

print(
    f"총 중복 삭제: "
    f"{removed_article_id + removed_url:,}건"
)

print(
    f"컬럼 수: "
    f"{len(merged_df.columns)}개"
)

if "보도일자" in merged_df.columns:

    print(
        f"기간: "
        f"{merged_df['보도일자'].min()} "
        f"~ "
        f"{merged_df['보도일자'].max()}"
    )


print(
    f"\n저장 위치:\n"
    f"{OUTPUT_FILE}"
)

print("=" * 70)