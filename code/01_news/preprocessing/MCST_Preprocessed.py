import pandas as pd
import re
import html
import unicodedata
from pathlib import Path


# ============================================================
# 1. 파일 경로
# ============================================================

INPUT_FILE = Path(
    r".\1_News\MCST_0627_2026\MCST_EcoTech.csv"
)

OUTPUT_FILE = (
    INPUT_FILE.parent
    / "MCST_EcoTech_preprocessed_20150227.csv"
)

CUTOFF_DATE = pd.Timestamp("2015-02-27")


# ============================================================
# 2. CSV 자동 읽기
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

            print(f"읽기 성공 인코딩: {enc}")

            return df

        except UnicodeDecodeError:
            continue

    raise ValueError(
        "CSV 파일의 인코딩을 확인할 수 없습니다."
    )


df = read_csv_auto(INPUT_FILE)


print("\n" + "=" * 70)
print("MCST 뉴스 전처리 시작")
print("=" * 70)

print(f"원본 뉴스 수: {len(df):,}건")
print(f"원본 컬럼 수: {len(df.columns)}")
print(f"원본 컬럼: {list(df.columns)}")


# ============================================================
# 3. 보도일시 처리
#
# 등록일이 아니라 실제 뉴스 보도일시 기준
# ============================================================

if "보도일시" not in df.columns:

    raise ValueError(
        "'보도일시' 컬럼이 없습니다."
    )


df["보도일시"] = pd.to_datetime(
    df["보도일시"],
    errors="coerce"
)


# 날짜 변환 실패
invalid_date_count = (
    df["보도일시"]
    .isna()
    .sum()
)

print(
    f"\n날짜 변환 실패: "
    f"{invalid_date_count:,}건"
)


# 날짜 없는 행 제거
df = df.dropna(
    subset=["보도일시"]
)


# ============================================================
# 4. 2015-02-27 이전 뉴스 삭제
# ============================================================

before_date_filter = len(df)


df = df[
    df["보도일시"] >= CUTOFF_DATE
].copy()


removed_old_news = (
    before_date_filter
    - len(df)
)


print(
    f"2015-02-27 이전 기사 삭제: "
    f"{removed_old_news:,}건"
)


# ============================================================
# 5. 텍스트 정제 함수
# ============================================================

def clean_news_text(text):

    if pd.isna(text):
        return ""

    text = str(text)


    # --------------------------------------------------------
    # Unicode 정규화
    # --------------------------------------------------------

    text = unicodedata.normalize(
        "NFKC",
        text
    )


    # --------------------------------------------------------
    # HTML entity 변환
    #
    # &amp; -> &
    # &quot; -> "
    # --------------------------------------------------------

    text = html.unescape(text)


    # --------------------------------------------------------
    # HTML 태그 제거
    # --------------------------------------------------------

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # --------------------------------------------------------
    # 보이지 않는 문자 / 깨진 문자 제거
    # --------------------------------------------------------

    invisible_chars = [
        "\ufeff",   # BOM
        "\u200b",   # zero-width space
        "\u200c",
        "\u200d",
        "\u2060",
        "\xa0",     # non-breaking space
        "�"         # replacement character
    ]

    for char in invisible_chars:

        text = text.replace(
            char,
            " "
        )


    # --------------------------------------------------------
    # 제어 문자 제거
    # --------------------------------------------------------

    text = re.sub(
        r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]",
        " ",
        text
    )


    # --------------------------------------------------------
    # 뉴스 장식용 기호 제거
    #
    # 예:
    # ❑ 北 경제 -> 北 경제
    # □ 삼성전자 -> 삼성전자
    # ■ 반도체 -> 반도체
    # --------------------------------------------------------

    decorative_symbols = (
        r"["
        r"❑❒"
        r"■□▪▫"
        r"◆◇"
        r"●○◎"
        r"▲△▼▽"
        r"▶▷◀◁"
        r"★☆"
        r"※"
        r"☞☜"
        r"→←↑↓"
        r"♠♣♥♦"
        r"☎☏"
        r"✓✔"
        r"◈◉"
        r"]"
    )

    text = re.sub(
        decorative_symbols,
        " ",
        text
    )


    # --------------------------------------------------------
    # 반복 기호 정리
    #
    # !!!!! -> !
    # ????? -> ?
    # ....... -> ...
    # --------------------------------------------------------

    text = re.sub(
        r"!{2,}",
        "!",
        text
    )

    text = re.sub(
        r"\?{2,}",
        "?",
        text
    )

    text = re.sub(
        r"\.{4,}",
        "...",
        text
    )


    # --------------------------------------------------------
    # 반복 하이픈 / 언더바 정리
    # --------------------------------------------------------

    text = re.sub(
        r"_{2,}",
        " ",
        text
    )

    text = re.sub(
        r"-{3,}",
        " ",
        text
    )


    # --------------------------------------------------------
    # 여러 공백 / 줄바꿈 -> 공백 하나
    # --------------------------------------------------------

    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# 6. 뉴스 텍스트 전처리
# ============================================================

TEXT_COLUMNS = [
    "제목(국문)",
    "내용",
    "요약내용"
]


for col in TEXT_COLUMNS:

    if col in df.columns:

        print(f"전처리 중: {col}")

        df[col] = df[col].apply(
            clean_news_text
        )


# ============================================================
# 7. 제목과 요약이 모두 없는 기사 제거
# ============================================================

title_empty = (
    df["제목(국문)"]
    .fillna("")
    .str.strip()
    .eq("")
)

summary_empty = (
    df["요약내용"]
    .fillna("")
    .str.strip()
    .eq("")
)


empty_news = (
    title_empty
    &
    summary_empty
)


empty_news_count = (
    empty_news.sum()
)


df = df[
    ~empty_news
].copy()


print(
    f"제목·요약 모두 없는 기사 삭제: "
    f"{empty_news_count:,}건"
)


# ============================================================
# 8. URL 기준 중복 제거
#
# URL이 있는 기사만 URL 중복 제거
# ============================================================

before_url_duplicate = len(df)


if "URL" in df.columns:

    has_url = (
        df["URL"]
        .notna()
        &
        df["URL"]
        .astype(str)
        .str.strip()
        .ne("")
    )


    df_with_url = (
        df[has_url]
        .drop_duplicates(
            subset=["URL"],
            keep="first"
        )
    )


    df_without_url = (
        df[~has_url]
    )


    df = pd.concat(
        [
            df_with_url,
            df_without_url
        ],
        ignore_index=True
    )


removed_url_duplicates = (
    before_url_duplicate
    - len(df)
)


print(
    f"URL 중복 기사 삭제: "
    f"{removed_url_duplicates:,}건"
)


# ============================================================
# 9. 제목 + 요약내용 완전 중복 제거
#
# URL이 다르거나 없는 경우에도
# 제목과 요약이 완전히 동일한 기사 제거
# ============================================================

before_text_duplicate = len(df)


df["_duplicate_title"] = (
    df["제목(국문)"]
    .fillna("")
    .str.strip()
)


df["_duplicate_summary"] = (
    df["요약내용"]
    .fillna("")
    .str.strip()
)


# 제목이나 요약이 어느 정도 존재하는 경우만 중복 판단
valid_text = (
    df["_duplicate_title"].ne("")
    |
    df["_duplicate_summary"].ne("")
)


duplicated_text_df = (
    df[valid_text]
    .drop_duplicates(
        subset=[
            "_duplicate_title",
            "_duplicate_summary"
        ],
        keep="first"
    )
)


non_text_df = (
    df[~valid_text]
)


df = pd.concat(
    [
        duplicated_text_df,
        non_text_df
    ],
    ignore_index=True
)


removed_text_duplicates = (
    before_text_duplicate
    - len(df)
)


# 임시 컬럼 제거
df = df.drop(
    columns=[
        "_duplicate_title",
        "_duplicate_summary"
    ]
)


print(
    f"제목+요약 완전중복 삭제: "
    f"{removed_text_duplicates:,}건"
)


# ============================================================
# 10. 불필요 컬럼 삭제
# ============================================================

DROP_COLUMNS = [
    "제목(현지)",
    "보도형식",
    "등록일"
]


df = df.drop(
    columns=[
        col
        for col in DROP_COLUMNS
        if col in df.columns
    ]
)


# ============================================================
# 11. 보도일시 -> 날짜만 남기기
#
# 2015-02-27 00:00
# ->
# 2015-02-27
# ============================================================

df["보도일시"] = pd.to_datetime(
    df["보도일시"],
    errors="coerce"
)


# 날짜순 정렬
df = (
    df
    .sort_values(
        "보도일시"
    )
    .reset_index(
        drop=True
    )
)


df["보도일시"] = (
    df["보도일시"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# 12. 최종 컬럼 순서 정리
# ============================================================

FINAL_COLUMNS = [
    "제목(국문)",
    "매체명",
    "국가",
    "대분류",
    "중분류",
    "보도일시",
    "URL",
    "내용",
    "요약내용"
]


FINAL_COLUMNS = [
    col
    for col in FINAL_COLUMNS
    if col in df.columns
]


df = df[
    FINAL_COLUMNS
]


# ============================================================
# 13. 최종 CSV 저장
# ============================================================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 14. 결과 출력
# ============================================================

print("\n" + "=" * 70)

print("MCST 뉴스 전처리 완료")

print("=" * 70)


print(
    f"최종 뉴스 수: "
    f"{len(df):,}건"
)


print(
    f"2015-02-27 이전 삭제: "
    f"{removed_old_news:,}건"
)


print(
    f"URL 중복 삭제: "
    f"{removed_url_duplicates:,}건"
)


print(
    f"제목+요약 중복 삭제: "
    f"{removed_text_duplicates:,}건"
)


print(
    f"최초 보도일자: "
    f"{df['보도일시'].min()}"
)


print(
    f"최종 보도일자: "
    f"{df['보도일시'].max()}"
)


print(
    f"\n최종 컬럼:"
)

for i, col in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i}. {col}"
    )


print(
    f"\n저장 위치:\n"
    f"{OUTPUT_FILE}"
)


print("=" * 70)