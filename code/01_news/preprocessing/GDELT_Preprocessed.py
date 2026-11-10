import re
import html
import hashlib
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# ============================================================
# 0. 경로
# ============================================================

INPUT_FILE = Path(
    r".\1_News\GDELT_API\GDELT.csv"
)

OUTPUT_FILE = (
    INPUT_FILE.parent
    / "GDELT_preprocessed.csv"
)


# ============================================================
# 1. 설정
# ============================================================

CHUNK_SIZE = 10000

# 중복 제거
REMOVE_DUPLICATE_URL = True
REMOVE_DUPLICATE_BODY = True


# ============================================================
# 2. 입력 파일 확인
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"파일 없음:\n{INPUT_FILE}"
    )


# 기존 결과 삭제
if OUTPUT_FILE.exists():

    print(
        f"기존 출력파일 삭제:\n"
        f"{OUTPUT_FILE}"
    )

    OUTPUT_FILE.unlink()


# ============================================================
# 3. 인코딩 자동 확인
# ============================================================

def detect_encoding(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
    ]

    for enc in encodings:

        try:

            pd.read_csv(
                file_path,
                encoding=enc,
                nrows=5,
                low_memory=False,
            )

            return enc

        except UnicodeDecodeError:
            continue

    raise ValueError(
        f"CSV 인코딩 확인 실패:\n{file_path}"
    )


ENCODING = detect_encoding(
    INPUT_FILE
)


# ============================================================
# 4. slug → 제목 대체값
# ============================================================

def clean_slug(value):

    if pd.isna(value):
        return ""

    text = str(value).strip()


    # -, _, / → 공백
    text = re.sub(
        r"[-_/]+",
        " ",
        text
    )


    # content 2464 ...
    text = re.sub(
        r"^content\s+\d+\s+",
        "",
        text,
        flags=re.IGNORECASE
    )


    # article 1234 ...
    text = re.sub(
        r"^article\s+\d+\s+",
        "",
        text,
        flags=re.IGNORECASE
    )


    # story 1234 ...
    text = re.sub(
        r"^story\s+\d+\s+",
        "",
        text,
        flags=re.IGNORECASE
    )


    # news 1234 ...
    text = re.sub(
        r"^news\s+\d+\s+",
        "",
        text,
        flags=re.IGNORECASE
    )


    # 맨 앞 숫자 ID
    text = re.sub(
        r"^\d+\s+",
        "",
        text
    )


    # 연속 공백
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# 5. 본문 전처리
# ============================================================

def clean_body(value):

    if pd.isna(value):
        return ""


    text = str(value)


    # HTML entity
    text = html.unescape(
        text
    )


    # HTML tag
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )


    # URL 제거
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )


    # 제어문자 제거
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f]",
        " ",
        text
    )


    # 연속 공백 정리
    text = re.sub(
        r"\s+",
        " ",
        text
    )


    return text.strip()


# ============================================================
# 6. 본문 Hash
#
# 최종 CSV에는 저장하지 않고
# 중복 검사에만 사용
# ============================================================

def make_hash(text):

    if not text:
        return ""

    return hashlib.md5(
        text.encode(
            "utf-8",
            errors="ignore"
        )
    ).hexdigest()


# ============================================================
# 7. 입력 데이터 확인
# ============================================================

sample = pd.read_csv(
    INPUT_FILE,
    encoding=ENCODING,
    nrows=5,
    low_memory=False
)


print("=" * 80)
print("GDELT 전처리")
print("=" * 80)


print(
    f"파일   : {INPUT_FILE}"
)

print(
    f"인코딩 : {ENCODING}"
)

print(
    f"출력   : {OUTPUT_FILE}"
)


print("\n원본 컬럼")

for i, col in enumerate(
    sample.columns,
    start=1
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# 8. 필요한 원본 컬럼만 확인
# ============================================================

REQUIRED_COLUMNS = [

    "datetime_utc",

    "url",

    "slug",

    "source",

    "tone",

    "body",
]


missing = [

    col

    for col in REQUIRED_COLUMNS

    if col not in sample.columns
]


if missing:

    raise ValueError(
        f"필수 컬럼 없음: {missing}"
    )


# ============================================================
# 9. 전체 기사 수
# ============================================================

total_rows = 0


for part in pd.read_csv(

    INPUT_FILE,

    encoding=ENCODING,

    usecols=["url"],

    chunksize=100000,

    low_memory=False,

):

    total_rows += len(part)


print(
    f"\n전체 원본 기사: "
    f"{total_rows:,}건"
)


# ============================================================
# 10. 필요한 컬럼만 읽기
#
# 아래 컬럼들은 처음부터 읽지 않음
#
# tier
# locations
# organizations
# themes
# ============================================================

USE_COLUMNS = [

    "datetime_utc",

    "url",

    "slug",

    "source",

    "tone",

    "body",
]


reader = pd.read_csv(

    INPUT_FILE,

    encoding=ENCODING,

    usecols=USE_COLUMNS,

    chunksize=CHUNK_SIZE,

    low_memory=False,
)


# ============================================================
# 11. 중복 검사용
# ============================================================

seen_urls = set()

seen_body_hashes = set()


write_header = True


body_missing_total = 0

duplicate_url_total = 0

duplicate_body_total = 0

removed_duplicate_total = 0

processed = 0


# ============================================================
# 12. Chunk 전처리
# ============================================================

for chunk_number, df in enumerate(

    tqdm(
        reader,
        desc="GDELT Preprocessing"
    ),

    start=1,
):


    df = df.copy()


    # ========================================================
    # 12-1. body 없는 기사 제거
    # ========================================================

    body_exists = (

        df["body"].notna()

        &

        df["body"]
        .astype(str)
        .str.strip()
        .ne("")
    )


    removed_missing = int(
        (~body_exists).sum()
    )


    body_missing_total += (
        removed_missing
    )


    df = (

        df[
            body_exists
        ]

        .copy()
    )


    if df.empty:

        continue


    # ========================================================
    # 12-2. 날짜
    #
    # 2015-02-19 00:00:00
    # →
    # 2015-02-19
    # ========================================================

    dt = pd.to_datetime(

        df[
            "datetime_utc"
        ],

        errors="coerce"
    )


    df[
        "datetime_utc"
    ] = (

        dt.dt.strftime(
            "%Y-%m-%d"
        )
    )


    # ========================================================
    # 12-3. slug → title_from_slug
    #
    # url_slug는 만들지 않음
    # ========================================================

    df[
        "title_from_slug"
    ] = (

        df[
            "slug"
        ]

        .apply(
            clean_slug
        )
    )


    # ========================================================
    # 12-4. body 전처리
    #
    # 원본 body를 전처리한 값으로 덮어씀
    #
    # body_clean 별도 컬럼 생성하지 않음
    # ========================================================

    df[
        "body"
    ] = (

        df[
            "body"
        ]

        .apply(
            clean_body
        )
    )


    # ========================================================
    # 12-5. 전처리 후 빈 body 제거
    # ========================================================

    clean_body_exists = (

        df[
            "body"
        ]

        .str.strip()

        .ne("")
    )


    removed_after_clean = int(

        (
            ~clean_body_exists
        ).sum()
    )


    body_missing_total += (
        removed_after_clean
    )


    df = (

        df[
            clean_body_exists
        ]

        .copy()
    )


    if df.empty:

        continue


    # ========================================================
    # 12-6. 본문 Hash
    #
    # 내부 중복 확인에만 사용
    # ========================================================

    body_hashes = (

        df[
            "body"
        ]

        .apply(
            make_hash
        )
    )


    # ========================================================
    # 12-7. URL 중복
    # ========================================================

    duplicate_url_flags = []


    for url in df["url"]:

        url = str(
            url
        ).strip()


        if url in seen_urls:

            duplicate_url_flags.append(
                True
            )


        else:

            duplicate_url_flags.append(
                False
            )

            seen_urls.add(
                url
            )


    duplicate_url_total += sum(
        duplicate_url_flags
    )


    # ========================================================
    # 12-8. 본문 중복
    # ========================================================

    duplicate_body_flags = []


    for body_hash in body_hashes:


        if body_hash in seen_body_hashes:

            duplicate_body_flags.append(
                True
            )


        else:

            duplicate_body_flags.append(
                False
            )

            seen_body_hashes.add(
                body_hash
            )


    duplicate_body_total += sum(
        duplicate_body_flags
    )


    # ========================================================
    # 12-9. 중복 제거
    # ========================================================

    keep_mask = pd.Series(

        True,

        index=df.index,
    )


    if REMOVE_DUPLICATE_URL:

        keep_mask &= (

            ~pd.Series(
                duplicate_url_flags,
                index=df.index
            )
        )


    if REMOVE_DUPLICATE_BODY:

        keep_mask &= (

            ~pd.Series(
                duplicate_body_flags,
                index=df.index
            )
        )


    before = len(
        df
    )


    df = (

        df[
            keep_mask
        ]

        .copy()
    )


    removed_duplicate_total += (

        before
        -
        len(df)
    )


    if df.empty:

        continue


    # ========================================================
    # 12-10. slug 원본 삭제
    # ========================================================

    df = df.drop(

        columns=[
            "slug"
        ],

        errors="ignore",
    )


    # ========================================================
    # 12-11. 최종 컬럼 순서
    #
    # body = 두 번째
    # url = 맨 마지막
    # ========================================================

    FINAL_COLUMNS = [

        # 1
        "datetime_utc",

        # 2
        "body",

        # 3
        "title_from_slug",

        # 4
        "source",

        # 5
        "tone",

        # 마지막
        "url",
    ]


    df = df[
        FINAL_COLUMNS
    ]


    # ========================================================
    # 12-12. 저장
    # ========================================================

    df.to_csv(

        OUTPUT_FILE,

        mode="a",

        header=write_header,

        index=False,

        encoding="utf-8-sig",
    )


    write_header = False


    processed += len(
        df
    )


# ============================================================
# 13. 종료
# ============================================================

print("\n" + "=" * 80)

print(
    "GDELT 전처리 완료"
)

print("=" * 80)


print(
    f"원본 기사            : "
    f"{total_rows:,}건"
)


print(
    f"body 없는 기사 제거  : "
    f"{body_missing_total:,}건"
)


print(
    f"중복 URL             : "
    f"{duplicate_url_total:,}건"
)


print(
    f"중복 본문            : "
    f"{duplicate_body_total:,}건"
)


print(
    f"중복 제거 기사       : "
    f"{removed_duplicate_total:,}건"
)


print(
    f"최종 저장 기사       : "
    f"{processed:,}건"
)


print(
    "\n최종 컬럼:"
)

print(
    "1. datetime_utc"
)

print(
    "2. body"
)

print(
    "3. title_from_slug"
)

print(
    "4. source"
)

print(
    "5. tone"
)

print(
    "6. url"
)


print(
    f"\n저장 파일:\n"
    f"{OUTPUT_FILE}"
)


print("=" * 80)