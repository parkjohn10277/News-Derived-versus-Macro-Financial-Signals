from pathlib import Path
import pandas as pd
import numpy as np
import re

# ============================================================
# 1. 경로 설정
# ============================================================

ROOT_DIR = Path("data/krx_raw")  # downloaded KRX files, see data/KRX_DOWNLOAD_GUIDE.md

OUTPUT_FILE = ROOT_DIR / "KRX_150_Prediction_ready_wide.csv"

if not ROOT_DIR.exists():
    raise FileNotFoundError(
        f"폴더를 찾을 수 없습니다: {ROOT_DIR}"
    )

print(f"데이터 경로: {ROOT_DIR}")


# ============================================================
# 2. KRX CSV 읽기 함수
# ============================================================

def read_krx_csv(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr"
    ]

    df = None

    for enc in encodings:

        try:
            df = pd.read_csv(
                file_path,
                encoding=enc
            )
            break

        except UnicodeDecodeError:
            continue

        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    if df is None:
        print(f"[ERROR] 읽기 실패: {file_path}")
        return pd.DataFrame()


    # --------------------------------------------------------
    # 컬럼명 정리
    # --------------------------------------------------------

    df.columns = [
        str(col)
        .replace("\ufeff", "")
        .strip()
        for col in df.columns
    ]


    # 빈 파일
    if df.empty:
        return pd.DataFrame()


    # --------------------------------------------------------
    # 필수 컬럼 확인
    # --------------------------------------------------------

    if "일자" not in df.columns:

        print(
            f"[SKIP] 일자 컬럼 없음: "
            f"{file_path.name}"
        )

        return pd.DataFrame()


    if "외국인 합계" not in df.columns:

        print(
            f"[SKIP] 외국인 합계 컬럼 없음: "
            f"{file_path.name}"
        )

        return pd.DataFrame()


    # --------------------------------------------------------
    # 날짜
    # --------------------------------------------------------

    df["일자"] = pd.to_datetime(
        df["일자"],
        errors="coerce"
    )

    df = df.dropna(
        subset=["일자"]
    )


    # --------------------------------------------------------
    # 외국인 순매수 금액
    #
    # 예:
    # "1,234,567" → 1234567
    # "-1,234,567" → -1234567
    # --------------------------------------------------------

    df["외국인 합계"] = (
        df["외국인 합계"]
        .astype(str)
        .str.replace(
            ",",
            "",
            regex=False
        )
        .str.replace(
            " ",
            "",
            regex=False
        )
    )

    df["외국인 합계"] = pd.to_numeric(
        df["외국인 합계"],
        errors="coerce"
    )


    return df[
        [
            "일자",
            "외국인 합계"
        ]
    ]


# ============================================================
# 3. 기업 폴더 탐색
#
# 예:
#
# 2_KRX_150
# ├─ 1~30
# │   ├─ 1_삼성전자
# │   ├─ 2_SK하이닉스
# │   └─ ...
# ├─ 31~60
# ├─ 61~100
# └─ 101~150
# ============================================================

company_folders = []

for group_folder in ROOT_DIR.iterdir():

    if not group_folder.is_dir():
        continue


    for company_folder in group_folder.iterdir():

        if not company_folder.is_dir():
            continue


        # 예: 1_삼성전자
        match = re.match(
            r"^(\d+)_(.+)$",
            company_folder.name
        )

        if match:

            company_no = int(
                match.group(1)
            )

            company_name = (
                match.group(2)
                .strip()
            )


            if 1 <= company_no <= 150:

                company_folders.append(
                    (
                        company_no,
                        company_name,
                        company_folder
                    )
                )


# 기업번호 순 정렬
company_folders.sort(
    key=lambda x: x[0]
)


print(
    f"\n탐색된 기업 폴더: "
    f"{len(company_folders)}개"
)


# ============================================================
# 4. 기업번호 → 기업명
# ============================================================

company_name_map = {}

for (
    company_no,
    company_name,
    company_folder
) in company_folders:

    company_name_map[
        company_no
    ] = company_name


# ============================================================
# 5. 기업별 6개 CSV 통합
# ============================================================

company_series = {}


for (
    company_no,
    company_name,
    company_folder
) in company_folders:


    csv_files = sorted(
        company_folder.rglob("*.csv")
    )


    print(
        f"[{company_no:03d}] "
        f"{company_name} "
        f"→ {len(csv_files)}개 파일"
    )


    frames = []


    for csv_file in csv_files:

        df = read_krx_csv(
            csv_file
        )


        # 상장 전 등 빈 파일
        if df.empty:
            continue


        frames.append(
            df
        )


    # --------------------------------------------------------
    # 데이터가 없는 기업
    # --------------------------------------------------------

    if not frames:

        print(
            f"   ⚠ 데이터 없음"
        )

        continue


    # --------------------------------------------------------
    # 기업별 6개 파일 합치기
    # --------------------------------------------------------

    company_df = pd.concat(
        frames,
        ignore_index=True
    )


    # --------------------------------------------------------
    # 날짜 중복 제거
    #
    # 다운로드 기간 경계가 겹칠 경우
    # 중복 합산 방지
    # --------------------------------------------------------

    company_df = (
        company_df
        .sort_values("일자")
        .drop_duplicates(
            subset=["일자"],
            keep="last"
        )
    )


    # --------------------------------------------------------
    # Series 생성
    #
    # index = 날짜
    # value = 외국인 순매수 금액
    # --------------------------------------------------------

    series = (
        company_df
        .set_index("일자")[
            "외국인 합계"
        ]
    )


    company_series[
        company_no
    ] = series


# ============================================================
# 6. 기업 1~150 WIDE 데이터 생성
# ============================================================

wide_data = {}


for company_no in range(
    1,
    151
):


    # --------------------------------------------------------
    # 기업명
    #
    # 폴더 자체가 없는 경우
    # "기업131"처럼 임시명 생성
    # --------------------------------------------------------

    company_name = (
        company_name_map.get(
            company_no,
            f"기업{company_no:03d}"
        )
    )


    column_name = (
        f"{company_no:03d}_"
        f"{company_name}"
    )


    # --------------------------------------------------------
    # 데이터가 있는 경우
    # --------------------------------------------------------

    if company_no in company_series:

        wide_data[
            column_name
        ] = company_series[
            company_no
        ]


    # --------------------------------------------------------
    # 데이터 없는 기업은 나중에 NaN 컬럼 생성
    # --------------------------------------------------------

    else:

        wide_data[
            column_name
        ] = pd.Series(
            dtype=float
        )


# ============================================================
# 7. 기업별 데이터를 날짜 기준으로 합치기
# ============================================================

wide_df = pd.DataFrame(
    wide_data
)


wide_df.index.name = "일자"


wide_df = (
    wide_df
    .sort_index()
)


# ============================================================
# 8. 날짜별 외국인 순매수 금액 총합
#
# 총합 =
# 삼성전자
# + SK하이닉스
# + ...
# + 150번 기업
#
# NaN은 상장 전 / 데이터 없음
# ============================================================

foreign_total = wide_df.sum(
    axis=1,
    skipna=True,
    min_count=1
)


# ============================================================
# 9. 최종 파일
#
# 컬럼 순서:
#
# 일자
# 외국인_순매수금액_총합
# 001_삼성전자
# 002_SK하이닉스
# ...
# 150_기업
# ============================================================

final_df = wide_df.copy()


final_df.insert(
    0,
    "외국인_순매수금액_총합",
    foreign_total
)


final_df = (
    final_df
    .reset_index()
)


# 날짜 형식
final_df["일자"] = (
    final_df["일자"]
    .dt.strftime(
        "%Y-%m-%d"
    )
)


# ============================================================
# 10. CSV 하나만 저장
# ============================================================

final_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. 결과 확인
# ============================================================

print("\n" + "=" * 70)

print(
    "KRX 150 통합 완료"
)

print("=" * 70)

print(
    f"거래일 수: "
    f"{len(final_df):,}"
)

print(
    f"전체 컬럼 수: "
    f"{len(final_df.columns)}"
)

print(
    f"기업 컬럼 수: "
    f"{len(final_df.columns) - 2}"
)

print(
    f"\n저장 위치:\n"
    f"{OUTPUT_FILE}"
)

print(
    "\n최종 구조:"
)

print(
    "일자 | "
    "외국인_순매수금액_총합 | "
    "001_기업 | "
    "002_기업 | ... | "
    "150_기업"
)

print(
    "\n완료."
)