import json
import re
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ============================================================
# 0. 경로 / 실행 설정
# ============================================================

NEWS_DIR = Path(
    r".\1_News"
)

# 우선 사용하려는 파일명
INPUT_FILE = NEWS_DIR / "MCST_EcoTech..csv"

# 위 파일명이 실제로 없으면 자동 검색
if not INPUT_FILE.exists():
    candidates = [
        p for p in NEWS_DIR.glob("MCST_EcoTech*.csv")
        if "EXAONE" not in p.name.upper()
        and "checkpoint" not in p.name.lower()
    ]

    if not candidates:
        raise FileNotFoundError(
            f"MCST_EcoTech*.csv 파일을 찾을 수 없습니다.\n"
            f"검색 위치: {NEWS_DIR}"
        )

    # 전처리 파일이 있으면 우선
    preprocessed = [
        p for p in candidates
        if "preprocessed" in p.name.lower()
    ]

    INPUT_FILE = (
        sorted(preprocessed)[0]
        if preprocessed
        else sorted(candidates)[0]
    )

OUTPUT_DIR = NEWS_DIR / "Sentimental Analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = (
    OUTPUT_DIR / "MCST_EcoTech_EXAONE_sentiment.csv"
)

ERROR_LOG = (
    OUTPUT_DIR / "MCST_EcoTech_EXAONE_errors.txt"
)

MODEL_NAME = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"

# EXAONE 3.5 초기 안정 버전 고정
MODEL_REVISION = "496aef060b296b34c6b0035149f5af9e2b8c168c"

# ------------------------------------------------------------
# 중요:
# 처음에는 10으로 테스트하세요.
# 10건 정상 확인 후 None으로 바꾸면 전체 304,215건 처리합니다.
# ------------------------------------------------------------
MAX_ARTICLES = None

# 기존 결과가 있으면 이어서 실행
RESUME = True

# 메모리 절약 + 중간 저장 단위
CHUNK_SIZE = 100

# 제목 + 요약 기준
MAX_INPUT_TOKENS = 768

# JSON + 짧은 판단근거
MAX_NEW_TOKENS = 80


# ============================================================
# 1. CSV 인코딩 확인
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
        f"CSV 인코딩을 확인할 수 없습니다:\n{file_path}"
    )


CSV_ENCODING = detect_encoding(INPUT_FILE)


# ============================================================
# 2. 입력 컬럼 확인
# ============================================================

sample_df = pd.read_csv(
    INPUT_FILE,
    encoding=CSV_ENCODING,
    nrows=5,
    low_memory=False,
)

REQUIRED_COLUMNS = [
    "제목(국문)",
    "요약내용",
]

missing_columns = [
    col for col in REQUIRED_COLUMNS
    if col not in sample_df.columns
]

if missing_columns:
    raise ValueError(
        f"필수 컬럼이 없습니다: {missing_columns}\n"
        f"현재 컬럼: {list(sample_df.columns)}"
    )


print("=" * 75)
print("EXAONE-3.5-7.8B-Instruct 감성분석")
print("=" * 75)
print(f"입력 파일 : {INPUT_FILE}")
print(f"인코딩    : {CSV_ENCODING}")
print(f"출력 파일 : {OUTPUT_FILE}")
print(f"모델       : {MODEL_NAME}")
print("=" * 75)


# ============================================================
# 3. GPU 확인
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 감지되지 않았습니다."
    )

GPU_NAME = torch.cuda.get_device_name(0)

GPU_VRAM = (
    torch.cuda.get_device_properties(0).total_memory
    / 1024**3
)

print(f"\nGPU  : {GPU_NAME}")
print(f"VRAM : {GPU_VRAM:.1f} GB")
print(f"CUDA : {torch.version.cuda}")


# ============================================================
# 4. 기존 결과 행 수 확인
# ============================================================

def count_csv_rows(file_path):
    """
    결과 CSV를 RAM에 전부 올리지 않고 행 수만 계산.
    """
    if not file_path.exists():
        return 0

    total = 0

    try:
        for part in pd.read_csv(
            file_path,
            encoding="utf-8-sig",
            usecols=[0],
            chunksize=100_000,
            low_memory=False,
        ):
            total += len(part)

    except pd.errors.EmptyDataError:
        return 0

    return total


if not RESUME and OUTPUT_FILE.exists():
    OUTPUT_FILE.unlink()

processed_rows = (
    count_csv_rows(OUTPUT_FILE)
    if RESUME
    else 0
)

print(
    f"\n기존 결과에 저장된 기사: "
    f"{processed_rows:,}건"
)


# ============================================================
# 5. RTX 4060 8GB용 4-bit NF4 설정
# ============================================================

COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)

print(
    f"4-bit compute dtype: "
    f"{COMPUTE_DTYPE}"
)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
    bnb_4bit_use_double_quant=True,
)


# ============================================================
# 6. Tokenizer / Model 로딩
# ============================================================

print("\nTokenizer 로딩 중...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    revision=MODEL_REVISION
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

tokenizer.truncation_side = "right"
print("EXAONE 4-bit 모델 로딩 중...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,

    # EXAONE 초기 안정 코드 고정
    revision=MODEL_REVISION,

    # EXAONE custom model code 사용
    trust_remote_code=True,

    # RTX 4060 8GB용 4-bit
    quantization_config=quantization_config,

    # GPU 0에 배치
    device_map={"": 0},

    low_cpu_mem_usage=True
)

model.eval()

print("\n모델 로딩 완료.")
print("Model device:", next(model.parameters()).device)
print("Device map:", getattr(model, "hf_device_map", "정보 없음"))

print(
    "Model first parameter device:",
    next(model.parameters()).device,
)

print(
    "Device map:",
    getattr(model, "hf_device_map", "정보 없음"),
)

# 모델 전체가 cuda:0에 올라갔는지 확인
device_map = getattr(model, "hf_device_map", {})

bad_devices = [
    value
    for value in device_map.values()
    if str(value) not in {"0", "cuda:0"}
]

if bad_devices:
    raise RuntimeError(
        "모델 일부가 GPU 이외의 장치에 배치되었습니다.\n"
        f"device_map={device_map}\n"
        "RTX 4060 8GB에서 4-bit 로딩 상태를 다시 확인하세요."
    )


# ============================================================
# 7. 고정 Prompt
# ============================================================

SYSTEM_PROMPT = """
당신은 금융시장과 반도체 산업을 분석하는 전문가이다.

주어진 뉴스의 제목과 요약내용만을 근거로,
해당 뉴스가 한국 반도체 산업에 갖는 정보적 함의를
Positive, Negative, Neutral 중 하나로 분류하라.

Positive:
한국 반도체 산업의 전망, 수요, 실적, 경쟁력,
투자환경 또는 사업환경의 개선을 시사하는 경우.

Negative:
한국 반도체 산업의 전망, 수요, 실적, 경쟁력,
투자환경 또는 사업환경의 악화를 시사하는 경우.

Neutral:
한국 반도체 산업과 관련된 정보이지만
긍정 또는 부정의 방향성이 명확하지 않은 경우.

반드시 제공된 뉴스의 제목과 요약내용만 사용하라.
외부 정보나 기사 작성 이후의 미래 정보를 사용하지 마라.

판단 근거는 한 문장으로 간결하게 작성하라.

반드시 아래 JSON 형식으로만 출력하라.

{
  "sentiment": "Positive | Negative | Neutral",
  "reason": "판단 근거"
}
""".strip()


def make_user_prompt(title, summary):
    title = (
        ""
        if pd.isna(title)
        else str(title).strip()
    )

    summary = (
        ""
        if pd.isna(summary)
        else str(summary).strip()
    )

    return f"""
[뉴스 제목]
{title}

[뉴스 요약]
{summary}

위 뉴스의 감성을 분류하라.
""".strip()


# ============================================================
# 8. 출력 파싱
# ============================================================

VALID_LABELS = {
    "positive": "Positive",
    "negative": "Negative",
    "neutral": "Neutral",
}


def parse_response(response):
    cleaned = (
        response
        .replace("```json", "")
        .replace("```JSON", "")
        .replace("```", "")
        .strip()
    )

    # JSON 우선
    match = re.search(
        r"\{.*?\}",
        cleaned,
        flags=re.DOTALL,
    )

    if match:
        try:
            obj = json.loads(match.group(0))

            raw_sentiment = str(
                obj.get("sentiment", "")
            ).strip()

            reason = str(
                obj.get("reason", "")
            ).strip()

            key = raw_sentiment.lower()

            if key in VALID_LABELS:
                return (
                    VALID_LABELS[key],
                    reason,
                )

        except Exception:
            pass

    # JSON 실패 fallback
    lower = cleaned.lower()

    found_labels = []

    for key, label in VALID_LABELS.items():
        if re.search(
            rf"\b{key}\b",
            lower,
        ):
            found_labels.append(label)

    if len(found_labels) == 1:
        return (
            found_labels[0],
            cleaned[:500],
        )

    return (
        "PARSE_ERROR",
        cleaned[:500],
    )


# ============================================================
# 9. 기사 1건 추론
# ============================================================

@torch.inference_mode()
def analyze_one(title, summary):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": make_user_prompt(
                title,
                summary,
            ),
        },
    ]

    # EXAONE chat template
    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        chat_text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_TOKENS,
        padding=False,
    )

    # 모델 전체가 cuda:0이므로 입력도 cuda:0
    inputs = {
        key: value.to("cuda:0")
        for key, value in inputs.items()
    }

    output_ids = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,

        # deterministic decoding
        do_sample=False,

        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
    )

    # 입력 prompt 토큰 제거
    new_tokens = output_ids[
        0,
        inputs["input_ids"].shape[1]:
    ]

    response = tokenizer.decode(
        new_tokens,
        skip_special_tokens=True,
    ).strip()

    return parse_response(response)


# ============================================================
# 10. GPU OOM 재시도
# ============================================================

def analyze_with_retry(title, summary):
    try:
        return analyze_one(
            title,
            summary,
        )

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()

        print(
            "\n[CUDA OOM] 요약내용을 줄여 1회 재시도"
        )

        shorter_summary = (
            ""
            if pd.isna(summary)
            else str(summary)[:700]
        )

        try:
            return analyze_one(
                title,
                shorter_summary,
            )

        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()

            return (
                "ERROR",
                "CUDA Out Of Memory",
            )


# ============================================================
# 11. 오류 로그 함수
# ============================================================

def write_error_log(
    row_number,
    exception,
):
    text = (
        f"row={row_number}\n"
        f"type={type(exception).__name__}\n"
        f"repr={repr(exception)}\n"
        + "-" * 70
        + "\n"
    )

    with open(
        ERROR_LOG,
        "a",
        encoding="utf-8",
    ) as f:
        f.write(text)


# ============================================================
# 12. Chunk 단위 분석
# ============================================================

reader = pd.read_csv(
    INPUT_FILE,
    encoding=CSV_ENCODING,
    chunksize=CHUNK_SIZE,
    low_memory=False,
)

global_start = 0
new_processed = 0

# 새 파일이면 헤더 기록
write_header = not OUTPUT_FILE.exists()

print("\n" + "=" * 75)
print("감성분석 시작")
print("=" * 75)

if MAX_ARTICLES is None:
    print("모드: 전체 기사 분석")
else:
    print(
        f"모드: 테스트 "
        f"{MAX_ARTICLES:,}건"
    )

stop_after_chunk = False


for chunk_number, chunk in enumerate(
    reader,
    start=1,
):
    original_chunk_length = len(chunk)

    chunk_start = global_start
    chunk_end = (
        global_start
        + original_chunk_length
    )

    global_start = chunk_end

    # 이미 결과에 저장된 구간
    if chunk_end <= processed_rows:
        continue

    # Resume가 chunk 중간인 경우
    if chunk_start < processed_rows:
        skip_count = (
            processed_rows
            - chunk_start
        )

        chunk = chunk.iloc[
            skip_count:
        ].copy()

    else:
        chunk = chunk.copy()

    if chunk.empty:
        continue

    # MAX_ARTICLES 테스트 제한
    if MAX_ARTICLES is not None:
        remaining = (
            MAX_ARTICLES
            - new_processed
        )

        if remaining <= 0:
            break

        if len(chunk) > remaining:
            chunk = chunk.iloc[
                :remaining
            ].copy()

            stop_after_chunk = True

    chunk[
        "EXAONE_Sentiment"
    ] = ""

    chunk[
        "EXAONE_Reason"
    ] = ""

    print(
        f"\nChunk {chunk_number} | "
        f"이번 분석 {len(chunk):,}건"
    )

    for idx in tqdm(
        chunk.index,
        desc=f"Chunk {chunk_number}",
    ):
        title = chunk.at[
            idx,
            "제목(국문)"
        ]

        summary = chunk.at[
            idx,
            "요약내용"
        ]

        title_empty = (
            pd.isna(title)
            or not str(title).strip()
        )

        summary_empty = (
            pd.isna(summary)
            or not str(summary).strip()
        )

        if title_empty and summary_empty:
            sentiment = "EMPTY"
            reason = (
                "제목과 요약내용이 모두 비어 있음"
            )

        else:
            try:
                sentiment, reason = (
                    analyze_with_retry(
                        title,
                        summary,
                    )
                )

            except Exception as e:
                sentiment = "ERROR"

                reason = (
                    f"{type(e).__name__}: "
                    f"{repr(e)[:350]}"
                )

                print(
                    f"\n[ERROR] row={idx}"
                )

                print(
                    "Exception Type:",
                    type(e).__name__,
                )

                print(
                    "Exception:",
                    repr(e),
                )

                write_error_log(
                    idx,
                    e,
                )

                torch.cuda.empty_cache()

        chunk.at[
            idx,
            "EXAONE_Sentiment"
        ] = sentiment

        chunk.at[
            idx,
            "EXAONE_Reason"
        ] = reason

    # ========================================================
    # Chunk 결과 즉시 저장
    # ========================================================

    chunk.to_csv(
        OUTPUT_FILE,
        mode="a",
        header=write_header,
        index=False,
        encoding="utf-8-sig",
    )

    write_header = False

    new_processed += len(chunk)

    print(
        f"저장 완료 | "
        f"이번 실행 처리: "
        f"{new_processed:,}건"
    )

    # 테스트 MAX_ARTICLES에 도달
    if stop_after_chunk:
        break


# ============================================================
# 13. 종료 결과
# ============================================================

total_saved = count_csv_rows(
    OUTPUT_FILE
)

print("\n" + "=" * 75)
print("실행 종료")
print("=" * 75)

print(
    f"이번 실행 새 처리: "
    f"{new_processed:,}건"
)

print(
    f"결과파일 누적: "
    f"{total_saved:,}건"
)

print(
    f"\n결과파일:\n"
    f"{OUTPUT_FILE}"
)

if ERROR_LOG.exists():
    print(
        f"\n오류 로그:\n"
        f"{ERROR_LOG}"
    )

print("=" * 75)

print(
    "\n※ 처음에는 MAX_ARTICLES = 10으로 "
    "정상 결과를 확인하세요."
)

print(
    "※ 정상 확인 후 MAX_ARTICLES = None으로 바꾸고 "
    "다시 실행하면 기존 결과 다음 기사부터 이어서 진행합니다."
)
