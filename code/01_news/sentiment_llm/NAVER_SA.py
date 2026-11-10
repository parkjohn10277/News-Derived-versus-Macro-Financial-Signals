import gc
import re
import time
from pathlib import Path

import pandas as pd
import torch
import transformers

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)


# ============================================================
# 0. 파일 경로
# ============================================================

INPUT_FILE = Path(
    r".\1_News\NAVER.csv"
)

OUTPUT_DIR = (
    INPUT_FILE.parent
    / "Sentimental Analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "NAVER_EXAONE_7.8B_sentiment_score.csv"
)

ERROR_LOG = (
    OUTPUT_DIR
    / "NAVER_EXAONE_7.8B_score_errors.txt"
)


# ============================================================
# 1. EXAONE 모델
# ============================================================

MODEL_NAME = (
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)

# 안정 revision
MODEL_REVISION = (
    "496aef060b296b34c6b0035149f5af9e2b8c168c"
)


# ============================================================
# 2. 실행 설정
# ============================================================

# None = NAVER.csv 전체 실행
MAX_ARTICLES = None

# 기존 결과가 있으면 이어서 실행
RESUME = True

# CSV 읽기 단위
CHUNK_SIZE = 500

# ------------------------------------------------------------
# RTX 4060 8GB 권장
#
# Reason generation이 없으므로 BATCH_SIZE 4부터 시작.
#
# OOM 발생 시:
# 4 -> 2 -> 1
# 자동 분할
# ------------------------------------------------------------

BATCH_SIZE = 4

# 제목 + 요약 최대 token
MAX_INPUT_TOKENS = 384

# tokenizer 이전 문자 제한
MAX_TEXT_CHARS = 1500


# ============================================================
# 3. 입력 파일 확인
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"NAVER.csv를 찾을 수 없습니다.\n"
        f"{INPUT_FILE}"
    )


# ============================================================
# 4. 시작 정보
# ============================================================

print("=" * 80)

print(
    "NAVER EXAONE-3.5-7.8B "
    "Probability-based Sentiment Scoring"
)

print("=" * 80)

print(
    f"Transformers : "
    f"{transformers.__version__}"
)

print(
    f"Input file   : "
    f"{INPUT_FILE}"
)

print(
    f"Output file  : "
    f"{OUTPUT_FILE}"
)

print(
    f"Model        : "
    f"{MODEL_NAME}"
)

print(
    f"Revision     : "
    f"{MODEL_REVISION}"
)

print("=" * 80)


# ============================================================
# 5. CSV 인코딩 확인
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

        except Exception:
            continue


    raise ValueError(
        f"CSV 인코딩 확인 실패:\n"
        f"{file_path}"
    )


CSV_ENCODING = detect_encoding(
    INPUT_FILE
)


print(
    f"\nCSV encoding: "
    f"{CSV_ENCODING}"
)


# ============================================================
# 6. NAVER.csv 컬럼 확인
# ============================================================

sample_df = pd.read_csv(
    INPUT_FILE,
    encoding=CSV_ENCODING,
    nrows=5,
    low_memory=False,
)


print("\nNAVER.csv columns")

for i, col in enumerate(
    sample_df.columns,
    start=1,
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# 7. 입력 컬럼 탐색
# ============================================================

def find_column(candidates):

    columns = list(
        sample_df.columns
    )


    # 정확 일치
    for candidate in candidates:

        candidate_lower = (
            str(candidate)
            .strip()
            .lower()
        )

        for col in columns:

            col_lower = (
                str(col)
                .strip()
                .lower()
            )

            if candidate_lower == col_lower:

                return col


    # 부분 일치
    for candidate in candidates:

        candidate_lower = (
            str(candidate)
            .strip()
            .lower()
        )

        for col in columns:

            col_lower = (
                str(col)
                .strip()
                .lower()
            )

            if candidate_lower in col_lower:

                return col


    return None


TITLE_COL = find_column([
    "제목",
    "제목(국문)",
    "기사제목",
    "뉴스제목",
    "title",
    "headline",
])


TEXT_COL = find_column([
    "요약내용",
    "요약",
    "summary",
    "description",
    "본문",
    "기사본문",
    "content",
    "body",
    "text",
])


ARTICLE_ID_COL = find_column([
    "article_id",
    "articleid",
])


print("\n" + "=" * 80)

print(
    f"Title column   : "
    f"{TITLE_COL}"
)

print(
    f"Summary column : "
    f"{TEXT_COL}"
)

print(
    f"Article ID     : "
    f"{ARTICLE_ID_COL}"
)

print("=" * 80)


if TITLE_COL is None:

    raise ValueError(
        "제목 컬럼을 찾지 못했습니다."
    )


if TEXT_COL is None:

    raise ValueError(
        "요약내용 컬럼을 찾지 못했습니다."
    )


if ARTICLE_ID_COL is None:

    raise ValueError(
        "article_id 컬럼을 찾지 못했습니다."
    )


# ============================================================
# 8. GPU 확인
# ============================================================

if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU가 감지되지 않았습니다."
    )


GPU_NAME = (
    torch.cuda.get_device_name(0)
)

GPU_VRAM = (
    torch.cuda
    .get_device_properties(0)
    .total_memory
    / 1024**3
)


print("\n" + "=" * 80)

print(
    f"GPU  : "
    f"{GPU_NAME}"
)

print(
    f"VRAM : "
    f"{GPU_VRAM:.1f} GB"
)

print(
    f"CUDA : "
    f"{torch.version.cuda}"
)

print("=" * 80)


# ============================================================
# 9. CUDA 속도 설정
# ============================================================

torch.set_grad_enabled(
    False
)

torch.backends.cuda.matmul.allow_tf32 = (
    True
)

torch.backends.cudnn.allow_tf32 = (
    True
)

torch.backends.cudnn.benchmark = (
    True
)


# ============================================================
# 10. 4-bit compute dtype
# ============================================================

COMPUTE_DTYPE = (

    torch.bfloat16

    if torch.cuda.is_bf16_supported()

    else torch.float16
)


print(
    f"\n4-bit compute dtype: "
    f"{COMPUTE_DTYPE}"
)


# ============================================================
# 11. BitsAndBytes 4-bit NF4
# ============================================================

quantization_config = (
    BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=(
            COMPUTE_DTYPE
        ),

        bnb_4bit_use_double_quant=True,
    )
)


# ============================================================
# 12. Tokenizer
# ============================================================

print(
    "\nTokenizer loading..."
)


tokenizer = (
    AutoTokenizer
    .from_pretrained(

        MODEL_NAME,

        revision=MODEL_REVISION,

        trust_remote_code=True,
    )
)


if tokenizer.pad_token_id is None:

    tokenizer.pad_token_id = (
        tokenizer.eos_token_id
    )


# Decoder-only batch inference
tokenizer.padding_side = "left"

tokenizer.truncation_side = "right"


print(
    "Tokenizer loaded."
)


# ============================================================
# 13. EXAONE 모델
# ============================================================

print(
    "\nLoading EXAONE-3.5-7.8B "
    "4-bit model..."
)


model = (
    AutoModelForCausalLM
    .from_pretrained(

        MODEL_NAME,

        revision=MODEL_REVISION,

        trust_remote_code=True,

        quantization_config=(
            quantization_config
        ),

        device_map={
            "": 0
        },

        low_cpu_mem_usage=True,
    )
)


model.eval()


print(
    "\nModel loaded."
)


print(
    "Model device:",
    next(
        model.parameters()
    ).device
)


# ============================================================
# 14. Classification Token
#
# 1 = Positive
# 2 = Neutral
# 3 = Negative
#
# generate() 사용 안 함.
# 다음 token logits만 사용.
# ============================================================

CLASS_TEXT = {

    "Positive": "1",

    "Neutral": "2",

    "Negative": "3",
}


CLASS_TOKEN_IDS = {}


for label, token_text in CLASS_TEXT.items():

    token_ids = tokenizer.encode(
        token_text,
        add_special_tokens=False,
    )

    if len(token_ids) != 1:

        raise RuntimeError(

            f"Class token '{token_text}' "
            f"is not a single token.\n"

            f"Token IDs = {token_ids}"
        )

    CLASS_TOKEN_IDS[
        label
    ] = token_ids[0]


POS_TOKEN_ID = (
    CLASS_TOKEN_IDS[
        "Positive"
    ]
)

NEU_TOKEN_ID = (
    CLASS_TOKEN_IDS[
        "Neutral"
    ]
)

NEG_TOKEN_ID = (
    CLASS_TOKEN_IDS[
        "Negative"
    ]
)


print("\nClassification tokens")

print(
    f"Positive = 1 "
    f"(token ID={POS_TOKEN_ID})"
)

print(
    f"Neutral  = 2 "
    f"(token ID={NEU_TOKEN_ID})"
)

print(
    f"Negative = 3 "
    f"(token ID={NEG_TOKEN_ID})"
)


# ============================================================
# 15. English Prompt
# ============================================================

SYSTEM_PROMPT = """
You are an expert in financial markets and the semiconductor industry.

Based only on the provided news title and summary, determine the informational implication of the news for the South Korean semiconductor industry.

Classify the news into exactly one of the following three categories.

1 = Positive

Select Positive when the news indicates an improvement in the demand, earnings, exports, investment, competitiveness, technological capabilities, supply-chain conditions, market environment, or business environment of the South Korean semiconductor industry.

2 = Neutral

Select Neutral when the news is related to the South Korean semiconductor industry but does not provide a clearly positive or negative directional implication.

Also select Neutral when the article mainly reports factual information, announcements, events, policies, or developments without sufficient evidence of improvement or deterioration.

3 = Negative

Select Negative when the news indicates a deterioration in the demand, earnings, exports, investment, competitiveness, technological capabilities, supply-chain conditions, market environment, or business environment of the South Korean semiconductor industry.

Use only the information contained in the provided news title and summary.

Do not use external knowledge.

Do not use information that became available after the article was published.

Do not infer facts that are not supported by the provided article.

Select exactly one class:

1 = Positive
2 = Neutral
3 = Negative

Return only the class number.
""".strip()


# ============================================================
# 16. 텍스트 정리
# ============================================================

def clean_text(value):

    if pd.isna(value):

        return ""

    text = str(value)


    # HTML tag
    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )


    # URL
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text,
    )


    # 제어문자
    text = re.sub(
        r"[\x00-\x1f\x7f-\x9f]",
        " ",
        text,
    )


    # 공백
    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return text.strip()


# ============================================================
# 17. Prompt 생성
# ============================================================

def make_prompt(
    title,
    text,
):

    title = clean_text(
        title
    )

    text = clean_text(
        text
    )


    # tokenizer 전 문자 길이 제한
    text = text[
        :MAX_TEXT_CHARS
    ]


    user_content = (
        f"[News Title]\n"
        f"{title}\n\n"

        f"[News Summary]\n"
        f"{text}\n\n"

        f"Classification:"
    )


    messages = [

        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },

        {
            "role": "user",
            "content": user_content,
        },
    ]


    return (
        tokenizer.apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


# ============================================================
# 18. 확률 기반 감성 분석
#
# Score =
#
# (+1 × P_positive)
# +
# (0 × P_neutral)
# +
# (-1 × P_negative)
#
# =
#
# P_positive - P_negative
#
#
# probability는 내부 계산에만 사용.
# CSV에는 저장하지 않음.
# ============================================================

@torch.inference_mode()
def analyze_batch(
    titles,
    texts,
):


    prompts = [

        make_prompt(
            title,
            text,
        )

        for title, text
        in zip(
            titles,
            texts,
        )
    ]


    inputs = tokenizer(

        prompts,

        return_tensors="pt",

        padding=True,

        truncation=True,

        max_length=(
            MAX_INPUT_TOKENS
        ),
    )


    inputs = {

        key: value.to(
            "cuda:0",
            non_blocking=True,
        )

        for key, value
        in inputs.items()
    }


    try:

        # ====================================================
        # 1회 forward pass
        #
        # Reason generation 없음
        # generate() 없음
        # ====================================================

        outputs = model(

            **inputs,

            use_cache=False,

            return_dict=True,
        )


        # ====================================================
        # Prompt 바로 다음 token의 logits
        # ====================================================

        next_token_logits = (
            outputs.logits[
                :,
                -1,
                :
            ]
        )


        # ====================================================
        # 3개 클래스 logit 추출
        #
        # 0 = Positive
        # 1 = Neutral
        # 2 = Negative
        # ====================================================

        class_logits = torch.stack(

            [

                next_token_logits[
                    :,
                    POS_TOKEN_ID
                ],

                next_token_logits[
                    :,
                    NEU_TOKEN_ID
                ],

                next_token_logits[
                    :,
                    NEG_TOKEN_ID
                ],

            ],

            dim=1,
        )


        # ====================================================
        # 세 클래스 안에서 normalized probability 계산
        # ====================================================

        probabilities = torch.softmax(

            class_logits.float(),

            dim=1,
        )


        probabilities_cpu = (
            probabilities
            .detach()
            .cpu()
        )


    except torch.cuda.OutOfMemoryError:

        del inputs

        torch.cuda.empty_cache()

        gc.collect()


        # ====================================================
        # OOM 자동 분할
        #
        # 4 -> 2 -> 1
        # ====================================================

        if len(titles) > 1:

            middle = (
                len(titles)
                // 2
            )


            print(
                "\n[CUDA OOM] "
                f"Batch {len(titles)} "
                f"-> {middle} + "
                f"{len(titles) - middle}"
            )


            left = analyze_batch(

                titles[
                    :middle
                ],

                texts[
                    :middle
                ],
            )


            right = analyze_batch(

                titles[
                    middle:
                ],

                texts[
                    middle:
                ],
            )


            return (
                left
                +
                right
            )


        return [
            (
                "ERROR",
                None,
            )
        ]


    # ========================================================
    # 최종 결과
    # ========================================================

    results = []


    for probs in probabilities_cpu:


        # ----------------------------------------------------
        # 내부 probability
        # ----------------------------------------------------

        p_positive = float(
            probs[0]
        )

        p_neutral = float(
            probs[1]
        )

        p_negative = float(
            probs[2]
        )


        # ----------------------------------------------------
        # 가장 높은 확률의 class 선택
        # ----------------------------------------------------

        predicted_index = int(
            torch.argmax(
                probs
            )
        )


        if predicted_index == 0:

            sentiment = (
                "Positive"
            )


        elif predicted_index == 1:

            sentiment = (
                "Neutral"
            )


        else:

            sentiment = (
                "Negative"
            )


        # ----------------------------------------------------
        # Continuous sentiment score
        #
        # Positive = +1
        # Neutral  =  0
        # Negative = -1
        #
        # Score =
        # P(Positive) - P(Negative)
        # ----------------------------------------------------

        sentiment_score = (
            p_positive
            -
            p_negative
        )


        # ----------------------------------------------------
        # probability는 저장하지 않음
        #
        # sentiment + score만 반환
        # ----------------------------------------------------

        results.append(

            (
                sentiment,

                round(
                    sentiment_score,
                    6
                ),
            )
        )


    # ========================================================
    # GPU 객체 삭제
    # ========================================================

    del inputs
    del outputs
    del next_token_logits
    del class_logits
    del probabilities


    return results


# ============================================================
# 19. 결과 CSV 행 수
# ============================================================

def count_csv_rows(
    file_path
):

    if not file_path.exists():

        return 0


    total = 0


    try:

        for part in pd.read_csv(

            file_path,

            encoding="utf-8-sig",

            usecols=[
                "article_id"
            ],

            chunksize=100_000,

            low_memory=False,
        ):

            total += len(part)


    except pd.errors.EmptyDataError:

        return 0


    return total


# ============================================================
# 20. 기존 결과 파일 구조 확인
# ============================================================

EXPECTED_OUTPUT_COLUMNS = [

    "article_id",

    "EXAONE_Sentiment",

    "EXAONE_Score",
]


if OUTPUT_FILE.exists():

    try:

        existing_header = pd.read_csv(
            OUTPUT_FILE,
            encoding="utf-8-sig",
            nrows=0,
        )


        existing_columns = list(
            existing_header.columns
        )


        if (
            existing_columns
            !=
            EXPECTED_OUTPUT_COLUMNS
        ):

            raise RuntimeError(

                "\n기존 결과파일의 컬럼 구조가 "
                "현재 코드와 다릅니다.\n\n"

                f"기존 파일:\n"
                f"{OUTPUT_FILE}\n\n"

                f"기존 컬럼:\n"
                f"{existing_columns}\n\n"

                f"현재 필요한 컬럼:\n"
                f"{EXPECTED_OUTPUT_COLUMNS}\n\n"

                "기존 파일을 삭제하거나 "
                "다른 OUTPUT_FILE 이름을 사용하세요."
            )


    except pd.errors.EmptyDataError:

        pass


# ============================================================
# 21. Resume
# ============================================================

if (
    not RESUME
    and
    OUTPUT_FILE.exists()
):

    OUTPUT_FILE.unlink()


processed_rows = (

    count_csv_rows(
        OUTPUT_FILE
    )

    if RESUME

    else 0
)


print(
    f"\nAlready processed : "
    f"{processed_rows:,}"
)


# ============================================================
# 22. 전체 기사 수
# ============================================================

def count_input_rows(
    file_path,
    encoding,
):

    total = 0


    for part in pd.read_csv(

        file_path,

        encoding=encoding,

        usecols=[
            ARTICLE_ID_COL
        ],

        chunksize=100_000,

        low_memory=False,
    ):

        total += len(part)


    return total


TOTAL_ARTICLES = (
    count_input_rows(

        INPUT_FILE,

        CSV_ENCODING,
    )
)


remaining_articles = max(

    TOTAL_ARTICLES
    -
    processed_rows,

    0,
)


print(
    f"Total articles    : "
    f"{TOTAL_ARTICLES:,}"
)

print(
    f"Remaining         : "
    f"{remaining_articles:,}"
)


# ============================================================
# 23. CSV Reader
#
# 필요한 3개 원본 컬럼만 읽음
#
# article_id
# 제목
# 요약내용
#
# 속도 / RAM 절약
# ============================================================

USE_COLUMNS = [

    ARTICLE_ID_COL,

    TITLE_COL,

    TEXT_COL,
]


reader = pd.read_csv(

    INPUT_FILE,

    encoding=CSV_ENCODING,

    usecols=USE_COLUMNS,

    chunksize=(
        CHUNK_SIZE
    ),

    low_memory=False,
)


global_start = 0

new_processed = 0

write_header = (
    not OUTPUT_FILE.exists()
)

start_time = (
    time.time()
)


# ============================================================
# 24. 분석 시작
# ============================================================

print("\n" + "=" * 80)

print(
    "EXAONE-3.5-7.8B "
    "NAVER Sentiment Scoring Started"
)

print("=" * 80)


print(
    f"Model          : "
    f"{MODEL_NAME}"
)

print(
    f"Batch size     : "
    f"{BATCH_SIZE}"
)

print(
    f"Chunk size     : "
    f"{CHUNK_SIZE}"
)

print(
    f"Input tokens   : "
    f"{MAX_INPUT_TOKENS}"
)

print(
    f"Total articles : "
    f"{TOTAL_ARTICLES:,}"
)

print(
    f"Resume from    : "
    f"{processed_rows:,}"
)

print(
    "Scoring        : "
    "P(Positive) - P(Negative)"
)

print(
    "Output columns : "
    "article_id, EXAONE_Sentiment, EXAONE_Score"
)

print("=" * 80)


# ============================================================
# 25. 전체 실행
# ============================================================

for chunk_number, chunk in enumerate(
    reader,
    start=1,
):


    original_length = (
        len(chunk)
    )


    chunk_start = (
        global_start
    )


    chunk_end = (
        chunk_start
        +
        original_length
    )


    global_start = (
        chunk_end
    )


    # ========================================================
    # 이미 처리 완료된 chunk
    # ========================================================

    if (
        chunk_end
        <=
        processed_rows
    ):

        continue


    # ========================================================
    # chunk 중간부터 Resume
    # ========================================================

    if (
        chunk_start
        <
        processed_rows
    ):

        skip_count = (
            processed_rows
            -
            chunk_start
        )


        chunk = (

            chunk.iloc[
                skip_count:
            ]

            .copy()
        )


    else:

        chunk = (
            chunk.copy()
        )


    if chunk.empty:

        continue


    # ========================================================
    # MAX_ARTICLES 옵션
    # ========================================================

    if MAX_ARTICLES is not None:


        remaining = (
            MAX_ARTICLES
            -
            new_processed
        )


        if remaining <= 0:

            break


        if len(chunk) > remaining:

            chunk = (

                chunk.iloc[
                    :remaining
                ]

                .copy()
            )


    # ========================================================
    # 결과 컬럼
    # ========================================================

    chunk[
        "EXAONE_Sentiment"
    ] = ""


    chunk[
        "EXAONE_Score"
    ] = float("nan")


    valid_indices = []


    # ========================================================
    # 빈 기사 검사
    # ========================================================

    for idx in chunk.index:


        title = (
            chunk.at[
                idx,
                TITLE_COL
            ]
        )


        text = (
            chunk.at[
                idx,
                TEXT_COL
            ]
        )


        title_empty = (

            pd.isna(title)

            or

            not str(
                title
            ).strip()
        )


        text_empty = (

            pd.isna(text)

            or

            not str(
                text
            ).strip()
        )


        if (
            title_empty
            and
            text_empty
        ):


            chunk.at[
                idx,
                "EXAONE_Sentiment"
            ] = "EMPTY"


        else:

            valid_indices.append(
                idx
            )


    print(
        f"\nChunk {chunk_number:,}"
        f" | rows {len(chunk):,}"
        f" | inference "
        f"{len(valid_indices):,}"
    )


    # ========================================================
    # Batch inference
    # ========================================================

    for batch_start in tqdm(

        range(
            0,
            len(valid_indices),
            BATCH_SIZE,
        ),

        desc=(
            f"Chunk {chunk_number}"
        ),

        unit="batch",
    ):


        batch_indices = (

            valid_indices[
                batch_start
                :
                batch_start
                +
                BATCH_SIZE
            ]
        )


        titles = []

        texts = []


        for idx in batch_indices:


            titles.append(

                chunk.at[
                    idx,
                    TITLE_COL
                ]
            )


            texts.append(

                chunk.at[
                    idx,
                    TEXT_COL
                ]
            )


        try:

            results = analyze_batch(

                titles,

                texts,
            )


        except Exception as e:


            results = [

                (
                    "ERROR",
                    None,
                )

                for _
                in batch_indices
            ]


            # =================================================
            # 오류 로그
            # =================================================

            with open(

                ERROR_LOG,

                "a",

                encoding="utf-8",

            ) as f:


                f.write(
                    f"Chunk="
                    f"{chunk_number}\n"
                )


                f.write(
                    f"Indices="
                    f"{list(batch_indices)}\n"
                )


                f.write(
                    f"Type="
                    f"{type(e).__name__}\n"
                )


                f.write(
                    f"Error="
                    f"{repr(e)}\n"
                )


                f.write(
                    "-" * 80
                    + "\n"
                )


            torch.cuda.empty_cache()

            gc.collect()


        # ====================================================
        # 결과 기록
        # ====================================================

        for (
            idx,
            result
        ) in zip(
            batch_indices,
            results,
        ):


            sentiment, score = (
                result
            )


            chunk.at[
                idx,
                "EXAONE_Sentiment"
            ] = sentiment


            if score is not None:

                chunk.at[
                    idx,
                    "EXAONE_Score"
                ] = score


    # ========================================================
    # 최종 저장용 3개 컬럼만 선택
    # ========================================================

    output_chunk = chunk[

        [
            ARTICLE_ID_COL,
            "EXAONE_Sentiment",
            "EXAONE_Score",
        ]

    ].copy()


    # article_id 컬럼명 강제 통일
    output_chunk = output_chunk.rename(

        columns={
            ARTICLE_ID_COL:
                "article_id"
        }
    )


    # ========================================================
    # Chunk 저장
    # ========================================================

    output_chunk.to_csv(

        OUTPUT_FILE,

        mode="a",

        header=(
            write_header
        ),

        index=False,

        encoding="utf-8-sig",
    )


    write_header = False


    new_processed += (
        len(chunk)
    )


    current_total = (
        processed_rows
        +
        new_processed
    )


    elapsed = (
        time.time()
        -
        start_time
    )


    articles_per_second = (

        new_processed
        /
        elapsed

        if elapsed > 0

        else 0
    )


    # ========================================================
    # GPU 메모리
    # ========================================================

    allocated = (
        torch.cuda.memory_allocated(0)
        /
        1024**3
    )


    reserved = (
        torch.cuda.memory_reserved(0)
        /
        1024**3
    )


    print(
        "\nSaved"
    )


    print(
        f"Progress       : "
        f"{current_total:,}"
        f"/{TOTAL_ARTICLES:,}"
    )


    print(
        f"This run       : "
        f"{new_processed:,}"
    )


    print(
        f"Average speed  : "
        f"{articles_per_second:.3f} "
        f"articles/sec"
    )


    print(
        f"GPU allocated  : "
        f"{allocated:.2f} GB"
    )


    print(
        f"GPU reserved   : "
        f"{reserved:.2f} GB"
    )


    # ========================================================
    # Chunk 종료 후 메모리 정리
    # ========================================================

    del output_chunk

    torch.cuda.empty_cache()

    gc.collect()


    # ========================================================
    # MAX_ARTICLES
    # ========================================================

    if (
        MAX_ARTICLES is not None
        and
        new_processed >= MAX_ARTICLES
    ):

        break


# ============================================================
# 26. 종료
# ============================================================

total_saved = (
    count_csv_rows(
        OUTPUT_FILE
    )
)


elapsed_total = (
    time.time()
    -
    start_time
)


print("\n" + "=" * 80)

print(
    "NAVER EXAONE-3.5-7.8B "
    "Sentiment Scoring Completed"
)

print("=" * 80)


print(
    f"Processed this run : "
    f"{new_processed:,}"
)


print(
    f"Total saved        : "
    f"{total_saved:,}"
)


if (
    elapsed_total > 0
    and
    new_processed > 0
):

    print(
        f"Average speed      : "
        f"{new_processed / elapsed_total:.3f} "
        f"articles/sec"
    )


print(
    "\nFinal columns:"
)

print(
    "1. article_id"
)

print(
    "2. EXAONE_Sentiment"
)

print(
    "3. EXAONE_Score"
)


print(
    f"\nOutput file:\n"
    f"{OUTPUT_FILE}"
)


if ERROR_LOG.exists():

    print(
        f"\nError log:\n"
        f"{ERROR_LOG}"
    )


print("=" * 80)