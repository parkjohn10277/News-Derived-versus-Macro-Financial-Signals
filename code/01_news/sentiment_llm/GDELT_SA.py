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
    r".\1_News\GDELT_API\GDELT_preprocessed.csv"
)

OUTPUT_DIR = (
    INPUT_FILE.parent.parent
    / "Sentimental Analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_FILE = (
    OUTPUT_DIR
    / "GDELT_EXAONE_7.8B_sentiment_score.csv"
)

ERROR_LOG = (
    OUTPUT_DIR
    / "GDELT_EXAONE_7.8B_score_errors.txt"
)


# ============================================================
# 1. GDELT 입력 컬럼
# ============================================================

TITLE_COL = "title_proxy"
TEXT_COL = "body"

# 전처리 파일에 article_id가 있으면 사용하고,
# 없으면 행 번호 기준으로 자동 생성
ARTICLE_ID_COL = "article_id"


# ============================================================
# 2. EXAONE 모델
# ============================================================

MODEL_NAME = (
    "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
)

MODEL_REVISION = (
    "496aef060b296b34c6b0035149f5af9e2b8c168c"
)


# ============================================================
# 3. 실행 설정
# ============================================================

# None = 전체 기사
MAX_ARTICLES = None

# 기존 결과 다음부터 이어서
RESUME = True

# CSV 읽기 단위
CHUNK_SIZE = 500


# RTX 4060 8GB
#
# Reason 생성 없음
# generate() 사용 안 함
#
# 우선 4
# OOM 시 자동:
# 4 -> 2 -> 1
BATCH_SIZE = 4


# 전체 prompt 최대 token
MAX_INPUT_TOKENS = 512

# 기사 본문을 tokenizer 전에 지나치게 길게
# 메모리에 올리지 않기 위한 문자 제한
MAX_BODY_CHARS = 5000


# ============================================================
# 4. 입력파일 확인
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"GDELT_preprocessed.csv를 찾을 수 없습니다.\n"
        f"{INPUT_FILE}"
    )


# ============================================================
# 5. CSV Encoding
# ============================================================

def detect_encoding(file_path):

    encodings = [
        "utf-8-sig",
        "utf-8",
        "cp949",
        "euc-kr",
        "latin1",
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


# ============================================================
# 6. 입력 컬럼 확인
# ============================================================

sample_df = pd.read_csv(
    INPUT_FILE,
    encoding=CSV_ENCODING,
    nrows=5,
    low_memory=False,
)


print("=" * 80)

print(
    "GDELT EXAONE-3.5-7.8B "
    "Sentiment Scoring"
)

print("=" * 80)

print(
    f"Transformers : {transformers.__version__}"
)

print(
    f"Input file   : {INPUT_FILE}"
)

print(
    f"Encoding     : {CSV_ENCODING}"
)

print(
    f"Output file  : {OUTPUT_FILE}"
)

print(
    f"Model        : {MODEL_NAME}"
)

print(
    f"Revision     : {MODEL_REVISION}"
)


print("\nCSV columns")

for i, col in enumerate(
    sample_df.columns,
    start=1,
):

    print(
        f"{i:02d}. {col}"
    )


# ============================================================
# 7. 필수 컬럼 확인
# ============================================================

REQUIRED_COLUMNS = [
    TITLE_COL,
    TEXT_COL,
]


missing_columns = [
    col
    for col in REQUIRED_COLUMNS
    if col not in sample_df.columns
]


if missing_columns:

    raise ValueError(
        f"필수 컬럼이 없습니다:\n"
        f"{missing_columns}\n\n"
        f"현재 컬럼:\n"
        f"{list(sample_df.columns)}"
    )


HAS_ARTICLE_ID = (
    ARTICLE_ID_COL
    in sample_df.columns
)


print("\n" + "=" * 80)

print("Detected columns")

print(
    f"Title proxy : {TITLE_COL}"
)

print(
    f"Article body: {TEXT_COL}"
)

print(
    f"Article ID  : "
    f"{ARTICLE_ID_COL if HAS_ARTICLE_ID else 'AUTO-GENERATED'}"
)

print("=" * 80)


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
    f"GPU  : {GPU_NAME}"
)

print(
    f"VRAM : {GPU_VRAM:.1f} GB"
)

print(
    f"CUDA : {torch.version.cuda}"
)

print("=" * 80)


# ============================================================
# 9. CUDA 속도 최적화
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
# 10. 4-bit dtype
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
# 11. 4-bit NF4
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


tokenizer.padding_side = "left"

tokenizer.truncation_side = "right"


print(
    "Tokenizer loaded."
)


# ============================================================
# 13. 모델 로딩
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
# 14. Classification Tokens
#
# 1 = Positive
# 2 = Neutral
# 3 = Negative
# ============================================================

def get_candidate_token_ids(
    candidates
):

    token_ids = []


    for candidate in candidates:

        ids = tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )


        if len(ids) == 1:

            token_id = ids[0]

            if token_id not in token_ids:

                token_ids.append(
                    token_id
                )


    if not token_ids:

        raise RuntimeError(
            "Single-token class candidate를 "
            "찾을 수 없습니다.\n"
            f"Candidates: {candidates}"
        )


    return token_ids


POS_TOKEN_IDS = (
    get_candidate_token_ids([
        "1",
        " 1",
    ])
)


NEU_TOKEN_IDS = (
    get_candidate_token_ids([
        "2",
        " 2",
    ])
)


NEG_TOKEN_IDS = (
    get_candidate_token_ids([
        "3",
        " 3",
    ])
)


print("\nClassification tokens")

print(
    f"Positive : {POS_TOKEN_IDS}"
)

print(
    f"Neutral  : {NEU_TOKEN_IDS}"
)

print(
    f"Negative : {NEG_TOKEN_IDS}"
)


# ============================================================
# 15. English Prompt
#
# NAVER / NYT와 동일한 기준
# ============================================================

SYSTEM_PROMPT = """
You are a financial expert specializing in financial markets and the semiconductor industry.

Based only on the provided news title and article text, evaluate the informational implication of the news for the South Korean semiconductor industry.

Classify the news from the perspective of its expected economic impact on the South Korean semiconductor industry.

1 = Positive

The news implies a favorable impact on the South Korean semiconductor industry, including improvements in business conditions, demand, earnings, exports, investment, competitiveness, technology, supply-chain conditions, or market conditions.

2 = Neutral

The news does not imply a clearly favorable or unfavorable impact on the South Korean semiconductor industry.

Assign Neutral when the information is ambiguous, primarily factual, or insufficient to determine a directional economic implication.

3 = Negative

The news implies an unfavorable impact on the South Korean semiconductor industry, including deterioration in business conditions, demand, earnings, exports, investment, competitiveness, technology, supply-chain conditions, or market conditions.

Use only the information contained in the provided title and article text.

Do not use external knowledge.

Do not use future information.

Do not infer facts that are not supported by the article.

Select exactly one class:

1 = Positive
2 = Neutral
3 = Negative

Return only the class number.
""".strip()


# ============================================================
# 16. 텍스트 전처리
# ============================================================

def clean_text(value):

    if pd.isna(value):

        return ""


    text = str(value)


    # HTML
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


    # 연속 공백
    text = re.sub(
        r"\s+",
        " ",
        text,
    )


    return text.strip()


# ============================================================
# 17. Token 기준 본문 제한
#
# GDELT body가 길기 때문에 단순 문자 수가 아니라
# 실제 token 수 기준으로 자름
# ============================================================

def truncate_text_tokens(
    text,
    max_tokens
):

    if not text:

        return ""


    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False,
    )


    if len(token_ids) <= max_tokens:

        return text


    token_ids = (
        token_ids[
            :max_tokens
        ]
    )


    return tokenizer.decode(
        token_ids,
        skip_special_tokens=True,
    )


# ============================================================
# 18. Prompt 생성
#
# 전체 입력이 MAX_INPUT_TOKENS를 넘지 않도록
# body token 수를 자동 조절
# ============================================================

def make_prompt(
    title,
    body,
):


    title = clean_text(
        title
    )


    body = clean_text(
        body
    )


    # 지나치게 큰 문자열을 우선 제한
    body = body[
        :MAX_BODY_CHARS
    ]


    # 제목도 비정상적으로 길 경우 제한
    title = truncate_text_tokens(
        title,
        60,
    )


    # 우선 본문 240 token
    body_token_budget = 240


    while True:


        body_short = truncate_text_tokens(
            body,
            body_token_budget,
        )


        user_content = (

            f"[News Title Proxy]\n"
            f"{title}\n\n"

            f"[Article Body]\n"
            f"{body_short}\n\n"

            f"Classification:"
        )


        messages = [

            {
                "role":
                    "system",

                "content":
                    SYSTEM_PROMPT,
            },

            {
                "role":
                    "user",

                "content":
                    user_content,
            },
        ]


        prompt = (
            tokenizer.apply_chat_template(

                messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )


        prompt_length = len(
            tokenizer.encode(
                prompt,
                add_special_tokens=False,
            )
        )


        if (
            prompt_length
            <= MAX_INPUT_TOKENS
        ):

            return prompt


        # prompt가 너무 길면 body를 조금씩 감소
        body_token_budget -= 20


        if body_token_budget <= 40:

            return prompt


# ============================================================
# 19. 여러 token 후보 logit 결합
# ============================================================

def aggregate_token_logits(
    logits,
    token_ids,
):


    selected = (
        logits[
            :,
            token_ids
        ]
    )


    if selected.shape[1] == 1:

        return (
            selected[
                :,
                0
            ]
        )


    return torch.logsumexp(
        selected,
        dim=1,
    )


# ============================================================
# 20. Sentiment & Scoring
#
# Score =
#
# (+1 × Ppositive)
# +
# ( 0 × Pneutral)
# +
# (-1 × Pnegative)
#
# =
#
# Ppositive - Pnegative
#
# generate() 없음
# reason 없음
# ============================================================

@torch.inference_mode()
def analyze_batch(
    titles,
    bodies,
):


    prompts = [

        make_prompt(
            title,
            body,
        )

        for (
            title,
            body
        ) in zip(
            titles,
            bodies,
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

        key:
            value.to(
                "cuda:0",
                non_blocking=True,
            )

        for (
            key,
            value
        ) in inputs.items()
    }


    try:


        outputs = model(

            **inputs,

            use_cache=False,

            return_dict=True,
        )


        next_token_logits = (
            outputs.logits[
                :,
                -1,
                :
            ]
        )


        pos_logits = (
            aggregate_token_logits(

                next_token_logits,

                POS_TOKEN_IDS,
            )
        )


        neu_logits = (
            aggregate_token_logits(

                next_token_logits,

                NEU_TOKEN_IDS,
            )
        )


        neg_logits = (
            aggregate_token_logits(

                next_token_logits,

                NEG_TOKEN_IDS,
            )
        )


        class_logits = (
            torch.stack(

                [
                    pos_logits,
                    neu_logits,
                    neg_logits,
                ],

                dim=1,
            )
        )


        probabilities = (
            torch.softmax(

                class_logits.float(),

                dim=1,
            )
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

                bodies[
                    :middle
                ],
            )


            right = analyze_batch(

                titles[
                    middle:
                ],

                bodies[
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
    # 결과 계산
    # ========================================================

    results = []


    for probs in probabilities_cpu:


        p_positive = float(
            probs[0]
        )


        p_neutral = float(
            probs[1]
        )


        p_negative = float(
            probs[2]
        )


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


        score = (
            p_positive
            -
            p_negative
        )


        results.append(

            (
                sentiment,

                round(
                    score,
                    6
                ),
            )
        )


    # ========================================================
    # GPU 메모리 정리
    # ========================================================

    del inputs
    del outputs
    del next_token_logits
    del pos_logits
    del neu_logits
    del neg_logits
    del class_logits
    del probabilities


    return results


# ============================================================
# 21. 결과 CSV 행 수
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
# 22. 기존 결과 구조
# ============================================================

EXPECTED_OUTPUT_COLUMNS = [

    "article_id",

    "EXAONE_Sentiment",

    "EXAONE_Score",
]


if OUTPUT_FILE.exists():

    try:

        existing_header = (
            pd.read_csv(

                OUTPUT_FILE,

                encoding="utf-8-sig",

                nrows=0,
            )
        )


        existing_columns = list(
            existing_header.columns
        )


        if (
            existing_columns
            != EXPECTED_OUTPUT_COLUMNS
        ):


            raise RuntimeError(

                "\n기존 GDELT 결과 파일의 "
                "컬럼 구조가 현재 코드와 다릅니다.\n\n"

                f"기존 파일:\n"
                f"{OUTPUT_FILE}\n\n"

                f"기존 컬럼:\n"
                f"{existing_columns}\n\n"

                f"현재 필요한 컬럼:\n"
                f"{EXPECTED_OUTPUT_COLUMNS}\n\n"

                "기존 파일을 삭제하거나 "
                "OUTPUT_FILE 이름을 변경하세요."
            )


    except pd.errors.EmptyDataError:

        pass


# ============================================================
# 23. Resume
# ============================================================

if (
    not RESUME
    and OUTPUT_FILE.exists()
):

    OUTPUT_FILE.unlink()


processed_rows = (

    count_csv_rows(
        OUTPUT_FILE
    )

    if RESUME

    else 0
)


# ============================================================
# 24. 전체 기사 수
# ============================================================

def count_input_rows(
    file_path,
    encoding,
):


    total = 0


    first_col = (
        sample_df.columns[0]
    )


    for part in pd.read_csv(

        file_path,

        encoding=encoding,

        usecols=[
            first_col
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


print("\n" + "=" * 80)

print(
    f"Total articles : "
    f"{TOTAL_ARTICLES:,}"
)

print(
    f"Processed      : "
    f"{processed_rows:,}"
)

print(
    f"Remaining      : "
    f"{remaining_articles:,}"
)

print("=" * 80)


# ============================================================
# 25. 실제 읽을 컬럼
# ============================================================

USE_COLUMNS = [

    TITLE_COL,

    TEXT_COL,
]


if HAS_ARTICLE_ID:

    USE_COLUMNS.append(
        ARTICLE_ID_COL
    )


reader = pd.read_csv(

    INPUT_FILE,

    encoding=CSV_ENCODING,

    usecols=USE_COLUMNS,

    chunksize=(
        CHUNK_SIZE
    ),

    low_memory=False,
)


# ============================================================
# 26. 전체 분석
# ============================================================

global_start = 0

new_processed = 0

write_header = (
    not OUTPUT_FILE.exists()
)

start_time = (
    time.time()
)


print("\n" + "=" * 80)

print(
    "GDELT EXAONE-3.5-7.8B "
    "Sentiment Scoring Started"
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
    "Input          : "
    "title_from_slug + body"
)

print(
    "Scoring        : "
    "P(Positive) - P(Negative)"
)

print(
    "Output         : "
    "article_id / EXAONE_Sentiment / EXAONE_Score"
)

print("=" * 80)


for (
    chunk_number,
    chunk
) in enumerate(
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
    # 이미 처리된 Chunk
    # ========================================================

    if (
        chunk_end
        <=
        processed_rows
    ):

        continue


    # ========================================================
    # 중간 Resume
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
    # MAX_ARTICLES
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
    # article_id
    # ========================================================

    if HAS_ARTICLE_ID:


        current_id_col = (
            ARTICLE_ID_COL
        )


    else:


        # 전처리 CSV의 원본 행 번호 기준
        # 첫 기사 = 1
        chunk[
            "article_id"
        ] = (
            chunk.index
            +
            1
        )


        current_id_col = (
            "article_id"
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
    # 빈 기사 확인
    #
    # 전처리 단계에서 body 없는 행을 제거했으므로
    # 일반적으로 모두 통과해야 함
    # ========================================================

    for idx in chunk.index:


        title = (
            chunk.at[
                idx,
                TITLE_COL
            ]
        )


        body = (
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


        body_empty = (

            pd.isna(body)

            or

            not str(
                body
            ).strip()
        )


        if (
            title_empty
            and
            body_empty
        ):


            chunk.at[
                idx,
                "EXAONE_Sentiment"
            ] = (
                "EMPTY"
            )


        else:


            valid_indices.append(
                idx
            )


    print(

        f"\nChunk "
        f"{chunk_number:,}"

        f" | rows "
        f"{len(chunk):,}"

        f" | inference "
        f"{len(valid_indices):,}"
    )


    # ========================================================
    # Batch
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


        titles = [

            chunk.at[
                idx,
                TITLE_COL
            ]

            for idx
            in batch_indices
        ]


        bodies = [

            chunk.at[
                idx,
                TEXT_COL
            ]

            for idx
            in batch_indices
        ]


        try:


            results = (
                analyze_batch(

                    titles,

                    bodies,
                )
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


            (
                sentiment,
                score
            ) = result


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
    # 최종 3개 컬럼
    # ========================================================

    output_chunk = (

        chunk[
            [
                current_id_col,
                "EXAONE_Sentiment",
                "EXAONE_Score",
            ]
        ]

        .copy()
    )


    if (
        current_id_col
        !=
        "article_id"
    ):


        output_chunk = (

            output_chunk.rename(

                columns={
                    current_id_col:
                        "article_id"
                }
            )
        )


    # ========================================================
    # 저장
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


    speed = (

        new_processed
        /
        elapsed

        if elapsed > 0

        else 0
    )


    allocated = (

        torch.cuda
        .memory_allocated(0)

        /
        1024**3
    )


    reserved = (

        torch.cuda
        .memory_reserved(0)

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
        f"{speed:.3f} "
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


    del output_chunk


    torch.cuda.empty_cache()


    gc.collect()


    if (
        MAX_ARTICLES is not None

        and

        new_processed
        >= MAX_ARTICLES
    ):

        break


# ============================================================
# 27. 종료
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
    "GDELT EXAONE-3.5-7.8B "
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