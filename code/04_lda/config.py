"""
Central configuration for the bilingual (KO/EN) news LDA pipeline.

Every tunable lives here so the scripts stay declarative. Paths are resolved
relative to this file, so the project can be moved without edits.

Methodology anchors:
  - Larsen & Thorsrud (2019, J. Econometrics)  -> soft topic proportions, YoY+standardize
  - Han & Kim (2021, J. Behavioral Finance)    -> Korean stack (Mecab/KoNLPy), Gensim LDA, K=20, LDAvis
  - KCI reference (표 8)                         -> Korean standard stopword list
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent

RAW_CSV = ROOT / "NEWS.csv"

OUTPUT_DIR = ROOT / "output"          # human-facing deliverables (html, csv, reports)
ARTIFACT_DIR = ROOT / "artifacts"     # machine intermediates (parquet, models, dictionaries)

KO_STOPWORDS_FILE = ROOT / "korean_stopwords.txt"

# Intermediate artifact files (stage -> file)
FILTERED_PARQUET = ARTIFACT_DIR / "01_filtered.parquet"          # domain-filtered + language-tagged
TOKENS_PARQUET   = ARTIFACT_DIR / "02_tokens.parquet"            # token lists per document

def corpus_paths(lang: str) -> dict:
    """Per-language artifact paths for the corpus / model / viz stages."""
    return {
        "dictionary": ARTIFACT_DIR / f"03_dict_{lang}.dict",
        "corpus":     ARTIFACT_DIR / f"03_bow_{lang}.mm",
        "docmeta":    ARTIFACT_DIR / f"03_docmeta_{lang}.parquet",   # article_id, date aligned to corpus rows
        "model":      ARTIFACT_DIR / f"04_lda_{lang}.model",
        "topterms":   OUTPUT_DIR   / f"04_topterms_{lang}.csv",
        "coherence":  OUTPUT_DIR   / f"04_coherence_{lang}.json",
        "ldavis":     OUTPUT_DIR   / f"05_ldavis_{lang}.html",
        "tr_raw":     OUTPUT_DIR   / f"06_topic_timeseries_{lang}_raw.csv",
        "tr_z":       OUTPUT_DIR   / f"06_topic_timeseries_{lang}_zscore.csv",
    }

# --------------------------------------------------------------------------- #
# CSV ingestion
# --------------------------------------------------------------------------- #
CSV_ENCODING = "utf-8-sig"            # strips the BOM on the header
CSV_CHUNKSIZE = 50_000
TEXT_FIELDS = ["title", "summary", "body"]   # concatenated (in order, non-empty) into the document text
DATE_FIELD = "id_date"               # canonical name after normalisation; source column is "date"
ID_FIELD = "article_id"

# --------------------------------------------------------------------------- #
# Language identification  (character-based, per project spec)
# --------------------------------------------------------------------------- #
# Two-sided, STRICT gate on the share of Hangul among all *scripted* characters
# (Hangul + Latin):
#   ratio >= LANG_KO_MIN_HANGUL_RATIO  -> 'ko'
#   ratio <= LANG_EN_MAX_HANGUL_RATIO  -> 'en'
#   anything in between                -> 'unknown'  (dropped by stage 01)
# Mixed-language articles (e.g. MCST rows that pair a Korean summary with the
# full English body, 10-30% Hangul) would otherwise leak English content words
# into the Korean model and form junk topics. The corpus is bimodal with an
# empty valley around 30-70% Hangul, so these cuts cost ~8% of articles and
# keep both models monolingual.
LANG_KO_MIN_HANGUL_RATIO = 0.80
LANG_EN_MAX_HANGUL_RATIO = 0.05
# Below this many scripted characters total, the text is too short to judge -> 'unknown'.
LANG_MIN_SCRIPTED_CHARS = 15

# --------------------------------------------------------------------------- #
# Domain filter  (Korean semiconductor market + global value-chain adjacency)
# --------------------------------------------------------------------------- #
# Set False to run LDA on the entire corpus (no domain restriction).
FILTER_ENABLED = False
# Match is case-insensitive; a document is kept if ANY term (KO or EN) appears
# in the concatenated text fields. Refine this list for your research scope.
FILTER_MATCH_MIN_HITS = 1

FILTER_TERMS_KO = [
    # core
    "반도체", "메모리반도체", "시스템반도체", "비메모리", "파운드리", "팹리스",
    "디램", "D램", "낸드", "낸드플래시", "HBM", "고대역폭메모리",
    "웨이퍼", "전공정", "후공정", "패키징", "첨단패키징", "노광", "극자외선",
    "소부장", "소재부품장비", "반도체장비", "반도체소재", "반도체공정",
    # firms
    "삼성전자", "SK하이닉스", "하이닉스", "DB하이텍", "마이크론",
    "TSMC", "엔비디아", "인텔", "ASML", "어플라이드머티리얼즈", "램리서치", "도쿄일렉트론",
    # policy / value chain
    "반도체수출", "수출규제", "수출통제", "반도체지원법", "칩스법", "반도체법",
    "반도체클러스터", "용인클러스터", "칩4", "칩4동맹", "반도체동맹", "위탁생산",
    "공급망", "글로벌공급망", "밸류체인", "가치사슬",
]

FILTER_TERMS_EN = [
    # core
    "semiconductor", "semiconductors", "chipmaker", "chipmakers", "microchip",
    "foundry", "fabless", "wafer", "fab", "chip fab", "dram", "nand",
    "hbm", "high bandwidth memory", "lithography", "euv", "nanometer", "nanometre",
    "memory chip", "logic chip", "advanced packaging", "chip packaging",
    "semiconductor equipment", "semiconductor supply chain", "chip shortage",
    # firms
    "samsung electronics", "sk hynix", "hynix", "tsmc", "micron", "nvidia",
    "asml", "applied materials", "lam research", "tokyo electron",
    # policy / value chain
    "chip war", "export controls", "chips act", "chip 4", "chip4",
    "global value chain", "global supply chain",
]

# --------------------------------------------------------------------------- #
# Preprocessing  --  Korean (kiwipiepy)
# --------------------------------------------------------------------------- #
KIWI_MODEL_TYPE = "cong"            # 'cong' = current best default (kiwipiepy >=0.22);
                                    # 'cong-global' higher accuracy/slower; 'knlm' fastest.
                                    # ('sbg' needs the extra skipbigram model file -- not bundled.)
KIWI_KEEP_TAGS = ("NNG", "NNP", "SL")   # common noun, proper noun, latin-script token
KO_MIN_TOKEN_LEN = 2                 # drop 1-character noun fragments

# Re-join noun morphemes that were written as ONE word, so terms come out as
# full words instead of fragments:
#   영업 + 이익   -> 영업이익      부동산 + 시장 -> 부동산시장
#   인공 + 지능   -> 인공지능      국내 + 기업   -> 국내기업
# Only morphemes that are adjacent in the source text (no space, no punctuation
# between them) are joined, so every emitted term literally appears in the
# article. Compounds already in semiconductor_userdict.txt are never split in
# the first place. Set False for the older morpheme-level behaviour.
KO_COMPOUND_NOUNS = True

# Latin-script (SL) tokens inside Korean text -- STRICT policy.
#   "allowlist": keep an SL token only if it is in KO_LATIN_ALLOWLIST (case-insensitive).
#   "stopwords": keep any SL token that is not an English stopword (older, looser behaviour).
# Domain compounds from semiconductor_userdict.txt (HBM, DRAM, TSMC, SK하이닉스 ...)
# are tagged NNP by kiwi, not SL, so they survive regardless of this policy.
KO_LATIN_POLICY = "allowlist"
KO_LATIN_ALLOWLIST = (
    # tech / AI
    "ai", "gpu", "cpu", "npu", "tpu", "hpc", "iot", "5g", "6g", "llm", "chatgpt", "openai",
    # chip & tech firms
    "tsmc", "asml", "amd", "intel", "nvidia", "micron", "qualcomm", "arm", "apple", "google",
    "microsoft", "amazon", "meta", "tesla", "sk", "lg", "kt", "naver", "kakao",
    # finance / markets
    "etf", "ipo", "ceo", "cfo", "m&a", "kospi", "kosdaq", "nasdaq", "fed", "imf", "wto", "oecd",
    # energy / misc
    "lng", "ev", "ess", "esg", "nasa", "spacex",
)
# Document purity gate: if SL tokens exceed this fraction of the noun-tagged
# tokens in a document, the document is treated as English and dropped from
# the Korean corpus (belt-and-braces behind the character-level detector).
# Measured on articles that already pass the 0.80 Hangul bar: p90 = 0.12,
# p99 = 0.21, so 0.25 removes only true outliers (~0.5%) while 0.10 would
# discard ~14% of genuine Korean news that merely cites acronyms (IRS, MLC ...).
KO_MAX_LATIN_TOKEN_FRAC = 0.25
KIWI_USER_DICT_FILE = ROOT / "semiconductor_userdict.txt"   # optional; created if you add one

# Journalistic / reporting artefacts -- the "domain-specific" layer the methods
# section promises, kept separate from the KCI 표 8 general list.
KO_STOPWORDS_DOMAIN = [
    "기자", "특파원", "앵커", "논설위원", "연합뉴스", "뉴시스", "뉴스", "뉴스1",
    "데일리", "이데일리", "헤럴드", "코리아", "저작권", "무단", "전재", "재배포",
    "배포", "제공", "사진", "그래픽", "기사", "리포트", "인터뷰", "보도", "취재",
    "제목", "주요내용", "내용", "관련", "이날", "지난해", "올해", "내년", "최근",
]

# --------------------------------------------------------------------------- #
# Preprocessing  --  English (NLTK)
# --------------------------------------------------------------------------- #
EN_TOKEN_PATTERN = r"[A-Za-z]+"      # RegexpTokenizer -- alphabetic only, drops digits/punct
EN_MIN_TOKEN_LEN = 3
# How inflected forms are folded together:
#   "lemma" : WordNet lemmatisation -> real dictionary words
#             (companies->company, studies->study, prices->price)
#   "stem"  : Snowball/Porter stemming -> truncated stems
#             (companies->compani, studies->studi, university->univers)
#   "none"  : keep surface forms as written
# Lemmatisation is the default so top-term tables are readable without
# mentally reversing the stemmer.
EN_NORMALIZER = "lemma"
EN_STEMMER = "snowball"              # only used when EN_NORMALIZER == "stem"
EN_STOPWORDS_DOMAIN = [
    "reuters", "bloomberg", "said", "say", "says", "according", "report",
    "reported", "monday", "tuesday", "wednesday", "thursday", "friday",
    "saturday", "sunday", "year", "week", "month", "day", "percent",
]

# --------------------------------------------------------------------------- #
# Dictionary pruning  (gensim Dictionary.filter_extremes)
# --------------------------------------------------------------------------- #
# no_below : absolute number of documents  (rare-term floor)
# no_above : fraction of the corpus         (generic-term ceiling -> methods: "more than 50%")
# keep_n   : hard cap on vocabulary size    (gensim default 100000)
DICT_NO_BELOW = 10
DICT_NO_ABOVE = 0.50
DICT_KEEP_N = 100_000

# --------------------------------------------------------------------------- #
# LDA  (gensim)
# --------------------------------------------------------------------------- #
NUM_TOPICS = 20                      # FIXED for now -- grid search comes later (see scripts/04)
LDA_MODEL_CLASS = "LdaMulticore"    # "LdaMulticore" (Online VB, parallel) or "LdaModel" (allows alpha='auto')
LDA_WORKERS = max(1, (os.cpu_count() or 2) - 1)
LDA_PASSES = 10
LDA_ITERATIONS = 400
LDA_CHUNKSIZE = 2000
LDA_EVAL_EVERY = None
LDA_RANDOM_STATE = 42
# Priors. LdaMulticore: alpha in {'symmetric','asymmetric'} or float; eta in {'symmetric','auto'} or float.
# LdaModel additionally allows alpha='auto'.
LDA_ALPHA = "asymmetric"
LDA_ETA = "symmetric"
# Coherence measures to report for the fixed K (and later across the grid).
COHERENCE_MEASURES = ("c_v", "u_mass")

# --------------------------------------------------------------------------- #
# Topic time series  (TR_{t,k})
# --------------------------------------------------------------------------- #
# Soft aggregation, per Larsen & Thorsrud (2019):
#   TR_{t,k} = (1 / N_t) * sum_{d in day t} theta_{d,k}
TR_STANDARDIZE = True               # z-score each topic column over time
TR_YOY_MODE = None                  # None | "diff" | "growth"  -- keep None for daily/weekly targets
TR_YOY_PERIODS = 365                # calendar days for the YoY lag if TR_YOY_MODE is set
TR_ROLLING_WINDOW = None            # e.g. 7 for a 7-day rolling mean of the raw series; None to disable
TR_RESAMPLE = "D"                   # pandas offset alias: "D" daily, "W" weekly, "M" monthly
