# Bilingual (KO / EN) News LDA Pipeline

Topic decomposition of a mixed Korean/English news corpus centred on the Korean
semiconductor market and its global-value-chain context, with per-document topic
proportions aggregated into daily topic-frequency series for downstream
econometric / ML models.

## Methodology anchors

| Reference | What we take from it |
|---|---|
| **Larsen & Thorsrud (2019)**, *J. Econometrics* | LDA decomposition of a news corpus; **soft** topic proportions θ_{d,k}; daily topic frequency → transform → standardise as continuous features |
| **Han & Kim (2021)**, *J. Behavioral Finance* | Korean stack (morph analysis, Gensim LDA), **K = 20**, LDAvis for interpretation, TR_{t,k} = daily topic rate, downstream regression/neural prediction |
| **KCI reference — 표 8** | Korean standard stopword list (`korean_stopwords.txt`) |

Two explicit deviations, both deliberate:

1. **TR definition.** We use the expected topic proportion (Larsen & Thorsrud),
   *not* the hard argmax topic count of Han & Kim. Preserves LDA's
   mixed-membership information.
2. **YoY differencing is OFF by default** (`config.TR_YOY_MODE = None`). L&T apply
   it because their targets are quarterly macro aggregates; for daily/weekly
   trading targets a 365-day difference destroys the short-horizon signal. Turn
   it on only for low-frequency targets.

## Setup

The conda env **`lda`** is already created (Python 3.11) with everything installed
and verified:

```
gensim 4.3.3 · numpy 1.26.4 · scipy 1.12.0 · pandas 2.2.3 · pyarrow 17.0.0
kiwipiepy 0.23.2 · pyLDAvis 3.4.1 · nltk 3.9 (+ stopwords corpus)
```

Activate it, or call its interpreter directly:

```powershell
conda activate lda
# or, without activating:
& "$env:USERPROFILE\.conda\envs\lda\python.exe" scripts\run_all.py
```

To rebuild from scratch:

```powershell
conda create -y -n lda -c conda-forge python=3.11 pip
& "$env:USERPROFILE\.conda\envs\lda\python.exe" -m pip install -r requirements.txt
& "$env:USERPROFILE\.conda\envs\lda\python.exe" -m nltk.downloader stopwords
```

Notes:
- **`kiwipiepy` model** — `config.KIWI_MODEL_TYPE = "cong"` (the current best default;
  `"sbg"` needs an extra model file not bundled with `kiwipiepy_model`).
- **Windows + `LdaMulticore`** — every entry point that trains a model must sit
  under `if __name__ == "__main__":` (all `scripts/*.py` already do). Don't call
  the stage `main()` functions from an unguarded module.
- `kiwipiepy` is pure-pip (no Mecab/MSVC toolchain). To use Mecab-ko instead,
  swap `src/preprocess_ko.py`; nothing else changes.

## Data

`NEWS.csv` (~1.67 GB) in the project root. Columns:
`article_id, dataset, date, title, summary, body, url, media, country`.
Sources seen: `NYT`, `GDELT` (English), `NAVER`, `MCST` (Korean). `body` is often
empty for NYT/GDELT (title+summary only); populated for NAVER/MCST.

## Run

```bash
python scripts/run_all.py             # full build, stages 00–06
python scripts/run_all.py 04 06       # just re-run LDA → viz → time series
```

or stage by stage:

| Stage | Script | Output |
|---|---|---|
| 0 | `00_diagnostic.py` | `output/00_diagnostic_report.txt`, `00_articles_per_day.csv` |
| 1 | `01_filter_and_split.py` | `artifacts/01_filtered.parquet` (domain-filtered, language-tagged) |
| 2 | `02_preprocess.py` | `artifacts/02_tokens.parquet` (token lists) |
| 3 | `03_build_corpus.py` | `artifacts/03_dict_{ko,en}.dict`, `03_bow_{ko,en}.mm`, `03_docmeta_{ko,en}.parquet` |
| 4 | `04_train_lda.py` | `artifacts/04_lda_{ko,en}.model`, `output/04_topterms_{ko,en}.csv`, `04_coherence_{ko,en}.json` |
| 5 | `05_visualize.py` | `output/05_ldavis_{ko,en}.html` |
| 6 | `06_timeseries.py` | `output/06_topic_timeseries_{ko,en}_{raw,zscore}.csv` |

**Always run stage 0 first** and read the report. It tells you: how many articles
survive the domain filter, the ko/en balance, `body` coverage, and the
articles/day distribution — which decides whether you need `TR_ROLLING_WINDOW` or
weekly resampling.

## Configuration — everything lives in `config.py`

- **Domain filter** — `FILTER_ENABLED`, `FILTER_TERMS_KO`, `FILTER_TERMS_EN`.
  The term lists are a **starting point**; refine them for your exact scope.
  Set `FILTER_ENABLED = False` to model the whole corpus.
- **Language detection** — two-sided and strict: a doc is Korean if Hangul is
  ≥ `LANG_KO_MIN_HANGUL_RATIO` (0.80) of its scripted characters, English if
  ≤ `LANG_EN_MAX_HANGUL_RATIO` (0.05), and **dropped** in between. The corpus is
  bimodal (an empty valley at 30–70 % Hangul), so this costs ~8 % of articles and
  stops mixed-language rows — a Korean summary glued to a full English body —
  from leaking English content words into the Korean model.
- **Korean tokens** — `KIWI_KEEP_TAGS = (NNG, NNP, SL)`: common noun, proper
  noun, Latin-script token (so `TSMC`, `HBM`, `EUV` survive).
  `KO_COMPOUND_NOUNS = True` re-joins noun morphemes that were written as one
  word (영업 + 이익 → `영업이익`, 부동산 + 시장 → `부동산시장`), so terms are full
  words, not fragments. Only morphemes whose character spans touch are joined,
  so every term literally occurs in the article (`미국·중국` stays two words).
  Latin-script tokens inside Korean text are held to `KO_LATIN_POLICY`
  (`"allowlist"` — only `KO_LATIN_ALLOWLIST` entries such as `ai`, `ceo`, `etf`
  survive), and a document whose Latin share exceeds `KO_MAX_LATIN_TOKEN_FRAC`
  (0.25) is dropped outright.
  `semiconductor_userdict.txt` (~185 entries, kiwipiepy tab format) is loaded
  automatically and keeps domain compounds intact — verified: `SK하이닉스`,
  `HBM3E`, `수출규제`, `3나노`, `노광장비`, `초미세공정`, `반도체지원법`,
  `투자세액공제`, `D램`, `AI가속기`. Extend it as your corpus reveals more
  split compounds.
- **English tokens** — RegexpTokenizer (alphabetic) → NLTK stopwords →
  `EN_NORMALIZER` (default `"lemma"`: WordNet lemmatisation, noun → verb →
  adjective, so terms stay real words — `companies` → `company`, `studies` →
  `study`). Set `"stem"` for the original Snowball stems (`compani`, `studi`) or
  `"none"` for surface forms. Follows the written methodology (all content words,
  not nouns-only). For POS symmetry with Korean, add an `nltk.pos_tag` NN\*
  filter in `src/preprocess_en.py`.
  Requires the `wordnet` corpus: `python -m nltk.downloader wordnet omw-1.4`
  (downloaded on first use if missing).
- **Vocabulary pruning** — `DICT_NO_BELOW` (abs. doc count), `DICT_NO_ABOVE = 0.50`
  (fraction; methods: "more than 50 %"), `DICT_KEEP_N`.
- **LDA** — `NUM_TOPICS = 20` (fixed; grid search deferred).
  `LDA_MODEL_CLASS`: `LdaMulticore` (Online VB, parallel; `alpha` ∈
  {symmetric, asymmetric}) or `LdaModel` (single-core, allows `alpha='auto'`).
- **Time series** — `TR_STANDARDIZE`, `TR_YOY_MODE`, `TR_ROLLING_WINDOW`,
  `TR_RESAMPLE`.

## Topic IDs

`topic_00 … topic_19` in the time-series CSVs are the **gensim** topic ids.
pyLDAvis displays them **1-indexed** (topic_00 → "Topic 1"). `sort_topics=False`
is set so the mapping stays fixed. Label topics from
`output/04_topterms_*.csv` + the pyLDAvis pages (relevance slider λ ≈ 0.6).

## The one unresolved design question

The KO and EN models are **independent** — `topic_07` (ko) ≠ `topic_07` (en).
Downstream you either (a) keep two parallel feature blocks, or (b) align topics
across languages by hand (label matching) / via a bilingual seed dictionary.
None of the three references addresses this (all are monolingual). Decide before
building the feature matrix.

## Later: grid search over K

`scripts/04_train_lda.py::train_one(lang, k, save=False)` is factored for it —
loop `k` over a range, collect `c_v` / `u_mass`, pick per language, then set
`config.NUM_TOPICS` (or make it per-language) and re-run 04–06.
