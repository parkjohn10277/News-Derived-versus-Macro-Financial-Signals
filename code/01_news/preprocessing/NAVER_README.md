# Naver News Dataset — Korean Semiconductor Sector

Korean-language news articles about the Korean semiconductor industry and its
major listed companies, collected from the **Naver News Search API**.

All files are CSV, UTF-8 with BOM (`utf-8-sig`), same 10-column schema.

---

## Which file to use

**`naver/naver.csv` is the master file.** It is the merged, de-duplicated
result of every collection run. The other CSVs are individual scrape runs that
were merged into it — keep them only as backups.

| | |
|---|---|
| **Rows** | **871,948** |
| Unique article URLs | ~465,100 |
| Unique (stock, URL) pairs | 786,146 |
| Date range | **2002-01-28 → 2026-03-19** |
| Practical start of dense data | ~2015 (≈91% of articles are 2015 or later; usable coverage from 2010) |
| Language | Korean (`한국어`) only |
| Entities covered | 22 (15 companies + 7 sector/macro topics) |

### Articles per year (naver.csv)

| Year | Articles | | Year | Articles |
|---|---|---|---|---|
| ≤2009 | 2,895 | | 2018 | 41,224 |
| 2010 | 10,629 | | 2019 | 55,771 |
| 2011 | 9,812 | | 2020 | 54,518 |
| 2012 | 12,746 | | 2021 | 62,446 |
| 2013 | 17,634 | | 2022 | 69,969 |
| 2014 | 21,453 | | 2023 | 87,505 |
| 2015 | 24,113 | | 2024 | 121,292 |
| 2016 | 26,318 | | 2025 | 148,316 |
| 2017 | 44,384 | | 2026 (partial) | 61,449 |

---

## Columns

| Column | Meaning |
|---|---|
| `종목코드` | Stock ticker code, or `MAC0xx` for a macro/sector topic |
| `종목명` | Company name or topic name |
| `제목` | Article headline |
| `요약내용` | Article summary / description (from the API, HTML stripped) |
| `내용길이` | Character length of `요약내용` |
| `보도일자` | Publication date (`YYYY-MM-DD`) |
| `URL` | Original article URL (publisher site) |
| `네이버링크` | Naver's link for the article |
| `언어` | Always `한국어` |
| `article_id` | `<code>_<hash of URL>` — **run-specific, not a stable key.** De-duplicate on `URL` (or `종목코드` + `URL`). |

---

## Entities

**Companies (15)** — ticker: name
`005930` 삼성전자 · `000660` SK하이닉스 · `042700` 한미반도체 ·
`058470` 리노공업 · `240810` 원익IPS · `039030` 이오테크닉스 ·
`000990` DB하이텍 · `403870` HPSP · `095340` ISC · `084370` 유진테크 ·
`036930` 주성엔지니어링 · `064760` 티씨케이 · `067310` 하나마이크론 ·
`082270` 젬백스 · `089030` 테크윙

> Note: in `naver.csv` some tickers appear both zero-padded and unpadded
> (e.g. `005930` and `5930`) because runs from different periods were merged.
> Strip leading zeros to normalize.

**Sector / macro topics (7)** — code: search term
`MAC001` 반도체 수출 (exports) · `MAC002` 반도체 생산 (production) ·
`MAC003` 반도체 설비투자 (capex) · `MAC004` 반도체 업황 (market cycle) ·
`MAC005` 반도체 재고 (inventory) · `MAC006` 반도체 (general) ·
`MAC007` 반도체 투자 (investment)

---

## API details

- **Source:** Naver Developers — News Search API
- **Endpoint:** `GET https://openapi.naver.com/v1/search/news.json`
- **Auth:** header `X-Naver-Client-Id` / `X-Naver-Client-Secret`
  (5 key pairs used in round-robin to spread the quota; keys are in the
  collection scripts)
- **Parameters used:** `query`, `display=100` (max per call), `start` (1–1000),
  `sort=date`
- **Hard limits:** max 100 results per call, max `start`=1000 → **only the
  first 1,000 results per query** are reachable. On HTTP 429 the scripts sleep
  5s and retry (up to 3 attempts per page).

### How history was collected

The API only returns recent results per query, so to reach older articles each
entity is queried **one calendar month at a time**, with the month name forced
into the query string, e.g.:

```
삼성전자 "2013년 4월"
```

Company runs also multiply each month by keyword filters
(`실적`, `반도체`, `투자`, `공장`, `매출`, `영업이익`, `증설`) to get past the
1,000-result cap. Results are de-duplicated by `URL` within each run.

### Collection scripts

| Script | Purpose |
|---|---|
| `articles_company.py` | Collect news for the 15 individual companies |
| `articles_semi.py` | Collect news for the 7 sector/macro topics |

Edit `START_DATE` / `END_DATE` in each script for the period to fetch. Each run
writes per-article JSON files into an output folder plus a checkpointed CSV.

---

## Files

| File | Rows | Notes |
|---|---|---|
| `naver/naver.csv` | **871,948** | The full merged & de-duplicated dataset (~465 MB) |

This is the single consolidated output. The individual per-run scrape CSVs that
were merged into it have been removed — they added only ~10,700 unique article
URLs beyond `naver.csv`, which is ~98% complete on its own. Re-run the
collection scripts to add newer articles.

---

## Quick start (Python)

```python
import pandas as pd

df = pd.read_csv("naver/naver.csv", encoding="utf-8-sig", parse_dates=["보도일자"])
df["종목코드"] = df["종목코드"].astype(str).str.lstrip("0")
df = df.drop_duplicates(subset=["종목코드", "URL"])

samsung = df[df["종목코드"] == "5930"]
```
