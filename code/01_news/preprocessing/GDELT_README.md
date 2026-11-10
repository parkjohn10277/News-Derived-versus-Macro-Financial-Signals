# GDELT2 — Foreign News on Korea / Semiconductors

Foreign (non-Korean) press coverage of the Korean semiconductor industry and its
big players (Samsung, SK Hynix, and the wider chip supply chain), built from
**GDELT 2.0 GKG** metadata and then scraped for the full article text.

All files are CSV, UTF-8 with BOM. Time span: **2015-02-19 → 2026-06-27**.

---

## The two files that matter

### `gdelt_scrape_target_with_body.csv` — MAIN DATASET

The filtered, cleaned set of articles **with the article text scraped in**.

- **Date range: 2015-02-19 → 2026-06-27**
- **583,389 rows total**
- **274,701 rows have a real article body** (~47%). The other ~308k could not be
  scraped (paywall, blocked, dead link, or junk extraction) and have `body = ""`.
- Use `body != ""` to get the usable articles.

Columns: `datetime_utc, url, slug, source, tone, themes, organizations,
locations, tier, body`

- `datetime_utc` – article time in UTC (15-min resolution)
- `tone` – GDELT tone score (negative = negative coverage)
- `themes / organizations / locations` – GDELT tag fields
- `tier` – 1 = Samsung/SK Hynix article, 2 = foreign chip company + Korea link
- `body` – the scraped article text (empty if scraping failed)

### `semi_korea_news.csv` — full candidate list (metadata only)

Everything GDELT returned before filtering. **13,309,435 rows**, same date range.

- **Metadata only — no article body.**
- It has a **`slug`** column: the URL path reduced to plain keywords
  (e.g. `business samsung to acquire apple pay competitor looppay`). The slug is
  what the filter uses to judge relevance and to drop duplicate/syndicated
  copies of the same story.
- Columns: `datetime_utc, url, slug, source, tone, themes, organizations,
  locations`

---

## How the main dataset was made

1. **Harvest** – download GDELT 2.0 GKG files (2015 → now), keep any article
   that mentions a semiconductor company, or an ICT/trade/sanctions topic plus a
   Korea/chip keyword in the slug → `semi_korea_news.csv` (13.3M rows).
2. **Filter** (`slug_filter.py`) – keep only genuinely Korea-related industry
   articles, drop junk domains, PR spam, and consumer/gadget content, then
   de-duplicate on slug → `gdelt_scrape_target.csv` (583,389 rows).
3. **Scrape** (`gdelt_fast_scraper_v3.py`) – fetch each article and extract the
   body text → **`gdelt_scrape_target_with_body.csv`** (274,701 with text).

---

## Other files (intermediate / audit — not needed for analysis)

| File | Rows | What it is |
|---|---|---|
| `gdelt_scrape_target.csv` | 583,389 | step 2 output, before body scraping |
| `gdelt_scrape_target_with_body.csv.jsonl` | 583,389 | scraper checkpoint (same data) |
| `gdelt_tier3_excluded.csv` | 5,835,332 | dropped – no company/keyword signal |
| `gdelt_no_korea_dropped.csv` | 4,424,794 | dropped – no Korea link |
| `gdelt_junk_dropped.csv` | 823,736 | dropped – junk / PR-spam domains |
| `gdelt_consumer_dropped.csv` | 492,775 | dropped – reviews / deals / gadget posts |
| `_pre_merge_backup/` | — | old pre-merge copies, redundant (~13 GB) |
| `*.py` | — | the pipeline scripts |

---

## Quick start

```python
import pandas as pd

df = pd.read_csv("gdelt_scrape_target_with_body.csv",
                 encoding="utf-8-sig", parse_dates=["datetime_utc"])

articles = df[df["body"].fillna("").str.len() > 0]   # 274,701 rows with text
```
