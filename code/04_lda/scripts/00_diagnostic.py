"""
Stage 0 -- corpus diagnostics. Reads NEWS.csv once (chunked, proper CSV parsing)
and reports the numbers that decide how the rest of the pipeline is tuned:

  * row count, date span, rows per source dataset
  * body-field coverage (LDA on title+summary only = very short documents)
  * language split (ko / en / unknown) by the character-based detector
  * domain-filter hit rate (how many articles survive FILTER_TERMS_*)
  * articles/day distribution overall and per language (drives TR smoothing)

Outputs:
  output/00_diagnostic_report.txt
  output/00_articles_per_day.csv
Nothing here is destructive; safe to re-run.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json
from collections import Counter

import pandas as pd

import config
from src import textutils

USECOLS = [config.ID_FIELD, "dataset", "date", "title", "summary", "body", "media", "country"]


def main() -> None:
    n_rows = 0
    body_nonempty = 0
    any_text_empty = 0
    ds_counter: Counter = Counter()
    lang_counter: Counter = Counter()
    lang_by_ds: dict[str, Counter] = {}
    kept_counter: Counter = Counter()          # lang -> kept-by-domain-filter count
    date_min, date_max = None, None
    per_day: Counter = Counter()               # (date, lang) -> count
    bad_dates = 0

    reader = pd.read_csv(
        config.RAW_CSV,
        encoding=config.CSV_ENCODING,
        chunksize=config.CSV_CHUNKSIZE,
        dtype=str,
        keep_default_na=False,
        usecols=lambda c: c in USECOLS,
        engine="c",
        on_bad_lines="warn",
    )

    for chunk in reader:
        n_rows += len(chunk)
        ds_counter.update(chunk["dataset"].tolist())
        body_nonempty += int((chunk["body"].str.len() > 0).sum())

        dates = textutils.parse_dates(chunk["date"])
        bad_dates += int(dates.isna().sum())
        valid = dates.dropna()
        if len(valid):
            lo, hi = valid.min(), valid.max()
            date_min = lo if date_min is None else min(date_min, lo)
            date_max = hi if date_max is None else max(date_max, hi)

        for row, dt in zip(chunk.to_dict("records"), dates):
            text = textutils.coalesce_text(row)
            if not text:
                any_text_empty += 1
            lang = textutils.detect_lang(text)
            lang_counter[lang] += 1
            lang_by_ds.setdefault(row["dataset"], Counter())[lang] += 1

            if textutils.passes_domain_filter(text):
                kept_counter[lang] += 1
                if pd.notna(dt):
                    per_day[(dt.date().isoformat(), lang)] += 1

    # ---- articles/day frame (domain-filtered) ---------------------------- #
    rows = [{"date": d, "lang": lg, "n": c} for (d, lg), c in per_day.items()]
    apd = pd.DataFrame(rows)
    if not apd.empty:
        apd = apd.pivot_table(index="date", columns="lang", values="n",
                              aggfunc="sum", fill_value=0).sort_index()
        apd["total"] = apd.sum(axis=1)
        apd.to_csv(config.OUTPUT_DIR / "00_articles_per_day.csv")

    def dist(s: pd.Series) -> dict:
        return {
            "n_days": int(s.shape[0]),
            "min": int(s.min()), "p05": float(s.quantile(.05)),
            "median": float(s.median()), "mean": round(float(s.mean()), 1),
            "p95": float(s.quantile(.95)), "max": int(s.max()),
            "days_with_zero": int((s == 0).sum()),
        }

    # ---- report --------------------------------------------------------- #
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("NEWS.csv  --  diagnostic report")
    lines.append("=" * 70)
    lines.append(f"total rows                : {n_rows:,}")
    lines.append(f"unparseable dates         : {bad_dates:,}")
    lines.append(f"date span (valid)         : {date_min}  ..  {date_max}")
    lines.append(f"rows with empty title+summary+body : {any_text_empty:,}")
    lines.append(f"rows with non-empty body  : {body_nonempty:,}  "
                 f"({body_nonempty / max(n_rows,1):.1%})")
    lines.append("")
    lines.append("rows per source dataset:")
    for ds, c in ds_counter.most_common():
        lines.append(f"  {ds:<24} {c:>12,}")
    lines.append("")
    lines.append("language split (character-based detector, whole corpus):")
    for lg in ("ko", "en", "unknown"):
        lines.append(f"  {lg:<8} {lang_counter[lg]:>12,}  ({lang_counter[lg]/max(n_rows,1):.1%})")
    lines.append("")
    lines.append("language by dataset:")
    for ds, cc in sorted(lang_by_ds.items(), key=lambda kv: -sum(kv[1].values())):
        tot = sum(cc.values())
        lines.append(f"  {ds:<24} ko={cc['ko']:>9,}  en={cc['en']:>9,}  "
                     f"unk={cc['unknown']:>8,}   (n={tot:,})")
    lines.append("")
    lines.append(f"domain filter                : ENABLED={config.FILTER_ENABLED}  "
                 f"min_hits={config.FILTER_MATCH_MIN_HITS}")
    kept_total = sum(kept_counter.values())
    lines.append(f"articles kept by domain filter: {kept_total:,}  "
                 f"({kept_total/max(n_rows,1):.1%} of corpus)")
    for lg in ("ko", "en", "unknown"):
        lines.append(f"  kept {lg:<8} {kept_counter[lg]:>12,}")
    lines.append("")
    if not apd.empty:
        lines.append("articles/day AFTER domain filter:")
        for col in [c for c in ("ko", "en", "total") if c in apd.columns]:
            lines.append(f"  [{col}] {json.dumps(dist(apd[col]))}")
        lines.append("")
        lines.append("  -> if median articles/day is low (<~10) or many zero days,")
        lines.append("     set config.TR_ROLLING_WINDOW (e.g. 7) or TR_RESAMPLE='W'.")
    lines.append("")
    lines.append("NEXT: review the kept-article count and language balance, then run")
    lines.append("      scripts/01_filter_and_split.py")

    report = "\n".join(lines)
    (config.OUTPUT_DIR / "00_diagnostic_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
