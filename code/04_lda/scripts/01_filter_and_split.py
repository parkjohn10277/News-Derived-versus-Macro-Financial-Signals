"""
Stage 1 -- domain filter + language split.

Reads NEWS.csv (chunked), concatenates the text fields, applies the domain
keyword filter (config.FILTER_*), tags each surviving article with a
character-detected language, and writes a single tidy parquet.

Output: artifacts/01_filtered.parquet
Columns: article_id, id_date (datetime64), dataset, media, country,
         lang ('ko'|'en'), text (raw concatenated), n_domain_hits
'unknown'-language rows are dropped (too short to model).
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src import textutils

USECOLS = [config.ID_FIELD, "dataset", "date", "title", "summary", "body", "media", "country"]

SCHEMA = pa.schema([
    ("article_id", pa.string()),
    ("id_date", pa.timestamp("ns")),
    ("dataset", pa.string()),
    ("media", pa.string()),
    ("country", pa.string()),
    ("lang", pa.string()),
    ("text", pa.string()),
    ("n_domain_hits", pa.int32()),
])


def main() -> None:
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

    config.FILTERED_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(config.FILTERED_PARQUET, SCHEMA, compression="zstd")

    seen = 0
    kept = 0
    by_lang = {"ko": 0, "en": 0}
    try:
        for chunk in reader:
            seen += len(chunk)
            dates = textutils.parse_dates(chunk["date"])

            recs = []
            for row, dt in zip(chunk.to_dict("records"), dates):
                if pd.isna(dt):
                    continue
                text = textutils.coalesce_text(row)
                if not text:
                    continue
                hits = textutils.domain_hits(text)
                if config.FILTER_ENABLED and hits < config.FILTER_MATCH_MIN_HITS:
                    continue
                lang = textutils.detect_lang(text)
                if lang not in ("ko", "en"):
                    continue
                recs.append({
                    "article_id": row.get(config.ID_FIELD, ""),
                    "id_date": dt,
                    "dataset": row.get("dataset", ""),
                    "media": row.get("media", ""),
                    "country": row.get("country", ""),
                    "lang": lang,
                    "text": text,
                    "n_domain_hits": int(hits),
                })
                by_lang[lang] += 1

            if recs:
                df = pd.DataFrame.from_records(recs)
                df["id_date"] = pd.to_datetime(df["id_date"])
                writer.write_table(pa.Table.from_pandas(df, schema=SCHEMA, preserve_index=False))
                kept += len(recs)

            print(f"  scanned {seen:,}  kept {kept:,}  (ko={by_lang['ko']:,} en={by_lang['en']:,})",
                  end="\r", flush=True)
    finally:
        writer.close()

    print()
    print(f"done. {kept:,} / {seen:,} articles kept -> {config.FILTERED_PARQUET}")
    print(f"  ko = {by_lang['ko']:,}")
    print(f"  en = {by_lang['en']:,}")
    if kept == 0:
        print("WARNING: 0 rows kept. Check config.FILTER_TERMS_* or set FILTER_ENABLED=False.")


if __name__ == "__main__":
    main()
