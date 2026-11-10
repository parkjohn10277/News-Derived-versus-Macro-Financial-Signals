"""
Stage 2 -- tokenisation.

Reads artifacts/01_filtered.parquet and runs the language-specific pipeline on
each document (Korean -> kiwipiepy nouns, compounds re-joined; English -> NLTK
stopword + WordNet lemma).

Output: artifacts/02_tokens.parquet
Columns: article_id, id_date, lang, tokens (list<string>), n_tokens
Documents that reduce to 0 tokens are dropped.
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import config
from src import preprocess_en, preprocess_ko

SCHEMA = pa.schema([
    ("article_id", pa.string()),
    ("id_date", pa.timestamp("ns")),
    ("lang", pa.string()),
    ("tokens", pa.list_(pa.string())),
    ("n_tokens", pa.int32()),
])

BATCH = 5000


def _run_lang(df: pd.DataFrame, lang: str, writer: pq.ParquetWriter) -> tuple[int, int]:
    tok_fn = preprocess_ko.tokenize_many if lang == "ko" else preprocess_en.tokenize_many
    sub = df[df["lang"] == lang].reset_index(drop=True)
    kept = 0
    for start in range(0, len(sub), BATCH):
        part = sub.iloc[start:start + BATCH]
        toks = list(tok_fn(part["text"].tolist()))
        recs = []
        for (_, row), tk in zip(part.iterrows(), toks):
            if not tk:
                continue
            recs.append({
                "article_id": row["article_id"],
                "id_date": row["id_date"],
                "lang": lang,
                "tokens": tk,
                "n_tokens": len(tk),
            })
        if recs:
            out = pd.DataFrame.from_records(recs)
            out["id_date"] = pd.to_datetime(out["id_date"])
            writer.write_table(pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False))
            kept += len(recs)
        print(f"  [{lang}] {min(start + BATCH, len(sub)):,}/{len(sub):,}  kept {kept:,}",
              end="\r", flush=True)
    print()
    return len(sub), kept


def main() -> None:
    df = pd.read_parquet(config.FILTERED_PARQUET)
    print(f"loaded {len(df):,} filtered articles "
          f"(ko={int((df.lang=='ko').sum()):,} en={int((df.lang=='en').sum()):,})")

    config.TOKENS_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(config.TOKENS_PARQUET, SCHEMA, compression="zstd")
    try:
        for lang in ("ko", "en"):
            seen, kept = _run_lang(df, lang, writer)
            print(f"  [{lang}] {kept:,}/{seen:,} documents survived tokenisation")
    finally:
        writer.close()
    print(f"done -> {config.TOKENS_PARQUET}")


if __name__ == "__main__":
    main()
