"""
Stage 3 -- dictionary + Bag-of-Words, per language.

For each language independently (per project spec -- the two LDA models never
share a vocabulary or topic space):
  * build a gensim Dictionary from the token lists
  * Dictionary.filter_extremes(no_below, no_above, keep_n)
        no_below : absolute #documents  (rare-term floor)
        no_above : fraction of corpus   (generic-term ceiling; methods: ">50%")
  * doc2bow -> Matrix Market corpus on disk
  * write doc metadata (article_id, id_date) row-aligned with the corpus

Outputs (per lang): artifacts/03_dict_{lang}.dict
                    artifacts/03_bow_{lang}.mm
                    artifacts/03_docmeta_{lang}.parquet
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus

import config


def build_for_lang(lang: str) -> None:
    paths = config.corpus_paths(lang)

    df = pd.read_parquet(config.TOKENS_PARQUET, filters=[("lang", "==", lang)])
    df = df.sort_values("id_date").reset_index(drop=True)
    docs = df["tokens"].tolist()
    print(f"[{lang}] {len(docs):,} documents")

    dictionary = Dictionary(docs)
    raw_vocab = len(dictionary)
    dictionary.filter_extremes(
        no_below=config.DICT_NO_BELOW,
        no_above=config.DICT_NO_ABOVE,
        keep_n=config.DICT_KEEP_N,
    )
    dictionary.compactify()
    if len(dictionary) == 0:
        raise RuntimeError(
            f"[{lang}] vocabulary is empty after filter_extremes. "
            f"Lower config.DICT_NO_BELOW (currently {config.DICT_NO_BELOW}) "
            f"or check that stage 2 produced tokens for this language."
        )
    _ = dictionary[0]  # force id2token to materialise
    print(f"[{lang}] vocabulary: {raw_vocab:,} -> {len(dictionary):,} "
          f"(no_below={config.DICT_NO_BELOW}, no_above={config.DICT_NO_ABOVE})")

    bow = [dictionary.doc2bow(d) for d in docs]

    # drop documents that became empty after pruning; keep metadata aligned
    keep = [i for i, b in enumerate(bow) if b]
    if len(keep) != len(bow):
        print(f"[{lang}] dropped {len(bow) - len(keep):,} docs emptied by vocab pruning")
    bow = [bow[i] for i in keep]
    meta = df.iloc[keep][["article_id", "id_date", "lang"]].reset_index(drop=True)

    paths["dictionary"].parent.mkdir(parents=True, exist_ok=True)
    dictionary.save(str(paths["dictionary"]))
    MmCorpus.serialize(str(paths["corpus"]), bow)
    meta.to_parquet(paths["docmeta"], index=False)
    print(f"[{lang}] wrote {len(bow):,} BoW docs -> {paths['corpus'].name}")


def main() -> None:
    for lang in ("ko", "en"):
        build_for_lang(lang)
    print("done.")


if __name__ == "__main__":
    main()
