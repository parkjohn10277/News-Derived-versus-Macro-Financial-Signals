"""
Stage 4 -- LDA, per language.

Trains one topic model per language with K fixed at config.NUM_TOPICS (20).
The grid search over K is deliberately deferred; `train_one(lang, k)` is factored
out so a later script can just loop it and compare coherence.

Model class is config-selected:
  * LdaMulticore  -- Online Variational Bayes, parallel (methods section).
                     alpha in {'symmetric','asymmetric'|float}; eta {'symmetric','auto'|float}
  * LdaModel      -- single-core; additionally allows alpha='auto', eta='auto'

Outputs (per lang): artifacts/04_lda_{lang}.model
                    output/04_topterms_{lang}.csv     (top-30 terms/topic, for labelling)
                    output/04_coherence_{lang}.json    (c_v, u_mass at K=20)
"""
from __future__ import annotations

import _bootstrap  # noqa: F401
import json

import pandas as pd
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel, LdaMulticore

import config


def _make_model(corpus, dictionary, k: int):
    common = dict(
        corpus=corpus,
        id2word=dictionary,
        num_topics=k,
        chunksize=config.LDA_CHUNKSIZE,
        passes=config.LDA_PASSES,
        iterations=config.LDA_ITERATIONS,
        eval_every=config.LDA_EVAL_EVERY,
        random_state=config.LDA_RANDOM_STATE,
    )
    if config.LDA_MODEL_CLASS == "LdaMulticore":
        return LdaMulticore(
            workers=config.LDA_WORKERS,
            alpha=config.LDA_ALPHA if config.LDA_ALPHA != "auto" else "symmetric",
            eta=config.LDA_ETA,
            **common,
        )
    return LdaModel(alpha=config.LDA_ALPHA, eta=config.LDA_ETA, **common)


def train_one(lang: str, k: int, save: bool = True):
    paths = config.corpus_paths(lang)
    dictionary = Dictionary.load(str(paths["dictionary"]))
    corpus = MmCorpus(str(paths["corpus"]))
    tokens = (pd.read_parquet(config.TOKENS_PARQUET, filters=[("lang", "==", lang)])
                [["article_id", "tokens"]].drop_duplicates("article_id"))
    # align token lists to the (possibly pruned) corpus via docmeta, preserving corpus order
    meta = pd.read_parquet(paths["docmeta"])
    tok_map = dict(zip(tokens["article_id"], tokens["tokens"]))
    texts = [list(tok_map.get(aid, [])) for aid in meta["article_id"]]

    print(f"[{lang}] training {config.LDA_MODEL_CLASS}  K={k}  "
          f"docs={len(corpus)}  vocab={len(dictionary)}")
    model = _make_model(corpus, dictionary, k)

    cm_cv = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence="c_v")
    cm_um = CoherenceModel(model=model, corpus=corpus, dictionary=dictionary, coherence="u_mass")
    scores = {"k": k, "c_v": float(cm_cv.get_coherence()), "u_mass": float(cm_um.get_coherence())}
    print(f"[{lang}] K={k}  c_v={scores['c_v']:.4f}  u_mass={scores['u_mass']:.4f}")

    if save:
        model.save(str(paths["model"]))
        rows = []
        for tid in range(k):
            for rank, (term, wt) in enumerate(model.show_topic(tid, topn=30)):
                rows.append({"topic": tid, "rank": rank, "term": term, "weight": round(float(wt), 5)})
        pd.DataFrame(rows).to_csv(paths["topterms"], index=False, encoding="utf-8-sig")
        paths["coherence"].write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(f"[{lang}] saved model + {paths['topterms'].name} + {paths['coherence'].name}")

    return model, scores


def main() -> None:
    for lang in ("ko", "en"):
        train_one(lang, config.NUM_TOPICS, save=True)
    print("done.  Inspect output/04_topterms_*.csv and assign topic labels.")


if __name__ == "__main__":
    main()
