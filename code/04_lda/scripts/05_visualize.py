"""
Stage 5 -- pyLDAvis, per language.

Builds the interactive intertopic-distance / salient-term visualisation for
qualitative validation and labelling (relevance metric lambda is the slider in
the resulting page; Sievert & Shirley 2014).

  pyLDAvis.gensim_models.prepare(model, corpus, dictionary, sort_topics=False)

`sort_topics=False` keeps pyLDAvis topic numbers identical to the gensim model
(and therefore to the TR_{t,k} columns from stage 6). Note pyLDAvis renders
topics 1-indexed; gensim is 0-indexed.

Outputs (per lang): output/05_ldavis_{lang}.html
"""
from __future__ import annotations

import _bootstrap  # noqa: F401

import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import LdaModel, LdaMulticore

import config


def _load_model(path: str):
    try:
        return LdaMulticore.load(path)
    except Exception:
        return LdaModel.load(path)


def visualize(lang: str) -> None:
    paths = config.corpus_paths(lang)
    dictionary = Dictionary.load(str(paths["dictionary"]))
    corpus = MmCorpus(str(paths["corpus"]))
    model = _load_model(str(paths["model"]))

    print(f"[{lang}] preparing pyLDAvis  (K={model.num_topics}, docs={len(corpus)}) ...")
    vis = gensimvis.prepare(model, corpus, dictionary, sort_topics=False, mds="pcoa")
    pyLDAvis.save_html(vis, str(paths["ldavis"]))
    print(f"[{lang}] -> {paths['ldavis']}")


def main() -> None:
    for lang in ("ko", "en"):
        visualize(lang)
    print("done. Open the HTML files in a browser; use the lambda slider (~0.6) to label topics.")


if __name__ == "__main__":
    main()
