"""
English preprocessing: case normalisation, RegexpTokenizer (alphabetic tokens),
NLTK stopword removal, lemmatisation.

Mirrors the methods section: "standard case normalization, stop-word removal via
NLTK, and [normalisation] to reduce inflected forms to their common base
[form]." The default is WordNet lemmatisation rather than stemming, so terms
stay readable dictionary words (company, study, price) instead of truncated
stems (compani, studi, price); set config.EN_NORMALIZER = "stem" to restore the
original Snowball behaviour.

Note the deliberate asymmetry with the Korean side (nouns-only there, all content
words here) -- this follows the written methodology. To enforce POS symmetry,
add an nltk.pos_tag filter for NN* tags in `tokenize()`.
"""
from __future__ import annotations

import functools

import config
from src.textutils import clean_common


@functools.lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    import nltk
    from nltk.corpus import stopwords

    try:
        base = set(stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        base = set(stopwords.words("english"))
    base.update(config.EN_STOPWORDS_DOMAIN)
    return frozenset(base)


@functools.lru_cache(maxsize=1)
def _load_tokenizer():
    from nltk.tokenize import RegexpTokenizer

    return RegexpTokenizer(config.EN_TOKEN_PATTERN)


@functools.lru_cache(maxsize=1)
def _load_stemmer():
    if config.EN_STEMMER == "porter":
        from nltk.stem import PorterStemmer

        return PorterStemmer()
    from nltk.stem.snowball import SnowballStemmer

    return SnowballStemmer("english")


@functools.lru_cache(maxsize=1)
def _load_lemmatizer():
    import nltk
    from nltk.stem import WordNetLemmatizer

    lem = WordNetLemmatizer()
    try:
        lem.lemmatize("tests")
    except LookupError:
        nltk.download("wordnet", quiet=True)
        nltk.download("omw-1.4", quiet=True)
        lem.lemmatize("tests")
    return lem


@functools.lru_cache(maxsize=1 << 18)
def _lemma(word: str) -> str:
    """
    WordNet lemma of *word*, trying noun -> verb -> adjective.

    No POS tagger is run (too slow for a corpus this size); the first reading
    that actually changes the word wins, which folds plurals and verb
    inflections while leaving genuine nouns alone.
    """
    lem = _load_lemmatizer()
    for pos in ("n", "v", "a"):
        out = lem.lemmatize(word, pos)
        if out != word:
            return out
    return word


@functools.lru_cache(maxsize=1)
def _load_normalizer():
    """Return the word -> base-form function selected by config.EN_NORMALIZER."""
    mode = config.EN_NORMALIZER
    if mode == "lemma":
        return _lemma
    if mode == "stem":
        return _load_stemmer().stem
    if mode == "none":
        return lambda w: w
    raise ValueError(
        f"config.EN_NORMALIZER must be 'lemma', 'stem' or 'none' (got {mode!r})"
    )


def tokenize(text: str) -> list[str]:
    """Clean -> lowercase -> tokenise -> stopword filter -> lemmatise."""
    text = clean_common(text)
    if not text:
        return []

    tokenizer = _load_tokenizer()
    stop = _load_stopwords()
    normalize = _load_normalizer()
    min_len = config.EN_MIN_TOKEN_LEN

    out: list[str] = []
    for tok in tokenizer.tokenize(text.lower()):
        if len(tok) < min_len or tok in stop:
            continue
        base = normalize(tok)
        if len(base) < min_len or base in stop:
            continue
        out.append(base)
    return out


def tokenize_many(texts: list[str]):
    """Yield token lists in input order (English path is pure-Python, no batching win)."""
    for t in texts:
        yield tokenize(t)
