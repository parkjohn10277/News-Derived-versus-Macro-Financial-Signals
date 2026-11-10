"""
Korean preprocessing: morphological analysis with kiwipiepy, noun extraction,
compound re-joining, stopword removal (KCI 표 8 general list + domain
journalistic layer).

Mirrors the methods section: "Only nouns (명사) are extracted ... Domain-specific
financial stop-words ... are systematically removed."  kiwipiepy stands in for
KoNLPy+Mecab-ko (equivalent morphological analysis, pip-installable on Windows).

Noun morphemes that were written as one word are re-joined so terms come out as
full words (영업이익, 부동산시장) rather than morpheme fragments -- see `_words`
and config.KO_COMPOUND_NOUNS.
"""
from __future__ import annotations

import functools

import config
from src.textutils import clean_common


@functools.lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    words: set[str] = set(config.KO_STOPWORDS_DOMAIN)
    if config.KO_STOPWORDS_FILE.exists():
        for ln in config.KO_STOPWORDS_FILE.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("#"):
                words.add(ln)
    return frozenset(words)


@functools.lru_cache(maxsize=1)
def _load_latin_stopwords() -> frozenset[str]:
    """
    English stopwords applied to Latin-script (SL) tokens only.

    Mixed-language articles pass the Hangul-ratio detector and their English
    function words ("the", "of", "and" ...) would otherwise survive as SL nouns
    and form a junk topic. Reuses the English pipeline's list (NLTK english +
    config.EN_STOPWORDS_DOMAIN) so both models filter the same words.
    """
    from src.preprocess_en import _load_stopwords as _en_stopwords

    return _en_stopwords()


@functools.lru_cache(maxsize=1)
def _load_latin_allowlist() -> frozenset[str]:
    return frozenset(w.lower() for w in config.KO_LATIN_ALLOWLIST)


_PUNCT = ".,;:!?'\"()[]"


def _words(tokens, keep_tags: set[str]) -> list[tuple[str, bool]]:
    """
    Group the noun-tagged tokens of a document into the WORDS they were written as.

    kiwipiepy returns morphemes, so a compound noun arrives in pieces
    (영업 + 이익, 인공 + 지능, 부동산 + 시장). Consecutive kept morphemes whose
    character spans touch -- no space and no punctuation between them -- were a
    single word in the source, so they are re-joined. Anything separated by a
    space, a comma or a dropped suffix stays apart, which guarantees every
    emitted word literally occurs in the article.

    Returns (surface_form, is_latin_only) pairs. When config.KO_COMPOUND_NOUNS
    is False each morpheme is returned on its own (the older behaviour).
    """
    if not config.KO_COMPOUND_NOUNS:
        return [(t.form, t.tag == "SL") for t in tokens if t.tag in keep_tags]

    out: list[tuple[str, bool]] = []
    parts: list[str] = []
    all_latin = True
    end = -1
    for tok in tokens:
        if tok.tag not in keep_tags:
            continue
        if parts and tok.start == end:
            parts.append(tok.form)
        else:
            if parts:
                out.append(("".join(parts), all_latin))
            parts = [tok.form]
            all_latin = True
        if tok.tag != "SL":
            all_latin = False
        end = tok.end
    if parts:
        out.append(("".join(parts), all_latin))
    return out


def _filter_tokens(tokens, keep_tags: set[str], stop: frozenset[str],
                   latin_stop: frozenset[str], min_len: int) -> list[str]:
    """
    Rebuild whole words, filter them, then apply the document purity gate.

    Purity gate: if Latin-only words make up more than
    config.KO_MAX_LATIN_TOKEN_FRAC of the noun words *before* filtering, the
    document is mostly English text that slipped past the character-level
    language detector. Return [] so stage 02 drops it from the Korean corpus.
    """
    words = _words(tokens, keep_tags)
    allowlist_only = config.KO_LATIN_POLICY == "allowlist"
    allow = _load_latin_allowlist()

    n_latin = 0
    out: list[str] = []
    for form, is_latin in words:
        if is_latin:
            n_latin += 1
            form = form.lower().strip(_PUNCT)
            if form in latin_stop:
                continue
            if allowlist_only and form not in allow:
                continue
        if len(form) < min_len or form in stop:
            continue
        out.append(form)
    if words and n_latin / len(words) > config.KO_MAX_LATIN_TOKEN_FRAC:
        return []
    return out


@functools.lru_cache(maxsize=1)
def _load_kiwi():
    from kiwipiepy import Kiwi

    kiwi = Kiwi(num_workers=config.LDA_WORKERS, model_type=config.KIWI_MODEL_TYPE)
    if config.KIWI_USER_DICT_FILE.exists():
        kiwi.load_user_dictionary(str(config.KIWI_USER_DICT_FILE))
    return kiwi


def tokenize(text: str) -> list[str]:
    """Clean -> morph-analyse -> keep configured noun tags -> stopword filter."""
    text = clean_common(text)
    if not text:
        return []

    kiwi = _load_kiwi()
    stop = _load_stopwords()
    latin_stop = _load_latin_stopwords()
    keep_tags = set(config.KIWI_KEEP_TAGS)
    min_len = config.KO_MIN_TOKEN_LEN

    return _filter_tokens(kiwi.tokenize(text, normalize_coda=True),
                          keep_tags, stop, latin_stop, min_len)


def tokenize_many(texts: list[str], batch_size: int = 2000):
    """
    Vectorised path using kiwi.tokenize's iterable input (releases the GIL,
    uses num_workers). Yields token lists in input order.
    """
    kiwi = _load_kiwi()
    stop = _load_stopwords()
    latin_stop = _load_latin_stopwords()
    keep_tags = set(config.KIWI_KEEP_TAGS)
    min_len = config.KO_MIN_TOKEN_LEN

    cleaned = [clean_common(t) for t in texts]
    for res in kiwi.tokenize(cleaned, normalize_coda=True):
        yield _filter_tokens(res, keep_tags, stop, latin_stop, min_len)
