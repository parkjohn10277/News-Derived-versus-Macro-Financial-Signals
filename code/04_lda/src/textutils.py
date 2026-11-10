"""
Language identification, text-field coalescing, light cleaning, and the
domain keyword filter.

Language ID is purely character-based (project spec): we count Hangul syllables
/ jamo vs. Latin letters and threshold the ratio. No external language-detection
model is used.
"""
from __future__ import annotations

import re
from typing import Iterable

import pandas as pd

import config


def parse_dates(s: "pd.Series") -> "pd.Series":
    """
    Robust date parsing for the mixed corpus. Tries the canonical YYYY-MM-DD
    first, then falls back to pandas' mixed parser (handles GDELT-style
    YYYYMMDD and any stray time components). Unparseable -> NaT.
    """
    out = pd.to_datetime(s, errors="coerce", format="%Y-%m-%d")
    bad = out.isna() & s.astype(str).str.strip().ne("")
    if bad.any():
        retry = pd.to_datetime(s[bad], errors="coerce", format="mixed")
        out.loc[bad] = retry
    return out

# --------------------------------------------------------------------------- #
# Character classes
# --------------------------------------------------------------------------- #
# Hangul syllables + conjoining/compat jamo blocks.
_HANGUL_RE = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏ꥠ-꥿ힰ-퟿]")
_LATIN_RE = re.compile(r"[A-Za-z]")

# URLs, e-mails, HTML entities, bracketed tags like [사진], (연합뉴스)
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_RE = re.compile(r"&[a-z]+;|<[^>]+>")
_MULTIWS_RE = re.compile(r"\s+")
# Korean report scaffolding seen in MCST/NAVER bodies.
_KO_SCAFFOLD_RE = re.compile(r"(제목\s*:|주요\s*내용|ᄋ|□|◦|▲|※|◇)")


def coalesce_text(row: dict, fields: Iterable[str] = None) -> str:
    """Join the non-empty text fields of a record, in the configured order."""
    fields = fields or config.TEXT_FIELDS
    parts = []
    for f in fields:
        v = row.get(f)
        if v:
            v = str(v).strip()
            if v:
                parts.append(v)
    return " ".join(parts)


def clean_common(text: str) -> str:
    """Cheap normalisation applied to both languages before tokenisation."""
    if not text:
        return ""
    text = _URL_RE.sub(" ", text)
    text = _EMAIL_RE.sub(" ", text)
    text = _HTML_RE.sub(" ", text)
    text = _KO_SCAFFOLD_RE.sub(" ", text)
    text = _MULTIWS_RE.sub(" ", text)
    return text.strip()


def script_counts(text: str) -> tuple[int, int]:
    """(hangul_chars, latin_chars) in *text*."""
    return len(_HANGUL_RE.findall(text)), len(_LATIN_RE.findall(text))


def detect_lang(text: str,
                min_ratio: float = None,
                min_scripted: int = None,
                en_max_ratio: float = None) -> str:
    """
    Return 'ko', 'en', or 'unknown' from character composition alone.

    'ko'  if hangul / (hangul + latin) >= min_ratio
    'en'  if hangul / (hangul + latin) <= en_max_ratio
    'unknown' if the ratio falls between the two bars (mixed-language article)
              or the text has too few scripted characters to judge
    """
    min_ratio = config.LANG_KO_MIN_HANGUL_RATIO if min_ratio is None else min_ratio
    en_max_ratio = config.LANG_EN_MAX_HANGUL_RATIO if en_max_ratio is None else en_max_ratio
    min_scripted = config.LANG_MIN_SCRIPTED_CHARS if min_scripted is None else min_scripted

    h, l = script_counts(text)
    if h + l < min_scripted:
        return "unknown"
    ratio = h / (h + l)
    if ratio >= min_ratio:
        return "ko"
    if ratio <= en_max_ratio:
        return "en"
    return "unknown"


# --------------------------------------------------------------------------- #
# Domain keyword filter
# --------------------------------------------------------------------------- #
def _compile_terms(terms: Iterable[str]) -> re.Pattern:
    # Sort longest-first so overlapping terms still count once per occurrence;
    # word boundaries only make sense for the ASCII side.
    esc_ascii = []
    esc_other = []
    for t in terms:
        t = t.strip()
        if not t:
            continue
        if re.fullmatch(r"[\x00-\x7f ]+", t):
            esc_ascii.append(re.escape(t.strip()))
        else:
            esc_other.append(re.escape(t))
    pat = []
    if esc_ascii:
        pat.append(r"\b(?:" + "|".join(sorted(esc_ascii, key=len, reverse=True)) + r")\b")
    if esc_other:
        pat.append(r"(?:" + "|".join(sorted(esc_other, key=len, reverse=True)) + r")")
    return re.compile("|".join(pat), re.IGNORECASE) if pat else re.compile(r"(?!x)x")


_FILTER_RE = _compile_terms(list(config.FILTER_TERMS_KO) + list(config.FILTER_TERMS_EN))


def domain_hits(text: str) -> int:
    """Number of domain-term matches in *text* (0 => article is out of scope)."""
    if not text:
        return 0
    return len(_FILTER_RE.findall(text))


def passes_domain_filter(text: str) -> bool:
    if not config.FILTER_ENABLED:
        return True
    return domain_hits(text) >= config.FILTER_MATCH_MIN_HITS
