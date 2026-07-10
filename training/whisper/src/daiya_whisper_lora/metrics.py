from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Any


TECHNICAL_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9_+#.-]*")
THAI_GAP_RE = re.compile(r"(?<=[\u0e00-\u0e7f]) +(?=[\u0e00-\u0e7f])")
COMMON_ENGLISH_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "but",
    "by",
    "context",
    "current",
    "do",
    "does",
    "first",
    "for",
    "from",
    "hint",
    "in",
    "into",
    "is",
    "it",
    "label",
    "labels",
    "latest",
    "memory",
    "metadata",
    "note",
    "notes",
    "of",
    "only",
    "on",
    "or",
    "previous",
    "prompt",
    "recent",
    "repeat",
    "said",
    "spoken",
    "static",
    "successful",
    "term",
    "terms",
    "text",
    "that",
    "the",
    "this",
    "to",
    "topic",
    "transcript",
    "unless",
    "use",
    "with",
    "word",
    "words",
    "you",
}


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_thai_spacing(text: str) -> str:
    thai_chars = sum("\u0e00" <= char <= "\u0e7f" for char in text)
    if thai_chars < 10:
        return text
    gaps = len(THAI_GAP_RE.findall(text))
    if gaps / thai_chars <= 0.12:
        return text
    return THAI_GAP_RE.sub("", text)


def normalize_no_space(text: str) -> str:
    return re.sub(r"\s+", "", normalize_text(text))


def is_combining_or_tone(char: str) -> bool:
    return unicodedata.category(char).startswith("M")


def is_thai(char: str) -> bool:
    return "\u0e00" <= char <= "\u0e7f"


def is_basic_latin(char: str) -> bool:
    return "A" <= char <= "Z" or "a" <= char <= "z"


def is_latin_or_digit(char: str) -> bool:
    return is_basic_latin(char) or char.isdigit()


def is_punctuation_or_symbol(char: str) -> bool:
    return unicodedata.category(char)[0] in {"P", "S"}


def token_units(text: str) -> list[str]:
    normalized = normalize_text(text)
    tokens: list[str] = []
    index = 0
    while index < len(normalized):
        char = normalized[index]
        if char.isspace() or is_punctuation_or_symbol(char):
            index += 1
            continue

        if is_latin_or_digit(char):
            start = index
            while index < len(normalized) and is_latin_or_digit(normalized[index]):
                index += 1
            tokens.append(normalized[start:index])
            continue

        if is_thai(char):
            unit = [char]
            index += 1
            while index < len(normalized) and is_thai(normalized[index]) and is_combining_or_tone(
                normalized[index]
            ):
                unit.append(normalized[index])
                index += 1
            tokens.append("".join(unit))
            continue

        if is_combining_or_tone(char) and tokens:
            tokens[-1] += char
        else:
            tokens.append(char)
        index += 1
    return tokens


def levenshtein(reference: list[str] | str, hypothesis: list[str] | str) -> int:
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for row_index, ref_item in enumerate(reference, start=1):
        current = [row_index]
        for col_index, hyp_item in enumerate(hypothesis, start=1):
            insertion = current[col_index - 1] + 1
            deletion = previous[col_index] + 1
            substitution = previous[col_index - 1] + (ref_item != hyp_item)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def error_rate(distance: int, reference_length: int) -> float | None:
    if reference_length == 0:
        return None
    return distance / reference_length


def text_metrics(reference: str, hypothesis: str) -> dict[str, Any]:
    ref_norm = normalize_text(reference)
    hyp_norm = normalize_text(hypothesis)
    ref_no_space = normalize_no_space(reference)
    hyp_no_space = normalize_no_space(hypothesis)
    ref_tokens = token_units(reference)
    hyp_tokens = token_units(hypothesis)

    cer_distance = levenshtein(ref_norm, hyp_norm)
    cer_no_space_distance = levenshtein(ref_no_space, hyp_no_space)
    wer_like_distance = levenshtein(ref_tokens, hyp_tokens)

    return {
        "cer_distance": cer_distance,
        "cer_reference_length": len(ref_norm),
        "cer": error_rate(cer_distance, len(ref_norm)),
        "cer_no_space_distance": cer_no_space_distance,
        "cer_no_space_reference_length": len(ref_no_space),
        "cer_no_space": error_rate(cer_no_space_distance, len(ref_no_space)),
        "wer_like_distance": wer_like_distance,
        "wer_like_reference_length": len(ref_tokens),
        "wer_like": error_rate(wer_like_distance, len(ref_tokens)),
    }


def nan_safe_mean(values: list[float | None]) -> float | None:
    real_values = [value for value in values if value is not None and math.isfinite(value)]
    if not real_values:
        return None
    return sum(real_values) / len(real_values)


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate_distance = sum(row["metrics"]["cer_distance"] for row in rows)
    aggregate_length = sum(row["metrics"]["cer_reference_length"] for row in rows)
    aggregate_no_space_distance = sum(row["metrics"]["cer_no_space_distance"] for row in rows)
    aggregate_no_space_length = sum(row["metrics"]["cer_no_space_reference_length"] for row in rows)
    aggregate_word_distance = sum(row["metrics"]["wer_like_distance"] for row in rows)
    aggregate_word_length = sum(row["metrics"]["wer_like_reference_length"] for row in rows)
    return {
        "count": len(rows),
        "mean_cer": nan_safe_mean([row["metrics"]["cer"] for row in rows]),
        "micro_cer": error_rate(aggregate_distance, aggregate_length),
        "mean_cer_no_space": nan_safe_mean([row["metrics"]["cer_no_space"] for row in rows]),
        "micro_cer_no_space": error_rate(aggregate_no_space_distance, aggregate_no_space_length),
        "mean_wer_like": nan_safe_mean([row["metrics"]["wer_like"] for row in rows]),
        "micro_wer_like": error_rate(aggregate_word_distance, aggregate_word_length),
    }


def english_terms_from_text(*texts: str) -> list[str]:
    terms: dict[str, str] = {}
    for text in texts:
        for match in TECHNICAL_TERM_RE.finditer(unicodedata.normalize("NFKC", text)):
            term = match.group(0).strip("._-")
            if is_useful_english_term(term):
                terms.setdefault(term.lower(), term)
    return sorted(terms.values(), key=lambda value: value.lower())


def is_useful_english_term(term: str) -> bool:
    normalized = term.lower()
    if len(term) < 2 or normalized in COMMON_ENGLISH_WORDS:
        return False
    if normalized.isdigit():
        return False
    if len(term) == 2 and not term.isupper():
        return False
    if "." in term or "_" in term or "+" in term or "#" in term:
        return True
    if term.isupper() and len(term) >= 2:
        return True
    return len(term) >= 4


def has_english_terms(row: dict[str, Any]) -> bool:
    text_parts = [str(row.get("text", ""))]
    text_parts.extend(str(row.get(key, "")) for key in ("context_before", "context_after", "notes"))
    return bool(english_terms_from_text(*text_parts))


def is_short_utterance(row: dict[str, Any], threshold_seconds: float) -> bool:
    if row.get("contains_short_utterance"):
        return True
    duration = first_number(row, "speech_duration", "audio_duration_seconds", "duration")
    return isinstance(duration, int | float) and duration <= threshold_seconds


def first_number(row: dict[str, Any], *keys: str) -> int | float | None:
    for key in keys:
        value = row.get(key)
        if isinstance(value, int | float):
            return value
    return None


def summarize_scored_rows(rows: list[dict[str, Any]], short_threshold_seconds: float) -> dict[str, Any]:
    scored = [row for row in rows if row.get("status") == "ok"]
    short_rows = [row for row in scored if is_short_utterance(row, short_threshold_seconds)]
    term_rows = [row for row in scored if row.get("english_terms")]
    return {
        "overall": aggregate_metric_rows(scored),
        "short_utterance_subset": aggregate_metric_rows(short_rows),
        "english_technical_term_subset": aggregate_metric_rows(term_rows),
    }


def count_probe_tags(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(row.get("probe_tags", []))
    return dict(sorted(counter.items()))
