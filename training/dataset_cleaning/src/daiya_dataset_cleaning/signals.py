"""Language-neutral baseline quality signals."""

from __future__ import annotations

from collections import Counter
import math
import unicodedata

from .models import Evidence, ReasonCode
from .normalize import normalize_text


def _script_name(char: str) -> str:
    codepoint = ord(char)
    if (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0x20000 <= codepoint <= 0x323AF
    ):
        return "han"
    name = unicodedata.name(char, "")
    return next(
        (
            script.lower()
            for script in ("THAI", "HIRAGANA", "KATAKANA", "HANGUL", "ARABIC", "HEBREW", "CYRILLIC", "LATIN")
            if script in name
        ),
        "other",
    )


def script_profile(text: object) -> dict[str, float]:
    """Return proportions of Unicode script families among letter characters."""
    counts: Counter[str] = Counter()
    for char in normalize_text(text):
        if not unicodedata.category(char).startswith("L"):
            continue
        counts[_script_name(char)] += 1
    total = sum(counts.values())
    return {key: count / total for key, count in sorted(counts.items())} if total else {}


def evaluate_signals(
    label: object,
    duration_seconds: object,
    *,
    expected_scripts: frozenset[str] | None = None,
    min_duration: float = 0.15,
    max_duration: float = 30.0,
    min_char_rate: float = 0.5,
    max_char_rate: float = 35.0,
) -> tuple[tuple[Evidence, ...], tuple[ReasonCode, ...]]:
    """Evaluate safe baselines. Missing/malformed inputs become reviewable signals."""
    text = normalize_text(label)
    evidence: list[Evidence] = [Evidence("normalized_length", len(text))]
    reasons: list[ReasonCode] = []
    if not text:
        reasons.append(ReasonCode.EMPTY_LABEL)
    try:
        duration = float(duration_seconds)
    except (TypeError, ValueError, OverflowError):
        duration = math.nan
    if not math.isfinite(duration) or duration <= 0:
        reasons.append(ReasonCode.INVALID_DURATION)
        evidence.append(Evidence("duration_seconds", None, detail="missing or invalid"))
    else:
        evidence.append(Evidence("duration_seconds", duration))
        if duration < min_duration:
            reasons.append(ReasonCode.TOO_SHORT)
        if duration > max_duration:
            reasons.append(ReasonCode.TOO_LONG)
        rate = len(text.replace(" ", "")) / duration
        evidence.append(Evidence("characters_per_second", rate))
        if text and rate < min_char_rate:
            reasons.append(ReasonCode.CHAR_RATE_LOW)
        if rate > max_char_rate:
            reasons.append(ReasonCode.CHAR_RATE_HIGH)
    profile = script_profile(text)
    evidence.extend(Evidence(f"script_fraction.{name}", value) for name, value in profile.items())
    if expected_scripts and profile and not any(profile.get(s.lower(), 0) > 0 for s in expected_scripts):
        reasons.append(ReasonCode.SCRIPT_MISMATCH)
    return tuple(evidence), tuple(dict.fromkeys(reasons))
