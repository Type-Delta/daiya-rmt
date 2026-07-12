"""Script-aware spelling validation with optional language-specific adapters.

Spelling is evidence for review only. This module never changes label text and
never turns a suggestion into a correction proposal.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Protocol, Sequence
import unicodedata

from .normalize import normalize_text
from .signals import _script_name


@dataclass(frozen=True, slots=True)
class ScriptSpan:
    text: str
    language: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class SpellingIssue:
    language: str
    checker: str
    unit: str
    suggestions: tuple[str, ...] = ()

    @property
    def unit_sha256(self) -> str:
        return hashlib.sha256(normalize_text(self.unit, casefold=True).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class SpanCheckResult:
    checked_units: int
    issues: tuple[SpellingIssue, ...] = ()


class SpanSpellChecker(Protocol):
    name: str
    language: str

    def check(self, text: str) -> SpanCheckResult: ...


@dataclass(frozen=True, slots=True)
class SpellingValidation:
    checked_units: int
    issues: tuple[SpellingIssue, ...]
    checker_names: tuple[str, ...]
    routed_span_counts: tuple[tuple[str, int], ...]
    checked_units_by_language: tuple[tuple[str, int], ...]

    @property
    def suspicious_units(self) -> int:
        return len(self.issues)

    @property
    def suspicious_ratio(self) -> float:
        return self.suspicious_units / self.checked_units if self.checked_units else 0.0

    def to_dict(self, *, include_issue_text: bool = False) -> dict[str, object]:
        issue_rows: list[dict[str, object]] = []
        for issue in self.issues:
            row: dict[str, object] = {
                "language": issue.language,
                "checker": issue.checker,
                "unit_sha256": issue.unit_sha256,
                "suggestion_count": len(issue.suggestions),
            }
            if include_issue_text:
                row["unit"] = issue.unit
                row["suggestions"] = list(issue.suggestions)
            issue_rows.append(row)
        issue_counts: dict[str, int] = {}
        for issue in self.issues:
            issue_counts[issue.language] = issue_counts.get(issue.language, 0) + 1
        by_language = {
            language: {
                "checked_units": checked,
                "suspicious_units": issue_counts.get(language, 0),
                "suspicious_ratio": issue_counts.get(language, 0) / checked if checked else 0.0,
            }
            for language, checked in self.checked_units_by_language
        }
        return {
            "checked_units": self.checked_units,
            "suspicious_units": self.suspicious_units,
            "suspicious_ratio": self.suspicious_ratio,
            "checker_names": list(self.checker_names),
            "routed_span_counts": dict(self.routed_span_counts),
            "by_language": by_language,
            "issues": issue_rows,
        }


def _language_for_char(char: str, active: str | None) -> str | None:
    if char.isspace():
        return None
    script = _script_name(char)
    if script == "thai":
        return "thai"
    if script in {"han", "hiragana", "katakana"} or char == "々":
        return "japanese"
    if script == "latin":
        return "english"
    category = unicodedata.category(char)
    if active and (category.startswith("M") or char.isdigit() or char in {"'", "’", "-", "_"}):
        return active
    return None


def route_script_spans(text: object) -> tuple[ScriptSpan, ...]:
    """Split mixed text into Thai, Japanese, and English checker spans."""
    value = normalize_text(text)
    spans: list[ScriptSpan] = []
    start = 0
    active: str | None = None
    for index, char in enumerate(value):
        language = _language_for_char(char, active)
        if language == active and language is not None:
            continue
        if active is not None:
            spans.append(ScriptSpan(value[start:index], active, start, index))
        if language is None:
            active = None
            start = index + 1
        else:
            active = language
            start = index
    if active is not None:
        spans.append(ScriptSpan(value[start:], active, start, len(value)))
    return tuple(span for span in spans if span.text)


def validate_spelling(
    text: object,
    checkers: Sequence[SpanSpellChecker],
    *,
    allowlist: frozenset[str] = frozenset(),
) -> SpellingValidation:
    """Run each checker only on spans for its declared language."""
    normalized_allowlist = {normalize_text(item, casefold=True) for item in allowlist}
    spans = route_script_spans(text)
    span_counts: dict[str, int] = {}
    for span in spans:
        span_counts[span.language] = span_counts.get(span.language, 0) + 1
    checked = 0
    checked_by_language: dict[str, int] = {}
    issues: list[SpellingIssue] = []
    names: list[str] = []
    for checker in checkers:
        names.append(checker.name)
        for span in spans:
            if span.language != checker.language:
                continue
            result = checker.check(span.text)
            checked += result.checked_units
            checked_by_language[checker.language] = (
                checked_by_language.get(checker.language, 0) + result.checked_units
            )
            issues.extend(
                issue
                for issue in result.issues
                if normalize_text(issue.unit, casefold=True) not in normalized_allowlist
            )
    return SpellingValidation(
        checked_units=checked,
        issues=tuple(issues),
        checker_names=tuple(dict.fromkeys(names)),
        routed_span_counts=tuple(sorted(span_counts.items())),
        checked_units_by_language=tuple(sorted(checked_by_language.items())),
    )


class PyThaiNLPChecker:
    """Optional PyThaiNLP tokenizer + spelling-engine adapter."""

    language = "thai"

    def __init__(self, engine: str = "pn") -> None:
        try:
            from pythainlp.spell import correct, spell
            from pythainlp.tokenize import word_tokenize
        except ImportError as exc:
            raise RuntimeError("PyThaiNLP spelling validation requires the optional 'pythainlp' package") from exc
        self.engine = engine
        self.name = f"pythainlp-{engine}"
        self._correct = correct
        self._spell = spell
        self._tokenize = word_tokenize

    def check(self, text: str) -> SpanCheckResult:
        tokens = [
            token
            for token in self._tokenize(text, engine="newmm", keep_whitespace=False)
            if len(token) >= 2 and token.strip() and any(_script_name(char) == "thai" for char in token)
        ]
        issues: list[SpellingIssue] = []
        for token in tokens:
            corrected = str(self._correct(token, engine=self.engine))
            if normalize_text(corrected) == normalize_text(token):
                continue
            suggestions = tuple(str(item) for item in self._spell(token, engine=self.engine)[:5])
            issues.append(SpellingIssue("thai", self.name, token, suggestions or (corrected,)))
        return SpanCheckResult(len(tokens), tuple(issues))


class SudachiJapaneseChecker:
    """Optional Sudachi dictionary/OOV adapter for Japanese spans."""

    language = "japanese"

    def __init__(self, dictionary: str = "core") -> None:
        try:
            from sudachipy import Dictionary
        except ImportError as exc:
            raise RuntimeError("Japanese spelling validation requires 'sudachipy' and a Sudachi dictionary") from exc
        self.name = f"sudachi-{dictionary}"
        self._tokenizer = Dictionary(dict=dictionary).create()

    def check(self, text: str) -> SpanCheckResult:
        morphemes = [
            morpheme
            for morpheme in self._tokenizer.tokenize(text)
            if any(_language_for_char(char, None) == "japanese" for char in str(morpheme.surface()))
        ]
        issues: list[SpellingIssue] = []
        for morpheme in morphemes:
            surface = str(morpheme.surface())
            if not surface or not morpheme.is_oov():
                continue
            normalized = str(morpheme.normalized_form())
            suggestions = (normalized,) if normalized and normalized != surface else ()
            issues.append(SpellingIssue("japanese", self.name, surface, suggestions))
        return SpanCheckResult(len(morphemes), tuple(issues))


class SymSpellEnglishChecker:
    """Optional SymSpell adapter using an explicit English frequency dictionary."""

    language = "english"
    _TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9'’_-]*")

    def __init__(self, dictionary_path: str, *, max_edit_distance: int = 2) -> None:
        try:
            from symspellpy import SymSpell, Verbosity
        except ImportError as exc:
            raise RuntimeError("English spelling validation requires the optional 'symspellpy' package") from exc
        self.name = "symspell-en"
        self._verbosity = Verbosity
        self._max_edit_distance = max_edit_distance
        self._checker = SymSpell(max_dictionary_edit_distance=max_edit_distance, prefix_length=7)
        if not self._checker.load_dictionary(dictionary_path, term_index=0, count_index=1):
            raise ValueError(f"could not load SymSpell dictionary: {dictionary_path}")

    def check(self, text: str) -> SpanCheckResult:
        tokens = [
            token
            for token in self._TOKEN.findall(text)
            if len(token) > 1 and not (token.isupper() and len(token) <= 8)
        ]
        issues: list[SpellingIssue] = []
        for token in tokens:
            suggestions = self._checker.lookup(
                token.casefold(),
                self._verbosity.CLOSEST,
                max_edit_distance=self._max_edit_distance,
                include_unknown=False,
            )
            terms = tuple(str(item.term) for item in suggestions[:5])
            if terms and terms[0].casefold() == token.casefold():
                continue
            issues.append(SpellingIssue("english", self.name, token, terms))
        return SpanCheckResult(len(tokens), tuple(issues))
