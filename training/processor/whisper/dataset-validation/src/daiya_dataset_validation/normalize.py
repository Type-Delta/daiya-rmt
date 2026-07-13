"""Deterministic normalization and identity helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import unicodedata

_SPACE = re.compile(r"\s+")


def normalize_text(value: object, *, casefold: bool = False) -> str:
    """Normalize Unicode and whitespace without language-specific substitutions."""
    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value))
    text = _SPACE.sub(" ", text).strip()
    return text.casefold() if casefold else text


def content_identity(label: object, *, casefold: bool = True) -> str:
    normalized = normalize_text(label, casefold=casefold)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def source_identity(uri: str | Path, *, record_id: str | None = None) -> str:
    """Create identity from provenance text; does not open or mutate the source."""
    canonical = unicodedata.normalize("NFC", str(uri).replace("\\", "/"))
    payload = f"{canonical}\0{record_id or ''}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

