from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Protocol


_WHITESPACE = re.compile(r"\s+")
_FUTURE_CONTEXT_FIELDS = {"context_after", "future_context", "right_context"}
_CONTEXT_FIELDS = {"context", "context_before", "rolling_context", *_FUTURE_CONTEXT_FIELDS}


class TokenizerLike(Protocol):
    bos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool = True) -> Any: ...

    def convert_tokens_to_ids(self, token: str) -> int | None: ...


@dataclass(frozen=True)
class PromptConditioningConfig:
    enabled: bool = False
    max_prompt_tokens: int = 64
    fields: tuple[str, ...] = ("context_before",)
    terms_only: bool = True
    allow_future_context: bool = False


@dataclass(frozen=True)
class PromptedLabels:
    labels: list[int]
    prompt_label_length: int


def parse_prompt_fields(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        fields = tuple(field.strip() for field in value.split(",") if field.strip())
    else:
        fields = tuple(str(field).strip() for field in value if str(field).strip())
    return fields or ("context_before",)


def validate_prompt_config(config: PromptConditioningConfig) -> None:
    if not config.enabled:
        return
    if config.max_prompt_tokens < 0:
        raise ValueError("prompt max tokens must be >= 0")
    future_fields = set(config.fields) & _FUTURE_CONTEXT_FIELDS
    if future_fields and not config.allow_future_context:
        names = ", ".join(sorted(future_fields))
        raise ValueError(
            f"Prompt fields include future context ({names}); pass --prompt-allow-future-context "
            "only for offline-labeling experiments."
        )


def build_prompt_text(example: dict[str, Any], config: PromptConditioningConfig) -> str:
    if not config.enabled:
        return ""

    validate_prompt_config(config)
    fragments: list[str] = []
    seen: set[str] = set()
    for field in config.fields:
        raw_value = example.get(field)
        if raw_value is None:
            continue
        raw_text = str(raw_value).strip()
        if not raw_text:
            continue
        fragment = _prompt_fragment(field, raw_text, config.terms_only)
        if not fragment or fragment in seen:
            continue
        seen.add(fragment)
        fragments.append(fragment)

    return "\n".join(fragments)


def encode_prompt_token_ids(tokenizer: TokenizerLike, prompt_text: str, max_prompt_tokens: int) -> list[int]:
    prompt_text = _normalize_text(prompt_text)
    if not prompt_text or max_prompt_tokens <= 0:
        return []

    body_ids = list(tokenizer(prompt_text, add_special_tokens=False).input_ids)
    body_ids = body_ids[:max_prompt_tokens]
    if not body_ids:
        return []

    start_of_previous = _start_of_previous_token_id(tokenizer)
    if start_of_previous is None:
        raise ValueError("Tokenizer does not define Whisper <|startofprev|> prompt token.")
    return [start_of_previous, *body_ids]


def build_prompted_labels(
    transcript_label_ids: list[int],
    prompt_token_ids: list[int],
    max_label_length: int,
    bos_token_id: int | None,
) -> PromptedLabels:
    transcript_label_ids = list(transcript_label_ids)
    if max_label_length <= 0:
        return PromptedLabels(labels=transcript_label_ids, prompt_label_length=0)

    prompt_budget = max(0, max_label_length - len(transcript_label_ids))
    bounded_prompt_ids = list(prompt_token_ids[:prompt_budget])
    labels = [*bounded_prompt_ids, *transcript_label_ids]

    mask_length = len(bounded_prompt_ids)
    if transcript_label_ids and transcript_label_ids[0] == bos_token_id:
        mask_length += 1

    return PromptedLabels(
        labels=labels,
        prompt_label_length=min(mask_length, len(labels)),
    )


def mask_prompt_labels(label_ids: list[int], prompt_label_length: int, ignore_index: int = -100) -> list[int]:
    labels = list(label_ids)
    for index in range(min(prompt_label_length, len(labels))):
        labels[index] = ignore_index
    return labels


def build_decoder_inputs_and_labels(
    label_ids: list[int],
    prompt_label_length: int,
    pad_token_id: int,
    ignore_index: int = -100,
) -> tuple[list[int], list[int]]:
    if not label_ids:
        return [], []

    decoder_input_ids = [*label_ids[:-1], pad_token_id]
    labels = [*label_ids[1:], ignore_index]
    return decoder_input_ids, mask_prompt_labels(labels, max(0, prompt_label_length - 1), ignore_index)


def _prompt_fragment(field: str, value: str, terms_only: bool) -> str:
    if terms_only and field in _CONTEXT_FIELDS:
        return _extract_terms(value)
    value = _normalize_text(value)
    if field in _CONTEXT_FIELDS:
        return value
    if field in {"previous_text", "previous_transcript", "previous_chunk_text"}:
        return f"Previous: {value}"
    if field == "notes":
        return f"Notes: {value}"
    return value


def _extract_terms(value: str) -> str:
    terms: list[str] = []
    for line in value.splitlines():
        line = line.strip()
        if not line.lower().startswith("terms:"):
            continue
        term_line = _normalize_text(line)
        if term_line and term_line not in terms:
            terms.append(term_line)
    return "\n".join(terms)


def _normalize_text(value: str) -> str:
    return _WHITESPACE.sub(" ", value).strip()


def _start_of_previous_token_id(tokenizer: TokenizerLike) -> int | None:
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if convert is None:
        return None
    token_id = convert("<|startofprev|>")
    return token_id if isinstance(token_id, int) and token_id >= 0 else None
