from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
import base64
from collections import defaultdict
from collections.abc import Callable
import json

from openai import OpenAI
from tqdm import tqdm

from .config import PipelineConfig
from .types import Chunk, LabeledChunk


SYSTEM_PROMPT = """You transcribe audio for a multilingual speech dataset.
Return only compact JSON with keys: text, language, notes, context.

Rules:
- Preserve the speaker's meaning and language switching.
- Keep real English technical words in English instead of spelling them phonetically in Thai or Japanese.
- Normalize punctuation and spacing.
- Do not translate Thai, Japanese, or English.
- Do not invent content that is not supported by the audio.
- If the speech is unintelligible, leave text empty and explain briefly in notes.
- Use the current source-file context only as a hint for ambiguous advanced terms, names, acronyms, tools, and topic direction.
- Return context as the full updated source-file context for later chunks.
- Keep context concise: source-specific terminology, names, acronyms, topic direction, and spelling hints.
- Do not include transcript text verbatim in context unless it is a term or short phrase worth preserving.
"""


@dataclass(frozen=True)
class TranscriptionContext:
    text: str = ""

    def trim(self, max_chars: int) -> "TranscriptionContext":
        if max_chars <= 0 or len(self.text) <= max_chars:
            return self
        return TranscriptionContext(self.text[-max_chars:].lstrip())


def _audio_part(path: Path, audio_format: str) -> dict[str, object]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "input_audio",
        "input_audio": {
            "data": data,
            "format": audio_format,
        },
    }


class OpenRouterAudioTranscriber:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.client = OpenAI(
            api_key=config.openrouter_api_key,
            base_url=config.openrouter_base_url,
            timeout=config.llm_timeout_seconds,
            default_headers={
                "HTTP-Referer": config.openrouter_site_url,
                "X-Title": config.openrouter_app_name,
            },
        )

    def transcribe(self, chunk: Chunk, context: TranscriptionContext | None = None) -> LabeledChunk:
        if not self.config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for audio LLM transcription")

        context_text = (context.text if context else "").strip()
        user_text = (
            "Transcribe this audio chunk for a mixed Thai-English or Japanese-English dataset. "
            "Return the best verbatim transcript suitable for model training.\n\n"
            f"Current source-file context:\n{context_text or '(none yet)'}"
        )
        response = self.client.chat.completions.create(
            model=self.config.openrouter_model,
            temperature=self.config.llm_temperature,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        _audio_part(chunk.chunk_path, self.config.llm_audio_format),
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or "{}"
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = {"text": content.strip(), "language": self.config.language_hint, "notes": "non_json_llm_response"}

        text = str(parsed.get("text") or "").strip()
        language = str(parsed.get("language") or self.config.language_hint).strip()
        notes = str(parsed.get("notes") or "").strip()
        next_context = str(parsed.get("context") or "").strip()
        if not next_context:
            next_context = str(parsed.get("context_update") or "").strip()
        return LabeledChunk(
            chunk=chunk,
            transcript_text=text,
            language=language,
            notes=notes,
            extra={
                "context_before": context_text,
                "context_after": next_context,
            },
        )


def _next_context(current: TranscriptionContext, labeled: LabeledChunk, max_chars: int) -> TranscriptionContext:
    next_context = str(labeled.extra.get("context_after") or "").strip()
    if not next_context:
        return current
    return TranscriptionContext(next_context).trim(max_chars)


def _transcribe_source_chunks(
    chunks: list[Chunk],
    transcriber: OpenRouterAudioTranscriber,
    config: PipelineConfig,
    on_chunk_transcribed: Callable[[], None] | None = None,
) -> list[LabeledChunk]:
    context = TranscriptionContext()
    labeled: list[LabeledChunk] = []
    for chunk in sorted(chunks, key=lambda item: item.index):
        item = transcriber.transcribe(chunk, context)
        context = _next_context(context, item, config.llm_context_max_chars)
        item.extra["context_after"] = context.text
        labeled.append(item)
        if on_chunk_transcribed:
            on_chunk_transcribed()
    return labeled


def transcribe_chunks(
    chunks: list[Chunk],
    transcriber: OpenRouterAudioTranscriber,
    config: PipelineConfig,
) -> list[LabeledChunk]:
    if not chunks:
        return []

    by_source: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        by_source[str(chunk.source.source_path)].append(chunk)

    labeled: list[LabeledChunk] = []
    with ThreadPoolExecutor(max_workers=config.llm_max_workers) as executor:
        with tqdm(total=len(chunks), desc="llm transcribe chunks") as progress:
            futures = [
                executor.submit(_transcribe_source_chunks, source_chunks, transcriber, config, progress.update)
                for source_chunks in by_source.values()
            ]
            for future in as_completed(futures):
                labeled.extend(future.result())
    return sorted(labeled, key=lambda item: (str(item.chunk.source.source_path), item.chunk.index))
