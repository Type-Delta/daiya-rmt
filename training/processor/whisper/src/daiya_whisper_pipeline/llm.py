from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import base64
from collections import defaultdict
from collections.abc import Callable
import json

from openai import OpenAI
from tqdm import tqdm

from .concurrency import bounded_ordered_map
from .config import PipelineConfig
from .types import Chunk, LabeledChunk


SYSTEM_PROMPT = """You transcribe audio for a multilingual speech dataset.
Return only compact JSON with keys: text, language, notes, context.

Rules:
- Preserve the speaker's meaning and language switching.
- Transcribe the words actually spoken. Never paraphrase, summarize, or substitute synonyms.
- Do not expand abbreviations or complete clipped words — write exactly the form spoken, even if a longer form seems more correct.
- The preceding-chunk transcript is context only, not part of this chunk. Transcribe exactly what is audible in this chunk's audio — including words that happen to repeat the previous chunk's ending — but never fill gaps with the previous transcript's words.
- Produce a clean read: drop stutters, filler repetitions ("พอ พอ พอ" -> "พอ"), and abandoned false starts; when the speaker corrects themself mid-word or mid-sentence, keep only the corrected version.
- Collapse immediately-doubled hesitation words and phrases ("ที่ที่" -> "ที่", "ในใน" -> "ใน"). Repeated connector or function words are always hesitation — collapse to one occurrence. Repeated content words may be deliberate emphasis — keep those only when clearly intentional. Apply all cleanup rules in the text field itself — never just describe the issue in notes.
- Do not drop intentional repetition that carries meaning or emphasis.
- The audio may contain multiple speakers taking turns; transcribe all speech in order. Do not attribute or label speakers.
- Keep real English technical words in English instead of spelling them phonetically in Thai or Japanese.
- Normalize punctuation and spacing.
- Write Thai as continuous text with spaces only at natural phrase or sentence boundaries. Never separate individual Thai words with spaces.
- Do not translate anything, keep all text in the original language(s) spoken.
- Do not invent content that is not supported by the audio.
- If the speech is unintelligible, leave text empty and explain briefly in notes.
- Use the current source-file context only as a hint for ambiguous advanced terms, names, acronyms, tools, and topic direction.
- Return context as the full updated source-file context for later chunks.
- Keep context concise: source-specific terminology, names, acronyms, topic direction, and spelling hints.
- In context, maintain a line starting with "Terms:" for cross-chunk consistency. Admit a term ONLY if it is specific (product names, system/acronym names, people, project-specific jargon) AND you are completely certain you heard it correctly. Never add generic words, and never add anything you might have misheard — a wrong term poisons every later chunk. Keep the list short; drop entries that turn out to be wrong.
- Do not include transcript text verbatim in context unless it is a term or short phrase worth preserving.

Examples of the required transcript style:

1. Disfluent speech (heard): "แล้วก็ แล้วก็ เรา เราจะ จะไปที่ ไปที่ออฟฟิศ อ่ะไม่ใช่ ไปที่ไซต์งานก่อนครับ"
   Correct text: "แล้วก็เราจะไปที่ไซต์งานก่อนครับ"
   (stutters and the abandoned "ออฟฟิศ" false start removed; only the corrected destination kept)

2. Word-by-word spaced Thai is wrong:
   Wrong: "เรา ใช้ ระบบ CI CD บน GitHub Actions ครับ"
   Correct: "เราใช้ระบบ CI/CD บน GitHub Actions ครับ"
   (Thai written continuously; English technical terms kept in English)

3. Intentional repetition kept, with a Terms line in context:
   Heard: "ดีมาก ดีมาก อ่า Deploy ตัว Daiya ขึ้น staging ได้เลย"
   Correct text: "ดีมาก ดีมาก อ่า Deploy ตัว Daiya ขึ้น staging ได้เลย"
   Context: "Topic: deploying the Daiya project. Terms: Daiya, staging, deploy"
   ("ดีมาก ดีมาก" is deliberate emphasis, not a stutter, so it stays)
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

    def transcribe(
        self,
        chunk: Chunk,
        context: TranscriptionContext | None = None,
        previous_text: str = "",
    ) -> LabeledChunk:
        if not self.config.openrouter_api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for audio LLM transcription")

        context_text = (context.text if context else "").strip()
        previous_tail = previous_text.strip()[-600:]
        user_text = (
            "Transcribe this audio chunk for a mixed Thai-English or Japanese-English dataset. "
            "Return a clean transcript suitable for model training: accurate to the audio, but with "
            "stutters, filler repetitions, and abandoned false starts removed.\n\n"
            f"Transcript of the immediately preceding chunk (continuity hint, do not repeat it):\n"
            f"{previous_tail or '(start of file)'}\n\n"
            f"Current source-file context:\n{context_text or '(none yet)'}"
        )
        extra_body: dict[str, object] = {}
        if self.config.llm_reasoning_effort:
            extra_body["reasoning"] = {"effort": self.config.llm_reasoning_effort}
        response = self.client.chat.completions.create(
            model=self.config.openrouter_model,
            temperature=self.config.llm_temperature,
            # cost fuse: transcripts are ~100 tokens; caps runaway thinking on reasoning models
            max_tokens=2000,
            extra_body=extra_body,
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
        previous_text = labeled[-1].transcript_text if labeled else ""
        item = transcriber.transcribe(chunk, context, previous_text=previous_text)
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

    source_groups = [
        (source_path, sorted(source_chunks, key=lambda item: item.index))
        for source_path, source_chunks in sorted(by_source.items())
    ]
    labeled: list[LabeledChunk] = []
    max_in_flight = config.llm_max_in_flight
    max_workers = min(config.llm_max_workers, max_in_flight)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        with tqdm(total=len(chunks), desc="llm transcribe chunks") as progress:
            results = bounded_ordered_map(
                executor,
                lambda group: _transcribe_source_chunks(group[1], transcriber, config, progress.update),
                source_groups,
                max_in_flight,
            )
            for source_labeled in results:
                labeled.extend(source_labeled)
    return labeled
