from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import base64
from collections import defaultdict
from collections.abc import Callable
from difflib import SequenceMatcher
import json
import unicodedata

from openai import OpenAI
from tqdm import tqdm

from .concurrency import bounded_ordered_map
from .config import PipelineConfig
from .evidence import TimestampWord, words_in_interval
from .types import Chunk, Interval, LabeledChunk


SYSTEM_PROMPT = """You transcribe audio for a multilingual speech dataset.
Return only compact JSON with keys: text, language, notes, context.

Rules:
- Preserve the speaker's meaning and language switching.
- Transcribe the words actually spoken. Never paraphrase, summarize, or substitute synonyms.
- Do not expand abbreviations or complete clipped words — write exactly the form spoken, even if a longer form seems more correct.
- The preceding-chunk transcript is context only, not part of this chunk. Transcribe exactly what is audible in this chunk's target audio — including words that happen to repeat the previous chunk's ending — but never fill gaps with the previous transcript's words.
- Some inputs declare an opening PRE-ROLL CONTEXT interval. It is audible continuity help only. When declared, return only text spoken after the target offset; never return timing or transcribe words heard exclusively in that pre-roll.
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


def _compact_alignment_text(text: str) -> str:
    """Compare scripts without imposing language-specific word boundaries."""
    return "".join(
        character.casefold()
        for character in unicodedata.normalize("NFKC", text)
        if unicodedata.category(character)[0] in {"L", "M", "N"}
    )


def _alignment_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    ratio = SequenceMatcher(a=left, b=right, autojunk=False).ratio()
    shorter, longer = sorted((left, right), key=len)
    if shorter in longer:
        ratio = max(ratio, len(shorter) / len(longer))
    return ratio


def _context_prefix_overlap(label: str, context: str) -> int:
    """Return a context-tail prefix copied into a proposed target label."""
    limit = min(len(label), len(context))
    for size in range(limit, 1, -1):
        if label.startswith(context[-size:]):
            return size
    return 0


def ownership_alignment_gate(chunk: Chunk, text: str, config: PipelineConfig) -> dict[str, object]:
    """Conservatively validate a pre-roll target without rewriting its label.

    Faster-Whisper text is used only as an independent local-ASR consistency
    signal.  It never replaces ``text`` and a failed/unavailable comparison
    leaves the row review-only.
    """
    if not chunk.has_labeling_preroll:
        return {"status": "not_required", "method": "owned_audio", "eligible": chunk.training_eligible}
    words = [word for word in chunk.alignment_words if isinstance(word, TimestampWord)]
    # A pre-roll label is only safe when the local timestamp evidence can be
    # assigned wholly to its owned interval.  Check both ownership edges:
    # checking only the leading handoff would let a word that spills into the
    # following row be represented by this row's audio/target ambiguously.
    ownership_edges = (chunk.start, chunk.end)
    straddling = [
        word
        for word in words
        if any(word.start < ownership_edge < word.end for ownership_edge in ownership_edges)
    ]
    target_words = words_in_interval(words, Interval(chunk.start, chunk.end))
    context_words = words_in_interval(words, Interval(chunk.labeling_start, chunk.start))
    target = _compact_alignment_text("".join(word.text for word in target_words))
    context = _compact_alignment_text("".join(word.text for word in context_words))
    label = _compact_alignment_text(text)
    base = {
        "method": "local_asr_timestamp_consistency",
        "target_asr_word_count": len(target_words),
        "context_asr_word_count": len(context_words),
        "threshold": float(getattr(config, "label_alignment_min_similarity", 0.45)),
        "boundary_straddling_timestamp_count": len(straddling),
    }
    if straddling:
        # A word whose timestamp crosses a fixed ownership handoff cannot be
        # assigned to either audio target.  Even a label that otherwise aligns
        # well is ambiguous until a human resolves that audible boundary.
        return {
            **base,
            "status": "review_required",
            "eligible": False,
            "reason": "timestamp_straddles_ownership_boundary",
        }
    if not target:
        return {**base, "status": "review_required", "eligible": False, "reason": "no_owned_timestamp_evidence"}
    if not label:
        return {**base, "status": "review_required", "eligible": False, "reason": "empty_label_with_preroll"}
    target_similarity = _alignment_similarity(label, target)
    context_similarity = _alignment_similarity(label, context)
    copied_context_prefix = _context_prefix_overlap(label, context)
    threshold = float(base["threshold"])
    if target_similarity < threshold:
        return {
            **base,
            "status": "review_required",
            "eligible": False,
            "reason": "owned_target_alignment_below_threshold",
            "target_similarity": round(target_similarity, 4),
            "context_similarity": round(context_similarity, 4),
        }
    # Pre-roll rows are intentionally conservative: context text that is at
    # least as plausible as the target is ambiguous even when the full label
    # also contains the target.  A copied suffix of the pre-roll at the start
    # of the label is direct evidence of the forbidden context+target form.
    if context and (
        context_similarity >= max(threshold, target_similarity - 0.08)
        or copied_context_prefix >= max(2, min(6, len(context) // 3))
    ):
        return {
            **base,
            "status": "review_required",
            "eligible": False,
            "reason": "label_contains_or_matches_preroll_context",
            "target_similarity": round(target_similarity, 4),
            "context_similarity": round(context_similarity, 4),
            "copied_context_prefix_characters": copied_context_prefix,
        }
    return {
        **base,
        "status": "passed",
        "eligible": True,
        "target_similarity": round(target_similarity, 4),
        "context_similarity": round(context_similarity, 4),
        "copied_context_prefix_characters": copied_context_prefix,
    }


def apply_ownership_alignment_gate(item: LabeledChunk, config: PipelineConfig) -> LabeledChunk:
    gate = ownership_alignment_gate(item.chunk, item.transcript_text, config)
    item.extra["ownership_alignment"] = gate
    eligible = bool(item.chunk.training_eligible or gate.get("eligible"))
    # The only baseline-quarantined state is pre-roll; a passed local gate plus
    # the already-exported owned crop is an explicit resolution of that state.
    if item.chunk.has_labeling_preroll:
        eligible = bool(gate.get("eligible"))
    item.extra["training_eligible"] = eligible
    item.extra["training_eligibility_reason"] = (
        "owned_target_alignment_passed" if eligible and item.chunk.has_labeling_preroll else item.chunk.eligibility_reason
    )
    return item


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
        ownership_instruction = ""
        if chunk.has_labeling_preroll:
            ownership_instruction = (
                f"\n\nOWNERSHIP CONTRACT: the first {chunk.target_offset_seconds:.3f} seconds of the supplied "
                f"audio ({chunk.labeling_start:.3f}s–{chunk.start:.3f}s of the normalized source) are PRE-ROLL "
                "CONTEXT ONLY. Return text only for the owned target beginning at that offset. Do not repeat or "
                "complete words that are audible only before the offset."
            )
        user_text = (
            "Transcribe this audio chunk for a mixed Thai-English or Japanese-English dataset. "
            "Return a clean transcript suitable for model training: accurate to the audio, but with "
            "stutters, filler repetitions, and abandoned false starts removed.\n\n"
            f"Transcript of the immediately preceding chunk (continuity hint, do not repeat it):\n"
            f"{previous_tail or '(start of file)'}\n\n"
            f"Current source-file context:\n{context_text or '(none yet)'}{ownership_instruction}"
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
                        _audio_part(chunk.labeling_path, self.config.llm_audio_format),
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
        return apply_ownership_alignment_gate(LabeledChunk(
            chunk=chunk,
            transcript_text=text,
            language=language,
            notes=notes,
            extra={
                "context_before": context_text,
                "context_after": next_context,
            },
        ), self.config)


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
