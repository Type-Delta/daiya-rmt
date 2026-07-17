from __future__ import annotations

from pathlib import Path
import errno
from hashlib import sha256
import json
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

from .concurrency import bounded_ordered_map
from .config import PipelineConfig, ensure_dirs
from .types import LabeledChunk

_THAI_GAP = re.compile(r"(?<=[฀-๿]) +(?=[฀-๿])")


# Standard Thai connectors; 3+ immediate repeats of these are always hesitation, never emphasis.
_CONNECTORS = "ซึ่ง|ที่|ถ้า|ใน|เป็น|ต้อง|ก็|คือ|แล้ว|และ|แต่|ว่า|จะ|กับ|ของ"
_REPEAT3 = re.compile(rf"({_CONNECTORS})(?:\s*\1){{2,}}")


def collapse_hesitation_repeats(text: str) -> str:
    """Collapse 3+ back-to-back repeats of a connector word (LLM misses these ~half the time)."""
    return _REPEAT3.sub(r"\1", text)


def normalize_thai_spacing(text: str) -> str:
    """Collapse word-by-word spaced Thai the transcription LLM sometimes emits."""
    thai_chars = sum("฀" <= ch <= "๿" for ch in text)
    if thai_chars < 10:
        return text
    gaps = len(_THAI_GAP.findall(text))
    if gaps / thai_chars <= 0.12:
        return text
    return _THAI_GAP.sub("", text)

def _atomic_copy(source: Path, target: Path) -> None:
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite export file: {target}")

    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        shutil.copy2(source, temporary)
        # The staging directory is private to this invocation, so a collision
        # is an error rather than an invitation to reuse an existing file.
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def _audio_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _interval_rows(intervals: tuple[object, ...]) -> list[dict[str, float]]:
    # This is intentionally duck-typed so the row builder stays independent of
    # an audio library and remains easy to inspect in migration tooling.
    return [
        {"start": round(float(interval.start), 6), "end": round(float(interval.end), 6)}
        for interval in intervals
    ]


def _row(item: LabeledChunk, target_name: str, config: PipelineConfig) -> dict[str, object]:
    chunk = item.chunk
    training_eligible = bool(item.extra.get("training_eligible", chunk.training_eligible))
    eligibility_reason = str(item.extra.get("training_eligibility_reason") or chunk.eligibility_reason)
    alignment = item.extra.get("ownership_alignment")
    overlap_seconds = sum(interval.duration for interval in chunk.overlap_intervals)
    review_signals: list[str] = []
    if chunk.overlap_intervals:
        review_signals.append("overlapped_speech_detected")
    if chunk.has_labeling_preroll:
        review_signals.extend(("continuous_speech_fallback", "labeling_preroll_context"))
        if not training_eligible:
            review_signals.append("owned_target_alignment_required")
    if chunk.context_overlap_after_seconds:
        review_signals.append("legacy_adjacent_context_overlap")
    return {
        "file_name": target_name.replace("\\", "/"),
        config.text_column: collapse_hesitation_repeats(normalize_thai_spacing(item.transcript_text)),
        "language": item.language or config.language_hint,
        "context_before": item.extra.get("context_before", ""),
        "context_after": item.extra.get("context_after", ""),
        "source_file": str(chunk.source.source_path),
        "source_id": chunk.source.source_id,
        # Six decimal places preserve exact normalized-audio timestamp intent;
        # no timestamp is reconstructed from a concatenated clip.
        "source_start": round(chunk.start, 6),
        "source_end": round(chunk.end, 6),
        # Explicit ownership schema. ``source_*`` remains the owned alias for
        # legacy tools, never the longer labeling input.
        "owned_source_start": round(chunk.start, 6),
        "owned_source_end": round(chunk.end, 6),
        "labeling_audio_source_start": round(chunk.labeling_start, 6),
        "labeling_audio_source_end": round(chunk.labeling_end, 6),
        "target_offset_seconds": round(chunk.target_offset_seconds, 6),
        "window_duration": round(chunk.duration, 6),
        "labeling_audio_duration": round(chunk.labeling_duration, 6),
        "speech_duration": round(chunk.speech_duration, 6),
        "audio_sha256": _audio_sha256(chunk.chunk_path),
        "segmentation": {
            "version": chunk.segmentation_version or "legacy-unknown",
            "config_id": chunk.segmentation_config_id or "legacy-unknown",
            "window_is_contiguous": len(chunk.intervals) == 1,
            "vad_speech_intervals": _interval_rows(chunk.speech_intervals),
            "overlap_intervals": _interval_rows(chunk.overlap_intervals),
            "overlap_seconds": round(overlap_seconds, 6),
            "context_overlap_before_seconds": round(chunk.context_overlap_before_seconds, 6),
            "context_overlap_after_seconds": round(chunk.context_overlap_after_seconds, 6),
            "ownership": {
                "owned_source_start": round(chunk.start, 6),
                "owned_source_end": round(chunk.end, 6),
                "labeling_audio_source_start": round(chunk.labeling_start, 6),
                "labeling_audio_source_end": round(chunk.labeling_end, 6),
                "target_offset_seconds": round(chunk.target_offset_seconds, 6),
                "training_artifact": "owned_audio_crop",
            },
            "boundary": {
                "method": chunk.boundary_method,
                "confidence": round(chunk.boundary_confidence, 6),
                "evidence": chunk.boundary_evidence,
            },
            "timestamp_evidence": chunk.evidence_provenance,
            "label_alignment": alignment,
            "training_eligible": training_eligible,
            "training_eligibility_reason": eligibility_reason,
        },
        "overlap_detected": bool(chunk.overlap_intervals),
        "review_signals": review_signals,
        # Context-fallback rows are visible to labelers but must be resolved
        # before a trainer treats their full-window label as a target.
        "training_eligible": training_eligible,
        "training_eligibility_reason": eligibility_reason,
        "ownership_alignment": alignment,
        "notes": item.notes,
    }


def _publish_no_clobber(staging_dir: Path, output_dir: Path) -> None:
    """Atomically publish a directory, refusing a destination collision."""
    if os.path.lexists(output_dir):
        raise FileExistsError(f"Export target appeared during publication: {output_dir}")

    # Windows os.rename already has no-replace semantics.  Linux has the same
    # guarantee through renameat2(RENAME_NOREPLACE); using it closes the final
    # check/replace race with processes that do not use our lock directory.
    if os.name == "nt":
        os.rename(staging_dir, output_dir)
        return

    if sys.platform == "linux":
        try:
            import ctypes

            libc = ctypes.CDLL(None, use_errno=True)
            renameat2 = libc.renameat2
        except (AttributeError, OSError):
            renameat2 = None
        if renameat2 is not None:
            renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            renameat2.restype = ctypes.c_int
            result = renameat2(
                -100, os.fsencode(staging_dir), -100, os.fsencode(output_dir), 1
            )
            if result == 0:
                return
            error_number = ctypes.get_errno()
            if error_number == errno.EEXIST:
                raise FileExistsError(f"Export target appeared during publication: {output_dir}")
            if error_number not in {errno.EINVAL, errno.ENOSYS, errno.ENOTSUP}:
                raise OSError(error_number, os.strerror(error_number), output_dir)

    # The publication lock still prevents races between all cooperating
    # exporters on platforms without an atomic no-replace directory move.
    os.rename(staging_dir, output_dir)


def export_audiofolder(labeled_chunks: list[LabeledChunk], config: PipelineConfig) -> Path:
    output_dir = config.output_dir
    output_parent = output_dir.parent
    ensure_dirs([output_parent])

    # The lock closes the check-then-publish window between separate pipeline
    # processes that use this exporter.  A stale lock is intentionally an
    # error: silently taking it could let two runs publish to the same target.
    lock_dir = output_parent / f".{output_dir.name}.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise FileExistsError(f"Another export is publishing {output_dir}") from error

    staging_dir: Path | None = None
    try:
        # Never inspect or reuse files in a pre-existing target.  The final
        # directory is published only once, after the complete dataset exists
        # in this unique sibling staging directory.
        if os.path.lexists(output_dir):
            raise FileExistsError(
                f"Export target already exists; choose a fresh output directory: {output_dir}"
            )

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_parent))
        split_dir = staging_dir / config.dataset_split
        split_dir.mkdir()
        metadata_path = staging_dir / "metadata.jsonl"

        entries = []
        seen_targets: set[str] = set()
        ordered_chunks = sorted(
            labeled_chunks,
            key=lambda item: (str(item.chunk.source.source_path), item.chunk.index),
        )
        for item in ordered_chunks:
            target_name = f"{config.dataset_split}/{item.chunk.source.source_id}_{item.chunk.index:05d}.wav"
            if target_name in seen_targets:
                raise ValueError(f"Duplicate export destination: {target_name}")
            seen_targets.add(target_name)
            entries.append((item, target_name, staging_dir / target_name))

        temporary_metadata = staging_dir / ".metadata.jsonl.tmp"
        max_in_flight = config.export_max_in_flight
        max_workers = min(config.export_max_workers, max_in_flight)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            copies = bounded_ordered_map(
                executor,
                lambda entry: _atomic_copy(entry[0].chunk.chunk_path, entry[2]),
                entries,
                max_in_flight,
            )
            with temporary_metadata.open("w", encoding="utf-8") as handle:
                for entry, _ in zip(entries, copies):
                    handle.write(json.dumps(_row(entry[0], entry[1], config), ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary_metadata, metadata_path)

        _publish_no_clobber(staging_dir, output_dir)
        published_metadata = output_dir / "metadata.jsonl"
        staging_dir = None
        return published_metadata
    finally:
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass
