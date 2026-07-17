from __future__ import annotations

import gc
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from .metrics import (
    count_probe_tags,
    english_terms_from_text,
    has_english_terms,
    is_short_utterance,
    normalize_thai_spacing,
    summarize_scored_rows,
    text_metrics,
)
from .prompt_conditioning import (
    PromptConditioningConfig,
    build_prompt_text,
    encode_prompt_token_ids,
    parse_prompt_fields,
    validate_prompt_config,
)
from .provenance import dataset_identity
from .split_manifest import apply_split_manifest, file_sha256, group_id_for_row, sample_id_for_row


PRIMARY_METRICS = {
    "micro_cer",
    "mean_cer",
    "micro_cer_no_space",
    "mean_cer_no_space",
    "micro_wer_like",
    "mean_wer_like",
}
PROMPT_STRATEGIES = {"isolated", "rolling-initial-prompt"}
GENERATION_FAILURE_POLICIES = {"raise", "eval-loss"}


@dataclass(frozen=True)
class ProbeConfig:
    run_dir: Path
    base_model: str
    dataset_dir: Path
    output_dir: Path
    candidates: tuple[Path, ...] = ()
    include_checkpoints: bool = True
    include_final: bool = True
    split: str | None = None
    split_manifest: Path | None = None
    selector_manifest: Path | None = None
    max_samples: int = 32
    min_short_samples: int = 8
    min_technical_term_samples: int = 8
    short_utterance_seconds: float = 3.0
    primary_metric: str = "micro_cer"
    generation_failure_policy: str = "raise"
    device: str = "auto"
    fp16: bool = False
    bf16: bool = False
    load_in_4bit: bool = False
    task: str = "transcribe"
    language: str | None = None
    language_policy: str = "metadata"
    generation_max_length: int = 225
    prompt_strategy: str = "isolated"
    prompt_max_tokens: int = 64
    prompt_fields: tuple[str, ...] = ("context_before",)
    prompt_terms_only: bool = True
    prompt_allow_future_context: bool = False
    prompt_include_row_context: bool = True
    rolling_prompt_turns: int = 3
    rolling_prompt_chars: int = 512


@dataclass(frozen=True)
class ProbeRows:
    rows: list[dict[str, Any]]
    dataset_identity: dict[str, Any]
    selector_identity: dict[str, Any] | None = None


def run_probe(config: ProbeConfig) -> dict[str, Any]:
    validate_probe_config(config)
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    candidates = discover_candidates(config)
    if not candidates:
        raise FileNotFoundError(f"No LoRA adapter candidates found under {config.run_dir}")

    probe_rows = load_probe_rows(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"{utc_stamp()}_{safe_name(config.run_dir.name)}"
    details_path = config.output_dir / f"details_{run_id}.jsonl"
    summary_path = config.output_dir / f"summary_{run_id}.json"
    generation_fingerprint_value = generation_fingerprint(config)
    candidate_fingerprints = {str(candidate.resolve()): adapter_fingerprint(candidate) for candidate in candidates}
    expected_freshness = {
        "dataset_fingerprint": dataset_fingerprint(probe_rows.dataset_identity),
        "generation_fingerprint": generation_fingerprint_value,
        "candidate_fingerprints": candidate_fingerprints,
    }

    all_details: list[dict[str, Any]] = []
    candidate_summaries: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"Probing {candidate.name} on {len(probe_rows.rows)} samples...")
        candidate_path = str(candidate.resolve())
        freshness = {
            "candidate_fingerprint": candidate_fingerprints[candidate_path],
            "dataset_fingerprint": expected_freshness["dataset_fingerprint"],
            "generation_fingerprint": generation_fingerprint_value,
        }
        try:
            details = score_candidate(candidate, probe_rows.rows, config)
        except Exception as exc:
            details = [
                {
                    "candidate": candidate.name,
                    "candidate_path": candidate_path,
                    "status": "candidate_failed",
                    "phase": "candidate_setup",
                    "error": repr(exc),
                }
            ]
        finally:
            release_model_memory()
        for detail in details:
            detail["freshness"] = freshness
        all_details.extend(details)
        scored_summary = summarize_scored_rows(details, config.short_utterance_seconds)
        status_counts = Counter(str(detail.get("status", "unknown")) for detail in details)
        candidate_summaries.append(
            {
                "name": candidate.name,
                "path": candidate_path,
                "attempted_count": len(details),
                "required_count": len(probe_rows.rows),
                "scored_count": scored_summary["overall"]["count"],
                "failed_count": len(details) - scored_summary["overall"]["count"],
                "status_counts": dict(sorted(status_counts.items())),
                "freshness": freshness,
                "eval_loss": eval_loss_for_candidate(config.run_dir, candidate),
                **scored_summary,
            }
        )

    final_summary = next((item for item in candidate_summaries if item["path"] == str(config.run_dir.resolve())), None)
    summary = {
        "status": "pending_selection",
        "selection_mode": "generation_gate",
        "run_id": run_id,
        "started_at": started_at,
        "elapsed_seconds": time.perf_counter() - started,
        "run_dir": str(config.run_dir.resolve()),
        "base_model": config.base_model,
        "dataset": probe_rows.dataset_identity,
        "dataset_dir": str(config.dataset_dir.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "details_path": str(details_path.resolve()),
        "summary_path": str(summary_path.resolve()),
        "primary_metric": config.primary_metric,
        "generation_failure_policy": config.generation_failure_policy,
        "selected_checkpoint": None,
        "final_adapter": final_summary,
        "selected_vs_final_delta": None,
        "probe": {
            "split": config.split,
            "split_manifest": str(config.split_manifest.resolve()) if config.split_manifest else None,
            "max_samples": config.max_samples,
            "short_utterance_seconds": config.short_utterance_seconds,
            "min_short_samples": config.min_short_samples,
            "min_technical_term_samples": config.min_technical_term_samples,
            "selected_count": len(probe_rows.rows),
            "tag_counts": count_probe_tags(probe_rows.rows),
            "selector": probe_rows.selector_identity,
        },
        "generation": generation_config_payload(config),
        "freshness": {
            "dataset_fingerprint": expected_freshness["dataset_fingerprint"],
            "generation_fingerprint": generation_fingerprint_value,
            "candidate_fingerprints": candidate_fingerprints,
        },
        "candidates": candidate_summaries,
    }
    write_jsonl(details_path, all_details)
    try:
        best = select_best_candidate(candidate_summaries, config.primary_metric, expected_freshness)
    except ValueError as exc:
        if config.generation_failure_policy == "eval-loss":
            try:
                best = select_best_eval_loss_candidate(candidate_summaries, expected_freshness)
            except ValueError as fallback_exc:
                summary["status"] = "failed"
                summary["selection_mode"] = "none"
                summary["failure"] = failure_payload(exc, fallback_exc, candidate_summaries, config.primary_metric)
                write_json(summary_path, summary)
                raise_probe_failure(exc, details_path, summary_path)
        else:
            summary["status"] = "failed"
            summary["selection_mode"] = "none"
            summary["failure"] = {
                "reason": str(exc),
                "generation_gate_failed": True,
                "eval_loss_fallback": "disabled",
                "candidate_status": candidate_selection_status(candidate_summaries, config.primary_metric),
            }
            write_json(summary_path, summary)
            raise_probe_failure(exc, details_path, summary_path)
    else:
        summary["status"] = "ok"

    if summary["status"] != "ok":
        summary["status"] = "fallback_eval_loss"
        summary["selection_mode"] = "eval_loss_fallback"
    summary["selected_checkpoint"] = best
    summary["selected_vs_final_delta"] = metric_delta(best, final_summary, config.primary_metric)
    write_json(summary_path, summary)
    print(f"Details: {details_path}")
    print(f"Summary: {summary_path}")
    print(selected_message(summary, best, config.primary_metric))
    return summary


def validate_probe_config(config: ProbeConfig) -> None:
    if config.primary_metric not in PRIMARY_METRICS:
        valid = ", ".join(sorted(PRIMARY_METRICS))
        raise ValueError(f"Unsupported primary metric {config.primary_metric!r}; expected one of: {valid}")
    if config.prompt_strategy not in PROMPT_STRATEGIES:
        valid = ", ".join(sorted(PROMPT_STRATEGIES))
        raise ValueError(f"Unsupported prompt strategy {config.prompt_strategy!r}; expected one of: {valid}")
    if config.generation_failure_policy not in GENERATION_FAILURE_POLICIES:
        valid = ", ".join(sorted(GENERATION_FAILURE_POLICIES))
        raise ValueError(
            f"Unsupported generation failure policy {config.generation_failure_policy!r}; expected one of: {valid}"
        )
    if config.rolling_prompt_turns < 0 or config.rolling_prompt_chars < 0:
        raise ValueError("rolling prompt turns/chars must be >= 0")
    validate_prompt_config(prompt_config_from_probe(config))


def discover_candidates(config: ProbeConfig) -> list[Path]:
    candidates: list[Path] = []
    for candidate in config.candidates:
        if is_lora_adapter(candidate):
            candidates.append(candidate)
        else:
            raise FileNotFoundError(f"Candidate is not a LoRA adapter directory: {candidate}")

    if config.include_checkpoints and config.run_dir.exists():
        candidates.extend(
            candidate
            for candidate in sorted(config.run_dir.glob("checkpoint-*"), key=checkpoint_sort_key)
            if is_lora_adapter(candidate)
        )

    if config.include_final and is_lora_adapter(config.run_dir):
        candidates.append(config.run_dir)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(candidate)
    return deduped


def is_lora_adapter(path: Path) -> bool:
    return path.is_dir() and (path / "adapter_config.json").exists()


def checkpoint_sort_key(path: Path) -> tuple[int, str]:
    match = re.search(r"checkpoint-(\d+)$", path.name)
    return (int(match.group(1)) if match else 10**12, path.name)


def load_probe_rows(config: ProbeConfig) -> ProbeRows:
    from datasets import Audio

    from .train import load_audiofolder_dataset

    dataset = load_audiofolder_dataset(config.dataset_dir)
    manifest_identity = None
    if config.split_manifest is not None:
        dataset, manifest_identity = apply_split_manifest(dataset, config.split_manifest)
    split_name = select_split_name(dataset, config.split)
    split = dataset[split_name]
    selection_rows = [row_with_duration_fallback(split[index], config.dataset_dir) for index in range(len(split))]
    selector_identity = None
    if config.selector_manifest is not None:
        selector_ids = load_selector_ids(config.selector_manifest)
        selected_indices = select_selector_indices(selection_rows, selector_ids)
        selector_identity = {
            "path": str(config.selector_manifest.resolve()),
            "sha256": file_sha256(config.selector_manifest),
            "count": len(selector_ids),
            "sample_set_sha256": stable_hash(sorted(selector_ids)),
        }
    else:
        selected_indices = select_probe_indices(
            selection_rows,
            max_samples=config.max_samples,
            min_short_samples=config.min_short_samples,
            min_technical_term_samples=config.min_technical_term_samples,
            short_utterance_seconds=config.short_utterance_seconds,
        )
    selected = split.select(selected_indices).cast_column("audio", Audio(sampling_rate=16_000))
    rows = [
        probe_row_from_sample(sample, split_name, source_index, config.short_utterance_seconds)
        for sample, source_index in zip(selected, selected_indices, strict=True)
    ]
    return ProbeRows(
        rows=rows,
        dataset_identity=dataset_identity(config.dataset_dir, manifest_identity),
        selector_identity=selector_identity,
    )


def load_selector_ids(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Selector manifest does not exist: {path}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                value = line
            if isinstance(value, dict):
                sample_id = value.get("sample_id") or value.get("file_name") or value.get("id")
            else:
                sample_id = value
            if not sample_id:
                raise ValueError(f"Selector row {path}:{line_number} has no sample ID.")
            ids.append(str(sample_id))
    if not ids:
        raise ValueError(f"Selector manifest is empty: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Selector manifest contains duplicate sample IDs: {path}")
    return ids


def select_selector_indices(rows: list[dict[str, Any]], selector_ids: list[str]) -> list[int]:
    by_id: dict[str, int] = {}
    for index, row in enumerate(rows):
        sample_id = sample_id_for_row(row)
        if sample_id in by_id:
            raise ValueError(f"Probe split contains duplicate sample ID {sample_id!r}.")
        by_id[sample_id] = index
    missing = [sample_id for sample_id in selector_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"Selector manifest IDs missing from probe split: {missing[:5]}")
    # Generation context must follow original source/time order, not manifest order.
    return sorted(by_id[sample_id] for sample_id in selector_ids)


def select_split_name(dataset: Any, requested: str | None) -> str:
    if requested:
        if requested not in dataset:
            raise ValueError(f"Split {requested!r} not found; available splits: {', '.join(dataset.keys())}")
        return requested
    for split_name in ("validation", "test", "benchmark", "eval"):
        if split_name in dataset:
            return split_name
    if "train" not in dataset:
        raise ValueError("Dataset has no validation/test/benchmark/eval/train split for probing.")
    print("Warning: dataset has no validation/test/eval split; probing train split.", file=sys.stderr)
    return "train"


def select_probe_indices(
    rows: list[dict[str, Any]],
    *,
    max_samples: int,
    min_short_samples: int,
    min_technical_term_samples: int,
    short_utterance_seconds: float,
) -> list[int]:
    if max_samples <= 0:
        raise ValueError("max_samples must be greater than zero.")

    selected: list[int] = []
    selected_set: set[int] = set()

    def add_matching(predicate: Any, limit: int) -> None:
        if limit <= 0:
            return
        added = 0
        for index, row in enumerate(rows):
            if index in selected_set or not predicate(row):
                continue
            selected.append(index)
            selected_set.add(index)
            added += 1
            if len(selected) >= max_samples or added >= limit:
                return

    add_matching(lambda row: is_short_utterance(row, short_utterance_seconds), min_short_samples)
    add_matching(has_english_terms, min_technical_term_samples)
    add_matching(lambda _row: True, max_samples)
    return sorted(selected[:max_samples])


def probe_row_from_sample(
    sample: dict[str, Any],
    split_name: str,
    source_index: int,
    short_utterance_seconds: float,
) -> dict[str, Any]:
    context_text = " ".join(str(sample.get(key, "")) for key in ("context_before", "context_after", "notes"))
    row = dict(sample)
    audio = row.get("audio")
    row["audio"] = audio
    row["split"] = split_name
    row["source_index"] = source_index
    row["reference"] = str(row.get("text", ""))
    row["audio_duration_seconds"] = audio_duration_seconds(row)
    row["english_terms"] = english_terms_from_text(row["reference"], context_text)
    row["probe_tags"] = probe_tags(row, short_utterance_seconds)
    return row


def probe_tags(row: dict[str, Any], short_utterance_seconds: float) -> list[str]:
    tags = ["overall"]
    if is_short_utterance(row, short_utterance_seconds):
        tags.append("short_utterance")
    if row.get("english_terms"):
        tags.append("english_technical_term")
    return tags


def score_candidate(candidate: Path, rows: list[dict[str, Any]], config: ProbeConfig) -> list[dict[str, Any]]:
    import torch
    from peft import PeftModel
    from transformers import BitsAndBytesConfig, WhisperForConditionalGeneration, WhisperProcessor

    dtype = torch.float32
    if config.fp16:
        dtype = torch.float16
    elif config.bf16:
        dtype = torch.bfloat16

    model_kwargs: dict[str, Any] = {}
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = dtype

    processor = WhisperProcessor.from_pretrained(config.base_model, task=config.task)
    model = WhisperForConditionalGeneration.from_pretrained(config.base_model, **model_kwargs)
    model.config.use_cache = True
    model.generation_config.task = config.task
    model.generation_config.language = None
    model.generation_config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(model, str(candidate))
    if not config.load_in_4bit:
        model.to(resolve_torch_device(config.device, torch))
    model.eval()
    input_device = model_input_device(model)

    details: list[dict[str, Any]] = []
    previous_predictions: list[str] = []
    previous_group: str | None = None
    with torch.no_grad():
        for row_number, row in enumerate(rows, start=1):
            current_group = group_id_for_row(row)
            if current_group != previous_group:
                previous_predictions = []
                previous_group = current_group
            prompt_text = generation_prompt_text(row, previous_predictions, config)
            try:
                audio = row["audio"]
                feature_batch = processor.feature_extractor(
                    audio["array"],
                    sampling_rate=audio["sampling_rate"],
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                features = feature_batch.input_features.to(device=input_device, dtype=model_input_dtype(model))
                attention_mask = feature_batch.attention_mask.to(device=input_device)
                generated_ids = model.generate(
                    input_features=features,
                    attention_mask=attention_mask,
                    max_length=config.generation_max_length,
                    task=config.task,
                    **language_kwargs(row, config),
                    **prompt_ids_kwargs(prompt_text, processor, config, torch, input_device),
                )
                prediction = normalize_thai_spacing(
                    processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                )
                detail = score_detail(
                        candidate=candidate,
                        row=row,
                        row_number=row_number,
                        status="ok",
                        prediction=prediction,
                    )
                detail["prompt_metadata"] = prompt_metadata(prompt_text, previous_predictions, current_group)
                details.append(detail)
                if config.prompt_strategy == "rolling-initial-prompt":
                    previous_predictions.append(prediction)
            except Exception as exc:
                detail = score_detail(
                        candidate=candidate,
                        row=row,
                        row_number=row_number,
                        status="failed",
                        error=repr(exc),
                    )
                detail["prompt_metadata"] = prompt_metadata(prompt_text, previous_predictions, current_group)
                details.append(detail)
    return details


def resolve_torch_device(device: str, torch: Any) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def model_input_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def model_input_dtype(model: Any) -> Any:
    dtype = getattr(model, "dtype", None)
    if dtype is not None:
        return dtype
    return next(model.parameters()).dtype


def release_model_memory() -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def language_kwargs(row: dict[str, Any], config: ProbeConfig) -> dict[str, str]:
    from .train import normalize_whisper_language

    if config.language_policy == "none":
        return {}
    if config.language_policy == "global":
        return {"language": config.language} if config.language else {}
    language = normalize_whisper_language(row.get("language")) or config.language
    return {"language": language} if language else {}


def generation_prompt_text(
    row: dict[str, Any],
    previous_predictions: list[str],
    config: ProbeConfig,
) -> str:
    if config.prompt_strategy == "isolated":
        return ""
    parts: list[str] = []
    if config.prompt_include_row_context:
        row_context = build_prompt_text(row, prompt_config_from_probe(config))
        if row_context:
            parts.append(row_context)
    recent = [text.strip() for text in previous_predictions[-config.rolling_prompt_turns :] if text.strip()]
    if recent and config.rolling_prompt_chars > 0:
        rolling = " ".join(recent)[-config.rolling_prompt_chars :]
        parts.append(f"Previous transcript: {rolling}")
    return "\n".join(parts)


def prompt_ids_kwargs(
    prompt_text: str,
    processor: Any,
    config: ProbeConfig,
    torch: Any,
    device: Any,
) -> dict[str, Any]:
    if not prompt_text:
        return {}
    prompt_ids = encode_prompt_token_ids(processor.tokenizer, prompt_text, config.prompt_max_tokens)
    if not prompt_ids:
        return {}
    return {"prompt_ids": torch.tensor(prompt_ids, dtype=torch.long, device=device)}


def prompt_metadata(prompt_text: str, previous_predictions: list[str], group: str) -> dict[str, Any]:
    return {
        "group": group,
        "non_empty": bool(prompt_text),
        "text_sha256": stable_hash(prompt_text) if prompt_text else None,
        "character_count": len(prompt_text),
        "previous_hypothesis_count": len(previous_predictions),
    }


def prompt_config_from_probe(config: ProbeConfig) -> PromptConditioningConfig:
    return PromptConditioningConfig(
        enabled=config.prompt_strategy == "rolling-initial-prompt",
        max_prompt_tokens=config.prompt_max_tokens,
        fields=parse_prompt_fields(config.prompt_fields),
        terms_only=config.prompt_terms_only,
        allow_future_context=config.prompt_allow_future_context,
    )


def score_detail(
    *,
    candidate: Path,
    row: dict[str, Any],
    row_number: int,
    status: str,
    prediction: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    reference = str(row.get("reference", ""))
    detail = {
        "candidate": candidate.name,
        "candidate_path": str(candidate.resolve()),
        "row_number": row_number,
        "split": row.get("split"),
        "source_index": row.get("source_index"),
        "file_name": row.get("file_name"),
        "language_label": row.get("language"),
        "speech_duration": row.get("speech_duration"),
        "probe_tags": row.get("probe_tags", []),
        "english_terms": row.get("english_terms", []),
        "reference": reference,
        "status": status,
    }
    if prediction is not None:
        detail["prediction"] = prediction
        detail["metrics"] = text_metrics(reference, prediction)
    if error is not None:
        detail["error"] = error
    return detail


def select_best_candidate(
    candidate_summaries: list[dict[str, Any]],
    primary_metric: str,
    expected_freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidate_summaries:
        raise ValueError("No candidate summaries to select from.")

    eligible = [
        summary
        for summary in candidate_summaries
        if candidate_has_complete_generation(summary)
        and is_finite_metric(summary.get("overall", {}).get(primary_metric))
        and candidate_is_fresh(summary, expected_freshness)
    ]
    if not eligible:
        statuses = candidate_selection_status(candidate_summaries, primary_metric, expected_freshness)
        raise ValueError(
            f"No fresh candidate has complete generation and a finite {primary_metric!r} value. "
            f"Candidate status: {json.dumps(statuses, sort_keys=True)}."
        )

    def sort_key(summary: dict[str, Any]) -> tuple[float, float, str]:
        overall = summary["overall"]
        primary = metric_or_infinity(overall.get(primary_metric))
        secondary = metric_or_infinity(overall.get("micro_cer"))
        return primary, secondary, str(summary["name"])

    return min(eligible, key=sort_key)


def select_best_eval_loss_candidate(
    candidate_summaries: list[dict[str, Any]],
    expected_freshness: dict[str, Any] | None = None,
) -> dict[str, Any]:
    eligible = [
        summary
        for summary in candidate_summaries
        if is_finite_metric(summary.get("eval_loss")) and candidate_is_fresh(summary, expected_freshness)
    ]
    if not eligible:
        raise ValueError("No fresh candidate has a finite eval_loss for explicit fallback selection.")
    best = min(eligible, key=lambda summary: (float(summary["eval_loss"]), str(summary["name"])))
    return {**best, "fallback_metric": "eval_loss", "fallback_metric_value": best["eval_loss"]}


def candidate_has_scored_rows(summary: dict[str, Any]) -> bool:
    count = summary.get("overall", {}).get("count")
    return isinstance(count, int) and not isinstance(count, bool) and count > 0


def candidate_has_complete_generation(summary: dict[str, Any]) -> bool:
    attempted = summary.get("attempted_count")
    required = summary.get("required_count", attempted)
    scored = summary.get("scored_count", summary.get("overall", {}).get("count"))
    failed = summary.get("failed_count", 0)
    return (
        isinstance(required, int)
        and required > 0
        and attempted == required
        and scored == required
        and failed == 0
    )


def candidate_is_fresh(summary: dict[str, Any], expected_freshness: dict[str, Any] | None) -> bool:
    if expected_freshness is None:
        return True
    freshness = summary.get("freshness")
    if not isinstance(freshness, dict):
        return False
    if freshness.get("dataset_fingerprint") != expected_freshness.get("dataset_fingerprint"):
        return False
    if freshness.get("generation_fingerprint") != expected_freshness.get("generation_fingerprint"):
        return False
    expected_candidates = expected_freshness.get("candidate_fingerprints") or {}
    if not isinstance(expected_candidates, dict):
        return False
    path = summary.get("path")
    fingerprint = freshness.get("candidate_fingerprint")
    if path in expected_candidates:
        return fingerprint == expected_candidates[path]
    return fingerprint in set(expected_candidates.values())


def candidate_selection_status(
    candidate_summaries: list[dict[str, Any]],
    primary_metric: str,
    expected_freshness: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    statuses = []
    for summary in candidate_summaries:
        metric_value = summary.get("overall", {}).get(primary_metric)
        statuses.append(
            {
                "name": summary.get("name"),
                "attempted_count": summary.get("attempted_count"),
                "required_count": summary.get("required_count"),
                "scored_count": summary.get("scored_count", summary.get("overall", {}).get("count", 0)),
                "failed_count": summary.get("failed_count"),
                "status_counts": summary.get("status_counts", {}),
                "primary_metric": primary_metric,
                "primary_metric_value": metric_value,
                "eval_loss": summary.get("eval_loss"),
                "fresh": candidate_is_fresh(summary, expected_freshness),
                "complete_generation": candidate_has_complete_generation(summary),
                "eligible": candidate_has_complete_generation(summary)
                and is_finite_metric(metric_value)
                and candidate_is_fresh(summary, expected_freshness),
            }
        )
    return statuses


def is_finite_metric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def metric_or_infinity(value: Any) -> float:
    if is_finite_metric(value):
        return float(value)
    return float("inf")


def metric_delta(
    selected: dict[str, Any] | None,
    final: dict[str, Any] | None,
    primary_metric: str,
) -> dict[str, float | None] | None:
    if selected is None or final is None:
        return None
    selected_value = selected.get("overall", {}).get(primary_metric)
    final_value = final.get("overall", {}).get(primary_metric)
    if not is_finite_metric(selected_value) or not is_finite_metric(final_value):
        return {"selected_minus_final": None}
    return {"selected_minus_final": selected_value - final_value}


def generation_config_payload(config: ProbeConfig) -> dict[str, Any]:
    return {
        "device": config.device,
        "fp16": config.fp16,
        "bf16": config.bf16,
        "load_in_4bit": config.load_in_4bit,
        "task": config.task,
        "language": config.language,
        "language_policy": config.language_policy,
        "generation_max_length": config.generation_max_length,
        "prompt_strategy": config.prompt_strategy,
        "prompt_max_tokens": config.prompt_max_tokens,
        "prompt_fields": list(parse_prompt_fields(config.prompt_fields)),
        "prompt_terms_only": config.prompt_terms_only,
        "prompt_allow_future_context": config.prompt_allow_future_context,
        "prompt_include_row_context": config.prompt_include_row_context,
        "rolling_prompt_turns": config.rolling_prompt_turns,
        "rolling_prompt_chars": config.rolling_prompt_chars,
        "selector_manifest_sha256": file_sha256(config.selector_manifest) if config.selector_manifest else None,
    }


def generation_fingerprint(config: ProbeConfig) -> str:
    payload = {
        "base_model": config.base_model,
        "generation": generation_config_payload(config),
    }
    return stable_hash(payload)


def dataset_fingerprint(identity: dict[str, Any]) -> str:
    return stable_hash(identity)


def adapter_fingerprint(path: Path) -> str:
    files = [
        path / "adapter_config.json",
        path / "adapter_model.safetensors",
        path / "adapter_model.bin",
        path / "trainer_state.json",
    ]
    payload = {
        "path": str(path.resolve()),
        "files": [
            {
                "name": file.name,
                "sha256": file_sha256(file),
            }
            for file in files
            if file.exists()
        ],
    }
    return stable_hash(payload)


def stable_hash(payload: Any) -> str:
    data = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=json_default).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def eval_loss_for_candidate(run_dir: Path, candidate: Path) -> float | None:
    step = checkpoint_step(candidate)
    for state_path in (candidate / "trainer_state.json", run_dir / "trainer_state.json"):
        value = eval_loss_from_trainer_state(state_path, step)
        if value is not None:
            return value
    return None


def checkpoint_step(candidate: Path) -> int | None:
    match = re.search(r"checkpoint-(\d+)$", candidate.name)
    return int(match.group(1)) if match else None


def eval_loss_from_trainer_state(path: Path, step: int | None) -> float | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    history = payload.get("log_history")
    if not isinstance(history, list):
        return None
    candidates = []
    for item in history:
        if not isinstance(item, dict) or "eval_loss" not in item:
            continue
        if step is not None and item.get("step") != step:
            continue
        if is_finite_metric(item.get("eval_loss")):
            candidates.append(float(item["eval_loss"]))
    if candidates:
        return candidates[-1]
    return None


def failure_payload(
    generation_error: Exception,
    fallback_error: Exception,
    candidate_summaries: list[dict[str, Any]],
    primary_metric: str,
) -> dict[str, Any]:
    return {
        "reason": str(generation_error),
        "generation_gate_failed": True,
        "eval_loss_fallback": "requested_but_unavailable",
        "fallback_reason": str(fallback_error),
        "candidate_status": candidate_selection_status(candidate_summaries, primary_metric),
    }


def raise_probe_failure(error: Exception, details_path: Path, summary_path: Path) -> None:
    message = (
        f"{error} Inspect per-sample errors in {details_path.resolve()} "
        f"and the failure summary in {summary_path.resolve()}."
    )
    print(message, file=sys.stderr)
    raise RuntimeError(message) from error


def selected_message(summary: dict[str, Any], selected: dict[str, Any], primary_metric: str) -> str:
    if summary["selection_mode"] == "eval_loss_fallback":
        return (
            f"Selected {selected['name']} by explicit eval_loss fallback: "
            f"eval_loss={selected.get('eval_loss')}"
        )
    return "Selected {name}: {metric}={value}".format(
        name=selected["name"],
        metric=primary_metric,
        value=selected["overall"].get(primary_metric),
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True, default=json_default))
            handle.write("\n")


def json_default(value: object) -> object:
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return]
    return str(value)


def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "run"


def row_with_duration_fallback(row: dict[str, Any], dataset_dir: Path) -> dict[str, Any]:
    if any(isinstance(row.get(key), int | float) for key in ("speech_duration", "audio_duration_seconds", "duration")):
        return row
    enriched = dict(row)
    duration = audio_duration_seconds(enriched, dataset_dir=dataset_dir)
    if duration is not None:
        enriched["audio_duration_seconds"] = duration
    return enriched


def audio_duration_seconds(row: dict[str, Any], dataset_dir: Path | None = None) -> float | None:
    for key in ("speech_duration", "audio_duration_seconds", "duration"):
        value = row.get(key)
        if isinstance(value, int | float):
            return float(value)

    audio = row.get("audio")
    if isinstance(audio, dict):
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if hasattr(array, "__len__") and isinstance(sampling_rate, int | float) and sampling_rate > 0:
            return len(array) / float(sampling_rate)
        path = audio.get("path")
    else:
        path = audio

    if path is None:
        path = row.get("file_name") or row.get("path")
    if path is None:
        return None

    audio_path = Path(str(path))
    if not audio_path.is_absolute() and dataset_dir is not None:
        audio_path = dataset_dir / audio_path
    if not audio_path.exists():
        return None

    try:
        import soundfile as sf
    except ImportError:
        return None

    try:
        info = sf.info(str(audio_path))
    except (RuntimeError, OSError):
        return None
    if info.samplerate <= 0:
        return None
    return info.frames / float(info.samplerate)
