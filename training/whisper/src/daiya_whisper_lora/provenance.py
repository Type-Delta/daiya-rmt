from __future__ import annotations

import json
import hashlib
from importlib import metadata as importlib_metadata
import os
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .split_manifest import SplitManifestIdentity, file_sha256


def collect_run_provenance(
    *,
    config: Any,
    dataset_identity: dict[str, Any],
    prompt_strategy: dict[str, Any],
    base_model_revision: str | None = None,
) -> dict[str, Any]:
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git": git_identity(),
        "base_model": getattr(config, "model_name_or_path", None),
        "base_model_revision": base_model_revision,
        "dataset": dataset_identity,
        "seed": getattr(config, "seed", None),
        "prompt": prompt_strategy,
        "hyperparameters": training_hyperparameters(config),
        "resolved_training_config": normalize_value(config),
        "cadence": {
            "eval_steps": getattr(config, "eval_steps", None),
            "save_steps": getattr(config, "save_steps", None),
            "logging_steps": getattr(config, "logging_steps", None),
            "max_steps": getattr(config, "max_steps", None),
            "num_train_epochs": getattr(config, "num_train_epochs", None),
        },
        "runtime": runtime_identity(),
    }


def dataset_identity(
    dataset_dir: Path,
    split_manifest: SplitManifestIdentity | None = None,
) -> dict[str, Any]:
    metadata_path = dataset_dir / "metadata.jsonl"
    identity: dict[str, Any] = {
        "dataset_dir": str(dataset_dir.resolve()),
        "metadata_jsonl": str(metadata_path.resolve()) if metadata_path.exists() else None,
        "metadata_jsonl_sha256": file_sha256(metadata_path) if metadata_path.exists() else None,
        "split_manifest": split_manifest.to_dict() if split_manifest is not None else None,
    }
    return identity


def write_run_provenance(path: Path, provenance: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, default=json_default)
        handle.write("\n")


def preprocessing_cache_key(
    *,
    config: Any,
    dataset: dict[str, Any],
    prompt: dict[str, Any],
    processor_identity: dict[str, Any],
) -> str:
    payload = {
        "dataset": dataset,
        "model": getattr(config, "model_name_or_path", None),
        "language": getattr(config, "language", None),
        "language_policy": getattr(config, "language_policy", None),
        "task": getattr(config, "task", None),
        "seed": getattr(config, "seed", None),
        "max_label_length": getattr(config, "max_label_length", None),
        "max_train_samples": getattr(config, "max_train_samples", None),
        "max_eval_samples": getattr(config, "max_eval_samples", None),
        "prompt": prompt,
        "processor": processor_identity,
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=json_default)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def git_identity() -> dict[str, Any]:
    return {
        "commit": run_git(["rev-parse", "HEAD"]),
        "is_dirty": git_is_dirty(),
    }


def run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or ""


def git_is_dirty() -> bool | None:
    status = run_git(["status", "--porcelain", "--untracked-files=normal"])
    return None if status is None else bool(status)


def training_hyperparameters(config: Any) -> dict[str, Any]:
    keys = (
        "learning_rate",
        "warmup_steps",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "lora_r",
        "lora_alpha",
        "lora_dropout",
        "lora_target_modules",
        "fp16",
        "bf16",
        "load_in_4bit",
        "load_in_8bit",
        "gradient_checkpointing",
        "predict_with_generate",
        "generation_max_length",
    )
    return {key: normalize_value(getattr(config, key, None)) for key in keys}


def runtime_identity() -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "pid": os.getpid(),
        "packages": package_versions(),
    }
    try:
        import torch
    except ImportError:
        identity["torch"] = None
        return identity

    cuda_available = torch.cuda.is_available()
    identity["torch"] = {
        "version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": getattr(torch.version, "cuda", None),
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "device_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    return identity


def package_versions() -> dict[str, str | None]:
    packages = ("transformers", "peft", "datasets", "accelerate", "bitsandbytes", "torch", "torchaudio")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib_metadata.version(package)
        except importlib_metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return list(value)
    if is_dataclass(value):
        return {key: normalize_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): normalize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_value(item) for item in value]
    return value


def json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()  # type: ignore[no-any-return]
    return str(value)
