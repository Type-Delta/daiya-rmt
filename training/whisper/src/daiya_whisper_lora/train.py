from __future__ import annotations

from hashlib import sha256
import json
import inspect
import random
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from rich.console import Console
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

console = Console()

LANGUAGE_ALIASES = {
    "en": "English",
    "eng": "English",
    "english": "English",
    "th": "Thai",
    "tha": "Thai",
    "thai": "Thai",
    "thai-english": "Thai",
    "th-en": "Thai",
    "th_en": "Thai",
    "ja": "Japanese",
    "jpn": "Japanese",
    "japanese": "Japanese",
    "japanese-english": "Japanese",
    "ja-en": "Japanese",
    "ja_en": "Japanese",
    "mixed": None,
    "none": None,
    "": None,
}


@dataclass(frozen=True)
class TrainingConfig:
    dataset_dir: Path
    model_name_or_path: str
    output_dir: Path
    language: str | None = None
    language_policy: str = "metadata"
    task: str = "transcribe"
    validation_size: float = 0.05
    seed: int = 42
    num_train_epochs: float = 3.0
    max_steps: int = -1
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    eval_steps: int = 250
    save_steps: int = 250
    logging_steps: int = 25
    predict_with_generate: bool = False
    generation_max_length: int = 225
    preprocessing_num_proc: int = 1
    dataloader_num_workers: int = 0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_label_length: int = 448
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    fp16: bool = False
    bf16: bool = False
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    gradient_checkpointing: bool = False
    load_best_model_at_end: bool = False
    resume_from_checkpoint: bool = False
    push_to_hub: bool = False
    hub_model_id: str | None = None
    include_ineligible_for_research: bool = False
    legacy_training_eligibility: str = "error"


@dataclass(frozen=True)
class TrainingSelection:
    included: tuple[str, ...]
    excluded: tuple[str, ...]
    legacy: tuple[str, ...]
    include_ineligible_for_research: bool
    legacy_policy: str

    def provenance(self) -> dict[str, Any]:
        return {
            "schema_version": "daiya-training-selection-1",
            "included_count": len(self.included),
            "excluded_count": len(self.excluded),
            "legacy_count": len(self.legacy),
            "included_identities": list(self.included),
            "excluded_identities": list(self.excluded),
            "legacy_identities": list(self.legacy),
            "include_ineligible_for_research": self.include_ineligible_for_research,
            "legacy_training_eligibility": self.legacy_policy,
        }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: WhisperProcessor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def inspect_dataset(
    dataset_dir: Path,
    *,
    include_ineligible_for_research: bool = False,
    legacy_training_eligibility: str = "error",
) -> None:
    dataset = load_audiofolder_dataset(
        dataset_dir,
        include_ineligible_for_research=include_ineligible_for_research,
        legacy_training_eligibility=legacy_training_eligibility,
    )
    console.print(f"[bold]Dataset:[/bold] {dataset_dir.resolve()}")
    for split, split_dataset in dataset.items():
        console.print(f"[bold]{split}[/bold]: {len(split_dataset)} rows")
        if len(split_dataset) == 0:
            continue
        sample = split_dataset[0]
        fields = ", ".join(sample.keys())
        console.print(f"  fields: {fields}")
        console.print(f"  text: {safe_for_console(sample.get('text', ''))}")
        console.print(f"  language: {safe_for_console(sample.get('language', ''))}")
        if "language" in split_dataset.column_names:
            languages = Counter(str(language or "") for language in split_dataset["language"]).most_common(12)
            console.print("  language counts:")
            for language, count in languages:
                console.print(f"    {safe_for_console(language)}: {count}")


def safe_for_console(value: Any) -> str:
    text = str(value)
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="backslashreplace").decode(encoding)


def source_split_indices(source_ids: list[str], *, validation_size: float, seed: int) -> tuple[list[int], list[int]]:
    """Choose validation rows by recording, never by adjacent clip."""
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be between 0 and 1 for source-disjoint splitting")
    groups: dict[str, list[int]] = {}
    for index, source_id in enumerate(source_ids):
        if not source_id:
            raise ValueError("Source-disjoint validation requires source_id or source_file for every row")
        groups.setdefault(source_id, []).append(index)
    if len(groups) < 2:
        raise ValueError(
            "Cannot create a source-disjoint validation split from fewer than two recordings. "
            "Provide explicit train/validation directories with disjoint source identities."
        )
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    validation_group_count = max(1, min(len(keys) - 1, round(len(keys) * validation_size)))
    validation_sources = set(keys[:validation_group_count])
    train = [index for index, source_id in enumerate(source_ids) if source_id not in validation_sources]
    validation = [index for index, source_id in enumerate(source_ids) if source_id in validation_sources]
    return train, validation


def _source_ids(dataset: Dataset) -> list[str]:
    source_ids = dataset["source_id"] if "source_id" in dataset.column_names else [None] * len(dataset)
    source_files = dataset["source_file"] if "source_file" in dataset.column_names else [None] * len(dataset)
    values = [str(source_id or source_file or "") for source_id, source_file in zip(source_ids, source_files, strict=True)]
    if any(not value for value in values):
        raise ValueError(
            "Source-disjoint validation requires source_id or source_file provenance for every row; "
            "regenerate legacy metadata or provide pre-split data with provenance."
        )
    return values


def _assert_source_disjoint(train: Dataset, validation: Dataset) -> None:
    shared = set(_source_ids(train)) & set(_source_ids(validation))
    if shared:
        preview = ", ".join(sorted(shared)[:5])
        raise ValueError(f"Train and validation share source recordings: {preview}")


def feature_cache_key(dataset: DatasetDict, config: TrainingConfig) -> str:
    """Hash selected rows and feature-affecting configuration for cache reuse."""
    digest = sha256()
    digest.update(json.dumps({
        "schema_version": "daiya-whisper-feature-cache-ownership-1",
        "model": config.model_name_or_path,
        "task": config.task,
        "language": config.language,
        "language_policy": config.language_policy,
        "max_label_length": config.max_label_length,
        "sample_rate": 16_000,
    }, sort_keys=True, ensure_ascii=False).encode("utf-8"))
    for split_name in sorted(dataset):
        split = dataset[split_name]
        columns = [
            column
            for column in (
                "file_name",
                "audio_sha256",
                "source_id",
                "source_file",
                "text",
                "language",
            )
            if column in split.column_names
        ]
        digest.update(split_name.encode("utf-8"))
        digest.update(str(len(split)).encode("ascii"))
        for index in range(len(split)):
            row = {column: split[column][index] for column in columns}
            digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()[:20]


_power_request_handle = None


def keep_system_awake() -> None:
    # Modern Standby ignores SetThreadExecutionState (see
    # docs/2026-07-04-overnight-training-kill-report.md); an ExecutionRequired power request is
    # the only API that keeps this process running through standby on this platform.
    if sys.platform != "win32":
        return
    import ctypes
    from ctypes import wintypes

    class ReasonContext(ctypes.Structure):
        _fields_ = [
            ("Version", wintypes.ULONG),
            ("Flags", wintypes.DWORD),
            ("SimpleReasonString", wintypes.LPWSTR),
        ]

    POWER_REQUEST_CONTEXT_SIMPLE_STRING = 0x1
    POWER_REQUEST_SYSTEM_REQUIRED = 1
    POWER_REQUEST_EXECUTION_REQUIRED = 3

    context = ReasonContext(0, POWER_REQUEST_CONTEXT_SIMPLE_STRING, "daiya whisper LoRA training")
    handle = ctypes.windll.kernel32.PowerCreateRequest(ctypes.byref(context))
    if handle and handle != -1:
        ctypes.windll.kernel32.PowerSetRequest(handle, POWER_REQUEST_SYSTEM_REQUIRED)
        ctypes.windll.kernel32.PowerSetRequest(handle, POWER_REQUEST_EXECUTION_REQUIRED)
        global _power_request_handle  # keep handle alive for the process lifetime
        _power_request_handle = handle

    ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
    ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def train(config: TrainingConfig) -> None:
    keep_system_awake()
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = config.dataset_dir / "metadata.jsonl"
    if metadata_path.is_file():
        selection = training_selection_from_metadata(
            metadata_path,
            include_ineligible_for_research=config.include_ineligible_for_research,
            legacy_training_eligibility=config.legacy_training_eligibility,
        )
        provenance_path = write_training_selection(selection, config.output_dir)
        console.print(
            f"[bold]Training selection:[/bold] included={len(selection.included)} "
            f"excluded={len(selection.excluded)} ({provenance_path.name})"
        )

    processor = load_processor(config)
    dataset = prepare_dataset(config, processor)
    model = load_lora_model(config)

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    compute_metrics = None
    if config.predict_with_generate:
        wer_metric = evaluate.load("wer")

        def compute_metrics(pred: Any) -> dict[str, float]:
            pred_ids = pred.predictions
            label_ids = pred.label_ids
            label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

            pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
            label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
            return {"wer": 100 * wer_metric.compute(predictions=pred_str, references=label_str)}

    training_args = build_training_args(config)

    trainer = Seq2SeqTrainer(
        **build_trainer_kwargs(
            training_args=training_args,
            model=model,
            dataset=dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            processor=processor,
        ),
    )

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint or None)
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))


def _row_identity(row: dict[str, Any], line_number: int) -> str:
    if row.get("file_name"):
        return str(row["file_name"])
    source_id = row.get("source_id")
    owned_start = row.get("owned_source_start", row.get("source_start"))
    owned_end = row.get("owned_source_end", row.get("source_end"))
    if source_id and owned_start is not None and owned_end is not None:
        return f"{source_id}@{owned_start}-{owned_end}"
    return str(source_id or f"metadata.jsonl#line={line_number}")


def select_training_rows(
    rows: list[tuple[int, dict[str, Any]]],
    *,
    include_ineligible_for_research: bool,
    legacy_training_eligibility: str,
) -> tuple[list[tuple[int, dict[str, Any]]], TrainingSelection]:
    """Select explicit ownership-safe rows; never guess a legacy field."""
    if legacy_training_eligibility not in {"error", "include", "exclude"}:
        raise ValueError("legacy_training_eligibility must be error, include, or exclude")
    selected: list[tuple[int, dict[str, Any]]] = []
    included: list[str] = []
    excluded: list[str] = []
    legacy: list[str] = []
    for line_number, row in rows:
        identity = _row_identity(row, line_number)
        value = row.get("training_eligible")
        if value is None:
            legacy.append(identity)
            if legacy_training_eligibility == "error":
                continue
            eligible = legacy_training_eligibility == "include"
            research_override = False
        elif isinstance(value, bool):
            eligible = value
            research_override = include_ineligible_for_research
        else:
            raise ValueError(f"training_eligible must be boolean at metadata.jsonl line {line_number}")
        if eligible or research_override:
            selected.append((line_number, row))
            included.append(identity)
        else:
            excluded.append(identity)
    if legacy and legacy_training_eligibility == "error":
        preview = ", ".join(legacy[:5])
        raise ValueError(
            "Dataset lacks training_eligible metadata for legacy rows. "
            "Regenerate with timestamp-ownership segmentation or explicitly set "
            f"--legacy-training-eligibility include/exclude. Examples: {preview}"
        )
    return selected, TrainingSelection(
        included=tuple(included),
        excluded=tuple(excluded),
        legacy=tuple(legacy),
        include_ineligible_for_research=include_ineligible_for_research,
        legacy_policy=legacy_training_eligibility,
    )


def training_selection_from_metadata(
    metadata_path: Path,
    *,
    include_ineligible_for_research: bool,
    legacy_training_eligibility: str,
) -> TrainingSelection:
    rows: list[tuple[int, dict[str, Any]]] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected object in {metadata_path}:{line_number}")
                rows.append((line_number, value))
    _, selection = select_training_rows(
        rows,
        include_ineligible_for_research=include_ineligible_for_research,
        legacy_training_eligibility=legacy_training_eligibility,
    )
    return selection


def write_training_selection(selection: TrainingSelection, output_dir: Path) -> Path:
    path = output_dir / "dataset-selection.json"
    path.write_text(json.dumps(selection.provenance(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def load_audiofolder_dataset(
    dataset_dir: Path,
    *,
    include_ineligible_for_research: bool = False,
    legacy_training_eligibility: str = "error",
) -> DatasetDict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    metadata_path = dataset_dir / "metadata.jsonl"
    if metadata_path.exists():
        return load_dataset_from_metadata(
            dataset_dir,
            metadata_path,
            include_ineligible_for_research=include_ineligible_for_research,
            legacy_training_eligibility=legacy_training_eligibility,
        )

    if legacy_training_eligibility == "error":
        raise ValueError(
            "Dataset has no metadata.jsonl and therefore no training_eligible ownership field. "
            "Pass --legacy-training-eligibility include only for an explicit legacy research run."
        )
    if legacy_training_eligibility == "exclude" and not include_ineligible_for_research:
        raise ValueError("Cannot select ownership-safe rows from metadata-free legacy audiofolder dataset")

    dataset = load_dataset("audiofolder", data_dir=str(dataset_dir))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    return dataset


def load_dataset_from_metadata(
    dataset_dir: Path,
    metadata_path: Path,
    *,
    include_ineligible_for_research: bool = False,
    legacy_training_eligibility: str = "error",
) -> DatasetDict:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    parsed_rows: list[tuple[int, dict[str, Any]]] = []
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Expected object in {metadata_path}:{line_number}")
            parsed_rows.append((line_number, row))

    selected_rows, _ = select_training_rows(
        parsed_rows,
        include_ineligible_for_research=include_ineligible_for_research,
        legacy_training_eligibility=legacy_training_eligibility,
    )
    for line_number, row in selected_rows:
        file_name = row.get("file_name")
        text = row.get("text")
        if not file_name or text is None:
            raise ValueError(f"Missing file_name or text in {metadata_path}:{line_number}")

        file_path = dataset_dir / file_name
        if not file_path.exists():
            raise FileNotFoundError(f"Audio file referenced by metadata does not exist: {file_path}")

        path_parts = Path(file_name).parts
        split = path_parts[0] if len(path_parts) > 1 else "train"
        row["audio"] = str(file_path)
        rows_by_split.setdefault(split, []).append(row)

    if not rows_by_split:
        raise ValueError(f"No rows found in metadata file: {metadata_path}")

    return DatasetDict({split: Dataset.from_list(rows) for split, rows in rows_by_split.items()})


def load_processor(config: TrainingConfig) -> WhisperProcessor:
    processor_kwargs = {"task": config.task}
    if config.language:
        processor_kwargs["language"] = config.language
    return WhisperProcessor.from_pretrained(config.model_name_or_path, **processor_kwargs)


def prepare_dataset(config: TrainingConfig, processor: WhisperProcessor) -> DatasetDict:
    if config.language_policy not in {"metadata", "global", "none"}:
        raise ValueError(f"Unsupported language policy: {config.language_policy}")

    dataset = load_audiofolder_dataset(
        config.dataset_dir,
        include_ineligible_for_research=config.include_ineligible_for_research,
        legacy_training_eligibility=config.legacy_training_eligibility,
    )

    if "validation" not in dataset:
        train_indices, validation_indices = source_split_indices(
            _source_ids(dataset["train"]), validation_size=config.validation_size, seed=config.seed
        )
        dataset = DatasetDict(
            train=dataset["train"].select(train_indices),
            validation=dataset["train"].select(validation_indices),
        )
    _assert_source_disjoint(dataset["train"], dataset["validation"])

    dataset = dataset.cast_column("audio", Audio(sampling_rate=16_000))

    if config.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(config.max_train_samples, len(dataset["train"]))))
    if config.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(config.max_eval_samples, len(dataset["validation"])))
        )

    columns_to_remove = {
        column
        for split_dataset in dataset.values()
        for column in split_dataset.column_names
        if column not in {"audio", "text"}
    }

    def prepare_example(example: dict[str, Any]) -> dict[str, Any]:
        audio = example["audio"]
        example["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        language = language_for_example(example, config)
        processor.tokenizer.set_prefix_tokens(language=language, task=config.task)
        example["labels"] = processor.tokenizer(example["text"]).input_ids
        return example

    # Disk-backed feature caches must never cross ownership selections or
    # label/config revisions.  Reusing fixed train.arrow filenames could feed
    # rows excluded by the current eligibility policy back into training.
    cache_dir = config.output_dir.parent / "feature_cache" / feature_cache_key(dataset, config)
    cache_dir.mkdir(parents=True, exist_ok=True)

    dataset = dataset.map(
        prepare_example,
        remove_columns=list(columns_to_remove | {"audio", "text"}),
        num_proc=config.preprocessing_num_proc,
        desc="Preparing Whisper features",
        cache_file_names={split: str(cache_dir / f"{split}.arrow") for split in dataset},
        writer_batch_size=32,
    )

    dataset = dataset.filter(
        lambda example: 0 < len(example["labels"]) <= config.max_label_length,
        desc="Filtering label length",
        cache_file_names={split: str(cache_dir / f"{split}.filtered.arrow") for split in dataset},
        writer_batch_size=32,
    )

    return dataset


def language_for_example(example: dict[str, Any], config: TrainingConfig) -> str | None:
    if config.language_policy == "none":
        return None
    if config.language_policy == "global":
        return config.language

    metadata_language = normalize_whisper_language(example.get("language"))
    return metadata_language or config.language


def normalize_whisper_language(value: Any) -> str | None:
    if value is None:
        return None
    key = str(value).strip().lower()
    key = key.replace(" / ", "-").replace("/", "-")
    return LANGUAGE_ALIASES.get(key, None)


def build_training_args(config: TrainingConfig) -> Seq2SeqTrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(config.output_dir),
        "per_device_train_batch_size": config.per_device_train_batch_size,
        "per_device_eval_batch_size": config.per_device_eval_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": config.learning_rate,
        "warmup_steps": config.warmup_steps,
        "num_train_epochs": config.num_train_epochs,
        "max_steps": config.max_steps,
        "gradient_checkpointing": config.gradient_checkpointing,
        # non-reentrant so checkpointed encoder blocks still get LoRA grads
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "fp16": config.fp16,
        "bf16": config.bf16,
        "eval_steps": config.eval_steps,
        "save_steps": config.save_steps,
        "logging_steps": config.logging_steps,
        "predict_with_generate": config.predict_with_generate,
        "generation_max_length": config.generation_max_length,
        "remove_unused_columns": False,
        "label_names": ["labels"],
        "dataloader_num_workers": config.dataloader_num_workers,
        "report_to": ["tensorboard"],
        "load_best_model_at_end": config.load_best_model_at_end,
        "metric_for_best_model": "wer" if config.predict_with_generate else "eval_loss",
        "greater_is_better": False,
        "push_to_hub": config.push_to_hub,
        "hub_model_id": config.hub_model_id,
    }

    parameters = inspect.signature(Seq2SeqTrainingArguments).parameters
    if "eval_strategy" in parameters:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    return Seq2SeqTrainingArguments(**kwargs)


def build_trainer_kwargs(
    training_args: Seq2SeqTrainingArguments,
    model: torch.nn.Module,
    dataset: DatasetDict,
    data_collator: DataCollatorSpeechSeq2SeqWithPadding,
    compute_metrics: Any | None,
    processor: WhisperProcessor,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "args": training_args,
        "model": model,
        "train_dataset": dataset["train"],
        "eval_dataset": dataset["validation"],
        "data_collator": data_collator,
    }
    if compute_metrics is not None:
        kwargs["compute_metrics"] = compute_metrics

    parameters = inspect.signature(Seq2SeqTrainer).parameters
    if "processing_class" in parameters:
        kwargs["processing_class"] = processor
    else:
        kwargs["tokenizer"] = processor.feature_extractor

    return kwargs


def load_lora_model(config: TrainingConfig) -> WhisperForConditionalGeneration:
    model_kwargs: dict[str, Any] = {}
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = "auto"
    elif config.load_in_8bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["device_map"] = "auto"

    model = WhisperForConditionalGeneration.from_pretrained(config.model_name_or_path, **model_kwargs)
    if config.load_in_4bit or config.load_in_8bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=config.gradient_checkpointing
        )
    model.config.use_cache = False
    model.generation_config.task = config.task
    if config.language_policy == "global" and config.language:
        model.generation_config.language = config.language
    else:
        model.generation_config.language = None
        model.generation_config.forced_decoder_ids = None

    if config.gradient_checkpointing:
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        target_modules=list(config.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model
