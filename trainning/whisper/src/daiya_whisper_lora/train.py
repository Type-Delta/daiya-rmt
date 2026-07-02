from __future__ import annotations

import json
import inspect
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import Audio, Dataset, DatasetDict, load_dataset
from peft import LoraConfig, get_peft_model
from rich.console import Console
from transformers import (
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
    gradient_checkpointing: bool = False
    load_best_model_at_end: bool = False
    push_to_hub: bool = False
    hub_model_id: str | None = None


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


def inspect_dataset(dataset_dir: Path) -> None:
    dataset = load_audiofolder_dataset(dataset_dir)
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


def train(config: TrainingConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)

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

    trainer.train()
    trainer.save_model(str(config.output_dir))
    processor.save_pretrained(str(config.output_dir))


def load_audiofolder_dataset(dataset_dir: Path) -> DatasetDict:
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    metadata_path = dataset_dir / "metadata.jsonl"
    if metadata_path.exists():
        return load_dataset_from_metadata(dataset_dir, metadata_path)

    dataset = load_dataset("audiofolder", data_dir=str(dataset_dir))
    if not isinstance(dataset, DatasetDict):
        dataset = DatasetDict({"train": dataset})
    return dataset


def load_dataset_from_metadata(dataset_dir: Path, metadata_path: Path) -> DatasetDict:
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    with metadata_path.open("r", encoding="utf-8") as metadata_file:
        for line_number, line in enumerate(metadata_file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
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

    dataset = load_audiofolder_dataset(config.dataset_dir)

    if "validation" not in dataset:
        split = dataset["train"].train_test_split(test_size=config.validation_size, seed=config.seed)
        dataset = DatasetDict(train=split["train"], validation=split["test"])

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

    dataset = dataset.map(
        prepare_example,
        remove_columns=list(columns_to_remove | {"audio", "text"}),
        num_proc=config.preprocessing_num_proc,
        desc="Preparing Whisper features",
    )

    dataset = dataset.filter(
        lambda example: 0 < len(example["labels"]) <= config.max_label_length,
        desc="Filtering label length",
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
    model = WhisperForConditionalGeneration.from_pretrained(config.model_name_or_path)
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
