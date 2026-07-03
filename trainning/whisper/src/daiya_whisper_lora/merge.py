from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_BASE_MODEL = "openai/whisper-medium"
DEFAULT_QUANTIZATION = "int8_float16"


def package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_adapter_path() -> Path:
    return package_root() / "runs" / "medium-real-iter4"


def default_merged_output_dir() -> Path:
    return package_root() / "runs" / "medium-real-iter4-merged"


def default_ct2_output_dir() -> Path:
    return package_root() / "runs" / "medium-real-iter4-ct2-int8_float16"


@dataclass(frozen=True)
class MergeConfig:
    adapter_path: Path = default_adapter_path()
    base_model: str = DEFAULT_BASE_MODEL
    merged_output_dir: Path = default_merged_output_dir()
    ct2_output_dir: Path = default_ct2_output_dir()
    quantization: str = DEFAULT_QUANTIZATION
    dataset_dir: Path | None = None
    max_eval_samples: int = 8
    allowed_absolute_wer_drift: float = 1.0
    skip_convert: bool = False
    skip_wer: bool = False
    force: bool = True
    device: str = "auto"
    task: str = "transcribe"
    language: str | None = None
    generation_max_length: int = 225


@dataclass(frozen=True)
class WerSanityResult:
    skipped: bool
    reason: str | None = None
    peft_wer: float | None = None
    ct2_wer: float | None = None
    absolute_drift: float | None = None
    sample_count: int = 0


@dataclass(frozen=True)
class MergeResult:
    merged_model_dir: Path
    ct2_model_dir: Path | None
    wer: WerSanityResult | None


def merge_and_convert(config: MergeConfig) -> MergeResult:
    merged_model_dir = merge_lora_adapter(config)

    ct2_model_dir: Path | None = None
    if not config.skip_convert:
        ct2_model_dir = convert_to_ct2(
            merged_model_dir=merged_model_dir,
            output_dir=config.ct2_output_dir,
            quantization=config.quantization,
            force=config.force,
        )
    elif config.ct2_output_dir.exists():
        ct2_model_dir = config.ct2_output_dir

    wer_result: WerSanityResult | None = None
    if not config.skip_wer:
        wer_result = run_wer_sanity_check(config, ct2_model_dir)
        if (
            not wer_result.skipped
            and wer_result.absolute_drift is not None
            and wer_result.absolute_drift > config.allowed_absolute_wer_drift
        ):
            raise RuntimeError(
                "Merged CT2 WER drift exceeded allowed threshold: "
                f"{wer_result.absolute_drift:.2f} > {config.allowed_absolute_wer_drift:.2f}"
            )

    return MergeResult(
        merged_model_dir=merged_model_dir,
        ct2_model_dir=ct2_model_dir,
        wer=wer_result,
    )


def merge_lora_adapter(config: MergeConfig) -> Path:
    if not config.adapter_path.exists():
        raise FileNotFoundError(f"LoRA adapter path does not exist: {config.adapter_path}")

    config.merged_output_dir.mkdir(parents=True, exist_ok=True)
    processor = load_processor(config.base_model, task=config.task, language=config.language)
    model = load_peft_model(config)

    if not hasattr(model, "merge_and_unload"):
        raise TypeError("Loaded PEFT model does not expose merge_and_unload().")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(str(config.merged_output_dir), safe_serialization=True)
    processor.save_pretrained(str(config.merged_output_dir))
    return config.merged_output_dir


def load_base_model(base_model: str, device: str = "auto") -> Any:
    import torch
    from transformers import WhisperForConditionalGeneration

    torch_dtype = torch.float16 if _resolve_torch_device(device) == "cuda" else torch.float32
    return WhisperForConditionalGeneration.from_pretrained(base_model, torch_dtype=torch_dtype)


def load_peft_model(config: MergeConfig) -> Any:
    from peft import PeftModel

    model = load_base_model(config.base_model, device=config.device)
    model.config.use_cache = True
    model.generation_config.task = config.task
    if config.language:
        model.generation_config.language = config.language
    else:
        model.generation_config.language = None
        model.generation_config.forced_decoder_ids = None
    return PeftModel.from_pretrained(model, str(config.adapter_path))


def load_processor(base_model: str, task: str = "transcribe", language: str | None = None) -> Any:
    from transformers import WhisperProcessor

    kwargs: dict[str, str] = {"task": task}
    if language:
        kwargs["language"] = language
    return WhisperProcessor.from_pretrained(base_model, **kwargs)


def convert_to_ct2(
    merged_model_dir: Path,
    output_dir: Path,
    quantization: str = DEFAULT_QUANTIZATION,
    force: bool = True,
) -> Path:
    if not merged_model_dir.exists():
        raise FileNotFoundError(f"Merged model directory does not exist: {merged_model_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from ctranslate2.converters import TransformersConverter
    except ImportError:
        converter_exe = shutil.which("ct2-transformers-converter")
        if converter_exe is None:
            raise RuntimeError(
                "CTranslate2 is required for conversion. Install ctranslate2 or rerun with --skip-convert."
            ) from None
        command = [
            converter_exe,
            "--model",
            str(merged_model_dir),
            "--output_dir",
            str(output_dir),
            "--quantization",
            quantization,
        ]
        if force:
            command.append("--force")
        subprocess.run(command, check=True)
    else:
        converter = TransformersConverter(str(merged_model_dir))
        converter.convert(
            output_dir=str(output_dir),
            quantization=quantization,
            force=force,
        )

    return output_dir


def run_wer_sanity_check(config: MergeConfig, ct2_model_dir: Path | None) -> WerSanityResult:
    if config.dataset_dir is None:
        return WerSanityResult(skipped=True, reason="No dataset directory was provided.")
    if not config.dataset_dir.exists():
        return WerSanityResult(skipped=True, reason=f"Dataset directory does not exist: {config.dataset_dir}")
    if ct2_model_dir is None or not ct2_model_dir.exists():
        return WerSanityResult(skipped=True, reason="Converted CT2 model directory is not available.")
    if config.max_eval_samples <= 0:
        return WerSanityResult(skipped=True, reason="max_eval_samples must be greater than zero.")

    try:
        import jiwer
        from datasets import Audio
        from faster_whisper import WhisperModel

        from .train import load_audiofolder_dataset
    except ImportError as exc:
        return WerSanityResult(skipped=True, reason=f"Missing WER dependency: {exc.name}")

    try:
        dataset = load_audiofolder_dataset(config.dataset_dir)
    except (FileNotFoundError, ValueError) as exc:
        return WerSanityResult(skipped=True, reason=str(exc))

    split = _select_eval_split(dataset, config.max_eval_samples)
    if split is None or len(split) == 0:
        return WerSanityResult(skipped=True, reason="Dataset has no usable evaluation rows.")

    split = split.cast_column("audio", Audio(sampling_rate=16_000))
    sample_count = min(config.max_eval_samples, len(split))
    samples = [split[index] for index in range(sample_count)]

    references = [str(sample.get("text", "")) for sample in samples]
    try:
        peft_predictions = _predict_with_peft(config, samples)
        ct2_predictions = _predict_with_ct2(config, ct2_model_dir, samples, WhisperModel)
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        return WerSanityResult(skipped=True, reason=f"Unable to run model inference for WER: {exc}")

    peft_wer = 100.0 * jiwer.wer(references, peft_predictions)
    ct2_wer = 100.0 * jiwer.wer(references, ct2_predictions)
    return WerSanityResult(
        skipped=False,
        peft_wer=peft_wer,
        ct2_wer=ct2_wer,
        absolute_drift=abs(ct2_wer - peft_wer),
        sample_count=sample_count,
    )


def _predict_with_peft(config: MergeConfig, samples: Iterable[dict[str, Any]]) -> list[str]:
    import torch

    processor = load_processor(config.base_model, task=config.task, language=config.language)
    model = load_peft_model(config)
    device = _resolve_torch_device(config.device)
    model.to(device)
    model.eval()

    predictions: list[str] = []
    with torch.no_grad():
        for sample in samples:
            audio = sample["audio"]
            features = processor.feature_extractor(
                audio["array"],
                sampling_rate=audio["sampling_rate"],
                return_tensors="pt",
            ).input_features.to(device)
            generated_ids = model.generate(
                input_features=features,
                max_length=config.generation_max_length,
            )
            predictions.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0])
    return predictions


def _predict_with_ct2(
    config: MergeConfig,
    ct2_model_dir: Path,
    samples: Iterable[dict[str, Any]],
    whisper_model_cls: Any,
) -> list[str]:
    model = whisper_model_cls(
        str(ct2_model_dir),
        device=config.device,
        compute_type=config.quantization,
    )

    predictions: list[str] = []
    for sample in samples:
        segments, _ = model.transcribe(
            sample["audio"]["array"],
            language=_language_code(config.language),
            task=config.task,
            beam_size=1,
            condition_on_previous_text=False,
        )
        predictions.append("".join(segment.text for segment in segments).strip())
    return predictions


def _select_eval_split(dataset: Any, max_eval_samples: int) -> Any | None:
    for split_name in ("validation", "test", "eval"):
        if split_name in dataset:
            return dataset[split_name]
    if "train" not in dataset:
        return None

    train_split = dataset["train"]
    if len(train_split) <= 1:
        return train_split
    held_out_size = min(max_eval_samples, len(train_split) - 1)
    return train_split.train_test_split(test_size=held_out_size, seed=42)["test"]


def _resolve_torch_device(device: str) -> str:
    if device != "auto":
        return device

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _language_code(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    return {
        "english": "en",
        "thai": "th",
        "japanese": "ja",
    }.get(normalized, normalized or None)
