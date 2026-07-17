from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from .merge import (
    DEFAULT_BASE_MODEL,
    DEFAULT_QUANTIZATION,
    MergeConfig,
    default_adapter_path,
    default_ct2_output_dir,
    default_merged_output_dir,
    merge_and_convert,
)

if TYPE_CHECKING:
    from .checkpoint_probe import ProbeConfig
    from .train import TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="daiya-whisper-lora",
        description="Fine-tune Whisper with LoRA on the local Daiya audiofolder dataset.",
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Print a short dataset summary.")
    add_dataset_args(inspect_parser)

    train_parser = subparsers.add_parser("train", help="Run LoRA fine-tuning.")
    add_dataset_args(train_parser)
    train_parser.add_argument("--model-name-or-path", default="openai/whisper-medium")
    train_parser.add_argument("--output-dir", type=Path, default=default_output_dir())
    train_parser.add_argument(
        "--language",
        default=None,
        help="Optional global Whisper language prompt or fallback for unmapped metadata rows, e.g. Thai.",
    )
    train_parser.add_argument(
        "--language-policy",
        choices=("metadata", "global", "none"),
        default="metadata",
        help="Use row metadata, one global language, or no language token when building Whisper labels.",
    )
    train_parser.add_argument("--task", default="transcribe")
    train_parser.add_argument("--validation-size", type=float, default=0.05)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--num-train-epochs", type=float, default=3.0)
    train_parser.add_argument("--max-steps", type=int, default=-1)
    train_parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    train_parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    train_parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=100)
    train_parser.add_argument("--eval-steps", type=int, default=250)
    train_parser.add_argument("--save-steps", type=int, default=250)
    train_parser.add_argument("--logging-steps", type=int, default=25)
    train_parser.add_argument("--predict-with-generate", action="store_true")
    train_parser.add_argument("--generation-max-length", type=int, default=225)
    train_parser.add_argument("--preprocessing-num-proc", type=int, default=1)
    train_parser.add_argument("--dataloader-num-workers", type=int, default=0)
    train_parser.add_argument("--max-train-samples", type=int, default=None)
    train_parser.add_argument("--max-eval-samples", type=int, default=None)
    train_parser.add_argument("--max-label-length", type=int, default=448)
    train_parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Optional explicit train/validation/test/benchmark split manifest grouped by source_file/conversation.",
    )
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    train_parser.add_argument("--fp16", action="store_true")
    train_parser.add_argument("--bf16", action="store_true")
    train_parser.add_argument("--load-in-4bit", action="store_true")
    train_parser.add_argument("--load-in-8bit", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--load-best-model-at-end", action="store_true")
    train_parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from the latest checkpoint in --output-dir.",
    )
    train_parser.add_argument("--push-to-hub", action="store_true")
    train_parser.add_argument("--hub-model-id", default=None)
    train_parser.add_argument(
        "--prompt-conditioning",
        action="store_true",
        help="Opt in to decoder prompt conditioning from metadata columns. Defaults stay unprompted.",
    )
    train_parser.add_argument(
        "--prompt-max-tokens",
        type=int,
        default=64,
        help="Maximum prompt body tokens before the transcript label. Transcript tokens keep priority.",
    )
    train_parser.add_argument(
        "--prompt-fields",
        default="context_before",
        help="Comma-separated metadata fields to build the prompt from; defaults to context_before.",
    )
    train_parser.add_argument(
        "--prompt-full-context",
        dest="prompt_terms_only",
        action="store_false",
        help="Use full selected context fields instead of extracting only Terms: fragments.",
    )
    train_parser.set_defaults(prompt_terms_only=True)
    train_parser.add_argument(
        "--prompt-allow-future-context",
        action="store_true",
        help="Allow context_after/right-context fields. Use only for explicit offline-labeling experiments.",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge a Whisper LoRA adapter, convert it to CTranslate2, and optionally run a WER gate.",
    )
    merge_parser.add_argument("--adapter-path", type=Path, default=default_adapter_path())
    merge_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    merge_parser.add_argument("--merged-output-dir", type=Path, default=default_merged_output_dir())
    merge_parser.add_argument("--ct2-output-dir", type=Path, default=default_ct2_output_dir())
    merge_parser.add_argument("--quantization", default=DEFAULT_QUANTIZATION)
    merge_parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    merge_parser.add_argument("--max-eval-samples", type=int, default=8)
    merge_parser.add_argument("--allowed-absolute-wer-drift", type=float, default=1.0)
    merge_parser.add_argument("--skip-convert", action="store_true")
    merge_parser.add_argument("--skip-wer", action="store_true")
    merge_parser.add_argument("--no-force", dest="force", action="store_false")
    merge_parser.set_defaults(force=True)
    merge_parser.add_argument("--device", default="auto", help="Torch/faster-whisper device, e.g. auto, cuda, cpu.")
    merge_parser.add_argument("--task", default="transcribe")
    merge_parser.add_argument("--language", default=None)
    merge_parser.add_argument("--generation-max-length", type=int, default=225)

    probe_parser = subparsers.add_parser(
        "probe-checkpoints",
        help="Score generated text quality for LoRA checkpoints and select the best generated-CER checkpoint.",
    )
    probe_parser.add_argument("--run-dir", type=Path, required=True, help="Training run containing checkpoint-* dirs.")
    probe_parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    probe_parser.add_argument("--dataset-dir", type=Path, default=default_dataset_dir())
    probe_parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_checkpoint_probe_output_dir(),
        help="Directory for checkpoint probe details JSONL and summary JSON.",
    )
    probe_parser.add_argument(
        "--candidate",
        dest="candidates",
        action="append",
        type=Path,
        default=[],
        help="Explicit adapter candidate to score. Can be repeated.",
    )
    probe_parser.add_argument("--no-checkpoints", dest="include_checkpoints", action="store_false")
    probe_parser.set_defaults(include_checkpoints=True)
    probe_parser.add_argument("--no-final", dest="include_final", action="store_false")
    probe_parser.set_defaults(include_final=True)
    probe_parser.add_argument("--split", default=None, help="Dataset split to probe; defaults to validation/test/benchmark/eval/train.")
    probe_parser.add_argument("--split-manifest", type=Path, default=None)
    probe_parser.add_argument(
        "--selector-manifest",
        type=Path,
        default=None,
        help="Frozen JSONL/plain sample-ID allowlist for exact checkpoint-gate rows.",
    )
    probe_parser.add_argument("--max-samples", type=int, default=32)
    probe_parser.add_argument("--min-short-samples", type=int, default=8)
    probe_parser.add_argument("--min-technical-term-samples", type=int, default=8)
    probe_parser.add_argument("--short-utterance-seconds", type=float, default=3.0)
    probe_parser.add_argument(
        "--primary-metric",
        choices=(
            "micro_cer",
            "mean_cer",
            "micro_cer_no_space",
            "mean_cer_no_space",
            "micro_wer_like",
            "mean_wer_like",
        ),
        default="micro_cer",
    )
    probe_parser.add_argument(
        "--generation-failure-policy",
        choices=("raise", "eval-loss"),
        default="raise",
        help="Generation gate failure behavior. eval-loss fallback is explicit and recorded in output.",
    )
    probe_parser.add_argument("--device", default="auto")
    probe_parser.add_argument("--fp16", action="store_true")
    probe_parser.add_argument("--bf16", action="store_true")
    probe_parser.add_argument("--load-in-4bit", action="store_true")
    probe_parser.add_argument("--task", default="transcribe")
    probe_parser.add_argument("--language", default=None)
    probe_parser.add_argument(
        "--language-policy",
        choices=("metadata", "global", "none"),
        default="metadata",
        help="Use row metadata, one global language, or no language hint during checkpoint generation.",
    )
    probe_parser.add_argument("--generation-max-length", type=int, default=225)
    probe_parser.add_argument(
        "--prompt-strategy",
        choices=("isolated", "rolling-initial-prompt"),
        default="isolated",
        help="Runtime generation prompt strategy for checkpoint probing.",
    )
    probe_parser.add_argument("--prompt-max-tokens", type=int, default=64)
    probe_parser.add_argument("--prompt-fields", default="context_before")
    probe_parser.add_argument("--prompt-full-context", dest="prompt_terms_only", action="store_false")
    probe_parser.set_defaults(prompt_terms_only=True)
    probe_parser.add_argument("--prompt-allow-future-context", action="store_true")
    probe_parser.add_argument(
        "--no-prompt-row-context",
        dest="prompt_include_row_context",
        action="store_false",
        help="For rolling probes, use prior hypotheses only (no per-row context_before terms).",
    )
    probe_parser.set_defaults(prompt_include_row_context=True)
    probe_parser.add_argument("--rolling-prompt-turns", type=int, default=3)
    probe_parser.add_argument("--rolling-prompt-chars", type=int, default=512)

    return parser


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir(),
        help="Path to the Hugging Face audiofolder dataset.",
    )


def default_dataset_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "dataset" / "hf_datasets" / "whisper"


def default_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "whisper-medium-lora"


def default_checkpoint_probe_output_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "runs" / "checkpoint_probes"


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
    from .train import TrainingConfig

    return TrainingConfig(
        dataset_dir=args.dataset_dir,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        language=args.language,
        language_policy=args.language_policy,
        task=args.task,
        validation_size=args.validation_size,
        seed=args.seed,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        preprocessing_num_proc=args.preprocessing_num_proc,
        dataloader_num_workers=args.dataloader_num_workers,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
        max_label_length=args.max_label_length,
        split_manifest=args.split_manifest,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(
            module.strip() for module in args.lora_target_modules.split(",") if module.strip()
        ),
        fp16=args.fp16,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        load_in_8bit=args.load_in_8bit,
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end=args.load_best_model_at_end,
        resume_from_checkpoint=args.resume_from_checkpoint,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        prompt_conditioning=args.prompt_conditioning,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_fields=tuple(field.strip() for field in args.prompt_fields.split(",") if field.strip()),
        prompt_terms_only=args.prompt_terms_only,
        prompt_allow_future_context=args.prompt_allow_future_context,
    )


def merge_config_from_args(args: argparse.Namespace) -> MergeConfig:
    return MergeConfig(
        adapter_path=args.adapter_path,
        base_model=args.base_model,
        merged_output_dir=args.merged_output_dir,
        ct2_output_dir=args.ct2_output_dir,
        quantization=args.quantization,
        dataset_dir=args.dataset_dir,
        max_eval_samples=args.max_eval_samples,
        allowed_absolute_wer_drift=args.allowed_absolute_wer_drift,
        skip_convert=args.skip_convert,
        skip_wer=args.skip_wer,
        force=args.force,
        device=args.device,
        task=args.task,
        language=args.language,
        generation_max_length=args.generation_max_length,
    )


def probe_config_from_args(args: argparse.Namespace) -> ProbeConfig:
    from .checkpoint_probe import ProbeConfig

    return ProbeConfig(
        run_dir=args.run_dir,
        base_model=args.base_model,
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        candidates=tuple(args.candidates),
        include_checkpoints=args.include_checkpoints,
        include_final=args.include_final,
        split=args.split,
        split_manifest=args.split_manifest,
        selector_manifest=args.selector_manifest,
        max_samples=args.max_samples,
        min_short_samples=args.min_short_samples,
        min_technical_term_samples=args.min_technical_term_samples,
        short_utterance_seconds=args.short_utterance_seconds,
        primary_metric=args.primary_metric,
        generation_failure_policy=args.generation_failure_policy,
        device=args.device,
        fp16=args.fp16,
        bf16=args.bf16,
        load_in_4bit=args.load_in_4bit,
        task=args.task,
        language=args.language,
        language_policy=args.language_policy,
        generation_max_length=args.generation_max_length,
        prompt_strategy=args.prompt_strategy,
        prompt_max_tokens=args.prompt_max_tokens,
        prompt_fields=tuple(field.strip() for field in args.prompt_fields.split(",") if field.strip()),
        prompt_terms_only=args.prompt_terms_only,
        prompt_allow_future_context=args.prompt_allow_future_context,
        prompt_include_row_context=args.prompt_include_row_context,
        rolling_prompt_turns=args.rolling_prompt_turns,
        rolling_prompt_chars=args.rolling_prompt_chars,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        from .train import inspect_dataset

        inspect_dataset(args.dataset_dir)
        return

    if args.command == "train":
        from .train import train

        train(config_from_args(args))
        return

    if args.command == "merge":
        result = merge_and_convert(merge_config_from_args(args))
        print(f"Merged model: {result.merged_model_dir}")
        if result.ct2_model_dir is not None:
            print(f"CT2 model: {result.ct2_model_dir}")
        if result.wer is not None:
            if result.wer.skipped:
                print(f"WER sanity check skipped: {result.wer.reason}")
            else:
                print(
                    "WER sanity check: "
                    f"PEFT={result.wer.peft_wer:.2f}, "
                    f"CT2={result.wer.ct2_wer:.2f}, "
                    f"drift={result.wer.absolute_drift:.2f}, "
                    f"samples={result.wer.sample_count}"
                )
        return

    if args.command == "probe-checkpoints":
        from .checkpoint_probe import run_probe

        result = run_probe(probe_config_from_args(args))
        selected = result["selected_checkpoint"]
        if result["selection_mode"] == "eval_loss_fallback":
            print(f"Selected checkpoint: {selected['name']} (eval_loss={selected['eval_loss']})")
        else:
            print(
                "Selected checkpoint: "
                f"{selected['name']} ({result['primary_metric']}={selected['overall'][result['primary_metric']]})"
            )
        return

    parser.print_help()


if __name__ == "__main__":
    main()
