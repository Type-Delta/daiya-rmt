from __future__ import annotations

import argparse
from pathlib import Path

from .train import TrainingConfig, inspect_dataset, train


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
    train_parser.add_argument("--lora-r", type=int, default=16)
    train_parser.add_argument("--lora-alpha", type=int, default=32)
    train_parser.add_argument("--lora-dropout", type=float, default=0.05)
    train_parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    train_parser.add_argument("--fp16", action="store_true")
    train_parser.add_argument("--bf16", action="store_true")
    train_parser.add_argument("--gradient-checkpointing", action="store_true")
    train_parser.add_argument("--load-best-model-at-end", action="store_true")
    train_parser.add_argument("--push-to-hub", action="store_true")
    train_parser.add_argument("--hub-model-id", default=None)

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


def config_from_args(args: argparse.Namespace) -> TrainingConfig:
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
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=tuple(
            module.strip() for module in args.lora_target_modules.split(",") if module.strip()
        ),
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        load_best_model_at_end=args.load_best_model_at_end,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect":
        inspect_dataset(args.dataset_dir)
        return

    if args.command == "train":
        train(config_from_args(args))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
