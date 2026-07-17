from __future__ import annotations

import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Label raw audio with an audio-capable LLM.")
    parser.add_argument("--env-file", type=Path, help="Path to a .env file. Defaults to this processor's .env.")
    parser.add_argument("--input-dir", type=Path, help="Override DAIYA_INPUT_DIR.")
    parser.add_argument("--output-dir", type=Path, help="Override DAIYA_OUTPUT_DIR.")
    parser.add_argument("--work-dir", type=Path, help="Override DAIYA_WORK_DIR.")
    parser.add_argument(
        "--no-overlap-filter",
        action="store_true",
        help="Skip pyannote overlap detection (overlap is preserved in audio by default).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig.load(args.env_file)

    updates = {}
    if args.input_dir:
        updates["input_dir"] = args.input_dir.resolve()
    if args.output_dir:
        updates["output_dir"] = args.output_dir.resolve()
    if args.work_dir:
        updates["work_dir"] = args.work_dir.resolve()
    if args.no_overlap_filter:
        updates["enable_overlap_filter"] = False
    if updates:
        from dataclasses import replace

        config = replace(config, **updates)

    run_pipeline(config)


if __name__ == "__main__":
    main()
