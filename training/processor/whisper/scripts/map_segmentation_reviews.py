from __future__ import annotations

import argparse
from pathlib import Path

from daiya_whisper_pipeline.review_mapping import build_mapping_report, write_mapping_report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Conservatively map labels/reviews between segmentation generations without copying decisions."
    )
    parser.add_argument("old_metadata", type=Path)
    parser.add_argument("new_metadata", type=Path)
    parser.add_argument("--old-reviews", type=Path)
    parser.add_argument("--old-audio-root", type=Path)
    parser.add_argument("--new-audio-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    report = build_mapping_report(
        args.old_metadata,
        args.new_metadata,
        old_reviews=args.old_reviews,
        old_audio_root=args.old_audio_root,
        new_audio_root=args.new_audio_root,
    )
    write_mapping_report(report, args.output)
    print(f"Wrote {args.output}")
    print(report["summary"])


if __name__ == "__main__":
    main()
