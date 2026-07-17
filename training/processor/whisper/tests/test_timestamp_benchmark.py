from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_timestamp_ownership import Segment, _intersect, _summary, _v2_segments  # noqa: E402
from daiya_whisper_pipeline.types import Interval  # noqa: E402


def test_reference_intersection_unions_overlapping_windows() -> None:
    # The wall-clock-v2 baseline may intentionally overlap labeler context.
    # Coverage cannot exceed the human-reference union merely because two
    # baseline windows both cover the same reference interval.
    assert _intersect(
        [Interval(0.0, 8.0), Interval(4.0, 10.0)],
        [Interval(2.0, 9.0)],
    ) == 7.0


def test_unprotected_boundary_count_includes_fallback_handoffs() -> None:
    summary = _summary(
        [
            Segment(0.0, 5.0, 0.0, False, True, "continuous_speech_fallback", 0.0),
            Segment(5.0, 10.0, 4.0, False, True, "source_end", 0.0),
        ],
        [Interval(2.0, 8.0)],
        0.25,
    )

    assert summary["unprotected_boundaries_inside_reference_speech"] == 1
    assert summary["duplicated_eligible_training_seconds"] == 0.0


def test_wall_clock_v2_context_is_counted_once_and_handoffs_are_not_lost() -> None:
    segments = _v2_segments(
        [Interval(0.0, 80.0)], target=18.0, maximum=25.0, minimum_silence=0.5, search=4.0, context=1.0
    )
    summary = _summary(segments, [], 0.25)

    assert [item.labeling_start for item in segments] == [item.start for item in segments]
    assert summary["fallback_handoffs"] == 3
    assert summary["fallback_rows"] == 4
    assert summary["duplicated_labeling_audio_seconds"] == 3.0
