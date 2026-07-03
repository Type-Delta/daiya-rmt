from __future__ import annotations

from dataclasses import dataclass

from .mux import CorrectionUpdate, TranscriptSegment


@dataclass
class NoOpCorrectionStage:
    """Future LLM correction hook; v0 deliberately emits no corrections."""

    def review(self, _segment: TranscriptSegment) -> list[CorrectionUpdate]:
        return []
