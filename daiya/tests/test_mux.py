from __future__ import annotations

import unittest

from daiya.mux import ASRSegment, DiarizationTurn, TranscriptMultiplexer


class TranscriptMultiplexerTest(unittest.TestCase):
    def test_partial_final_and_later_relabel(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=False))

        partial = mux.ingest_asr(ASRSegment(0.2, 1.2, "hello world"))
        self.assertEqual(partial[0].type, "transcript.partial")
        self.assertEqual(partial[0].segment.speaker_id, "SPEAKER_001")

        final = mux.ingest_diarization(
            DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=True, version=2)
        )
        self.assertEqual([event.type for event in final], ["transcript.final"])
        self.assertTrue(final[0].segment.final)

        update = mux.ingest_diarization(
            DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_002", final=True, version=3)
        )
        self.assertEqual([event.type for event in update], ["transcript.update"])
        self.assertEqual(update[0].segment.speaker_id, "SPEAKER_002")

    def test_speaker_uses_max_temporal_overlap(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))
        mux.ingest_diarization(DiarizationTurn("b", 1.0, 3.0, "B"))

        event = mux.ingest_asr(ASRSegment(0.5, 2.5, "mixed"))[0]
        self.assertEqual(event.segment.speaker_id, "B")

    def test_asr_after_commit_emits_partial_then_final(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=False))
        mux.ingest_diarization(DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=True))

        events = mux.ingest_asr(ASRSegment(0.2, 1.2, "late asr"))
        self.assertEqual([event.type for event in events], ["transcript.partial", "transcript.final"])


if __name__ == "__main__":
    unittest.main()
