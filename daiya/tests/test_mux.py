from __future__ import annotations

import unittest

from daiya.mux import ASRSegment, CorrectionUpdate, DiarizationTurn, TranscriptMultiplexer, WordTimestamp


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

    def test_no_word_fallback_preserves_any_positive_segment_overlap(self) -> None:
        mux = TranscriptMultiplexer(min_word_overlap_seconds=0.2)
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))

        event = mux.ingest_asr(ASRSegment(0.99, 1.5, "tail"))[0]

        self.assertEqual(event.segment.speaker_id, "A")

    def test_word_speakers_override_whole_segment_overlap(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.2, "A"))
        mux.ingest_diarization(DiarizationTurn("b", 1.2, 2.0, "B"))

        event = mux.ingest_asr(
            ASRSegment(
                0.0,
                2.0,
                "okay ครับ",
                words=(
                    WordTimestamp("okay", 0.1, 0.3),
                    WordTimestamp("ครับ", 1.3, 1.75),
                ),
            )
        )[0]

        self.assertEqual(event.segment.speaker_id, "B")
        self.assertEqual([word.speaker_id for word in event.segment.words], ["A", "B"])

    def test_word_speaker_serializes_as_wire_speaker_field(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))

        event = mux.ingest_asr(
            ASRSegment(0.0, 1.0, "hello", words=(WordTimestamp("hello", 0.1, 0.5),))
        )[0]
        payload = event.to_dict()

        self.assertEqual(payload["words"][0]["speaker"], "A")  # type: ignore[index]
        self.assertNotIn("speaker_id", payload["words"][0])  # type: ignore[index]

    def test_text_only_correction_preserves_existing_speaker(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))
        segment = mux.ingest_asr(ASRSegment(0.0, 1.0, "helo"))[0].segment
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "B", version=2))

        update = mux.apply_correction(CorrectionUpdate(segment.segment_id, text="hello"))

        self.assertEqual(update[0].segment.text, "hello")
        self.assertEqual(update[0].segment.speaker_id, "A")

    def test_word_correction_reassigns_word_and_segment_speakers(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))
        mux.ingest_diarization(DiarizationTurn("b", 1.0, 2.0, "B"))
        segment = mux.ingest_asr(ASRSegment(0.0, 2.0, "rough"))[0].segment

        update = mux.apply_correction(
            CorrectionUpdate(
                segment.segment_id,
                text="hello ครับ",
                words=(
                    WordTimestamp("hello", 0.1, 0.4),
                    WordTimestamp("ครับ", 1.1, 1.7),
                ),
            )
        )

        self.assertEqual(update[0].segment.speaker_id, "B")
        self.assertEqual([word.speaker_id for word in update[0].segment.words], ["A", "B"])

    def test_nearest_turn_collar_labels_short_boundary_backchannel(self) -> None:
        mux = TranscriptMultiplexer(nearest_turn_collar_seconds=0.12)
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A"))

        event = mux.ingest_asr(
            ASRSegment(1.03, 1.08, "อ๋อ", words=(WordTimestamp("อ๋อ", 1.03, 1.08),))
        )[0]

        self.assertEqual(event.segment.speaker_id, "A")
        self.assertEqual(event.segment.words[0].speaker_id, "A")

    def test_nearest_turn_collar_labels_short_leading_backchannel(self) -> None:
        mux = TranscriptMultiplexer(nearest_turn_collar_seconds=0.12)
        mux.ingest_diarization(DiarizationTurn("a", 1.0, 2.0, "A"))

        event = mux.ingest_asr(
            ASRSegment(0.92, 0.97, "อ๋อ", words=(WordTimestamp("อ๋อ", 0.92, 0.97),))
        )[0]

        self.assertEqual(event.segment.speaker_id, "A")
        self.assertEqual(event.segment.words[0].speaker_id, "A")

    def test_finalization_reassigns_words_from_committed_turns(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("draft", 0.0, 2.0, "A"))
        mux.ingest_asr(
            ASRSegment(
                0.0,
                2.0,
                "hello ครับ",
                words=(
                    WordTimestamp("hello", 0.1, 0.9),
                    WordTimestamp("ครับ", 1.1, 1.6),
                ),
            )
        )

        self.assertEqual(mux.segments[0].speaker_id, "A")
        self.assertEqual([word.speaker_id for word in mux.segments[0].words], ["A", "A"])

        self.assertEqual(
            mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A", final=True)),
            [],
        )
        final = mux.ingest_diarization(DiarizationTurn("b", 1.0, 2.0, "B", final=True))

        self.assertEqual([event.type for event in final], ["transcript.final"])
        self.assertEqual(final[0].segment.speaker_id, "A")
        self.assertEqual([word.speaker_id for word in final[0].segment.words], ["A", "B"])

    def test_final_diarization_update_emits_when_only_word_speaker_changes(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("draft", 0.0, 2.0, "A"))
        mux.ingest_asr(
            ASRSegment(
                0.0,
                2.0,
                "hello ครับ",
                words=(
                    WordTimestamp("hello", 0.1, 0.9),
                    WordTimestamp("ครับ", 1.1, 1.6),
                ),
            )
        )
        mux.ingest_diarization(DiarizationTurn("a", 0.0, 1.0, "A", final=True))
        mux.ingest_diarization(DiarizationTurn("b", 1.0, 2.0, "B", final=True))

        update = mux.ingest_diarization(DiarizationTurn("b", 1.0, 2.0, "A", final=True, version=2))

        self.assertEqual([event.type for event in update], ["transcript.update"])
        self.assertEqual(update[0].segment.speaker_id, "A")
        self.assertEqual(update[0].previous.speaker_id, "A")
        self.assertEqual([word.speaker_id for word in update[0].previous.words], ["A", "B"])
        self.assertEqual([word.speaker_id for word in update[0].segment.words], ["A", "A"])

    def test_asr_after_commit_emits_partial_then_final(self) -> None:
        mux = TranscriptMultiplexer()
        mux.ingest_diarization(DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=False))
        mux.ingest_diarization(DiarizationTurn("turn_1", 0.0, 2.0, "SPEAKER_001", final=True))

        events = mux.ingest_asr(ASRSegment(0.2, 1.2, "late asr"))
        self.assertEqual([event.type for event in events], ["transcript.partial", "transcript.final"])


if __name__ == "__main__":
    unittest.main()
