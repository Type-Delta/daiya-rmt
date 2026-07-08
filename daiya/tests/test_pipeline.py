from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from daiya.audio import SAMPLE_RATE, PCMChunk
from daiya.mux import ASRSegment
from daiya.pipeline import PipelineConfig, StreamingPipeline


def _bypass_pipeline() -> StreamingPipeline:
    return StreamingPipeline(PipelineConfig(enable_asr=False, enable_diarization=False))


class EngineToggleTests(unittest.TestCase):
    def test_asr_off_emits_textless_speaker_events(self) -> None:
        pipeline = StreamingPipeline(
            PipelineConfig(
                enable_asr=False,
                diarization_backend="null",
                commit_delay_seconds=0.0,
            )
        )
        self.assertIsNone(pipeline.segmenter)

        chunk = PCMChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), start_time=0.0)
        payloads = pipeline.accept_chunk(chunk) + pipeline.flush()

        ticks = [p for p in payloads if p["type"] == "tick"]
        turns = [p for p in payloads if p["type"] != "tick"]
        self.assertEqual(len(ticks), 1)
        self.assertEqual(ticks[0]["time"], chunk.end_time)
        self.assertTrue(turns)
        for payload in turns:
            self.assertIn(payload["type"], ("transcript.partial", "transcript.final"))
            self.assertEqual(payload["source"], "diarizer")
            self.assertEqual(payload["text"], "")
            self.assertTrue(payload["speaker"])

    def test_diarization_off_finalizes_asr_segments_immediately(self) -> None:
        pipeline = _bypass_pipeline()
        self.assertIsNone(pipeline.diarizer)

        class StubASR:
            def transcribe_utterance(self, utterance: object, **_kwargs: object) -> list[ASRSegment]:
                return [ASRSegment(start=0.0, end=1.5, text="hello world")]

        pipeline.asr = StubASR()  # type: ignore[assignment]
        payloads = pipeline._transcribe_utterance(object())

        self.assertEqual(len(payloads), 1)
        payload = payloads[0]
        self.assertEqual(payload["type"], "transcript.final")
        self.assertEqual(payload["text"], "hello world")
        self.assertIsNone(payload["speaker"])
        self.assertTrue(payload["final"])

    def test_both_off_produces_only_ticks(self) -> None:
        pipeline = _bypass_pipeline()
        chunk = PCMChunk(samples=np.zeros(SAMPLE_RATE, dtype=np.float32), start_time=0.0)
        self.assertEqual(pipeline.accept_chunk(chunk), [{"type": "tick", "time": chunk.end_time}])
        self.assertEqual(pipeline.flush(), [])


class SegmenterConfigTests(unittest.TestCase):
    def test_pipeline_passes_segmenter_configuration(self) -> None:
        with patch("daiya.pipeline.create_utterance_segmenter") as create_segmenter:
            create_segmenter.return_value = None
            StreamingPipeline(
                PipelineConfig(
                    enable_asr=True,
                    enable_diarization=False,
                    segmenter_backend="auto",
                    vad_threshold=0.42,
                    vad_min_speech_seconds=0.11,
                    vad_min_silence_seconds=0.33,
                    vad_speech_padding_seconds=0.07,
                    utterance_cap_seconds=3.5,
                )
            )

        create_segmenter.assert_called_once_with(
            backend="auto",
            threshold=0.42,
            min_speech_seconds=0.11,
            min_silence_seconds=0.33,
            speech_padding_seconds=0.07,
            max_utterance_seconds=3.5,
        )


if __name__ == "__main__":
    unittest.main()
