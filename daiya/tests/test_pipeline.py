from __future__ import annotations

import unittest

import numpy as np

from daiya.asr import Utterance
from daiya.audio import SAMPLE_RATE, PCMChunk
from daiya.mux import ASRSegment, WordTimestamp
from daiya.pipeline import ASRPromptMemory, PipelineConfig, StreamingPipeline


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


class ContextAwareASRTests(unittest.TestCase):
    def test_only_rolling_prompt_memory_is_enabled_by_default(self) -> None:
        config = PipelineConfig()

        self.assertTrue(config.asr_prompt_memory_enabled)
        self.assertFalse(config.asr_left_context_enabled)
        self.assertFalse(config.asr_delayed_correction_enabled)
        self.assertFalse(config.asr_tiny_utterance_merge_enabled)

    def test_prompt_memory_filters_control_labels_from_terms(self) -> None:
        memory = ASRPromptMemory(static_prompt="Terms: AI, Topic: founder project")

        memory.remember("Previous transcript: relation business first successful Topic Terms")

        terms = memory.terms()
        self.assertIn("AI", terms)
        self.assertIn("founder", terms)
        self.assertIn("project", terms)
        self.assertIn("relation", terms)
        self.assertIn("business", terms)
        self.assertNotIn("Terms", terms)
        self.assertNotIn("Topic", terms)
        self.assertNotIn("first", terms)
        self.assertNotIn("successful", terms)

    def test_prompt_memory_preserves_terms_when_transcript_is_long(self) -> None:
        memory = ASRPromptMemory(
            static_prompt="the the the the the the the the the the the the",
            tail_chars=500,
            max_prompt_chars=130,
            max_terms=5,
        )

        memory.remember("AI relation business founder project " + ("ครับ " * 80))

        prompt = memory.build_prompt() or ""
        self.assertLessEqual(len(prompt), 130)
        self.assertIn("Terms:", prompt)
        self.assertIn("AI", prompt)
        self.assertIn("business", prompt)
        self.assertIn("founder", prompt)
        self.assertIn("project", prompt)
        self.assertIn("relation", prompt)
        self.assertIn("Recent transcript:", prompt)
        self.assertNotIn("Use this only as context", prompt)

    def test_rolling_prompt_memory_is_passed_to_next_utterance(self) -> None:
        pipeline = StreamingPipeline(
            PipelineConfig(
                enable_asr=False,
                enable_diarization=False,
                initial_prompt="Daiya technical meeting",
                asr_left_context_enabled=False,
                asr_delayed_correction_enabled=False,
            )
        )

        class StubASR:
            def __init__(self) -> None:
                self.prompts: list[str | None] = []

            def transcribe_utterance(self, utterance: object, **kwargs: object) -> list[ASRSegment]:
                self.prompts.append(kwargs.get("initial_prompt"))  # type: ignore[arg-type]
                return [ASRSegment(start=0.0, end=1.0, text="relation status")]

        stub = StubASR()
        pipeline.asr = stub  # type: ignore[assignment]
        utterance = Utterance(samples=np.ones(SAMPLE_RATE, dtype=np.float32), start=0.0, end=1.0)

        pipeline._transcribe_utterance(utterance)
        pipeline._transcribe_utterance(utterance)

        self.assertIn("Daiya technical meeting", stub.prompts[0] or "")
        self.assertIn("relation status", stub.prompts[1] or "")
        self.assertIn("Terms:", stub.prompts[1] or "")

    def test_tiny_utterance_waits_and_merges_nearby_context(self) -> None:
        pipeline = StreamingPipeline(
            PipelineConfig(
                enable_asr=False,
                enable_diarization=False,
                asr_left_context_enabled=False,
                asr_delayed_correction_enabled=False,
                asr_tiny_utterance_merge_enabled=True,
                asr_tiny_utterance_seconds=0.55,
                asr_tiny_utterance_max_gap_seconds=0.25,
            )
        )

        class StubASR:
            def __init__(self) -> None:
                self.utterances: list[Utterance] = []

            def transcribe_utterance(self, utterance: Utterance, **_kwargs: object) -> list[ASRSegment]:
                self.utterances.append(utterance)
                return [ASRSegment(start=utterance.start, end=utterance.end, text="merged")]

        stub = StubASR()
        pipeline.asr = stub  # type: ignore[assignment]

        tiny = Utterance(samples=np.ones(int(0.4 * SAMPLE_RATE), dtype=np.float32), start=0.0, end=0.4)
        next_utterance = Utterance(
            samples=np.ones(int(0.4 * SAMPLE_RATE), dtype=np.float32),
            start=0.6,
            end=1.0,
        )

        self.assertEqual(pipeline._handle_utterance(tiny), [])
        payloads = pipeline._handle_utterance(next_utterance)

        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(stub.utterances), 1)
        self.assertEqual(stub.utterances[0].start, 0.0)
        self.assertEqual(stub.utterances[0].end, 1.0)

    def test_left_context_retry_keeps_only_current_window_words(self) -> None:
        pipeline = StreamingPipeline(
            PipelineConfig(
                enable_asr=False,
                enable_diarization=False,
                asr_left_context_enabled=True,
                asr_left_context_seconds=2.0,
                asr_left_context_short_utterance_seconds=1.0,
                asr_delayed_correction_enabled=False,
                asr_tiny_utterance_merge_enabled=False,
            )
        )
        pipeline._remember_audio(
            PCMChunk(samples=np.ones(2 * SAMPLE_RATE, dtype=np.float32), start_time=0.0)
        )

        class StubASR:
            def __init__(self) -> None:
                self.calls = 0

            def transcribe_utterance(self, utterance: Utterance, **_kwargs: object) -> list[ASRSegment]:
                self.calls += 1
                if self.calls == 1:
                    return [ASRSegment(start=utterance.start, end=utterance.end, text="อ่า", confidence=-2.0)]
                return [
                    ASRSegment(
                        start=0.4,
                        end=1.0,
                        text="previous",
                        words=(WordTimestamp("previous", 0.4, 1.0),),
                    ),
                    ASRSegment(
                        start=2.1,
                        end=2.5,
                        text="percent",
                        words=(WordTimestamp("percent", 2.1, 2.5),),
                    ),
                ]

        stub = StubASR()
        pipeline.asr = stub  # type: ignore[assignment]
        current = Utterance(samples=np.ones(int(0.6 * SAMPLE_RATE), dtype=np.float32), start=2.0, end=2.6)

        payloads = pipeline._transcribe_utterance(current)

        self.assertEqual(stub.calls, 2)
        self.assertEqual(payloads[0]["text"], "percent")


if __name__ == "__main__":
    unittest.main()
