from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from daiya.asr import (
    ASRUnavailableError,
    EnergyUtteranceSegmenter,
    FasterWhisperASR,
    SileroUtteranceSegmenter,
    Utterance,
    create_utterance_segmenter,
    is_low_confidence_segment,
    low_confidence_words,
)
from daiya.audio import SAMPLE_RATE, PCMChunk
from daiya.mux import ASRSegment, WordTimestamp


class FakeWhisperModel:
    def __init__(self, segments: list[object]) -> None:
        self.segments = segments
        self.calls: list[dict[str, object]] = []

    def transcribe(self, audio: np.ndarray, **kwargs: object) -> tuple[list[object], object]:
        self.calls.append({"audio": audio, **kwargs})
        return self.segments, object()


def _asr_with_model(model: FakeWhisperModel, *, initial_prompt: str | None = "default prompt") -> FasterWhisperASR:
    asr = FasterWhisperASR.__new__(FasterWhisperASR)
    asr.model_path = "fake-model"
    asr.language = "th"
    asr.initial_prompt = initial_prompt
    asr._model = model
    return asr


def _utterance(duration_seconds: float = 1.0) -> Utterance:
    samples = np.ones(int(SAMPLE_RATE * duration_seconds), dtype=np.float32)
    return Utterance(samples=samples, start=10.0, end=10.0 + duration_seconds)


class FakeVADIterator:
    def __init__(self, events: dict[int, dict[str, int]] | None = None) -> None:
        self.events = events or {}
        self.calls: list[np.ndarray] = []
        self.reset_count = 0

    def __call__(
        self,
        samples: np.ndarray,
        *,
        return_seconds: bool = False,
    ) -> dict[str, int] | None:
        if return_seconds:
            raise AssertionError("runtime must request sample timestamps")
        self.calls.append(np.asarray(samples).copy())
        return self.events.get(len(self.calls))

    def reset_states(self) -> None:
        self.reset_count += 1
        self.calls = []


class FasterWhisperASRTest(unittest.TestCase):
    def test_per_call_initial_prompt_honors_empty_string_override(self) -> None:
        model = FakeWhisperModel([])
        asr = _asr_with_model(model)

        asr.transcribe_utterance(_utterance(), initial_prompt="")

        self.assertEqual(model.calls[0]["initial_prompt"], "")

    def test_per_call_initial_prompt_defaults_only_when_none(self) -> None:
        model = FakeWhisperModel([])
        asr = _asr_with_model(model)

        asr.transcribe_utterance(_utterance())

        self.assertEqual(model.calls[0]["initial_prompt"], "default prompt")

    def test_left_context_audio_is_prepended_and_timestamps_are_clipped(self) -> None:
        segments = [
            SimpleNamespace(start=0.1, end=0.8, text="context only", words=[]),
            SimpleNamespace(
                start=0.8,
                end=1.4,
                text="crossing in",
                words=[
                    SimpleNamespace(word="old", start=0.2, end=0.6, probability=0.9),
                    SimpleNamespace(word="now", start=0.95, end=1.2, probability=0.8),
                ],
            ),
            SimpleNamespace(
                start=2.7,
                end=3.2,
                text="crossing out",
                words=[SimpleNamespace(word="tail", start=2.8, end=3.1, probability=0.7)],
            ),
            SimpleNamespace(start=3.2, end=3.4, text="after", words=[]),
        ]
        model = FakeWhisperModel(segments)
        asr = _asr_with_model(model)
        utterance = _utterance(duration_seconds=2.0)
        left_context = np.zeros(SAMPLE_RATE, dtype=np.float32)

        decoded = asr.transcribe_utterance(utterance, left_context_samples=left_context)

        self.assertEqual(model.calls[0]["audio"].shape, (SAMPLE_RATE * 3,))
        self.assertEqual(len(decoded), 2)
        self.assertEqual((decoded[0].start, decoded[0].end), (10.0, 10.4))
        self.assertEqual((decoded[1].start, decoded[1].end), (11.7, 12.0))
        self.assertEqual(len(decoded[0].words), 1)
        self.assertEqual(decoded[0].words[0].word, "now")
        self.assertEqual((decoded[0].words[0].start, decoded[0].words[0].end), (10.0, 10.2))
        self.assertEqual((decoded[1].words[0].start, decoded[1].words[0].end), (11.8, 12.0))

    def test_low_confidence_helpers_flag_segment_and_words(self) -> None:
        segment = ASRSegment(
            start=0.0,
            end=1.0,
            text="hello",
            confidence=-0.2,
            words=(
                WordTimestamp("hello", 0.0, 0.5, probability=0.9),
                WordTimestamp("world", 0.5, 1.0, probability=0.2),
            ),
        )

        self.assertTrue(is_low_confidence_segment(segment))
        self.assertEqual([word.word for word in low_confidence_words(segment)], ["world"])


class UtteranceSegmenterTest(unittest.TestCase):
    def test_factory_defaults_to_energy_segmenter(self) -> None:
        self.assertIsInstance(create_utterance_segmenter(), EnergyUtteranceSegmenter)

    def test_factory_falls_back_to_energy_when_silero_is_unavailable(self) -> None:
        with patch(
            "daiya.asr.SileroUtteranceSegmenter",
            side_effect=ASRUnavailableError("missing torch"),
        ):
            segmenter = create_utterance_segmenter(backend="auto")

        self.assertIsInstance(segmenter, EnergyUtteranceSegmenter)

    def test_factory_prefer_silero_falls_back_to_energy_when_unavailable(self) -> None:
        with patch(
            "daiya.asr.SileroUtteranceSegmenter",
            side_effect=ASRUnavailableError("missing torch"),
        ):
            segmenter = create_utterance_segmenter(prefer_silero=True)

        self.assertIsInstance(segmenter, EnergyUtteranceSegmenter)

    def test_silero_events_and_timestamps_span_small_pcm_chunks(self) -> None:
        vad = FakeVADIterator({2: {"start": 640}, 4: {"end": 1792}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.01,
            speech_padding_seconds=0.008,
            max_utterance_seconds=2.0,
            vad_iterator=vad,
        )
        samples = np.arange(2048, dtype=np.float32)
        completed: list[Utterance] = []
        for index, offset in enumerate(range(0, samples.size, 160)):
            completed.extend(
                segmenter.accept(
                    PCMChunk(
                        samples[offset : offset + 160],
                        start_time=10.0 + (offset / SAMPLE_RATE),
                        index=index,
                    )
                )
            )

        self.assertEqual(len(completed), 1)
        utterance = completed[0]
        self.assertAlmostEqual(utterance.start, 10.04)
        self.assertAlmostEqual(utterance.end, 10.112)
        np.testing.assert_array_equal(utterance.samples, samples[640:1792])
        self.assertTrue(all(call.shape == (512,) for call in vad.calls))

    def test_silero_flush_processes_residual_frame_and_resets_iterator(self) -> None:
        vad = FakeVADIterator({1: {"start": 64}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.01,
            speech_padding_seconds=0.0,
            vad_iterator=vad,
        )
        samples = np.arange(400, dtype=np.float32)
        self.assertEqual(segmenter.accept(PCMChunk(samples, start_time=3.0)), [])
        completed = segmenter.flush()

        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(completed[0].start, 3.004)
        self.assertAlmostEqual(completed[0].end, 3.025)
        np.testing.assert_array_equal(completed[0].samples, samples[64:])
        self.assertEqual(vad.reset_count, 1)

    def test_silero_max_duration_splits_without_gap_or_overlap(self) -> None:
        vad = FakeVADIterator({1: {"start": 0}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.01,
            speech_padding_seconds=0.0,
            max_utterance_seconds=0.064,
            vad_iterator=vad,
        )
        samples = np.arange(1600, dtype=np.float32)

        completed = segmenter.accept(PCMChunk(samples, start_time=1.0))
        completed.extend(segmenter.flush())

        self.assertEqual(len(completed), 2)
        self.assertAlmostEqual(completed[0].start, 1.0)
        self.assertAlmostEqual(completed[0].end, 1.064)
        self.assertAlmostEqual(completed[1].start, 1.064)
        self.assertAlmostEqual(completed[1].end, 1.1)
        np.testing.assert_array_equal(
            np.concatenate([item.samples for item in completed]),
            samples,
        )

    def test_silero_discards_short_event_after_removing_padding(self) -> None:
        vad = FakeVADIterator({1: {"start": 0}, 2: {"end": 640}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.04,
            speech_padding_seconds=0.008,
            vad_iterator=vad,
        )

        completed = segmenter.accept(PCMChunk(np.ones(1024, dtype=np.float32)))

        self.assertEqual(completed, [])

    def test_silero_min_speech_handles_padding_clamped_at_stream_start(self) -> None:
        vad = FakeVADIterator({1: {"start": 0}, 2: {"end": 640}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.03,
            speech_padding_seconds=0.008,
            vad_iterator=vad,
        )

        completed = segmenter.accept(PCMChunk(np.ones(1024, dtype=np.float32)))

        self.assertEqual(len(completed), 1)
        self.assertEqual(completed[0].samples.size, 640)

    def test_silero_reset_discards_audio_and_restarts_timestamp_origin(self) -> None:
        vad = FakeVADIterator({1: {"start": 0}})
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.01,
            speech_padding_seconds=0.0,
            vad_iterator=vad,
        )
        segmenter.accept(PCMChunk(np.ones(512, dtype=np.float32), start_time=1.0))

        segmenter.reset()
        segmenter.accept(PCMChunk(np.ones(512, dtype=np.float32), start_time=5.0))
        completed = segmenter.flush()

        self.assertEqual(vad.reset_count, 2)
        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(completed[0].start, 5.0)
        self.assertAlmostEqual(completed[0].end, 5.032)


if __name__ == "__main__":
    unittest.main()
