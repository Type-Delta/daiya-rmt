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


def _speech_ranges(samples: np.ndarray, *_args: object, **_kwargs: object) -> list[dict[str, int]]:
    voiced = np.flatnonzero(np.abs(samples) > 0.01)
    if voiced.size == 0:
        return []
    return [{"start": int(voiced[0]), "end": int(voiced[-1] + 1)}]


class UtteranceSegmenterTest(unittest.TestCase):
    def test_factory_defaults_to_energy_segmenter(self) -> None:
        segmenter = create_utterance_segmenter()

        self.assertIsInstance(segmenter, EnergyUtteranceSegmenter)

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

    def test_silero_segmenter_uses_detector_and_padding_across_chunks(self) -> None:
        segmenter = SileroUtteranceSegmenter(
            threshold=0.4,
            min_speech_seconds=0.05,
            min_silence_seconds=0.10,
            speech_padding_seconds=0.05,
            max_utterance_seconds=2.0,
            model=object(),
            get_speech_timestamps=_speech_ranges,
        )
        silence = np.zeros(int(0.10 * SAMPLE_RATE), dtype=np.float32)
        leading = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
        speech = np.full(int(0.20 * SAMPLE_RATE), 0.2, dtype=np.float32)

        self.assertEqual(segmenter.accept(PCMChunk(silence, start_time=0.0)), [])
        self.assertEqual(
            segmenter.accept(PCMChunk(np.concatenate([leading, speech]), start_time=0.10)),
            [],
        )
        completed = segmenter.accept(PCMChunk(np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32), start_time=0.35))

        self.assertEqual(len(completed), 1)
        utterance = completed[0]
        self.assertAlmostEqual(utterance.start, 0.10, places=3)
        self.assertAlmostEqual(utterance.end, 0.40, places=3)
        self.assertEqual(utterance.samples.shape[0], int(0.30 * SAMPLE_RATE))

    def test_silero_segmenter_uses_previous_chunk_for_speech_padding(self) -> None:
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.05,
            min_silence_seconds=0.10,
            speech_padding_seconds=0.05,
            max_utterance_seconds=2.0,
            model=object(),
            get_speech_timestamps=_speech_ranges,
        )
        silence = np.zeros(int(0.05 * SAMPLE_RATE), dtype=np.float32)
        speech = np.full(int(0.20 * SAMPLE_RATE), 0.2, dtype=np.float32)

        self.assertEqual(segmenter.accept(PCMChunk(silence, start_time=0.0)), [])
        self.assertEqual(segmenter.accept(PCMChunk(speech, start_time=0.05)), [])
        completed = segmenter.accept(PCMChunk(np.zeros(int(0.12 * SAMPLE_RATE), dtype=np.float32), start_time=0.25))

        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(completed[0].start, 0.0, places=3)
        self.assertAlmostEqual(completed[0].end, 0.30, places=3)
        self.assertEqual(completed[0].samples.shape[0], int(0.30 * SAMPLE_RATE))

    def test_silero_segmenter_flushes_active_speech(self) -> None:
        segmenter = SileroUtteranceSegmenter(
            min_speech_seconds=0.05,
            min_silence_seconds=0.20,
            speech_padding_seconds=0.0,
            model=object(),
            get_speech_timestamps=_speech_ranges,
        )

        segmenter.accept(
            PCMChunk(np.full(int(0.20 * SAMPLE_RATE), 0.2, dtype=np.float32), start_time=1.0)
        )
        completed = segmenter.flush()

        self.assertEqual(len(completed), 1)
        self.assertAlmostEqual(completed[0].start, 1.0)
        self.assertAlmostEqual(completed[0].end, 1.2)


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


if __name__ == "__main__":
    unittest.main()
