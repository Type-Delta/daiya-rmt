from __future__ import annotations

from types import SimpleNamespace
import unittest

import numpy as np

from daiya.asr import FasterWhisperASR, Utterance, is_low_confidence_segment, low_confidence_words
from daiya.audio import SAMPLE_RATE
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

    def test_right_context_audio_is_appended_and_timestamps_are_clipped(self) -> None:
        segments = [
            SimpleNamespace(
                start=0.2,
                end=0.8,
                text="target",
                words=[SimpleNamespace(word="target", start=0.2, end=0.8, probability=0.9)],
            ),
            SimpleNamespace(
                start=1.8,
                end=2.3,
                text="crossing out",
                words=[SimpleNamespace(word="tail", start=1.8, end=2.3, probability=0.7)],
            ),
            SimpleNamespace(start=2.3, end=2.8, text="future only", words=[]),
        ]
        model = FakeWhisperModel(segments)
        asr = _asr_with_model(model)
        utterance = _utterance(duration_seconds=2.0)
        right_context = np.zeros(SAMPLE_RATE, dtype=np.float32)

        decoded = asr.transcribe_utterance(utterance, right_context_samples=right_context)

        self.assertEqual(model.calls[0]["audio"].shape, (SAMPLE_RATE * 3,))
        self.assertEqual(len(decoded), 2)
        self.assertEqual([segment.text for segment in decoded], ["target", "crossing out"])
        self.assertEqual((decoded[0].start, decoded[0].end), (10.2, 10.8))
        self.assertEqual((decoded[1].start, decoded[1].end), (11.8, 12.0))
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
