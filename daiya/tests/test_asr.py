from __future__ import annotations

import unittest

import numpy as np

from daiya.asr import FasterWhisperASR, Utterance, decoder_options_for_duration
from daiya.audio import SAMPLE_RATE


class FakeWhisperModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def transcribe(self, _audio: np.ndarray, **kwargs: object):
        self.calls.append(kwargs)
        return [], object()


def _utterance(duration: float = 2.0) -> Utterance:
    return Utterance(
        samples=np.zeros(round(SAMPLE_RATE * duration), dtype=np.float32),
        start=4.0,
        end=4.0 + duration,
    )


def _asr_with_model(model: FakeWhisperModel) -> FasterWhisperASR:
    asr = FasterWhisperASR.__new__(FasterWhisperASR)
    asr._model = model
    asr.language = "th"
    asr.initial_prompt = None
    asr.decoding_policy = "baseline"
    asr.short_utterance_seconds = 3.0
    return asr


class FasterWhisperASRTest(unittest.TestCase):
    def test_baseline_preserves_current_decoder_options(self) -> None:
        first = decoder_options_for_duration("baseline", 0.5)
        self.assertEqual(first, {"temperature": [0.0, 0.8, 1.0]})
        first["temperature"] = []
        self.assertEqual(
            decoder_options_for_duration("baseline", 0.5),
            {"temperature": [0.0, 0.8, 1.0]},
        )

    def test_policy_boundaries_and_validation(self) -> None:
        self.assertEqual(
            decoder_options_for_duration("short_beam", 3.0),
            {"temperature": [0.0, 0.8, 1.0], "beam_size": 8, "patience": 1.2},
        )
        self.assertEqual(
            decoder_options_for_duration("short_greedy", 2.0),
            {"temperature": [0.0, 0.8, 1.0], "beam_size": 1, "best_of": 1},
        )
        self.assertNotIn("beam_size", decoder_options_for_duration("short_beam", 3.001))
        with self.assertRaisesRegex(ValueError, "unknown ASR decoding policy"):
            decoder_options_for_duration("mystery", 1.0)
        for threshold in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and greater than zero"):
                decoder_options_for_duration(
                    "baseline", 1.0, short_utterance_seconds=threshold
                )

    def test_selected_decoder_options_are_forwarded(self) -> None:
        model = FakeWhisperModel()
        asr = _asr_with_model(model)
        asr.decoding_policy = "short_beam"

        asr.transcribe_utterance(_utterance(3.0))

        self.assertEqual(model.calls[0]["beam_size"], 8)
        self.assertEqual(model.calls[0]["patience"], 1.2)
        self.assertEqual(model.calls[0]["temperature"], [0.0, 0.8, 1.0])


if __name__ == "__main__":
    unittest.main()
