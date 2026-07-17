from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import wave

from lab.asr_eval import evaluate_single_chunk_strategy, transcribe_one


class ASREvalRightContextTests(unittest.TestCase):
    def test_transcribe_one_scores_only_target_window_for_right_context(self) -> None:
        class DummyModel:
            def transcribe(self, _audio: str, **kwargs: object) -> tuple[list[object], object]:
                self.kwargs = kwargs
                return (
                    [
                        SimpleNamespace(
                            start=0.1,
                            end=0.4,
                            text="inside",
                            words=[SimpleNamespace(word="inside", start=0.1, end=0.4, probability=0.9)],
                        ),
                        SimpleNamespace(
                            start=0.8,
                            end=1.2,
                            text="boundary",
                            words=[SimpleNamespace(word="boundary", start=0.8, end=1.2, probability=0.8)],
                        ),
                        SimpleNamespace(
                            start=1.1,
                            end=1.4,
                            text="future",
                            words=[SimpleNamespace(word="future", start=1.1, end=1.4, probability=0.7)],
                        ),
                    ],
                    SimpleNamespace(language="th", language_probability=0.95, duration=1.4),
                )

        args = Namespace(
            language=None,
            initial_prompt=None,
            beam_size=5,
            no_condition_on_previous_text=True,
        )
        model = DummyModel()

        prediction, info = transcribe_one(
            model,
            Path("ignored.wav"),
            args,
            include_before_seconds=1.0,
        )

        self.assertEqual(prediction, "inside boundary")
        self.assertTrue(model.kwargs["word_timestamps"])
        self.assertEqual(
            [word["included_in_prediction"] for word in info["segments"][1]["words"]],
            [True],
        )
        self.assertEqual(
            [word["included_in_prediction"] for word in info["segments"][2]["words"]],
            [False],
        )

    def test_right_context_strategy_borrows_following_audio_but_scores_target(self) -> None:
        class DummyModel:
            def __init__(self) -> None:
                self.transcribed_paths: list[str] = []

            def transcribe(self, audio: str, **_kwargs: object) -> tuple[list[object], object]:
                self.transcribed_paths.append(audio)
                return (
                    [
                        SimpleNamespace(
                            start=0.1,
                            end=0.4,
                            text="target",
                            words=[SimpleNamespace(word="target", start=0.1, end=0.4, probability=0.9)],
                        ),
                        SimpleNamespace(
                            start=0.8,
                            end=1.1,
                            text="boundary",
                            words=[SimpleNamespace(word="boundary", start=0.8, end=1.1, probability=0.8)],
                        ),
                        SimpleNamespace(
                            start=1.2,
                            end=1.4,
                            text="future",
                            words=[SimpleNamespace(word="future", start=1.2, end=1.4, probability=0.7)],
                        ),
                    ],
                    SimpleNamespace(language="th", language_probability=0.95, duration=1.5),
                )

        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            dataset_dir = root / "dataset"
            temp_dir = root / "tmp"
            dataset_dir.mkdir()
            temp_dir.mkdir()
            _write_silent_wav(dataset_dir / "row1.wav", seconds=1.0)
            _write_silent_wav(dataset_dir / "row2.wav", seconds=1.0)
            metadata = [
                {
                    "_line_number": 1,
                    "file_name": "row1.wav",
                    "text": "target boundary",
                    "source_file": "meeting.wav",
                    "source_start": 0.0,
                    "source_end": 1.0,
                    "speech_duration": 1.0,
                },
                {
                    "_line_number": 2,
                    "file_name": "row2.wav",
                    "text": "future",
                    "source_file": "meeting.wav",
                    "source_start": 1.1,
                    "source_end": 2.1,
                    "speech_duration": 1.0,
                },
            ]
            args = Namespace(
                dataset_dir=dataset_dir,
                language=None,
                initial_prompt=None,
                beam_size=5,
                no_condition_on_previous_text=True,
                right_audio_context_seconds=0.5,
                left_audio_context_seconds=4.0,
                context_max_gap_seconds=0.2,
                short_utterance_seconds=3.0,
            )
            model = DummyModel()

            details = evaluate_single_chunk_strategy(
                model=model,
                metadata=metadata,
                args=args,
                strategy="right_audio_context",
                temp_dir=temp_dir,
            )

        self.assertEqual(details[0]["status"], "ok")
        self.assertEqual(details[0]["prediction"], "target boundary")
        self.assertAlmostEqual(details[0]["right_audio_context"]["audio_seconds"], 0.5)
        self.assertEqual(details[0]["right_audio_context"]["file_names"], ["row2.wav"])
        self.assertIn("right-context-00001.wav", model.transcribed_paths[0])


def _write_silent_wav(path: Path, *, seconds: float) -> None:
    sample_rate = 16_000
    frames = int(round(seconds * sample_rate))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)


if __name__ == "__main__":
    unittest.main()
