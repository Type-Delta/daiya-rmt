from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from daiya.asr import (
    ASRUnavailableError,
    EnergyUtteranceSegmenter,
    SileroUtteranceSegmenter,
    Utterance,
    create_utterance_segmenter,
)
from daiya.audio import SAMPLE_RATE, PCMChunk


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
