from __future__ import annotations

import unittest

import numpy as np
import torch

from backends import AudioWindow, DiarizationWindowResult
from realtime import RealtimeDiarizationConfig, RealtimeDiarizationDriver
from speaker_memory import CommittedSpeakerEvidence, SpeakerMemory
from timeline import SpeakerSegment, TimelineStore


def unit(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


class SpeakerMemoryTests(unittest.TestCase):
    def test_committed_evidence_is_idempotent(self) -> None:
        memory = SpeakerMemory(min_new_profile_seconds=1.0)
        vector = unit(np.ones(4))
        evidence = CommittedSpeakerEvidence(
            evidence_id="same-region",
            speaker_id="UNASSIGNED_A",
            vector=vector,
            start=0.0,
            end=2.0,
            clean_speech_seconds=2.0,
            overlap_ratio=0.0,
            confidence=1.0,
            source="test",
            local_label="A",
        )

        first = memory.commit_evidence([evidence])
        second = memory.commit_evidence([evidence])

        speaker_id = first["same-region"]
        self.assertEqual(second, {})
        self.assertEqual(memory.profiles[speaker_id].total_speech_seconds, 2.0)
        self.assertEqual(memory.update_count, 1)

    def test_candidate_promotes_only_after_distinct_evidence(self) -> None:
        memory = SpeakerMemory(
            min_new_profile_seconds=10.0,
            candidate_promote_seconds=2.0,
            candidate_promote_observations=2,
        )
        vector = unit(np.ones(4))
        first = CommittedSpeakerEvidence(
            evidence_id="region-1",
            speaker_id="UNASSIGNED_A",
            vector=vector,
            start=0.0,
            end=1.0,
            clean_speech_seconds=1.0,
            overlap_ratio=0.0,
            confidence=1.0,
            source="test",
            local_label="A",
        )
        first_result = memory.commit_evidence([first])
        candidate_id = first_result["region-1"]
        duplicate = memory.commit_evidence([first])
        self.assertEqual(duplicate, {})
        self.assertIn(candidate_id, memory.candidates)
        self.assertFalse(memory.profiles)

        second = CommittedSpeakerEvidence(
            evidence_id="region-2",
            speaker_id=candidate_id,
            vector=vector,
            start=1.0,
            end=2.0,
            clean_speech_seconds=1.0,
            overlap_ratio=0.0,
            confidence=1.0,
            source="test",
            local_label="A",
        )
        second_result = memory.commit_evidence([second])
        self.assertTrue(second_result["region-2"].startswith("SPEAKER_"))
        self.assertFalse(memory.candidates)
        self.assertEqual(len(memory.profiles), 1)

    def test_two_local_speakers_do_not_map_to_one_profile(self) -> None:
        memory = SpeakerMemory(min_new_profile_seconds=1.0, min_assignment_confidence=0.0)
        base = unit(np.array([1.0, 0.0, 0.0, 0.0]))
        memory.commit_evidence(
            [
                CommittedSpeakerEvidence(
                    evidence_id="profile",
                    speaker_id="UNASSIGNED_A",
                    vector=base,
                    start=0.0,
                    end=2.0,
                    clean_speech_seconds=2.0,
                    overlap_ratio=0.0,
                    confidence=1.0,
                    source="test",
                    local_label="A",
                )
            ]
        )

        labels = ["LOCAL_A", "LOCAL_B"]
        centroids = np.vstack([base, unit(np.array([0.99, 0.01, 0.0, 0.0]))])
        assignments = memory.match(labels, centroids, {"LOCAL_A": 1.0, "LOCAL_B": 1.0})
        mapped_profiles = [
            assignment.speaker_id
            for assignment in assignments.values()
            if assignment.speaker_id.startswith("SPEAKER_")
        ]

        self.assertEqual(len(mapped_profiles), 1)


class TimelineStoreTests(unittest.TestCase):
    def test_overlapping_window_updates_turn_instead_of_duplicating(self) -> None:
        timeline = TimelineStore()
        first_events, first_stats = timeline.update_region(
            1.0,
            2.0,
            [SpeakerSegment(1.0, 2.0, "SPEAKER_000", 0.8, "A", "e1")],
        )
        second_events, second_stats = timeline.update_region(
            1.0,
            2.0,
            [SpeakerSegment(1.1, 2.0, "SPEAKER_000", 0.9, "A", "e2")],
        )

        self.assertEqual(first_stats.created, 1)
        self.assertEqual(second_stats.updated, 1)
        self.assertEqual(len(timeline.turns), 1)
        self.assertEqual(first_events[0].type, "turn.created")
        self.assertEqual(second_events[0].type, "turn.updated")

    def test_committed_turns_are_not_mutated_by_normal_updates(self) -> None:
        timeline = TimelineStore()
        timeline.update_region(
            1.0,
            2.0,
            [SpeakerSegment(1.0, 2.0, "SPEAKER_000", 0.8, "A", "e1")],
        )
        commit_events, _ = timeline.commit_before(3.0)
        update_events, stats = timeline.update_region(
            1.0,
            2.0,
            [SpeakerSegment(1.0, 2.0, "SPEAKER_001", 0.8, "B", "e2")],
        )

        self.assertEqual(commit_events[0].type, "turn.committed")
        self.assertEqual(stats.created, 1)
        self.assertEqual(update_events[0].type, "turn.created")
        self.assertEqual(len([turn for turn in timeline.turns if turn.final]), 1)


class FakeBackend:
    def __init__(self, vector: np.ndarray) -> None:
        self.vector = vector

    def process_window(self, window: AudioWindow) -> DiarizationWindowResult:
        segment_start = window.end - 1.0
        return DiarizationWindowResult(
            window=window,
            segments=[
                SpeakerSegment(
                    start=segment_start,
                    end=window.end,
                    speaker_id="LOCAL_A",
                    speaker_confidence=1.0,
                    local_label="LOCAL_A",
                )
            ],
            local_labels=["LOCAL_A"],
            local_centroids=np.asarray([self.vector]),
            speech_seconds={"LOCAL_A": 1.0},
            pipeline_started_at=0.0,
            pipeline_finished_at=0.1,
        )


class RealtimeDriverTests(unittest.TestCase):
    def test_committed_evidence_uses_emit_slice_duration(self) -> None:
        vector = unit(np.ones(4))
        memory = SpeakerMemory(min_new_profile_seconds=0.5)
        driver = RealtimeDiarizationDriver(
            backend=FakeBackend(vector),
            memory=memory,
            config=RealtimeDiarizationConfig(
                window_seconds=3.0,
                hop_seconds=1.0,
                latency_seconds=1.0,
                commit_delay_seconds=1.0,
            ),
        )

        for index, end in enumerate([3.0, 4.0, 5.0]):
            driver.process_window(
                AudioWindow(
                    index=index,
                    waveform=torch.zeros((1, 48000)),
                    sample_rate=16000,
                    start=end - 3.0,
                    end=end,
                )
            )

        total_speech = sum(
            profile.total_speech_seconds for profile in memory.profiles.values()
        )
        self.assertLessEqual(total_speech, 2.01)


if __name__ == "__main__":
    unittest.main()
