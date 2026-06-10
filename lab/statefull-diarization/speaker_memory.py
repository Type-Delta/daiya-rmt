from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return vector
    return vector / norm


@dataclass
class SpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    observations: int = 1
    total_speech_seconds: float = 0.0
    last_seen_at: float = 0.0
    aliases: set[str] = field(default_factory=set)


class SpeakerMemory:
    """Small persistent speaker identity store for pyannote segment outputs."""

    def __init__(
        self,
        match_threshold: float = 0.38,
        update_alpha: float = 0.9,
        min_update_seconds: float = 1.0,
    ) -> None:
        self.match_threshold = match_threshold
        self.update_alpha = update_alpha
        self.min_update_seconds = min_update_seconds
        self._next_id = 0
        self.profiles: dict[str, SpeakerProfile] = {}

    def assign(
        self,
        local_labels: list[str],
        local_centroids: np.ndarray,
        speech_seconds: dict[str, float] | None = None,
        segment_end: float = 0.0,
    ) -> dict[str, str]:
        """Map pyannote-local labels to stable global labels and update memory."""

        if local_centroids is None or len(local_labels) == 0:
            return {}

        speech_seconds = speech_seconds or {}
        local_centroids = np.asarray(local_centroids)
        local_centroids = np.vstack([_normalize(c) for c in local_centroids])

        mapping: dict[str, str] = {}
        unmatched_local = set(range(len(local_labels)))
        created_speaker_ids: set[str] = set()

        if self.profiles:
            profile_ids = list(self.profiles)
            profile_centroids = np.vstack(
                [self.profiles[speaker_id].centroid for speaker_id in profile_ids]
            )
            distances = 1.0 - np.clip(local_centroids @ profile_centroids.T, -1.0, 1.0)
            rows, cols = linear_sum_assignment(distances)

            for row, col in zip(rows, cols):
                if distances[row, col] <= self.match_threshold:
                    mapping[local_labels[row]] = profile_ids[col]
                    unmatched_local.discard(row)

        for local_index in sorted(unmatched_local):
            speaker_id = f"SPEAKER_{self._next_id:03d}"
            self._next_id += 1
            mapping[local_labels[local_index]] = speaker_id
            created_speaker_ids.add(speaker_id)
            self.profiles[speaker_id] = SpeakerProfile(
                speaker_id=speaker_id,
                centroid=local_centroids[local_index],
                last_seen_at=segment_end,
            )

        for local_index, local_label in enumerate(local_labels):
            speaker_id = mapping[local_label]
            seconds = speech_seconds.get(local_label, 0.0)
            profile = self.profiles[speaker_id]
            profile.aliases.add(local_label)
            profile.total_speech_seconds += seconds
            profile.last_seen_at = segment_end

            if speaker_id not in created_speaker_ids and seconds >= self.min_update_seconds:
                profile.centroid = _normalize(
                    self.update_alpha * profile.centroid
                    + (1.0 - self.update_alpha) * local_centroids[local_index]
                )
                profile.observations += 1

        return mapping

    def debug_table(self) -> str:
        rows = []
        for speaker_id, profile in self.profiles.items():
            rows.append(
                f"{speaker_id}: observations={profile.observations}, "
                f"speech={profile.total_speech_seconds:.1f}s, "
                f"last_seen={profile.last_seen_at:.1f}s"
            )
        return "\n".join(rows)
