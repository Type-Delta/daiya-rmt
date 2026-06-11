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


@dataclass
class CandidateProfile:
    candidate_id: str
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
        min_new_profile_seconds: float = 6.0,
        candidate_promote_seconds: float = 3.0,
        candidate_promote_observations: int = 2,
    ) -> None:
        self.match_threshold = match_threshold
        self.update_alpha = update_alpha
        self.min_update_seconds = min_update_seconds
        self.min_new_profile_seconds = min_new_profile_seconds
        self.candidate_promote_seconds = candidate_promote_seconds
        self.candidate_promote_observations = candidate_promote_observations
        self._next_id = 0
        self._next_candidate_id = 0
        self.profiles: dict[str, SpeakerProfile] = {}
        self.candidates: dict[str, CandidateProfile] = {}

    def _distance_matrix(
        self,
        local_indices: list[int],
        local_centroids: np.ndarray,
        target_centroids: np.ndarray,
    ) -> np.ndarray:
        return 1.0 - np.clip(local_centroids[local_indices] @ target_centroids.T, -1.0, 1.0)

    def _create_profile(
        self,
        centroid: np.ndarray,
        seconds: float,
        segment_end: float,
        alias: str,
    ) -> str:
        speaker_id = f"SPEAKER_{self._next_id:03d}"
        self._next_id += 1
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=centroid,
            total_speech_seconds=seconds,
            last_seen_at=segment_end,
            aliases={alias},
        )
        return speaker_id

    def _create_candidate(
        self,
        centroid: np.ndarray,
        seconds: float,
        segment_end: float,
        alias: str,
    ) -> str:
        candidate_id = f"CANDIDATE_{self._next_candidate_id:03d}"
        self._next_candidate_id += 1
        self.candidates[candidate_id] = CandidateProfile(
            candidate_id=candidate_id,
            centroid=centroid,
            total_speech_seconds=seconds,
            last_seen_at=segment_end,
            aliases={alias},
        )
        return candidate_id

    def _promote_candidate(self, candidate_id: str) -> str:
        candidate = self.candidates.pop(candidate_id)
        speaker_id = f"SPEAKER_{self._next_id:03d}"
        self._next_id += 1
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=candidate.centroid,
            observations=candidate.observations,
            total_speech_seconds=candidate.total_speech_seconds,
            last_seen_at=candidate.last_seen_at,
            aliases=set(candidate.aliases),
        )
        return speaker_id

    def _should_promote(self, candidate: CandidateProfile) -> bool:
        return (
            candidate.total_speech_seconds >= self.candidate_promote_seconds
            and candidate.observations >= self.candidate_promote_observations
        )

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
        speech_by_index = [
            max(0.0, float(speech_seconds.get(label, 0.0))) for label in local_labels
        ]
        visible_indices = [index for index, seconds in enumerate(speech_by_index) if seconds > 0.0]
        hidden_indices = [index for index, seconds in enumerate(speech_by_index) if seconds <= 0.0]
        unmatched_visible = set(visible_indices)
        fresh_speaker_ids: set[str] = set()

        for local_index in hidden_indices:
            mapping[local_labels[local_index]] = "OVERLAP_ONLY"

        if self.profiles:
            profile_ids = list(self.profiles)
            profile_centroids = np.vstack(
                [self.profiles[speaker_id].centroid for speaker_id in profile_ids]
            )

            if visible_indices:
                distances = self._distance_matrix(
                    visible_indices, local_centroids, profile_centroids
                )
                rows, cols = linear_sum_assignment(distances)

                for row, col in zip(rows, cols):
                    if distances[row, col] <= self.match_threshold:
                        local_index = visible_indices[row]
                        mapping[local_labels[local_index]] = profile_ids[col]
                        unmatched_visible.discard(local_index)

            for local_index in hidden_indices:
                distances = self._distance_matrix(
                    [local_index], local_centroids, profile_centroids
                )[0]
                best_index = int(np.argmin(distances))
                if distances[best_index] <= self.match_threshold:
                    mapping[local_labels[local_index]] = profile_ids[best_index]
                else:
                    mapping[local_labels[local_index]] = "OVERLAP_ONLY"

        if self.candidates and unmatched_visible:
            candidate_ids = list(self.candidates)
            candidate_centroids = np.vstack(
                [self.candidates[candidate_id].centroid for candidate_id in candidate_ids]
            )
            visible_to_match = sorted(unmatched_visible)
            distances = self._distance_matrix(
                visible_to_match, local_centroids, candidate_centroids
            )
            rows, cols = linear_sum_assignment(distances)

            for row, col in zip(rows, cols):
                if distances[row, col] > self.match_threshold:
                    continue

                local_index = visible_to_match[row]
                local_label = local_labels[local_index]
                candidate_id = candidate_ids[col]
                candidate = self.candidates[candidate_id]
                seconds = speech_by_index[local_index]
                candidate.aliases.add(local_label)
                candidate.total_speech_seconds += seconds
                candidate.last_seen_at = segment_end
                candidate.centroid = _normalize(
                    self.update_alpha * candidate.centroid
                    + (1.0 - self.update_alpha) * local_centroids[local_index]
                )
                candidate.observations += 1

                if self._should_promote(candidate):
                    speaker_id = self._promote_candidate(candidate_id)
                    mapping[local_label] = speaker_id
                    fresh_speaker_ids.add(speaker_id)
                else:
                    mapping[local_label] = candidate_id

                unmatched_visible.discard(local_index)

        for local_index in sorted(unmatched_visible):
            seconds = speech_by_index[local_index]
            local_label = local_labels[local_index]

            if seconds >= self.min_new_profile_seconds:
                speaker_id = self._create_profile(
                    local_centroids[local_index],
                    seconds,
                    segment_end,
                    local_label,
                )
                mapping[local_label] = speaker_id
                fresh_speaker_ids.add(speaker_id)
            else:
                mapping[local_label] = self._create_candidate(
                    local_centroids[local_index],
                    seconds,
                    segment_end,
                    local_label,
                )

        for local_index, local_label in enumerate(local_labels):
            speaker_id = mapping[local_label]
            if not speaker_id.startswith("SPEAKER_"):
                continue

            seconds = speech_by_index[local_index]
            if seconds <= 0.0:
                continue

            if speaker_id in fresh_speaker_ids:
                continue

            profile = self.profiles[speaker_id]
            profile.aliases.add(local_label)
            profile.total_speech_seconds += seconds
            profile.last_seen_at = segment_end

            if seconds >= self.min_update_seconds:
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
        if not rows:
            rows.append("(no permanent speaker profiles)")

        if self.candidates:
            rows.append("\n=== candidates ===")
            for candidate_id, candidate in self.candidates.items():
                rows.append(
                    f"{candidate_id}: observations={candidate.observations}, "
                    f"speech={candidate.total_speech_seconds:.1f}s, "
                    f"last_seen={candidate.last_seen_at:.1f}s"
                )
        return "\n".join(rows)
