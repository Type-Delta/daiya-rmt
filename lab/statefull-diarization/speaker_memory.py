from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(vector)
    if norm == 0 or not np.isfinite(norm):
        return np.zeros_like(vector)
    return vector / norm


@dataclass(frozen=True)
class AssignmentConstraints:
    mutually_exclusive_local_labels: set[tuple[str, str]] = field(default_factory=set)
    blocked_speaker_ids: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class Assignment:
    local_label: str
    speaker_id: str
    decision: str
    distance: float | None
    confidence: float
    speech_seconds: float


@dataclass(frozen=True)
class CommittedSpeakerEvidence:
    evidence_id: str
    speaker_id: str
    vector: np.ndarray
    start: float
    end: float
    clean_speech_seconds: float
    overlap_ratio: float
    confidence: float
    source: str
    local_label: str | None = None


@dataclass(frozen=True)
class SpeakerObservation:
    evidence_id: str
    vector: np.ndarray
    start: float
    end: float
    clean_speech_seconds: float
    overlap_ratio: float
    confidence: float
    source: str


@dataclass
class SpeakerProfile:
    speaker_id: str
    centroid: np.ndarray
    observations: list[SpeakerObservation] = field(default_factory=list)
    total_speech_seconds: float = 0.0
    last_seen_at: float = 0.0
    aliases: set[str] = field(default_factory=set)

    @property
    def observation_count(self) -> int:
        return len(self.observations)

    @property
    def high_quality_observation_count(self) -> int:
        return sum(1 for observation in self.observations if observation.confidence >= 0.6)


@dataclass
class CandidateProfile:
    candidate_id: str
    centroid: np.ndarray
    observations: list[SpeakerObservation] = field(default_factory=list)
    total_speech_seconds: float = 0.0
    last_seen_at: float = 0.0
    aliases: set[str] = field(default_factory=set)

    @property
    def observation_count(self) -> int:
        return len(self.observations)


class SpeakerMemory:
    """Persistent speaker identity store with read-only matching and commit updates."""

    def __init__(
        self,
        match_threshold: float = 0.38,
        update_alpha: float = 0.9,
        min_update_seconds: float = 1.0,
        min_new_profile_seconds: float = 6.0,
        candidate_promote_seconds: float = 3.0,
        candidate_promote_observations: int = 2,
        min_observation_clean_speech: float = 1.0,
        max_observation_overlap_ratio: float = 0.3,
        min_assignment_confidence: float = 0.6,
        reservoir_size: int = 32,
    ) -> None:
        self.match_threshold = match_threshold
        self.update_alpha = update_alpha
        self.min_update_seconds = min_update_seconds
        self.min_new_profile_seconds = min_new_profile_seconds
        self.candidate_promote_seconds = candidate_promote_seconds
        self.candidate_promote_observations = candidate_promote_observations
        self.min_observation_clean_speech = min_observation_clean_speech
        self.max_observation_overlap_ratio = max_observation_overlap_ratio
        self.min_assignment_confidence = min_assignment_confidence
        self.reservoir_size = reservoir_size
        self._next_id = 0
        self._next_candidate_id = 0
        self.profiles: dict[str, SpeakerProfile] = {}
        self.candidates: dict[str, CandidateProfile] = {}
        self.committed_evidence_ids: set[str] = set()
        self.match_count = 0
        self.update_count = 0

    def _distance_matrix(
        self,
        local_indices: list[int],
        local_centroids: np.ndarray,
        target_centroids: np.ndarray,
    ) -> np.ndarray:
        return 1.0 - np.clip(local_centroids[local_indices] @ target_centroids.T, -1.0, 1.0)

    def _confidence(self, distance: float | None) -> float:
        if distance is None:
            return 0.0
        if self.match_threshold <= 0:
            return 1.0 if distance <= 0 else 0.0
        return float(np.clip(1.0 - (distance / self.match_threshold), 0.0, 1.0))

    def _profile_id(self) -> str:
        speaker_id = f"SPEAKER_{self._next_id:03d}"
        self._next_id += 1
        return speaker_id

    def _candidate_id(self) -> str:
        candidate_id = f"CANDIDATE_{self._next_candidate_id:03d}"
        self._next_candidate_id += 1
        return candidate_id

    def _make_observation(self, evidence: CommittedSpeakerEvidence) -> SpeakerObservation:
        return SpeakerObservation(
            evidence_id=evidence.evidence_id,
            vector=_normalize(np.asarray(evidence.vector)),
            start=evidence.start,
            end=evidence.end,
            clean_speech_seconds=max(0.0, float(evidence.clean_speech_seconds)),
            overlap_ratio=max(0.0, float(evidence.overlap_ratio)),
            confidence=float(np.clip(evidence.confidence, 0.0, 1.0)),
            source=evidence.source,
        )

    def _is_high_quality(self, observation: SpeakerObservation) -> bool:
        return (
            observation.clean_speech_seconds >= self.min_observation_clean_speech
            and observation.overlap_ratio <= self.max_observation_overlap_ratio
            and observation.confidence >= self.min_assignment_confidence
        )

    def _bounded_observations(
        self, observations: list[SpeakerObservation]
    ) -> list[SpeakerObservation]:
        if len(observations) <= self.reservoir_size:
            return observations
        high_quality = [obs for obs in observations if self._is_high_quality(obs)]
        weak = [obs for obs in observations if not self._is_high_quality(obs)]
        kept = high_quality[-self.reservoir_size :]
        if len(kept) < self.reservoir_size:
            kept = weak[-(self.reservoir_size - len(kept)) :] + kept
        return kept

    def _recompute_centroid(self, observations: list[SpeakerObservation], fallback: np.ndarray) -> np.ndarray:
        high_quality = [obs for obs in observations if self._is_high_quality(obs)]
        source = high_quality or observations
        if not source:
            return fallback

        vectors = np.vstack([obs.vector for obs in source])
        weights = np.asarray(
            [
                max(0.1, obs.clean_speech_seconds)
                * max(0.1, obs.confidence)
                * (1.0 - min(0.9, obs.overlap_ratio))
                for obs in source
            ],
            dtype=float,
        )
        return _normalize(np.average(vectors, axis=0, weights=weights))

    def _add_profile_observation(
        self,
        speaker_id: str,
        observation: SpeakerObservation,
        alias: str | None,
    ) -> None:
        profile = self.profiles[speaker_id]
        if alias:
            profile.aliases.add(alias)
        profile.observations.append(observation)
        profile.observations = self._bounded_observations(profile.observations)
        profile.total_speech_seconds += observation.clean_speech_seconds
        profile.last_seen_at = max(profile.last_seen_at, observation.end)

        if observation.clean_speech_seconds >= self.min_update_seconds:
            profile.centroid = self._recompute_centroid(profile.observations, profile.centroid)

    def _create_profile(self, observation: SpeakerObservation, alias: str | None) -> str:
        speaker_id = self._profile_id()
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=observation.vector,
            observations=[observation],
            total_speech_seconds=observation.clean_speech_seconds,
            last_seen_at=observation.end,
            aliases={alias} if alias else set(),
        )
        return speaker_id

    def _create_candidate(self, observation: SpeakerObservation, alias: str | None) -> str:
        candidate_id = self._candidate_id()
        self.candidates[candidate_id] = CandidateProfile(
            candidate_id=candidate_id,
            centroid=observation.vector,
            observations=[observation],
            total_speech_seconds=observation.clean_speech_seconds,
            last_seen_at=observation.end,
            aliases={alias} if alias else set(),
        )
        return candidate_id

    def _add_candidate_observation(
        self,
        candidate_id: str,
        observation: SpeakerObservation,
        alias: str | None,
    ) -> str:
        candidate = self.candidates[candidate_id]
        if alias:
            candidate.aliases.add(alias)
        candidate.observations.append(observation)
        candidate.observations = self._bounded_observations(candidate.observations)
        candidate.total_speech_seconds += observation.clean_speech_seconds
        candidate.last_seen_at = max(candidate.last_seen_at, observation.end)
        candidate.centroid = self._recompute_centroid(candidate.observations, candidate.centroid)

        if self._should_promote(candidate):
            return self._promote_candidate(candidate_id)
        return candidate_id

    def _promote_candidate(self, candidate_id: str) -> str:
        candidate = self.candidates.pop(candidate_id)
        speaker_id = self._profile_id()
        self.profiles[speaker_id] = SpeakerProfile(
            speaker_id=speaker_id,
            centroid=candidate.centroid,
            observations=list(candidate.observations),
            total_speech_seconds=candidate.total_speech_seconds,
            last_seen_at=candidate.last_seen_at,
            aliases=set(candidate.aliases),
        )
        return speaker_id

    def _should_promote(self, candidate: CandidateProfile) -> bool:
        return (
            candidate.total_speech_seconds >= self.candidate_promote_seconds
            and candidate.observation_count >= self.candidate_promote_observations
        )

    def _prepare_inputs(
        self,
        local_labels: list[str],
        local_centroids: np.ndarray | None,
        speech_seconds: dict[str, float] | None,
    ) -> tuple[list[str], np.ndarray | None, list[float], dict[str, Assignment]]:
        if len(local_labels) == 0:
            return [], None, [], {}

        if local_centroids is None:
            return [], None, [], {
                label: Assignment(label, "MISSING_EMBEDDING", "invalid", None, 0.0, 0.0)
                for label in local_labels
            }

        speech_seconds = speech_seconds or {}
        local_centroids = np.asarray(local_centroids)
        assignments: dict[str, Assignment] = {}

        if local_centroids.ndim != 2:
            return [], None, [], {
                label: Assignment(label, "INVALID_EMBEDDING", "invalid", None, 0.0, 0.0)
                for label in local_labels
            }

        if local_centroids.shape[0] < len(local_labels):
            for label in local_labels[local_centroids.shape[0] :]:
                assignments[label] = Assignment(
                    label, "MISSING_EMBEDDING", "invalid", None, 0.0, 0.0
                )
            local_labels = local_labels[: local_centroids.shape[0]]
        elif local_centroids.shape[0] > len(local_labels):
            local_centroids = local_centroids[: len(local_labels)]

        local_centroids = np.vstack([_normalize(c) for c in local_centroids])
        speech_by_index = [
            max(0.0, float(speech_seconds.get(label, 0.0))) for label in local_labels
        ]
        return local_labels, local_centroids, speech_by_index, assignments

    def match(
        self,
        local_labels: list[str],
        local_centroids: np.ndarray,
        speech_seconds: dict[str, float] | None = None,
        segment_end: float = 0.0,
        constraints: AssignmentConstraints | None = None,
    ) -> dict[str, Assignment]:
        """Read-only local-to-global assignment."""

        del segment_end
        constraints = constraints or AssignmentConstraints()
        local_labels, local_centroids, speech_by_index, mapping = self._prepare_inputs(
            local_labels, local_centroids, speech_seconds
        )
        if local_centroids is None or not local_labels:
            return mapping

        self.match_count += 1
        centroid_norms = np.linalg.norm(local_centroids, axis=1)
        invalid_indices = {
            index
            for index, norm in enumerate(centroid_norms)
            if norm <= 1e-12 or not np.isfinite(norm)
        }
        visible_indices = [
            index
            for index, seconds in enumerate(speech_by_index)
            if seconds > 0.0 and index not in invalid_indices
        ]
        hidden_indices = [
            index
            for index, seconds in enumerate(speech_by_index)
            if seconds <= 0.0 and index not in invalid_indices
        ]
        unmatched_visible = set(visible_indices)

        for local_index in invalid_indices:
            label = local_labels[local_index]
            mapping[label] = Assignment(label, "INVALID_EMBEDDING", "invalid", None, 0.0, 0.0)

        for local_index in hidden_indices:
            label = local_labels[local_index]
            mapping[label] = Assignment(label, "OVERLAP_ONLY", "overlap_only", None, 0.0, 0.0)

        profile_ids = [
            speaker_id
            for speaker_id in self.profiles
            if speaker_id not in constraints.blocked_speaker_ids
        ]
        if profile_ids and visible_indices:
            profile_centroids = np.vstack(
                [self.profiles[speaker_id].centroid for speaker_id in profile_ids]
            )
            distances = self._distance_matrix(visible_indices, local_centroids, profile_centroids)
            rows, cols = linear_sum_assignment(distances)

            for row, col in zip(rows, cols):
                distance = float(distances[row, col])
                local_index = visible_indices[row]
                label = local_labels[local_index]
                confidence = self._confidence(distance)
                if distance <= self.match_threshold and confidence >= self.min_assignment_confidence:
                    mapping[label] = Assignment(
                        label,
                        profile_ids[col],
                        "match",
                        distance,
                        confidence,
                        speech_by_index[local_index],
                    )
                    unmatched_visible.discard(local_index)
                elif distance <= self.match_threshold * 1.25:
                    mapping[label] = Assignment(
                        label,
                        profile_ids[col],
                        "ambiguous",
                        distance,
                        confidence,
                        speech_by_index[local_index],
                    )
                    unmatched_visible.discard(local_index)

        if profile_ids and hidden_indices:
            profile_centroids = np.vstack(
                [self.profiles[speaker_id].centroid for speaker_id in profile_ids]
            )
            for local_index in hidden_indices:
                distances = self._distance_matrix([local_index], local_centroids, profile_centroids)[0]
                best_index = int(np.argmin(distances))
                distance = float(distances[best_index])
                label = local_labels[local_index]
                if distance <= self.match_threshold:
                    mapping[label] = Assignment(
                        label,
                        profile_ids[best_index],
                        "match",
                        distance,
                        self._confidence(distance),
                        0.0,
                    )

        candidate_ids = [
            candidate_id
            for candidate_id in self.candidates
            if candidate_id not in constraints.blocked_speaker_ids
        ]
        if candidate_ids and unmatched_visible:
            candidate_centroids = np.vstack(
                [self.candidates[candidate_id].centroid for candidate_id in candidate_ids]
            )
            visible_to_match = sorted(unmatched_visible)
            distances = self._distance_matrix(visible_to_match, local_centroids, candidate_centroids)
            rows, cols = linear_sum_assignment(distances)

            for row, col in zip(rows, cols):
                distance = float(distances[row, col])
                if distance > self.match_threshold:
                    continue
                local_index = visible_to_match[row]
                label = local_labels[local_index]
                mapping[label] = Assignment(
                    label,
                    candidate_ids[col],
                    "candidate",
                    distance,
                    self._confidence(distance),
                    speech_by_index[local_index],
                )
                unmatched_visible.discard(local_index)

        for local_index in sorted(unmatched_visible):
            seconds = speech_by_index[local_index]
            label = local_labels[local_index]
            decision = "new" if seconds >= self.min_new_profile_seconds else "candidate"
            mapping[label] = Assignment(
                label,
                f"UNASSIGNED_{label}",
                decision,
                None,
                1.0 if decision == "new" else 0.5,
                seconds,
            )

        return mapping

    def commit_evidence(
        self,
        assignments: list[CommittedSpeakerEvidence],
    ) -> dict[str, str]:
        """Idempotently update durable profiles/candidates from committed evidence."""

        committed: dict[str, str] = {}
        for evidence in assignments:
            if evidence.evidence_id in self.committed_evidence_ids:
                if evidence.speaker_id in self.profiles or evidence.speaker_id in self.candidates:
                    committed[evidence.evidence_id] = evidence.speaker_id
                continue

            observation = self._make_observation(evidence)
            alias = evidence.local_label
            target_id = evidence.speaker_id

            if target_id in self.profiles:
                self._add_profile_observation(target_id, observation, alias)
                actual_id = target_id
            elif target_id in self.candidates:
                actual_id = self._add_candidate_observation(target_id, observation, alias)
            elif observation.clean_speech_seconds >= self.min_new_profile_seconds:
                actual_id = self._create_profile(observation, alias)
            else:
                actual_id = self._create_candidate(observation, alias)

            self.committed_evidence_ids.add(evidence.evidence_id)
            self.update_count += 1
            committed[evidence.evidence_id] = actual_id

        return committed

    def assign(
        self,
        local_labels: list[str],
        local_centroids: np.ndarray,
        speech_seconds: dict[str, float] | None = None,
        segment_end: float = 0.0,
    ) -> dict[str, str]:
        """Legacy compatibility wrapper: match, then immediately commit evidence."""

        assignments = self.match(
            local_labels,
            local_centroids,
            speech_seconds=speech_seconds,
            segment_end=segment_end,
        )
        centroid_by_label: dict[str, np.ndarray] = {}
        if local_centroids is not None:
            array = np.asarray(local_centroids)
            if array.ndim == 2:
                for label, vector in zip(local_labels, array):
                    centroid_by_label[label] = _normalize(vector)

        evidence_items: list[CommittedSpeakerEvidence] = []
        evidence_id_by_label: dict[str, str] = {}
        for label, assignment in assignments.items():
            if assignment.decision in {"invalid", "overlap_only", "ambiguous"}:
                continue
            vector = centroid_by_label.get(label)
            if vector is None:
                continue
            evidence_id = f"legacy:{segment_end:.3f}:{label}:{assignment.speaker_id}"
            evidence_id_by_label[label] = evidence_id
            evidence_items.append(
                CommittedSpeakerEvidence(
                    evidence_id=evidence_id,
                    speaker_id=assignment.speaker_id,
                    vector=vector,
                    start=max(0.0, segment_end - assignment.speech_seconds),
                    end=segment_end,
                    clean_speech_seconds=assignment.speech_seconds,
                    overlap_ratio=0.0,
                    confidence=max(assignment.confidence, 0.5),
                    source="legacy_assign",
                    local_label=label,
                )
            )

        committed = self.commit_evidence(evidence_items)
        mapping: dict[str, str] = {}
        for label, assignment in assignments.items():
            evidence_id = evidence_id_by_label.get(label)
            mapping[label] = committed.get(evidence_id, assignment.speaker_id)
        return mapping

    def debug_table(self) -> str:
        rows = []
        for speaker_id, profile in self.profiles.items():
            rows.append(
                f"{speaker_id}: observations={profile.observation_count}, "
                f"high_quality={profile.high_quality_observation_count}, "
                f"speech={profile.total_speech_seconds:.1f}s, "
                f"last_seen={profile.last_seen_at:.1f}s"
            )
        if not rows:
            rows.append("(no permanent speaker profiles)")

        if self.candidates:
            rows.append("\n=== candidates ===")
            for candidate_id, candidate in self.candidates.items():
                rows.append(
                    f"{candidate_id}: observations={candidate.observation_count}, "
                    f"speech={candidate.total_speech_seconds:.1f}s, "
                    f"last_seen={candidate.last_seen_at:.1f}s"
                )
        return "\n".join(rows)
