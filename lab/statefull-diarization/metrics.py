from __future__ import annotations

import csv
import statistics
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass(frozen=True)
class HopMetrics:
    window_index: int
    audio_now: float
    window_start: float
    window_end: float
    emit_start: float
    emit_end: float
    pipeline_started_at: float
    pipeline_finished_at: float
    pipeline_runtime_seconds: float
    emit_wall_time: float
    emit_latency_seconds: float
    num_local_speakers: int
    num_global_speakers: int
    num_candidates: int
    num_turns_created: int
    num_turns_updated: int
    num_turns_committed: int
    num_speaker_flips: int
    memory_match_count: int
    memory_update_count: int


class MetricsRecorder:
    def __init__(self, output_path: Path | str | None = None) -> None:
        if output_path is None:
            stamp = time.strftime("%Y%m%d-%H%M%S")
            output_path = Path("artifacts") / f"realtime-diarization-{stamp}.csv"
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.output_path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[field.name for field in fields(HopMetrics)],
        )
        self._writer.writeheader()
        self.rows: list[HopMetrics] = []

    def record(self, metrics: HopMetrics) -> None:
        self.rows.append(metrics)
        self._writer.writerow(asdict(metrics))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def summary(self) -> str:
        if not self.rows:
            return f"metrics={self.output_path} no hops recorded"

        runtimes = [row.pipeline_runtime_seconds for row in self.rows]
        latencies = [row.emit_latency_seconds for row in self.rows]
        return (
            f"metrics={self.output_path} hops={len(self.rows)} "
            f"pipeline_p50={_percentile(runtimes, 50):.3f}s "
            f"pipeline_p95={_percentile(runtimes, 95):.3f}s "
            f"emit_latency_p50={_percentile(latencies, 50):.3f}s "
            f"emit_latency_p95={_percentile(latencies, 95):.3f}s"
        )

    def __enter__(self) -> "MetricsRecorder":
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    if percentile == 50:
        return statistics.median(values)
    sorted_values = sorted(values)
    rank = (len(sorted_values) - 1) * percentile / 100
    lower = int(rank)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = rank - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
