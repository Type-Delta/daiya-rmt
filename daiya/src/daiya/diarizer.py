from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np

from .audio import PCMChunk, SAMPLE_RATE
from .mux import DiarizationTurn


class DiarizerUnavailableError(RuntimeError):
    """Raised when the optional lab/pyannote diarizer cannot be configured."""


@dataclass(frozen=True)
class DiarizerConfig:
    profile: str = "balanced"
    window_seconds: float | None = None
    hop_seconds: float | None = None
    latency_seconds: float | None = None
    commit_delay_seconds: float | None = None
    match_threshold: float | None = None


class NullDiarizer:
    """Fallback diarizer that produces UNKNOWN turns so the mux path still runs."""

    def __init__(self, *, commit_delay_seconds: float = 0.0, speaker_id: str = "UNKNOWN") -> None:
        self.commit_delay_seconds = commit_delay_seconds
        self.speaker_id = speaker_id
        self._next_id = 0
        self._open_turns: list[DiarizationTurn] = []

    def accept(self, chunk: PCMChunk) -> list[DiarizationTurn]:
        turn = DiarizationTurn(
            turn_id=f"null_{self._next_id:06d}",
            start=chunk.start_time,
            end=chunk.end_time,
            speaker_id=self.speaker_id,
            confidence=0.0,
            final=False,
        )
        self._next_id += 1
        self._open_turns.append(turn)
        events = [turn]
        horizon = chunk.end_time - self.commit_delay_seconds
        remaining: list[DiarizationTurn] = []
        for open_turn in self._open_turns:
            if open_turn.end <= horizon:
                events.append(_as_final(open_turn))
            else:
                remaining.append(open_turn)
        self._open_turns = remaining
        return events

    def flush(self) -> list[DiarizationTurn]:
        events = [_as_final(turn) for turn in self._open_turns]
        self._open_turns = []
        return events


class LabRealtimeDiarizer:
    """Adapter over lab/statefull-diarization without modifying lab files."""

    def __init__(self, backend: object, *, config: DiarizerConfig | None = None) -> None:
        modules = load_lab_modules()
        realtime = modules["realtime"]
        speaker_memory = modules["speaker_memory"]
        config = config or DiarizerConfig()
        lab_config = _make_lab_config(realtime, config)
        self._scheduler = realtime.RollingWindowScheduler(SAMPLE_RATE, lab_config, channels=1)
        memory_kwargs = {}
        if config.match_threshold is not None:
            memory_kwargs["match_threshold"] = config.match_threshold
        self._driver = realtime.RealtimeDiarizationDriver(
            backend=backend,
            memory=speaker_memory.SpeakerMemory(**memory_kwargs),
            config=lab_config,
        )

    def accept(self, chunk: PCMChunk) -> list[DiarizationTurn]:
        events: list[DiarizationTurn] = []
        block = np.asarray(chunk.samples, dtype=np.float32)
        for window in self._scheduler.append(block):
            hop = self._driver.process_window(window)
            events.extend(_timeline_event_to_turns(hop.events))
        return events

    def flush(self) -> list[DiarizationTurn]:
        return []


def create_diarizer(
    *,
    backend: object | None = None,
    config: DiarizerConfig | None = None,
    prefer_real: bool = True,
    allow_null: bool = True,
) -> LabRealtimeDiarizer | NullDiarizer:
    if backend is not None:
        return LabRealtimeDiarizer(backend, config=config)
    if prefer_real:
        try:
            return LabRealtimeDiarizer(create_lab_pyannote_backend(), config=config)
        except DiarizerUnavailableError:
            if not allow_null:
                raise
    if allow_null:
        delay = 0.0 if config is None else float(config.commit_delay_seconds or 0.0)
        return NullDiarizer(commit_delay_seconds=delay)
    raise DiarizerUnavailableError(
        "pyannote/lab diarization backend is not configured; pass a lab backend or allow null fallback"
    )


def create_lab_pyannote_backend(root: Path | None = None) -> object:
    """Load the lab pyannote pipeline with the same patches as the lab demo."""

    lab_root = root or _default_lab_root()
    _load_env_file(lab_root / ".env")
    try:
        modules = load_lab_modules(lab_root)
        demo = modules["demo"]
        backends = modules["backends"]
        pipeline = demo.load_pyannote_pipeline()
        return backends.PyannotePipelineBackend(pipeline)
    except SystemExit as exc:
        raise DiarizerUnavailableError(str(exc)) from exc
    except DiarizerUnavailableError:
        raise
    except Exception as exc:
        raise DiarizerUnavailableError(f"failed to load lab pyannote backend: {exc}") from exc


def load_lab_modules(root: Path | None = None) -> dict[str, ModuleType]:
    lab_root = root or _default_lab_root()
    if not lab_root.exists():
        raise DiarizerUnavailableError(f"lab diarization path not found: {lab_root}")
    inserted = False
    if str(lab_root) not in sys.path:
        sys.path.insert(0, str(lab_root))
        inserted = True
    try:
        return {
            name: _load_module_from_path(f"daiya_lab_statefull_{name}", lab_root / f"{name}.py")
            for name in ("timeline", "backends", "metrics", "speaker_memory", "realtime", "demo")
        }
    finally:
        if inserted:
            try:
                sys.path.remove(str(lab_root))
            except ValueError:
                pass


def _load_module_from_path(name: str, path: Path) -> ModuleType:
    if not path.exists():
        raise DiarizerUnavailableError(f"expected lab module does not exist: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise DiarizerUnavailableError(f"cannot load lab module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        raise DiarizerUnavailableError(
            f"cannot import lab diarization module {path.name}: {exc}"
        ) from exc
    return module


def _default_lab_root() -> Path:
    return Path(__file__).resolve().parents[3] / "lab" / "statefull-diarization"


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def _make_lab_config(realtime: ModuleType, config: DiarizerConfig) -> object:
    lab_config = realtime.RealtimeDiarizationConfig.for_profile(config.profile)
    values = {
        "window_seconds": config.window_seconds,
        "hop_seconds": config.hop_seconds,
        "latency_seconds": config.latency_seconds,
        "commit_delay_seconds": config.commit_delay_seconds,
    }
    for field, value in values.items():
        if value is not None:
            lab_config = _replace_dataclass(lab_config, field, float(value))
    return lab_config


def _replace_dataclass(instance: object, field: str, value: float) -> object:
    from dataclasses import replace

    return replace(instance, **{field: value})


def _timeline_event_to_turns(events: Iterable[object]) -> list[DiarizationTurn]:
    turns: list[DiarizationTurn] = []
    for event in events:
        turn = getattr(event, "turn", None)
        if turn is None:
            continue
        event_type = str(getattr(event, "type", ""))
        if event_type == "turn.deleted":
            continue
        turns.append(
            DiarizationTurn(
                turn_id=str(getattr(turn, "turn_id")),
                start=float(getattr(turn, "start")),
                end=float(getattr(turn, "end")),
                speaker_id=str(getattr(turn, "speaker_id")),
                confidence=float(getattr(turn, "speaker_confidence", 0.0)),
                final=bool(getattr(turn, "final", False)),
                version=int(getattr(turn, "version", 1)),
            )
        )
    return turns


def _as_final(turn: DiarizationTurn) -> DiarizationTurn:
    return DiarizationTurn(
        turn_id=turn.turn_id,
        start=turn.start,
        end=turn.end,
        speaker_id=turn.speaker_id,
        confidence=turn.confidence,
        final=True,
        version=turn.version + 1,
    )
