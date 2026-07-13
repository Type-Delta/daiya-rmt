"""Local-only API and command runner for the Daiya manual-labeling workbench.

It never passes a shell command to the operating system and never writes source
audio or input labels. Validation and human review outputs live in new,
timestamped directories selected by the researcher.
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import mimetypes
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, urlparse
from uuid import uuid4


WEB_ROOT = Path(__file__).resolve().parent
WHISPER_ROOT = WEB_ROOT.parent
REPOSITORY_ROOT = WEB_ROOT.parents[3]
APP_VERSION = "daiya-labeling-workbench/0.1.0"
MAX_REQUEST_BODY_BYTES = 1024 * 1024


class RequestError(ValueError):
    """A bad request that can be reported safely to the UI."""


class RequestTooLarge(RequestError):
    """A request body larger than the local API accepts."""


def read_processor_environment() -> dict[str, str]:
    """Read the processor's local configuration without importing its dependencies."""

    values: dict[str, str] = {}
    env_file = WHISPER_ROOT / ".env"
    if env_file.is_file():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    values.update({key: value for key, value in os.environ.items() if key.startswith(("DAIYA_", "OPENROUTER_"))})
    return values


def configured_path(environment: dict[str, str], name: str, default: str) -> Path:
    value = Path(environment.get(name, default)).expanduser()
    return value.resolve() if value.is_absolute() else (WHISPER_ROOT / value).resolve()


def project_relative_path(path: Path) -> str:
    """Display project files relative to the repository root when possible."""

    try:
        return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix() or "."
    except ValueError:
        return str(path.resolve())


def configured_optional_path(environment: dict[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    return project_relative_path(configured_path(environment, name, "")) if value else ""


def web_configuration() -> dict[str, Any]:
    """Return GUI defaults from the same .env configuration the processor uses."""

    environment = read_processor_environment()
    input_dir = configured_path(environment, "DAIYA_INPUT_DIR", "../../dataset/raw")
    output_dir = configured_path(environment, "DAIYA_OUTPUT_DIR", "output/hf_dataset")
    work_dir = configured_path(environment, "DAIYA_WORK_DIR", "work")
    validation_root = configured_path(environment, "DAIYA_VALIDATION_OUTPUT_DIR", "output/validation")
    review_root = configured_path(environment, "DAIYA_REVIEW_OUTPUT_DIR", "web/human-reviews")
    return {
        "projectRoot": ".",
        "autoLabel": {
            "inputDir": project_relative_path(input_dir),
            "outputDir": project_relative_path(output_dir),
            "workDir": project_relative_path(work_dir),
            "noOverlapFilter": environment.get("DAIYA_ENABLE_OVERLAP_FILTER", "true").strip().lower() in {"0", "false", "no", "off"},
        },
        "validation": {
            "metadataPath": project_relative_path(output_dir / "metadata.jsonl"),
            "audioRoot": project_relative_path(output_dir),
            "outputRoot": project_relative_path(validation_root),
            "datasetVersion": environment.get("DAIYA_VALIDATION_DATASET_VERSION", f"local-{datetime.now().date().isoformat()}"),
            "thaiEngine": environment.get("DAIYA_VALIDATION_THAI_ENGINE", "pn"),
            "expectedScripts": environment.get("DAIYA_VALIDATION_EXPECTED_SCRIPTS", "thai,latin"),
            "reviewThreshold": environment.get("DAIYA_VALIDATION_REVIEW_THRESHOLD", "0.2"),
            "minIssues": environment.get("DAIYA_VALIDATION_MIN_ISSUES", "1"),
            "allowlist": configured_optional_path(environment, "DAIYA_VALIDATION_ALLOWLIST"),
            "japaneseDictionary": environment.get("DAIYA_VALIDATION_JAPANESE_DICTIONARY", ""),
            "englishDictionary": configured_optional_path(environment, "DAIYA_VALIDATION_ENGLISH_DICTIONARY"),
        },
        "review": {
            "metadataPath": project_relative_path(output_dir / "metadata.jsonl"),
            "manifestPath": "",
            "audioRoot": project_relative_path(output_dir),
            "reviewRoot": project_relative_path(review_root),
            "reviewer": environment.get("DAIYA_REVIEWER", os.getenv("USERNAME", "local-reviewer")),
        },
    }


def iso_now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def is_loopback_host(host: str) -> bool:
    """Accept literal loopback addresses and the conventional localhost name."""

    candidate = host.strip().strip("[]")
    if candidate.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def as_path(value: object, *, label: str, must_exist: bool = True, directory: bool | None = None) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RequestError(f"{label} is required.")
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    if must_exist and not path.exists():
        raise RequestError(f"{label} does not exist: {path}")
    if directory is True and path.exists() and not path.is_dir():
        raise RequestError(f"{label} must be a directory: {path}")
    if directory is False and path.exists() and not path.is_file():
        raise RequestError(f"{label} must be a file: {path}")
    return path


def validate_ui_path(payload: dict[str, Any]) -> dict[str, Any]:
    """Check a GUI path without creating or mutating anything."""

    value = payload.get("path")
    kind = str(payload.get("kind") or "directory")
    allow_missing = bool(payload.get("allowMissing"))
    if kind not in {"file", "directory"}:
        raise RequestError("Path kind must be file or directory.")
    if not isinstance(value, str) or not value.strip():
        return {"valid": False, "exists": False, "message": "A path is required."}
    candidate = Path(value).expanduser()
    path = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
    if not path.exists():
        return {
            "valid": allow_missing,
            "exists": False,
            "path": project_relative_path(path),
            "message": "A new path will be created." if allow_missing else "This path does not exist.",
        }
    expected = path.is_dir() if kind == "directory" else path.is_file()
    return {
        "valid": expected,
        "exists": True,
        "path": project_relative_path(path),
        "message": "Ready." if expected else f"This path is not a {kind}.",
    }


def pick_ui_path(payload: dict[str, Any]) -> dict[str, Any]:
    """Open a native local picker; browsers cannot reveal real file paths."""

    kind = str(payload.get("kind") or "directory")
    if kind not in {"file", "directory"}:
        raise RequestError("Path kind must be file or directory.")
    initial = payload.get("initialPath")
    initial_dir = REPOSITORY_ROOT
    if isinstance(initial, str) and initial.strip():
        candidate = Path(initial).expanduser()
        candidate = candidate.resolve() if candidate.is_absolute() else (REPOSITORY_ROOT / candidate).resolve()
        initial_dir = candidate if candidate.is_dir() else candidate.parent
    if not initial_dir.is_dir():
        initial_dir = REPOSITORY_ROOT
    try:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askdirectory(parent=window, initialdir=str(initial_dir)) if kind == "directory" else filedialog.askopenfilename(parent=window, initialdir=str(initial_dir))
        window.destroy()
    except Exception as exc:  # pragma: no cover - platform GUI availability
        raise RequestError(f"Unable to open the native {kind} picker: {exc}") from exc
    return {"path": project_relative_path(Path(selected)) if selected else None}


def child_of(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def paths_overlap(first: Path, second: Path) -> bool:
    """Return whether two paths are equal or one is an ancestor of the other."""

    return child_of(first, second) or child_of(second, first)


def require_separate_paths(*, first: Path, second: Path, first_name: str, second_name: str) -> None:
    if paths_overlap(first, second):
        raise RequestError(
            f"{second_name} must not overlap {first_name}; choose separate paths that are not equal or ancestors/descendants."
        )


def require_fresh_directory(path: Path, *, label: str) -> None:
    """Require a caller-selected output directory to be absent or empty."""

    if not path.exists():
        return
    if not path.is_dir():
        raise RequestError(f"{label} must be a directory: {path}")
    try:
        if next(path.iterdir(), None) is not None:
            raise RequestError(f"{label} must be absent or an empty directory: {path}")
    except OSError as exc:
        raise RequestError(f"Unable to inspect {label}: {path}") from exc


def require_absent_directory(path: Path, *, label: str) -> None:
    """Require an unpublished output path so pipeline publication is atomic."""

    if path.exists():
        raise RequestError(f"{label} must not exist yet: {path}")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RequestError(f"{path.name}:{line_number} is not valid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise RequestError(f"{path.name}:{line_number} must be a JSON object.")
            rows.append(row)
    return rows


def metadata_key(row: dict[str, Any], index: int) -> str:
    return str(row.get("file_name") or row.get("uri") or row.get("id") or f"line-{index}").replace("\\", "/")


def list_value(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        return [str(item) for item in decoded] if isinstance(decoded, list) else []
    return []


def evidence_value(row: dict[str, Any]) -> list[dict[str, Any]]:
    value = row.get("evidence", row.get("evidence_json", []))
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return [{"name": "unparsed_evidence", "value": value, "source": "manifest"}]
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def nested_source(row: dict[str, Any]) -> dict[str, Any]:
    source = row.get("source")
    return source if isinstance(source, dict) else {}


def normalise_rows(metadata_path: Path, manifest_path: Path | None, audio_root: Path) -> list[dict[str, Any]]:
    metadata = read_jsonl(metadata_path)
    metadata_by_key: dict[str, dict[str, Any]] = {}
    for index, metadata_row in enumerate(metadata, 1):
        key = metadata_key(metadata_row, index)
        if key in metadata_by_key:
            raise RequestError(f"Duplicate canonical metadata identity {key!r} in {metadata_path.name}.")
        metadata_by_key[key] = metadata_row
    manifest = read_jsonl(manifest_path) if manifest_path else []
    source_rows = manifest or metadata
    normalised: list[dict[str, Any]] = []
    row_ids: set[str] = set()
    for index, row in enumerate(source_rows, 1):
        source = nested_source(row)
        uri = str(row.get("source_uri") or source.get("uri") or row.get("file_name") or row.get("uri") or "").replace("\\", "/")
        meta = metadata_by_key.get(uri, row)
        row_id = str(row.get("source_id") or source.get("source_id") or meta.get("id") or f"line-{index}")
        if row_id in row_ids:
            raise RequestError(f"Duplicate canonical row identity {row_id!r} in the loaded review queue.")
        row_ids.add(row_id)
        original = row.get("original_label", meta.get("text", meta.get("label", "")))
        proposed = row.get("proposed_label")
        proposed_text = proposed.get("text") if isinstance(proposed, dict) else None
        audio_path = (audio_root / uri).resolve() if uri else None
        normalised.append({
            "id": row_id,
            "index": index,
            "sourceUri": uri,
            "audioPath": str(audio_path) if audio_path and audio_path.is_file() else None,
            "audioAvailable": bool(audio_path and audio_path.is_file()),
            "disposition": str(row.get("disposition") or "unclassified"),
            "originalLabel": str(original or ""),
            "proposedLabel": str(proposed_text) if proposed_text is not None else None,
            "language": str(meta.get("language") or "mixed"),
            "duration": meta.get("speech_duration", meta.get("duration_seconds")),
            "reasons": list_value(row.get("reasons", [])),
            "evidence": evidence_value(row),
            "sourceStart": meta.get("source_start"),
            "sourceEnd": meta.get("source_end"),
        })
    return normalised


class AppState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.jobs: dict[str, dict[str, Any]] = {}
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.sessions: dict[str, dict[str, Any]] = {}
        self.allowed_audio_roots: set[Path] = set()

    def add_job(self, name: str, commands: list[list[str]], outputs: dict[str, str]) -> dict[str, Any]:
        job = {
            "id": uuid4().hex,
            "name": name,
            "status": "queued",
            "createdAt": iso_now(),
            "finishedAt": None,
            "commands": commands,
            "outputs": outputs,
            "log": [],
            "cancelRequested": False,
            "progress": {"current": 0, "total": max(1, len(commands)), "fraction": 0.0, "detail": "Queued"},
        }
        with self.lock:
            self.jobs[job["id"]] = job
        threading.Thread(target=self._run_job, args=(job["id"],), daemon=True).start()
        return job

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            if job["cancelRequested"] or job["status"] == "cancelled":
                return
            job["status"] = "running"
        for index, command in enumerate(job["commands"]):
            if self._cancel_requested(job_id):
                self._finish_job(job_id, "cancelled")
                return
            self._set_progress(job_id, index, index / max(1, len(job["commands"])), f"Step {index + 1} of {len(job['commands'])}")
            displayed_command = [project_relative_path(Path(part)) if Path(part).is_absolute() else part for part in command]
            self._append_log(job_id, "$ " + subprocess.list2cmdline(displayed_command))
            try:
                process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            except OSError as exc:
                self._append_log(job_id, f"Unable to start command: {exc}")
                self._finish_job(job_id, "failed")
                return
            with self.lock:
                self.processes[job_id] = process
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
                self._progress_from_output(job_id, index, len(job["commands"]), line)
            return_code = process.wait()
            process.stdout.close()
            with self.lock:
                self.processes.pop(job_id, None)
            if self._cancel_requested(job_id):
                self._append_log(job_id, "Cancellation confirmed; subprocess stopped.")
                self._finish_job(job_id, "cancelled")
                return
            if return_code != 0:
                self._append_log(job_id, f"Command exited with code {process.returncode}.")
                self._finish_job(job_id, "failed")
                return
            self._set_progress(job_id, index + 1, (index + 1) / max(1, len(job["commands"])), f"Step {index + 1} complete")
        self._finish_job(job_id, "completed")

    def _append_log(self, job_id: str, line: str) -> None:
        with self.lock:
            log = self.jobs[job_id]["log"]
            log.append(line)
            if len(log) > 250:
                del log[:-250]

    def _finish_job(self, job_id: str, status: str) -> None:
        with self.lock:
            self.jobs[job_id]["status"] = status
            self.jobs[job_id]["finishedAt"] = iso_now()
            if status == "completed":
                self.jobs[job_id]["progress"] = {**self.jobs[job_id]["progress"], "fraction": 1.0, "detail": "Complete"}

    def _set_progress(self, job_id: str, current: int, fraction: float, detail: str) -> None:
        with self.lock:
            if job_id in self.jobs:
                self.jobs[job_id]["progress"] = {"current": current, "total": self.jobs[job_id]["progress"]["total"], "fraction": max(0.0, min(1.0, fraction)), "detail": detail}

    def _progress_from_output(self, job_id: str, index: int, total: int, line: str) -> None:
        match = re.search(r"(?<!\d)(100|[1-9]?\d)%", line)
        if match:
            percent = int(match.group(1)) / 100
            self._set_progress(job_id, index, (index + percent) / max(1, total), f"Step {index + 1} of {total}: {match.group(1)}%")

    def _cancel_requested(self, job_id: str) -> bool:
        with self.lock:
            return bool(self.jobs[job_id]["cancelRequested"])

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        with self.lock:
            job = self.jobs.get(job_id)
            if job is None:
                raise RequestError("Job was not found.")
            if job["status"] not in {"queued", "running"}:
                raise RequestError("Only queued or running jobs can be cancelled.")
            job["cancelRequested"] = True
            job["progress"] = {**job["progress"], "detail": "Cancelling…"}
            process = self.processes.get(job_id)
            if job["status"] == "queued":
                job["status"] = "cancelled"
                job["finishedAt"] = iso_now()
        if process is not None:
            process.terminate()
        return dict(job, log=list(job["log"]))

    def job_list(self) -> list[dict[str, Any]]:
        with self.lock:
            return [dict(job, log=list(job["log"])) for job in sorted(self.jobs.values(), key=lambda item: item["createdAt"], reverse=True)]

    def add_session(self, session: dict[str, Any], audio_root: Path) -> None:
        with self.lock:
            self.sessions[session["id"]] = session
            self.allowed_audio_roots.add(audio_root)

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self.lock:
            return self.sessions.get(session_id)

    def allows_audio(self, audio: Path) -> bool:
        with self.lock:
            return any(child_of(audio, root) for root in self.allowed_audio_roots)


STATE = AppState()


def create_session(
    payload: dict[str, Any],
    metadata_path: Path,
    manifest_path: Path | None,
    audio_root: Path,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    reviewer = str(payload.get("reviewer") or os.getenv("USERNAME") or "local-reviewer").strip()
    default_directory = WEB_ROOT / "human-reviews" / f"human-review-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:6]}"
    directory = as_path(payload.get("reviewRoot") or str(default_directory), label="review output directory", must_exist=False, directory=True)
    require_separate_paths(first=audio_root, second=directory, first_name="audio root", second_name="review output directory")
    if directory.is_dir() and next(directory.iterdir(), None) is not None:
        return resume_session(directory, metadata_path, manifest_path, audio_root, rows)
    require_fresh_directory(directory, label="review output directory")
    directory.mkdir(parents=True, exist_ok=True)
    canonical_rows: dict[str, dict[str, Any]] = {}
    for row in rows:
        row_id = row["id"]
        if row_id in canonical_rows:
            raise RequestError(f"Duplicate canonical row identity {row_id!r} in the loaded review queue.")
        canonical_rows[row_id] = dict(row)
    metadata = {
        "schema_version": "daiya-human-review-session-1",
        "created_at": iso_now(),
        "reviewer": reviewer,
        "app_version": APP_VERSION,
        "metadata_path": str(metadata_path),
        "metadata_sha256": digest(metadata_path),
        "manifest_path": str(manifest_path) if manifest_path else None,
        "manifest_sha256": digest(manifest_path) if manifest_path else None,
        "audio_root": str(audio_root),
        "review_events": "reviews.jsonl",
        "current_reviews": "current-reviews.json",
    }
    atomic_json(directory / "session.json", metadata)
    atomic_json(directory / "current-reviews.json", {})
    return {
        "id": uuid4().hex,
        "directory": str(directory),
        "reviewer": reviewer,
        "reviews": {},
        "rows": canonical_rows,
        "lock": threading.Lock(),
        "meta": metadata,
        "resumed": False,
    }


def resume_session(directory: Path, metadata_path: Path, manifest_path: Path | None, audio_root: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Restore a durable review projection into a new in-memory API session."""

    session_file = directory / "session.json"
    projection_file = directory / "current-reviews.json"
    if not session_file.is_file() or not projection_file.is_file():
        raise RequestError("Review output directory is not empty and does not contain a resumable Daiya review session.")
    try:
        metadata = json.loads(session_file.read_text(encoding="utf-8"))
        persisted_reviews = json.loads(projection_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RequestError("Existing review session metadata is not valid JSON.") from exc
    if not isinstance(metadata, dict) or metadata.get("schema_version") != "daiya-human-review-session-1":
        raise RequestError("Existing review output directory is not a Daiya human-review session.")
    if metadata.get("metadata_sha256") != digest(metadata_path):
        raise RequestError("The selected metadata JSONL does not match the existing review session.")
    expected_manifest_hash = metadata.get("manifest_sha256")
    actual_manifest_hash = digest(manifest_path) if manifest_path else None
    if expected_manifest_hash != actual_manifest_hash:
        raise RequestError("The selected candidate manifest does not match the existing review session.")
    if not isinstance(persisted_reviews, dict):
        raise RequestError("Existing review projection must be a JSON object.")
    canonical_rows = {str(row["id"]): dict(row) for row in rows}
    restored_reviews = {
        row_id: event
        for row_id, event in persisted_reviews.items()
        if row_id in canonical_rows and isinstance(event, dict) and isinstance(event.get("human"), dict)
    }
    return {
        "id": uuid4().hex,
        "directory": str(directory),
        "reviewer": str(metadata.get("reviewer") or os.getenv("USERNAME") or "local-reviewer"),
        "reviews": restored_reviews,
        "rows": canonical_rows,
        "lock": threading.Lock(),
        "meta": metadata,
        "resumed": True,
    }


def atomic_json(path: Path, data: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _sync_directory(directory: Path) -> None:
    """Persist a replacement directory entry where the platform permits it."""

    if os.name == "nt":
        return
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def save_review(session: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    row_id = payload.get("rowId")
    if not isinstance(row_id, str) or not row_id.strip():
        raise RequestError("A loaded row ID is required to save a review.")
    row = session["rows"].get(row_id)
    if row is None:
        raise RequestError("Review row is not part of this active session.")
    text = payload.get("text")
    if not isinstance(text, str):
        raise RequestError("Review text must be a string.")
    action = str(payload.get("action") or ("edited" if text != row.get("originalLabel", "") else "confirmed"))
    if action not in {"confirmed", "edited", "skipped"}:
        raise RequestError("Review action must be confirmed, edited, or skipped.")
    event = {
        "schema_version": "daiya-human-review-1",
        "event_id": uuid4().hex,
        "saved_at": iso_now(),
        "reviewer": session["reviewer"],
        "app_version": APP_VERSION,
        "source": {"row_id": row["id"], "source_uri": row.get("sourceUri"), "audio_path": row.get("audioPath")},
        "automatic": {"disposition": row.get("disposition"), "original_label": row.get("originalLabel"), "proposed_label": row.get("proposedLabel"), "reasons": row.get("reasons", [])},
        "human": {"action": action, "label": text},
    }
    directory = Path(session["directory"])
    with session["lock"]:
        with (directory / "reviews.jsonl").open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        current = dict(session["reviews"])
        current[row_id] = event
        atomic_json(directory / "current-reviews.json", current)
        session["reviews"] = current
    return event


def build_auto_label_job(payload: dict[str, Any]) -> tuple[str, list[list[str]], dict[str, str]]:
    input_dir = as_path(payload.get("inputDir"), label="input audio directory", directory=True)
    output_dir = as_path(payload.get("outputDir"), label="dataset output directory", must_exist=False, directory=True)
    work_dir = as_path(payload.get("workDir"), label="pipeline work directory", must_exist=False, directory=True)
    require_separate_paths(first=input_dir, second=output_dir, first_name="input audio directory", second_name="dataset output directory")
    require_separate_paths(first=input_dir, second=work_dir, first_name="input audio directory", second_name="pipeline work directory")
    require_separate_paths(first=output_dir, second=work_dir, first_name="dataset output directory", second_name="pipeline work directory")
    require_absent_directory(output_dir, label="dataset output directory")
    require_fresh_directory(work_dir, label="pipeline work directory")
    command = ["uv", "run", "--directory", str(WHISPER_ROOT), "auto-label", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--work-dir", str(work_dir)]
    if bool(payload.get("noOverlapFilter")):
        command.append("--no-overlap-filter")
    return "Auto-label audio", [command], {"metadataPath": project_relative_path(output_dir / "metadata.jsonl"), "audioRoot": project_relative_path(output_dir)}


def build_validation_job(payload: dict[str, Any]) -> tuple[str, list[list[str]], dict[str, str]]:
    metadata = as_path(payload.get("metadataPath"), label="metadata JSONL", directory=False)
    audio_root = as_path(payload.get("audioRoot"), label="audio root", directory=True)
    output_root = as_path(payload.get("outputRoot"), label="validation output root", must_exist=False, directory=True)
    require_separate_paths(first=audio_root, second=output_root, first_name="audio root", second_name="validation output root")
    require_fresh_directory(output_root, label="validation output root")
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / f"dataset-validation-{datetime.now().strftime('%Y%m%dT%H%M%S')}-{uuid4().hex[:6]}"
    run_dir.mkdir()
    spelling = run_dir / "spelling-evidence.jsonl"
    manifest = run_dir / "candidate-manifest.jsonl"
    thai_engine = str(payload.get("thaiEngine") or "pn")
    japanese = payload.get("japaneseDictionary")
    english = payload.get("englishDictionary")
    if thai_engine not in {"pn", "symspellpy", "phunspell", "none"}:
        raise RequestError("Thai spellcheck engine is not supported.")
    spell_command = ["uv", "run", "--directory", str(WHISPER_ROOT), "--extra", "spelling", "python", str(WHISPER_ROOT / "scripts" / "run_spelling_validation.py"), str(metadata), str(spelling)]
    if thai_engine != "none":
        spell_command += ["--thai-engine", thai_engine]
    if japanese:
        spell_command += ["--japanese-dictionary", str(japanese)]
    if english:
        spell_command += ["--english-dictionary", str(as_path(english, label="English frequency dictionary", directory=False))]
    if thai_engine == "none" and not japanese and not english:
        raise RequestError("Enable at least one spellcheck engine.")
    if payload.get("allowlist"):
        spell_command += ["--allowlist", str(as_path(payload["allowlist"], label="allowlist", directory=False))]
    spell_command += ["--review-threshold", str(payload.get("reviewThreshold") or 0.2), "--min-issues", str(payload.get("minIssues") or 1)]
    dataset_version = str(payload.get("datasetVersion") or f"local-{datetime.now().date().isoformat()}")
    manifest_command = ["uv", "run", "--directory", str(WHISPER_ROOT), "--extra", "spelling", "python", str(WHISPER_ROOT / "scripts" / "build_candidate_manifest.py"), str(metadata), str(audio_root), str(manifest), "--dataset-version", dataset_version, "--spelling-results", str(spelling), "--spelling-review-threshold", str(payload.get("reviewThreshold") or 0.2), "--spelling-min-issues", str(payload.get("minIssues") or 1)]
    scripts = payload.get("expectedScripts")
    if isinstance(scripts, list):
        for script in scripts:
            if str(script).strip():
                manifest_command += ["--expected-script", str(script).strip()]
    return "Validate labels and spelling", [spell_command, manifest_command], {
        "metadataPath": project_relative_path(metadata),
        "audioRoot": project_relative_path(audio_root),
        "manifestPath": project_relative_path(manifest),
        "spellingPath": project_relative_path(spelling),
        "runDirectory": project_relative_path(run_dir),
    }


class Handler(BaseHTTPRequestHandler):
    server_version = "DaiyaLabeling/0.1"

    def log_message(self, format: str, *args: object) -> None:
        return

    def send_json(self, status: HTTPStatus, data: object) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise RequestError("Request body must include Content-Length.")
        try:
            length = int(raw_length)
            if length < 0:
                raise ValueError
        except (ValueError, json.JSONDecodeError) as exc:
            raise RequestError("Request body must be JSON.") from exc
        if length > MAX_REQUEST_BODY_BYTES:
            raise RequestTooLarge(f"Request body exceeds the {MAX_REQUEST_BODY_BYTES // (1024 * 1024)} MiB limit.")
        try:
            data = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise RequestError("Request body must be JSON.") from exc
        if not isinstance(data, dict):
            raise RequestError("Request body must be an object.")
        return data

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self.send_json(HTTPStatus.OK, {"ok": True, "version": APP_VERSION, "processorRoot": str(WHISPER_ROOT)})
            elif parsed.path == "/api/config":
                self.send_json(HTTPStatus.OK, web_configuration())
            elif parsed.path == "/api/jobs":
                self.send_json(HTTPStatus.OK, {"jobs": STATE.job_list()})
            elif parsed.path == "/api/audio":
                query = parse_qs(parsed.query)
                raw = query.get("path", [""])[0]
                audio = as_path(raw, label="audio path", directory=False)
                if not STATE.allows_audio(audio):
                    raise RequestError("Audio path is outside the loaded dataset roots.")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", mimetypes.guess_type(audio.name)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(audio.stat().st_size))
                self.end_headers()
                with audio.open("rb") as handle:
                    shutil.copyfileobj(handle, self.wfile)
            else:
                self.send_static(parsed.path)
        except RequestError as exc:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if isinstance(exc, RequestTooLarge) else HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def do_POST(self) -> None:  # noqa: N802
        try:
            payload = self.body()
            if self.path == "/api/jobs/autolabel":
                name, commands, outputs = build_auto_label_job(payload)
                self.send_json(HTTPStatus.ACCEPTED, {"job": STATE.add_job(name, commands, outputs)})
            elif self.path == "/api/jobs/validate":
                name, commands, outputs = build_validation_job(payload)
                self.send_json(HTTPStatus.ACCEPTED, {"job": STATE.add_job(name, commands, outputs)})
            elif self.path.startswith("/api/jobs/") and self.path.endswith("/cancel"):
                job_id = self.path.removeprefix("/api/jobs/").removesuffix("/cancel").strip("/")
                self.send_json(HTTPStatus.OK, {"job": STATE.cancel_job(job_id)})
            elif self.path == "/api/path/validate":
                self.send_json(HTTPStatus.OK, validate_ui_path(payload))
            elif self.path == "/api/path/pick":
                self.send_json(HTTPStatus.OK, pick_ui_path(payload))
            elif self.path == "/api/dataset/load":
                metadata = as_path(payload.get("metadataPath"), label="metadata JSONL", directory=False)
                manifest = as_path(payload["manifestPath"], label="candidate manifest", directory=False) if payload.get("manifestPath") else None
                audio_root = as_path(payload.get("audioRoot"), label="audio root", directory=True)
                rows = normalise_rows(metadata, manifest, audio_root)
                session = create_session(payload, metadata, manifest, audio_root, rows)
                STATE.add_session(session, audio_root)
                public_session = {key: value for key, value in session.items() if key not in {"reviews", "rows", "lock"}}
                public_session["directory"] = project_relative_path(Path(session["directory"]))
                reviews = {row_id: event["human"] for row_id, event in session["reviews"].items() if isinstance(event, dict) and isinstance(event.get("human"), dict)}
                self.send_json(HTTPStatus.OK, {"rows": rows, "session": public_session, "reviews": reviews})
            elif self.path == "/api/review/save":
                session_id = str(payload.get("sessionId") or "")
                session = STATE.get_session(session_id)
                if session is None:
                    raise RequestError("Review session is not active. Reload the dataset to start a new versioned session.")
                event = save_review(session, payload)
                self.send_json(HTTPStatus.OK, {"review": event})
            else:
                self.send_json(HTTPStatus.NOT_FOUND, {"error": "Unknown API endpoint."})
        except RequestError as exc:
            self.send_json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE if isinstance(exc, RequestTooLarge) else HTTPStatus.BAD_REQUEST, {"error": str(exc)})

    def send_static(self, request_path: str) -> None:
        dist = WEB_ROOT / "dist"
        if not dist.is_dir():
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "Frontend build missing. Run npm run dev or npm run build."})
            return
        target = (dist / request_path.lstrip("/")).resolve()
        if not child_of(target, dist) or not target.is_file():
            target = dist / "index.html"
        content = target.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="Defaults to loopback so paths and review data stay local.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--allow-unsafe-network-host",
        action="store_true",
        help="Allow a non-loopback host. This exposes local paths and review data to the network.",
    )
    args = parser.parse_args()
    if not is_loopback_host(args.host) and not args.allow_unsafe_network_host:
        parser.error("--host must be a loopback address; pass --allow-unsafe-network-host only when deliberate network exposure is required.")
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Daiya labeling API listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
