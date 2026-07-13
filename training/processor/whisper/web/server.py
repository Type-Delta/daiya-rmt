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
VALIDATION_ROOT = WHISPER_ROOT / "dataset-validation"
APP_VERSION = "daiya-labeling-workbench/0.1.0"
MAX_REQUEST_BODY_BYTES = 1024 * 1024


class RequestError(ValueError):
    """A bad request that can be reported safely to the UI."""


class RequestTooLarge(RequestError):
    """A request body larger than the local API accepts."""


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
    path = Path(value).expanduser().resolve()
    if must_exist and not path.exists():
        raise RequestError(f"{label} does not exist: {path}")
    if directory is True and path.exists() and not path.is_dir():
        raise RequestError(f"{label} must be a directory: {path}")
    if directory is False and path.exists() and not path.is_file():
        raise RequestError(f"{label} must be a file: {path}")
    return path


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
        self.sessions: dict[str, dict[str, Any]] = {}
        self.allowed_audio_roots: set[Path] = set()

    def add_job(self, name: str, commands: list[list[str]], outputs: dict[str, str]) -> dict[str, Any]:
        job = {"id": uuid4().hex, "name": name, "status": "queued", "createdAt": iso_now(), "finishedAt": None, "commands": commands, "outputs": outputs, "log": []}
        with self.lock:
            self.jobs[job["id"]] = job
        threading.Thread(target=self._run_job, args=(job["id"],), daemon=True).start()
        return job

    def _run_job(self, job_id: str) -> None:
        with self.lock:
            job = self.jobs[job_id]
            job["status"] = "running"
        for command in job["commands"]:
            self._append_log(job_id, "$ " + subprocess.list2cmdline(command))
            try:
                process = subprocess.Popen(command, cwd=REPOSITORY_ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
            except OSError as exc:
                self._append_log(job_id, f"Unable to start command: {exc}")
                self._finish_job(job_id, "failed")
                return
            assert process.stdout is not None
            for line in process.stdout:
                self._append_log(job_id, line.rstrip())
            if process.wait() != 0:
                self._append_log(job_id, f"Command exited with code {process.returncode}.")
                self._finish_job(job_id, "failed")
                return
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
    command = ["uv", "run", "--no-project", "--with-editable", str(WHISPER_ROOT), "auto-label", "--input-dir", str(input_dir), "--output-dir", str(output_dir), "--work-dir", str(work_dir)]
    if bool(payload.get("noOverlapFilter")):
        command.append("--no-overlap-filter")
    return "Auto-label audio", [command], {"metadataPath": str(output_dir / "metadata.jsonl"), "audioRoot": str(output_dir)}


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
    spell_command = ["uv", "run", "--no-project", "--with-editable", f"{VALIDATION_ROOT}[spelling]", "python", str(VALIDATION_ROOT / "scripts" / "run_spelling_validation.py"), str(metadata), str(spelling)]
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
    manifest_command = ["uv", "run", "--no-project", "--with-editable", str(VALIDATION_ROOT), "python", str(VALIDATION_ROOT / "scripts" / "build_candidate_manifest.py"), str(metadata), str(audio_root), str(manifest), "--dataset-version", dataset_version, "--spelling-results", str(spelling), "--spelling-review-threshold", str(payload.get("reviewThreshold") or 0.2), "--spelling-min-issues", str(payload.get("minIssues") or 1)]
    scripts = payload.get("expectedScripts")
    if isinstance(scripts, list):
        for script in scripts:
            if str(script).strip():
                manifest_command += ["--expected-script", str(script).strip()]
    return "Validate labels and spelling", [spell_command, manifest_command], {"metadataPath": str(metadata), "audioRoot": str(audio_root), "manifestPath": str(manifest), "spellingPath": str(spelling), "runDirectory": str(run_dir)}


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
                self.send_json(HTTPStatus.OK, {"ok": True, "version": APP_VERSION, "validationRoot": str(VALIDATION_ROOT)})
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
            elif self.path == "/api/dataset/load":
                metadata = as_path(payload.get("metadataPath"), label="metadata JSONL", directory=False)
                manifest = as_path(payload["manifestPath"], label="candidate manifest", directory=False) if payload.get("manifestPath") else None
                audio_root = as_path(payload.get("audioRoot"), label="audio root", directory=True)
                rows = normalise_rows(metadata, manifest, audio_root)
                session = create_session(payload, metadata, manifest, audio_root, rows)
                STATE.add_session(session, audio_root)
                self.send_json(HTTPStatus.OK, {"rows": rows, "session": {key: value for key, value in session.items() if key not in {"reviews", "rows", "lock"}}})
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
