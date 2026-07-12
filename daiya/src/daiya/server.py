from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .audio import FileReplayAudioSource, chunk_from_pcm_bytes
from .pipeline import PipelineConfig, StreamingPipeline

LOGGER = logging.getLogger("daiya")


class ReplayRequest(BaseModel):
    path: str
    pace: bool = True
    chunk_seconds: float = 0.5
    config: dict[str, Any] = {}


class CaptureRequest(BaseModel):
    source: str = "default"
    config: dict[str, Any] = {}


class LogBroadcaster(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        payload = {
            "type": "log",
            "level": record.levelname.lower(),
            "logger": record.name,
            "message": self.format(record),
        }
        for queue in list(self.queues):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                pass

    def subscribe(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=256)
        self.queues.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.queues.discard(queue)


log_broadcaster = LogBroadcaster()


async def _loop_lag_watchdog() -> None:
    # debug: if uvicorn's ws keepalive dies, this shows whether the event loop stalled.
    while True:
        t0 = time.monotonic()
        await asyncio.sleep(1)
        lag = time.monotonic() - t0 - 1
        if lag > 0.5:
            LOGGER.warning("event loop stalled for %.2fs", lag)


def create_app() -> FastAPI:
    app = FastAPI(title="Daiya v0 Prototype")
    _install_logging()

    @app.on_event("startup")
    async def start_watchdog() -> None:
        asyncio.create_task(_loop_lag_watchdog())

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/logs")
    async def logs(websocket: WebSocket) -> None:
        await websocket.accept()
        queue = log_broadcaster.subscribe()
        try:
            while True:
                await websocket.send_json(await queue.get())
        except WebSocketDisconnect:
            pass
        finally:
            log_broadcaster.unsubscribe(queue)

    @app.websocket("/ws/stream")
    async def stream(websocket: WebSocket) -> None:
        await websocket.accept()
        pipeline: StreamingPipeline | None = None
        next_index = 0
        next_start = 0.0
        LOGGER.info("stream connected")

        # Drain the socket in its own task: if slow pipeline turns block reads, the
        # TCP window fills and the client's keepalive pong can't get through — uvicorn
        # then kills the connection (CLOSE 1011 keepalive ping timeout).
        # ponytail: unbounded queue — audio backlog is bounded by session length;
        # coalesce/drop frames if the backlog warning below fires in practice.
        inbox: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        async def drain_socket() -> None:
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        LOGGER.info(
                            "stream received disconnect: code=%s reason=%r",
                            message.get("code"),
                            message.get("reason"),
                        )
                        break
                    inbox.put_nowait(message)
            finally:
                inbox.put_nowait(None)

        reader = asyncio.create_task(drain_socket())
        try:
            # Model load can take tens of seconds on first request — keep it off
            # the event loop so keepalive pings still flow and the socket stays alive.
            await websocket.send_json({"type": "log", "level": "info", "source": "stream", "message": "loading models..."})
            try:
                pipeline = await asyncio.to_thread(StreamingPipeline, _config_from_query(websocket.query_params))
            except Exception as exc:
                LOGGER.exception("pipeline init failed")
                await websocket.send_json({"type": "error", "source": "stream", "message": f"pipeline init failed: {exc}"})
                await websocket.close()
                return
            await websocket.send_json({"type": "log", "level": "info", "source": "stream", "message": "models ready"})
            while True:
                message = await inbox.get()
                if message is None:
                    break
                backlog = inbox.qsize()
                if backlog and backlog % 100 == 0:
                    LOGGER.warning("pipeline running behind realtime: %d frames queued", backlog)
                if message.get("bytes") is not None:
                    try:
                        chunk = chunk_from_pcm_bytes(
                            message["bytes"],
                            start_time=next_start,
                            index=next_index,
                            source="websocket",
                        )
                    except Exception as exc:
                        await websocket.send_json({"type": "error", "source": "audio", "message": str(exc)})
                        continue
                    next_index += 1
                    next_start = chunk.end_time
                    t0 = time.monotonic()
                    payloads = await asyncio.to_thread(pipeline.accept_chunk, chunk)
                    elapsed = time.monotonic() - t0
                    if elapsed > 1.0:
                        LOGGER.warning("accept_chunk #%d took %.2fs", next_index - 1, elapsed)
                    for payload in payloads:
                        await websocket.send_json(payload)
                    continue

                text = message.get("text")
                if text is None:
                    continue
                for payload in await _handle_text_message(text, pipeline):
                    await websocket.send_json(payload)
        except WebSocketDisconnect:
            pass
        finally:
            reader.cancel()
            LOGGER.info("stream disconnected")
            if pipeline is not None:
                for payload in await asyncio.to_thread(pipeline.flush):
                    LOGGER.debug("flush event after disconnect: %s", payload)

    @app.post("/api/replay")
    async def replay(request: Request) -> StreamingResponse:
        replay_request, cleanup_path = await _replay_request_from_http(request)

        async def stream_events() -> AsyncIterator[bytes]:
            pipeline = StreamingPipeline(_config_from_dict(replay_request.config))
            source = FileReplayAudioSource(
                replay_request.path,
                chunk_seconds=replay_request.chunk_seconds,
                pace=replay_request.pace,
            )
            try:
                async for chunk in source:
                    for payload in await asyncio.to_thread(pipeline.accept_chunk, chunk):
                        yield _json_line(payload)
                for payload in await asyncio.to_thread(pipeline.flush):
                    yield _json_line(payload)
            except Exception as exc:
                LOGGER.exception("replay failed")
                yield _json_line({"type": "error", "source": "replay", "message": str(exc)})
            finally:
                if cleanup_path is not None:
                    cleanup_path.unlink(missing_ok=True)

        return StreamingResponse(stream_events(), media_type="application/x-ndjson")

    @app.post("/api/capture/start")
    async def capture_start(_request: CaptureRequest) -> JSONResponse:
        return JSONResponse(
            {
                "type": "error",
                "source": "capture",
                "message": "server-side capture is a v0 placeholder; use browser mic or file replay",
            },
            status_code=501,
        )

    @app.post("/api/capture/stop")
    async def capture_stop() -> dict[str, str]:
        return {"status": "stopped"}

    _mount_web_build(app)
    return app

async def _handle_text_message(text: str, pipeline: StreamingPipeline) -> list[dict[str, Any]]:
    try:
        message = json.loads(text)
    except json.JSONDecodeError:
        return [{"type": "error", "source": "config", "message": "text frames must be JSON"}]

    message_type = message.get("type")
    if message_type == "ping":
        LOGGER.debug("heartbeat ping received")
        return [{"type": "pong"}]
    if message_type == "flush":
        return await asyncio.to_thread(pipeline.flush)
    if message_type == "config":
        return [
            {
                "type": "log",
                "level": "warning",
                "source": "config",
                "message": "runtime config updates are acknowledged but not hot-applied in this prototype",
            }
        ]
    if message_type == "source":
        selected_source = message.get("source", "unknown")
        return [
            {
                "type": "log",
                "level": "info",
                "source": "stream",
                "message": f"selected source: {selected_source}",
            }
        ]
    return [{"type": "error", "source": "config", "message": f"unknown message type: {message_type}"}]


async def _replay_request_from_http(request: Request) -> tuple[ReplayRequest, Path | None]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("file")
        if upload is None or not hasattr(upload, "filename"):
            raise ValueError("multipart replay requires a file field")
        suffix = Path(str(getattr(upload, "filename", ""))).suffix
        temp = tempfile.NamedTemporaryFile(prefix="daiya-replay-", suffix=suffix, delete=False)
        temp_path = Path(temp.name)
        try:
            with temp:
                while chunk := await upload.read(1024 * 1024):  # type: ignore[attr-defined]
                    temp.write(chunk)
            config = _json_form_field(form.get("config"), default={})
            return (
                ReplayRequest(
                    path=str(temp_path),
                    pace=_bool_form_field(form.get("pace"), default=True),
                    chunk_seconds=float(form.get("chunk_seconds") or 0.5),
                    config=config,
                ),
                temp_path,
            )
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise

    return ReplayRequest.model_validate(await request.json()), None


def _json_form_field(value: Any, *, default: dict[str, Any]) -> dict[str, Any]:
    if value in (None, ""):
        return default
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("expected JSON object form field")


def _bool_form_field(value: Any, *, default: bool) -> bool:
    if value in (None, ""):
        return default
    return str(value).lower() in {"1", "true", "yes", "on"}


def _config_from_query(query: Any) -> PipelineConfig:
    return _config_from_dict(dict(query))


def _config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    data = {
        **data,
        "commit_delay_seconds": data.get("commit_delay_seconds", data.get("diarization_commit_delay_seconds")),
    }
    fields = PipelineConfig.__dataclass_fields__
    filtered = {key: value for key, value in data.items() if key in fields and value not in (None, "")}
    for key in (
        "vad_threshold",
        "utterance_cap_seconds",
        "diarization_commit_delay_seconds",
        "window_seconds",
        "hop_seconds",
        "latency_seconds",
        "commit_delay_seconds",
        "match_threshold",
        "asr_left_context_seconds",
        "asr_left_context_short_utterance_seconds",
        "asr_low_confidence_threshold",
        "asr_delayed_correction_window_seconds",
        "asr_tiny_utterance_seconds",
        "asr_tiny_utterance_max_gap_seconds",
        "asr_tiny_utterance_max_delay_seconds",
    ):
        if key in filtered:
            filtered[key] = float(filtered[key])
    for key in ("asr_prompt_tail_chars", "asr_prompt_max_chars", "asr_prompt_max_terms"):
        if key in filtered:
            filtered[key] = int(filtered[key])
    for key in (
        "asr_model",
        "asr_device",
        "asr_compute_type",
        "language",
        "initial_prompt",
        "diarization_profile",
        "diarization_backend",
    ):
        if key in filtered:
            filtered[key] = str(filtered[key])
    for key in (
        "enable_asr",
        "enable_diarization",
        "asr_prompt_memory_enabled",
        "asr_left_context_enabled",
        "asr_delayed_correction_enabled",
        "asr_tiny_utterance_merge_enabled",
    ):
        if key in filtered:
            filtered[key] = str(filtered[key]).lower() not in {"0", "false", "no", "off"}
    return PipelineConfig(**filtered)


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _install_logging() -> None:
    root = logging.getLogger()
    if log_broadcaster not in root.handlers:
        root.addHandler(log_broadcaster)
    LOGGER.setLevel(logging.DEBUG if os.environ.get("DAIYA_DEBUG") else logging.INFO)


def _mount_web_build(app: FastAPI) -> None:
    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")


app = create_app()
