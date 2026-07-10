from __future__ import annotations

import asyncio
import json
import logging
import tempfile
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


def create_app() -> FastAPI:
    app = FastAPI(title="Daiya v0 Prototype")
    _install_logging()

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
        pipeline = StreamingPipeline(_config_from_query(websocket.query_params))
        next_index = 0
        next_start = 0.0
        LOGGER.info("stream connected")
        try:
            while True:
                message = await websocket.receive()
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
                    for payload in await asyncio.to_thread(pipeline.accept_chunk, chunk):
                        await websocket.send_json(payload)
                    continue

                text = message.get("text")
                if text is None:
                    continue
                for payload in await _handle_text_message(text, pipeline):
                    await websocket.send_json(payload)
        except WebSocketDisconnect:
            LOGGER.info("stream disconnected")
        finally:
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
        "vad_min_speech_seconds",
        "vad_min_silence_seconds",
        "vad_speech_padding_seconds",
        "utterance_cap_seconds",
        "diarization_commit_delay_seconds",
        "window_seconds",
        "hop_seconds",
        "latency_seconds",
        "commit_delay_seconds",
        "match_threshold",
    ):
        if key in filtered:
            filtered[key] = float(filtered[key])
    for key in (
        "asr_model",
        "asr_device",
        "asr_compute_type",
        "language",
        "initial_prompt",
        "segmenter_backend",
        "diarization_profile",
        "diarization_backend",
    ):
        if key in filtered:
            filtered[key] = str(filtered[key])
    for key in ("enable_asr", "enable_diarization"):
        if key in filtered:
            filtered[key] = str(filtered[key]).lower() not in {"0", "false", "no", "off"}
    return PipelineConfig(**filtered)


def _json_line(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")


def _install_logging() -> None:
    root = logging.getLogger()
    if log_broadcaster not in root.handlers:
        root.addHandler(log_broadcaster)
    LOGGER.setLevel(logging.INFO)


def _mount_web_build(app: FastAPI) -> None:
    web_dist = Path(__file__).resolve().parents[2] / "web" / "dist"
    if web_dist.exists():
        app.mount("/", StaticFiles(directory=web_dist, html=True), name="web")


app = create_app()
