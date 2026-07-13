# Daiya Audio Dataset Pipeline

This processor turns a raw audio folder into a HuggingFace `audiofolder` dataset using an audio-capable LLM for transcription.

## Setup

```powershell
cd C:\JokaMain\ProjectShowRoom\daiya-rmt\training\processor\whisper
uv sync
Copy-Item .env.example .env
```

Edit `.env`, especially `DAIYA_INPUT_DIR`, `DAIYA_OUTPUT_DIR`, `OPENROUTER_API_KEY`, and model choices.

Set `DAIYA_LLM_CONTEXT_MAX_CHARS` to control how much source-file-specific transcription context is carried into later chunks. Context is built independently for each source file and is not shared across recordings.

Concurrency is deliberately bounded and synchronous. FFmpeg normalization uses `DAIYA_FFMPEG_MAX_WORKERS` and `DAIYA_FFMPEG_MAX_IN_FLIGHT` (defaults: 4/4); LLM source-file jobs use `DAIYA_LLM_MAX_WORKERS` and `DAIYA_LLM_MAX_IN_FLIGHT` (2/2); and dataset audio copies use `DAIYA_EXPORT_MAX_WORKERS` and `DAIYA_EXPORT_MAX_IN_FLIGHT` (4/4). Results and metadata rows are emitted in deterministic source/chunk order even when work completes out of order. Temporary media files are created beside their destinations and published atomically; failed runs remove the complete private staging tree.

`DAIYA_OUTPUT_DIR` must name a fresh directory that does not already exist. The exporter builds a unique sibling staging directory, takes a cross-process publication lock, and atomically publishes the completed dataset. Existing output is never reused or overwritten, and a failed multi-item export leaves the previous publication untouched.

GPU inference is sequential by design: Silero VAD and the pyannote overlap detector run one audio file at a time in the pipeline loop. The worker settings above do not parallelize or make those GPU clients asynchronous. The LLM client remains the synchronous OpenAI client, called from bounded worker threads only for independent source files; chunks within one source retain sequential context.

FFmpeg must be available on `PATH`, or set `DAIYA_FFMPEG_BIN`.

This project pins CUDA PyTorch wheels through the `pytorch-cu128` uv index. If your driver cannot run CUDA 12.8 wheels, change the `[[tool.uv.index]]` URL in `pyproject.toml` to the matching PyTorch wheel index before running `uv sync`.

`datasets` is pinned below `4.0` for HuggingFace audiofolder export compatibility. `pyannote.audio` uses the 4.x API and receives preloaded waveform tensors, matching the lab diarization demo so pyannote does not rely on TorchCodec for decoding normalized WAV files.

## Run

```powershell
uv run auto-label
```

You can override paths at runtime:

```powershell
uv run auto-label --input-dir C:\datasets\raw --output-dir C:\datasets\daiya_hf
```

## Output

The exporter writes:

- `metadata.jsonl`
- `<split>/*.wav`

The result can be loaded with:

```python
from datasets import load_dataset

dataset = load_dataset("audiofolder", data_dir=r"C:\datasets\daiya_hf")
```

Each metadata row includes `file_name`, `text`, LLM transcript text, language hint, source-file context before/after, source file, timestamps, and chunk duration.

## Dataset validation

Dataset validation now lives in this processor workspace, alongside the pipeline
that produces `metadata.jsonl`. Its audit-oriented tools are packaged as
`daiya_dataset_validation`; see [dataset validation notes](docs-dataset-validation.md)
for the CLI, spelling adapters, and manifest workflow.
