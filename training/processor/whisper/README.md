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

FFmpeg must be available on `PATH`, or set `DAIYA_FFMPEG_BIN`.

This project pins CUDA PyTorch wheels through the `pytorch-cu128` uv index. If your driver cannot run CUDA 12.8 wheels, change the `[[tool.uv.index]]` URL in `pyproject.toml` to the matching PyTorch wheel index before running `uv sync`.

`datasets` is pinned below `4.0` for HuggingFace audiofolder export compatibility. `pyannote.audio` uses the 4.x API and receives preloaded waveform tensors, matching the lab diarization demo so pyannote does not rely on TorchCodec for decoding normalized WAV files.

## Run

```powershell
uv run daiya-audio-label
```

You can override paths at runtime:

```powershell
uv run daiya-audio-label --input-dir C:\datasets\raw --output-dir C:\datasets\daiya_hf
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
