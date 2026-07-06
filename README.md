# Daiya-RMT

Daiya-RMT is a research prototype for near-real-time mixed-lingual transcription, focused on Thai-English and Japanese-English conversations with speaker diarization and context-aware correction. The current repo contains the v0 API/UI prototype, Whisper LoRA training tools, dataset preparation utilities, and diarization experiments.

The project uses a root `uv` workspace. Run commands from the repository root unless a command explicitly says otherwise.

## Workspace Layout

- `daiya/` - v0 transcription prototype: FastAPI backend, CLI replay tool, and React web UI.
- `daiya/web/` - Vite/React frontend served by the backend after `npm run build`.
- `training/whisper/` - Whisper LoRA fine-tuning and merge-to-CTranslate2 tooling.
- `training/processor/whisper/` - raw audio to Hugging Face `audiofolder` dataset pipeline.
- `lab/statefull-diarization/` - throwaway pyannote speaker-memory diarization demo.
- `docs/` - design and implementation plans.

## Setup

```powershell
uv sync
```

For frontend work:

```powershell
cd daiya\web
npm install
```

The training projects pin CUDA PyTorch wheels through the `pytorch-cu128` uv index. If the local GPU driver cannot run CUDA 12.8 wheels, change the relevant `[[tool.uv.index]]` URL before syncing those workspace members.

## Script Index

Python entry points:

| Command | Package | Purpose |
| --- | --- | --- |
| `daiya` | `daiya` | Offline audio replay through the v0 transcription pipeline. |
| `daiya-whisper-lora` | `daiya-whisper-lora` | Inspect/train/merge Whisper LoRA models. |
| `start` | `daiya-whisper-lora` | Alias for `daiya-whisper-lora`. |
| `daiya-audio-label` | `whisper-dataset-pipeline` | Label raw audio and export a Whisper training dataset. |
| `daiya-whisper-clean` | `whisper-dataset-pipeline` | Alias to the dataset pipeline CLI. |
| `start` | `whisper-dataset-pipeline` | Alias to the dataset pipeline CLI. |
| `demo` | `statefull-diarization` | Stateful diarization lab demo. |

Frontend package scripts in `daiya/web`:

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite dev server. |
| `npm run build` | Type-check and build the web UI into `daiya/web/dist`. |
| `npm run preview` | Preview the built frontend with Vite. |

## V0 App Commands

Build the web UI:

```powershell
uv run --package daiya web-build
```

Run the web UI in Vite dev mode:

```powershell
uv run --package daiya web dev
```

Run the API and serve the built UI:

```powershell
uv run --package daiya uvicorn daiya.server:app --host 0.0.0.0 --port 8000
```

Open:

```text
http://127.0.0.1:8000/
```

For LAN devices, use the host machine IP, for example:

```text
http://192.168.1.168:8000/
```

Run the offline replay CLI:

```powershell
uv run --package daiya daiya training\dataset\raw\whisper\Th-En_sample_02.mp3 --no-pace --json
```

Run replay with the full workspace installed when you want the real lab pyannote
diarization backend instead of the `UNKNOWN` fallback:

```powershell
uv run --all-packages daiya training\dataset\raw\whisper\Th-En_sample_02.mp3 `
  --asr-model training\whisper\runs\medium-real-iter4-ct2-int8_float16 `
  --device cuda `
  --compute-type int8_float16 `
  --language th `
  --no-pace `
  --json
```

Useful CLI options:

- `--asr-model` - faster-whisper model name or CTranslate2 model path.
- `--device` - faster-whisper device, for example `auto`, `cpu`, or `cuda`.
- `--compute-type` - faster-whisper compute type, default `int8_float16`.
- `--language` - optional language hint.
- `--initial-prompt` - optional ASR prompt/context.
- `--chunk-seconds` - replay chunk size.
- `--no-pace` - process as fast as possible.
- `--json` - print raw JSON events.

## ASR Model Configuration

The v0 backend uses faster-whisper. Model selection is resolved in this order:

1. `DAIYA_ASR_MODEL`
2. local converted CT2 model at `training\whisper\runs\medium-real-iter4-ct2-int8_float16`
3. fallback model name `medium`

Other environment knobs:

```powershell
$env:DAIYA_ASR_MODEL='training\whisper\runs\medium-real-iter4-ct2-int8_float16'
$env:DAIYA_ASR_DEVICE='auto'
$env:DAIYA_ASR_COMPUTE_TYPE='int8_float16'
$env:DAIYA_ASR_LANGUAGE=''
$env:DAIYA_ASR_INITIAL_PROMPT=''
```

For quick smoke tests, use a smaller model:

```powershell
$env:DAIYA_ASR_MODEL='tiny'
$env:DAIYA_ASR_COMPUTE_TYPE='int8'
```

## Whisper LoRA Commands

Inspect the local Whisper dataset:

```powershell
uv run --package daiya-whisper-lora daiya-whisper-lora inspect
```

Train a LoRA adapter:

```powershell
uv run --package daiya-whisper-lora daiya-whisper-lora train `
  --output-dir training\whisper\runs\whisper-medium-lora `
  --num-train-epochs 3 `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 4 `
  --learning-rate 1e-4 `
  --fp16
```

Merge the trained adapter and convert it to CTranslate2:

```powershell
uv run --package daiya-whisper-lora daiya-whisper-lora merge `
  --adapter-path training\whisper\runs\medium-real-iter4 `
  --ct2-output-dir training\whisper\runs\medium-real-iter4-ct2-int8_float16 `
  --quantization int8_float16
```

Skip the WER gate when you only need a serving artifact quickly:

```powershell
uv run --package daiya-whisper-lora daiya-whisper-lora merge `
  --adapter-path training\whisper\runs\medium-real-iter4 `
  --ct2-output-dir training\whisper\runs\medium-real-iter4-ct2-int8_float16 `
  --quantization int8_float16 `
  --skip-wer
```

The converted serving model should contain:

```text
training\whisper\runs\medium-real-iter4-ct2-int8_float16\model.bin
```

## Dataset Pipeline Commands

The dataset processor converts raw audio into a Hugging Face `audiofolder` dataset for Whisper training.

```powershell
uv run --package whisper-dataset-pipeline daiya-audio-label
```

Override paths:

```powershell
uv run --package whisper-dataset-pipeline daiya-audio-label `
  --input-dir C:\datasets\raw `
  --output-dir C:\datasets\daiya_hf
```

Required configuration is normally placed in `training\processor\whisper\.env`. Important values include `DAIYA_INPUT_DIR`, `DAIYA_OUTPUT_DIR`, `OPENROUTER_API_KEY`, model choices, and optionally `DAIYA_FFMPEG_BIN`.

## Diarization Lab Commands

Run the stateful diarization demo:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python demo.py
```

Run synthetic tests:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python -m unittest
```

Run live capture:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python demo.py --live
```

List audio devices:

```powershell
uv run --package statefull-diarization --directory lab\statefull-diarization python -m sounddevice
```

## Tests and Validation

Backend compile check:

```powershell
python -m compileall daiya\src
```

Backend unit tests:

```powershell
$env:PYTHONPATH='daiya/src'
python -m unittest discover -s daiya\tests
```

Lockfile check:

```powershell
uv lock --check
```

Frontend build:

```powershell
uv run --package daiya web-build
```

## Current Prototype Notes

- The UI supports browser mic, server mic placeholder, desktop audio placeholder, and replay-file testing.
- `/api/replay` accepts multipart file uploads and streams NDJSON transcript events.
- `/ws/stream` accepts browser PCM chunks and JSON control messages.
- The replay and WebSocket ASR paths offload blocking pipeline work to worker threads so the API can keep serving health checks during long transcriptions.
- Speaker diarization in the v0 app is still a prototype integration; the lab contains the richer speaker-memory experiment.
- Context-aware correction is currently represented by a no-op correction stage in the v0 pipeline.

## Useful Paths

- Raw sample audio: `training\dataset\raw\whisper`
- Whisper HF dataset: `training\dataset\hf_datasets\whisper`
- LoRA adapter runs: `training\whisper\runs`
- Default CT2 ASR model: `training\whisper\runs\medium-real-iter4-ct2-int8_float16`
- Web build output: `daiya\web\dist`
- Design plan: `docs\v0-prototype-plan.md`
