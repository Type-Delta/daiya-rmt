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

## Segmentation and migration safety

The offline processor exports **contiguous, owned wall-clock windows**. VAD only selects a window; it never causes separate speech islands to be concatenated. Brief natural pauses and short VAD false-negative gaps therefore remain audible to the labeling model.

- The default offline profile is `threshold=0.5`, `min speech=250 ms`, `min silence=150 ms`, and `speech pad=80 ms`. The source-11 benchmark retained it because the new wall-clock construction achieved zero missed human-reference speech with fewer context fallbacks than the PR #10 profile variants; boundary padding remains configurable for recordings that need more lead/trail context.
- `DAIYA_MERGE_GAP_SECONDS` (default `0.8`) bridges nearby VAD islands into a single source window and preserves the gap in the WAV.
- The 18-second target and 25-second maximum are planning goals. A deterministic scorer searches toward a configurable 30-second-or-less hard ceiling and prefers agreement among Silero VAD gaps, waveform low-energy gaps, and local Faster-Whisper word/phrase timing.
- The audio-labeling LLM remains text-only. Faster-Whisper timing is cached outside the dataset with a local-model artifact fingerprint, library/decode/acoustic settings, and source hash; it is boundary and post-label validation evidence, never an exported transcript label.
- If continuous speech still has no grounded safe boundary, the previous row owns audio through the handoff and can remain eligible. Only the next labeling input gets bounded pre-roll; its exported training WAV is a distinct owned crop and it stays review-only until the local-ASR consistency gate resolves a target-only label.
- pyannote overlap remains audible. The default `DAIYA_OVERLAP_MODE=preserve` records overlap intervals and `overlapped_speech_detected` in metadata for review. `legacy-exclude` is the only destructive mode and exists solely for controlled comparisons; it is not appropriate for new labeling.

Every new row contains exact owned and labeling-audio source timestamps (six decimal places), target offset, source identity, output audio SHA-256, segmentation version/config ID, boundary method/confidence/evidence summary, VAD/overlap evidence, local timestamp-evidence provenance, and eligibility reason. This makes regenerated runs auditable without treating an approximate timestamp match as identical audio.

Before replacing an existing dataset, build it at a fresh output path and produce a conservative migration report:

```powershell
uv run python scripts/map_segmentation_reviews.py `
  C:\datasets\old\metadata.jsonl C:\datasets\regenerated\metadata.jsonl `
  --old-audio-root C:\datasets\old --new-audio-root C:\datasets\regenerated `
  --old-reviews web\human-reviews\reviews.jsonl `
  --output output\migration\segmentation-map.json
```

The report never copies text or a human review. Only a matching exported-audio SHA-256 becomes `unchanged` or `safely_reusable_review`; changed boundaries require relabeling and multi-match cases are `ambiguous`.

To reproduce an old-versus-new segmentation benchmark on a continuous human-reference window, first make a temporary 16 kHz WAV and then run the diagnostic (the output is intentionally not committed):

```powershell
$tmp = 'C:\Temp\daiya-segmentation-benchmark'
New-Item -ItemType Directory -Force $tmp | Out-Null
ffmpeg -ss 3600 -t 600 -i C:\datasets\raw\Th-En_sample_11.m4a `
  -ac 1 -ar 16000 -c:a pcm_s16le "$tmp\sample11-3600-4200.wav"
uv run python scripts/benchmark_timestamp_ownership.py "$tmp\sample11-3600-4200.wav" `
  --reference-labels ..\..\dataset\manual-label\m2-label-ref\ref_labels.txt `
  --timestamp-model C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\m3.1\converted\checkpoint-588-ct2-int8_float16 `
  --output output\benchmarks\sample11-timestamp-ownership.json
```

The timestamp benchmark compares a faithful wall-clock-v2 baseline with ownership segmentation. It reports retained/missed human-reference speech, unprotected boundaries inside reference speech, fallback handoffs/rows, duplicated labeling versus eligible-training seconds, duration distributions, evidence coverage/disagreement, changed rows, and concrete handoffs. It does not claim LLM timestamps, transcription quality, or universal reference coverage from a limited window.

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

Each metadata row includes `file_name`, `text`, LLM transcript text, language hint, source-file context before/after, source file, timestamps, contiguous-window duration, VAD/overlap evidence, SHA-256, segmentation provenance, and training-review status.

## Dataset validation

Dataset validation now lives in this processor workspace, alongside the pipeline
that produces `metadata.jsonl`. Its audit-oriented tools are packaged as
`daiya_dataset_validation`; see [dataset validation notes](docs-dataset-validation.md)
for the CLI, spelling adapters, and manifest workflow.
