# Daiya manual-labeling workbench

A local React workbench for reviewing auto-labeled Whisper chunks. It runs the
existing `daiya_whisper_pipeline`, drives the migrated
`daiya_dataset_validation` spelling/manifest commands, plays chunk audio, and
writes human review decisions as a separate append-only artifact.

It does not write source audio, `metadata.jsonl`, or a candidate manifest. The
pipeline and validation tools each write their own new outputs; every loaded
queue creates a fresh timestamped `human-review-*` directory containing
`session.json`, append-only `reviews.jsonl`, and a latest-state
`current-reviews.json` projection. The active review session and loaded rows
are intentionally memory-only: after the API server stops, start a new session
and use the saved artifacts as the durable record rather than expecting an
in-app resume.

## One-time setup

From this directory:

```powershell
npm install
```

Warm the processor and validator environments from the repository root. These
commands deliberately use `--no-project` so an unrelated broken root-workspace
member cannot prevent the local tools from running. The server uses the same
isolated project invocation.

```powershell
uv run --no-project --with-editable training/processor/whisper daiya-audio-label --help
uv run --no-project --with-editable "training/processor/whisper/dataset-validation[spelling]" python -c "import daiya_dataset_validation"
```

Copy `training/processor/whisper/.env.example` to `.env` and set the API,
model, and FFmpeg values needed by the auto-labeler. A Thai spellcheck engine is
enabled by default. For English spellcheck, supply a SymSpell frequency
dictionary path in the UI; for Japanese, set the dictionary level to `small`,
`core`, or `full` in the optional fields.

## Run

Start the local API first. It intentionally binds to `127.0.0.1`, so local file
paths and review data are not exposed on the network.

```powershell
cd training/processor/whisper/web
python server.py
```

The server rejects non-loopback hosts by default. Network exposure requires an
explicit acknowledgement because this API accepts local paths and serves review
data:

```powershell
python server.py --host 192.168.1.20 --allow-unsafe-network-host
```

In another terminal, start Vite:

```powershell
cd training/processor/whisper/web
npm run dev
```

Open the printed Vite URL (normally `http://localhost:5173`). The dev server
proxies `/api` to `http://127.0.0.1:8765`. Set `DAIYA_LABEL_SERVER` before
running Vite only when the local API uses another port.

For a single local server after building:

```powershell
npm run build
python server.py
# http://127.0.0.1:8765
```

## Workflow

1. In **Auto-label audio**, choose an existing input directory, a dataset output
   path that does not exist yet, and a new or empty work directory. The paths
   must also be separate from each other. The server runs:

   ```text
   uv run --no-project --with-editable training/processor/whisper daiya-audio-label --input-dir … --output-dir … --work-dir …
   ```

2. In **Validate with spellcheck**, use the resulting `metadata.jsonl` and its
   audiofolder root. Select a new or empty validation output root. The local API runs
   `dataset-validation/scripts/run_spelling_validation.py` followed by
   `build_candidate_manifest.py`; both import `daiya_dataset_validation` from
   `training/processor/whisper/dataset-validation`.

3. Copy the completed job outputs into **Load the review queue** (or press
   **Use outputs**). Loading creates a new versioned review directory. Supply a
   new or empty review output directory if the default is not appropriate.

4. Filter every automatic disposition, including **Keep**, listen to a chunk,
   and edit or confirm the human label. **Save human review** writes a provenance
   event; it never mutates the automatic source row. `Alt` + `Enter` saves the
   active clip.

The source root is deliberately rejected as an auto-label output, pipeline work,
validation output, or review output location, including equal and
ancestor/descendant paths. Audio playback is restricted to the audio root
registered when the queue was loaded, and subprocesses are launched as argument
lists without a shell. Jobs can be observed in the UI but cannot be cancelled or
resumed there; keep the API process running until a job completes.

## Verification

```powershell
cd training/processor/whisper/web
python -m unittest -v test_server.py
npm run build
```
