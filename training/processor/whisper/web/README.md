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

Warm the unified Whisper processor environment from the repository root. Dataset
validation is now an optional capability of that same project, so the pipeline
and validator share one `uv` environment.

```powershell
uv run --directory training/processor/whisper auto-label --help
uv run --directory training/processor/whisper --extra spelling python -c "import daiya_dataset_validation"
```

Copy `training/processor/whisper/.env.example` to `.env` and set the API,
model, and FFmpeg values needed by the auto-labeler. A Thai spellcheck engine is
enabled by default. For English spellcheck, supply a SymSpell frequency
dictionary path in the UI; for Japanese, set the dictionary level to `small`,
`core`, or `full` in the optional fields.

All paths shown in the workbench are relative to the repository root
(`daiya-rmt`). Relative paths entered in the GUI use that same root, while an
absolute path remains available for data stored outside the project.

The native folder/file buttons open a picker on the local workstation. Each path
is checked as it is entered; output directories may be new, while inputs must
already exist.

## Run

Start the development GUI with one command. It starts the local label server
automatically, then starts Vite. The server intentionally binds to `127.0.0.1`,
so local file paths and review data are not exposed on the network.

```powershell
cd training/processor/whisper/web
npm run dev
```

The server rejects non-loopback hosts by default. Network exposure requires an
explicit acknowledgement because this API accepts local paths and serves review
data:

```powershell
python server.py --host 192.168.1.20 --allow-unsafe-network-host
```

Open the printed Vite URL (normally `http://localhost:5173`). The dev server
proxies `/api` to `http://127.0.0.1:8765`. Set `DAIYA_LABEL_SERVER` before
running Vite only when the local API uses another port.

For a single local server after building:

```powershell
npm run build
npm start
# http://127.0.0.1:8765
```

## Workflow

1. In **Auto-label audio**, the processor's `.env` has already supplied the
   input, output, and work paths. Click **Run auto-labeling**. The paths must be
   separate; the output must not exist yet and the work directory must be new or
   empty. The server runs:

   ```text
   uv run --directory training/processor/whisper auto-label --input-dir … --output-dir … --work-dir …
   ```

2. In **Validate with spellcheck**, the completed auto-label outputs are already
   carried into the form. Click **Run validation**. The local API runs
   `scripts/run_spelling_validation.py` followed by
   `scripts/build_candidate_manifest.py` from the same processor environment.

3. The completed validation outputs are also carried into **Load the review
   queue**. Click **Open workbench**. Loading creates a new versioned review
   directory; the configured default can be overridden when needed. To resume a
   paused review, use its existing review directory along with the same metadata
   and candidate manifest. Saved decisions are restored from
   `current-reviews.json`.

4. Filter every automatic disposition, including **Keep**, listen to a chunk,
   and edit or confirm the human label. **Save human review** writes a provenance
   event; it never mutates the automatic source row. `Ctrl` + `S` saves the
   active clip. `Ctrl` + `Q` toggles **Drop chunk**; automatic drops start
   checked, and unchecking it enables the human label so a reviewer can keep or
   correct the chunk. `Alt` + `Up` / `Alt` + `Down` (or `Alt` + `A` / `Alt` +
   `D`) changes clips. `Ctrl` + `E` toggles focus between the human-label editor
   and the main panel; from the main panel, `Space` starts the active clip from
   its beginning, or stops and rewinds it when already playing. Outside editable
   fields, the normal number row and numpad `0`–`9` seek and play at 0%–90%.

The configuration screen is `/`; the active review screen is `/workbench`. The
configuration fields are saved locally in the browser, so returning to `/` after
closing a tab retains the paths needed to reopen the workbench. Once a queue has
been opened, reloading `/workbench` restores its saved review directory
automatically, along with the last selected chunk for that review directory.

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
