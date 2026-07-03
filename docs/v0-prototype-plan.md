# Daiya v0-Prototype Implementation Plan

This document is the agreed plan for the first end-to-end prototype: combine
the realtime stateful diarization lab (`lab/statefull-diarization`) with the
fine-tuned Whisper LoRA (`trainning/whisper/runs/medium-real-iter4`) into a
working app containing all four architecture components from `AGENTS.md`.

The v0 goal is a **real-world testing ground**: a web app that streams live
transcription with speaker labels, plus a file-replay mode that pushes a file
through the exact same code path as live audio at 1x wall-clock speed, so
realtime behavior can be tested reproducibly without a live conversation.

## Decisions (agreed 2026-07-02)

| Decision | Choice |
|---|---|
| LLM correction pass (pass 3) | **Not in v0-Prototype.** Pipeline keeps a no-op correction stage so the LLM slots in later as a drop-in. |
| ASR runtime | **Merge LoRA into base, convert to CTranslate2, serve with faster-whisper.** Merge/convert script lives in `trainning/whisper` so future adapters convert the same way. Gate: WER sanity check vs the PEFT model. |
| LoRA checkpoint | `trainning/whisper/runs/medium-real-iter4` |
| Live audio input | **Both**: browser mic → WebSocket, and server-side capture (mic + desktop/loopback via the existing sounddevice code). |
| Frontend | **Vite + React** app — the frontend is not throwaway; it will grow tuning knobs and diagnostics. Tailwind CSS + Phosphor icons, **no UI component library** (hand-rolled components). |
| Server console in UI | The frontend streams the server's console/log output live (a `log` event stream over WebSocket), so tuning sessions don't need a terminal next to the phone. |
| Deployment | Server binds to LAN so phones/other devices can join. Browsers require HTTPS (or localhost) for `getUserMedia`, so include a self-signed-cert dev setup for phone-mic testing. |

## Hardware reality

RTX 5060 Laptop, 8 GB VRAM. whisper-medium CT2 int8_float16 (~1.5 GB) +
pyannote (~1 GB) fit together. If contention appears, whisper drops to int8.

## Layout

```
daiya/                      # new uv workspace member: the product (lab/ stays a lab)
  src/daiya/
    audio.py                # AudioSource protocol: WebSocketMic, ServerCapture, FileReplay
    asr.py                  # faster-whisper wrapper + Silero-VAD utterance segmenter
    diarizer.py             # thin adapter over lab/statefull-diarization realtime driver
    mux.py                  # Multiplexer — the only genuinely new component
    correct.py              # no-op correction stage (future LLM pass)
    server.py               # FastAPI: WS /stream, replay + capture endpoints, serves web build
  web/                      # Vite + React frontend (built output served by FastAPI)
trainning/whisper/
  src/daiya_whisper_lora/merge.py   # new CLI subcommand: merge → CT2 convert → WER check
```

Component mapping to `AGENTS.md` architecture: `asr.py` = Transcription
Engine, `diarizer.py` = Speaker Diarization Engine, `mux.py` = Multiplexer,
`server.py` + `web/` = Interface Layer.

## Data flow

16 kHz mono PCM chunks — identical path whether they came from browser WS,
server capture, or file replay — fan out to two consumers:

- **Diarization**: the existing rolling-window driver, unmodified
  (window/hop/latency/commit-delay as already configured). It emits
  provisional and committed speaker segments with stable persistent IDs.
- **ASR**: Silero VAD gates utterance boundaries; transcribe on utterance end
  or an 8 s cap, with word timestamps.

**Multiplexer contract** (this is also the WS wire protocol):

- Each ASR segment gets a `segment_id` and is emitted immediately as
  `transcript.partial` with a provisional speaker (max temporal overlap with
  the *provisional* diarization timeline).
- When the diarization commit horizon passes the segment's time range, it is
  re-emitted as `transcript.final` with the committed speaker.
- Later speaker relabels (diarizer revisions) or future LLM corrections emit
  `transcript.update` for that `segment_id`.

The partial → final → update protocol *is* the "follow-up corrections"
architecture; pass 3 later becomes just another producer of
`transcript.update`.

## Phases

Each phase is independently runnable/testable.

1. **Merge + convert tooling** (`trainning/whisper`): new `merge` subcommand —
   merge iter4 adapter into `openai/whisper-medium`, convert to CT2, run WER
   on a held-out dataset slice comparing merged-CT2 vs PEFT.
   *Gate: merged WER within ~1 % absolute of PEFT; otherwise fall back to the
   transformers+PEFT runtime (slower, same pipeline shape).*
2. **Offline pipeline** (`daiya/`): audio file in → speaker-labeled transcript
   on stdout, no server. Proves the mux with reproducible input, verifies
   pyannote runs on the workspace-resolved torch (==2.11), and doubles as the
   CLI interface for free.
3. **Server**: FastAPI with the WS event protocol, file-replay endpoint
   (1x wall-clock pacing through the live path), and server-capture
   start/stop. A logging handler mirrors the server's console output as `log`
   events on a WS endpoint so the frontend can show it live. Testable with a
   script client before any UI exists.
4. **Web app** (`web/`): Vite + React with Tailwind CSS and Phosphor icons —
   no UI component library, components are hand-rolled. Mic capture via
   getUserMedia + AudioWorklet resampling to 16 k; transcript view with
   speaker colors, dim partials → solid finals, in-place updates; source
   picker (browser mic / server mic / desktop / replay file); collapsible
   server-console panel fed by the `log` event stream. Self-signed
   HTTPS dev setup for LAN phone testing.
5. **Tuning surface**: the knobs real-world testing needs — VAD threshold,
   utterance cap, diarization window/hop/commit-delay, match threshold —
   passed as WS-connect params and exposed as a simple settings panel.

## Known risks

- **Torch resolution**: one workspace venv must satisfy diarization's
  `torch>=2.8` and training's `==2.11` (resolves to 2.11). Verify pyannote on
  2.11 early, in phase 2.
- **Merged-LoRA quality drift**: gated by phase 1's WER check.
- **GPU contention** between CT2 and pyannote torch on 8 GB: mitigation is
  int8 whisper; worst case, serialize inference behind a lock.
- **getUserMedia over LAN** requires HTTPS: handled by the self-signed cert
  dev setup in phase 4.

## Deliberately out of scope for v0

- LLM correction pass (no-op stage keeps the slot).
- Separate CLI interface (phase 2's offline runner covers it).
- Auth, multi-session handling, config persistence.
- 3-pass transcription experiments (research Q4) — v0 is the testbed that
  makes those experiments possible, not the experiment itself.
