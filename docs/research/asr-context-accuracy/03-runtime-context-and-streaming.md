# Runtime Context & Streaming for Whisper ASR (Daiya-RMT)

Scope: runtime/decoding techniques only — **no training/fine-tuning**. This report maps published streaming-ASR practice onto Daiya's per-utterance `faster-whisper` path (`daiya/src/daiya/asr.py`, `pipeline.py`, `lab/asr_eval.py`) and proposes concrete experiments. Biasing/hotwords is only pointed at here; a separate report covers it in depth.

Daiya's current shape (from code): energy VAD segments utterances → one `WhisperModel.transcribe()` call per utterance → mux aligns to diarization by time → events emitted. `condition_on_previous_text` is **not passed** in `asr.py` (so it uses the faster-whisper default `True`), `vad_filter=False`, `word_timestamps=True`, sparse temperature ladder `[0.0, 0.8, 1.0]`, and a rolling `initial_prompt` built by `ASRPromptMemory`. In `lab/asr_eval.py`, `condition_on_previous_text` is explicitly disabled (`--no-condition-on-previous-text`) for independent per-chunk scoring.

---

## 1. `initial_prompt` vs `condition_on_previous_text` — exactly what each does

### Mechanism

Whisper's decoder prompt has three regions before generation starts: an optional **previous-text** region (the `<|startofprev|>` block), the `<|startoftranscript|>` special token, then the language/task tokens. Both `initial_prompt` and `condition_on_previous_text` write into that *previous-text* region — they are the same slot, filled from different sources.

- **`initial_prompt`** (str or token ids): tokenized via `tokenizer.encode(...)` and injected into the previous-text region of the **first window only** in `WhisperModel` (the non-batched path). It biases *style, spelling, and vocabulary* — it is context, not a transcript to reproduce. In `BatchedInferencePipeline` it is applied to **each** window instead. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))
- **`condition_on_previous_text`** (default `True` in `WhisperModel`, `False` in `BatchedInferencePipeline`): after each 30 s window, the model's **own decoded output** is fed back into the next window's previous-text region. This is what "conditioning" means — it only matters for multi-window (long-form) decodes within a single `transcribe()` call. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))
- **`prefix`** (str): different slot — inserted *after* `<|startoftranscript|>` as forced decoder output that the model must continue from (a true prefix of the transcript, not context). `hotwords` has "no effect if prefix is not None". First window only. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))
- **`prompt_reset_on_temperature`** (default `0.5`): when temperature fallback climbs above this, the accumulated previous-text prompt is **discarded** for that window. Only has effect if `condition_on_previous_text=True`. This is the built-in guard against error snowballing under high-temp sampling. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))

### The 224-token limit

Whisper's context is **448 tokens** total (`max_length=448`). The reference implementation reserves **half for the prompt: 224 tokens**. If the previous-text region (initial_prompt + carried-over text) exceeds 224 tokens, **only the last 224 are kept; earlier tokens are silently dropped** — the newest text wins. ([openai/whisper #1386](https://github.com/openai/whisper/discussions/1386), [#1824](https://github.com/openai/whisper/discussions/1824)) Note this is **tokens, not characters**, and Thai/Japanese are far less token-efficient than English, so Daiya's char-based budgets (`asr_prompt_tail_chars=420`, `asr_prompt_max_chars=900`) can translate to well over 224 tokens for Thai text — meaning the *front* of the prompt (which Daiya puts the static prompt + Terms) may be the part that gets truncated away, not the recent tail.

### Known failure modes

- **Prompt bleeding / hallucination**: the decoder can *reproduce* prompt text that was never spoken, especially on unclear/quiet audio. Whisper "starts to make things up based on previous results" when it can't recognize the audio; a standard fix is to run without prompt history (`--context 0`) when hallucination appears. ([Prompt Engineering in Whisper](https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d), [whisper.cpp #2286](https://github.com/ggml-org/whisper.cpp/discussions/2286))
- **Repetition loops**: repetitive audio input, code-switching, and fragmented/noisy audio all trigger autoregressive cycling (a phrase repeated many times). Silence at clip start makes this worse. ([Memo AI: Whisper hallucinations](https://memo.ac/blog/whisper-hallucinations), [whisper.cpp #3744](https://github.com/ggml-org/whisper.cpp/issues/3744))
- **`condition_on_previous_text` snowball**: one bad window's output poisons every subsequent window in a long decode; `prompt_reset_on_temperature` mitigates but doesn't eliminate.

### Best practices

- Keep `initial_prompt` as *terse context* (glossary + short recent tail), not a long transcript. Add an explicit instruction discouraging repetition (Daiya already does: "Do not repeat text unless it is spoken again" in `ASRPromptMemory.build_prompt`).
- Budget the prompt in **tokens**, favoring the glossary/terms and only a short tail. Because truncation keeps the *last* 224 tokens, put the most important content (terms) *last*, not first.
- For **independent** short-utterance decoding (Daiya's per-utterance calls), `condition_on_previous_text` is irrelevant within one call (single ≤30 s window) but is still `True` by default in `asr.py`; it is harmless there but should be explicitly set for clarity.

### Relevance to Daiya

`ASRPromptMemory` is exactly the "rolling `initial_prompt`" pattern that Whisper-Streaming uses (last-N confirmed words as the `prompt` param — see §2). It's the right idea. Two concrete risks in the current implementation:
1. **Token-budget inversion** (above): terms are placed at the *front* of the prompt string in `build_prompt()` (`Terms:` then `Recent transcript:`), but truncation drops the front. On Thai-heavy tails the Terms list may be silently dropped. **Fix candidate: order terms *after* the tail, or hard-cap the tail token count and always append terms last.**
2. Prompt text is the model's *own* prior output; on a hallucinated segment it feeds the hallucination forward. A confidence gate on what gets `remember()`-ed (skip segments below `logprob`/word-prob thresholds) would reduce bleed.

**Experiment**: In `lab/asr_eval.py`, add a strategy variant of `rolling_initial_prompt` that (a) measures actual prompt token length via the model tokenizer, (b) orders terms-last, and (c) gates remembered text on `avg_logprob > -1.0`. Compare CER/WER-like and count repetition hallucinations (segments with `compression_ratio`-style repetition) against the current rolling prompt.

---

## 2. Streaming policies (partials → finals → corrections)

Daiya wants **streaming partials + finals + later correction events**. The dominant open technique is **LocalAgreement-n**, from Whisper-Streaming (Macháček et al., 2023).

### LocalAgreement-2 (Whisper-Streaming / `ufal/whisper_streaming`)

**Mechanism**: re-transcribe a growing audio buffer each time a new chunk arrives. Emit as *confirmed* the **longest common prefix** that agrees across **2 consecutive** re-decodes. Tokens not yet agreed stay *unconfirmed* (partials). A timestamp marks the last confirmed word; on later passes, changes *inside* the already-confirmed region are ignored ("often insignificant"). ([ufal/whisper_streaming README](https://github.com/ufal/whisper_streaming/blob/main/README.md), [Macháček et al. 2023, arXiv:2307.14743](https://arxiv.org/html/2307.14743))

**Context fed to Whisper each pass**: the full audio **buffer (≤ ~30 s)** plus a **text prompt of the last ~200 confirmed words** — i.e. exactly the rolling-prompt pattern, used to "maintain consistency … style, terminology, and inter-sentence references." ([arXiv:2307.14743](https://arxiv.org/html/2307.14743))

**Buffer trimming**: buffer is capped near 30 s. `--buffer_trimming` = `segment` (default, trim at end of confirmed Whisper segments) or `sentence` (trim at punctuation via a sentence segmenter). `--buffer_trimming_sec` is the length threshold that triggers trimming. ([ufal README](https://github.com/ufal/whisper_streaming/blob/main/README.md))

**Concrete numbers** (`--min-chunk-size`, English, computationally-aware latency on A40; from the paper):

| MinChunkSize | WER | Latency |
|---|---|---|
| 0.5 s | 8.5% | 3.27 s |
| 1.0 s | 8.1% | 3.62 s |
| 2.0 s | 8.0% | 5.45 s |

German/Czech: 4.1–5.9 s latency, similar WER shape. Streaming WER over offline: +~2% (En), +0.2% (De), +6% (Cs). Headline: **~3.3 s latency** on unsegmented long-form. Self-adaptive: if a re-decode takes longer than MinChunkSize, the next runs immediately on accumulated audio. ([arXiv:2307.14743](https://arxiv.org/html/2307.14743))

### HypothesisBuffer / confirmed vs unconfirmed (WhisperLiveKit implementation)

WhisperLiveKit's LocalAgreement backend makes the token states explicit — useful as an implementation template for Daiya's mux:
- `committed_in_buffer` (finalized), `buffer` (unconfirmed from previous pass), `new` (current pass).
- Commit = exact match of `new` prefix against previous `buffer`. Optional `confidence_validation`: tokens with **probability > 0.95 are committed immediately**, bypassing the 2-pass agreement (lower latency at some risk).
- Unmatched tokens stay unconfirmed and become the next `buffer`.
- Trimming: `sentence` (Moses/WtP tokenizer) or `segment` (when buffer > `buffer_trimming_sec`); on trim, drop the corresponding audio and bump a `global_time_offset`. ([WhisperLiveKit LocalAgreement backend, DeepWiki](https://deepwiki.com/QuentinFuxa/WhisperLiveKit/3.2-localagreement-backend), [WhisperLiveKit repo](https://github.com/QuentinFuxa/WhisperLiveKit))

### AlignAtt / SimulStreaming (2025, successor)

Whisper-Streaming's author now recommends **SimulStreaming** (AlignAtt policy): uses the decoder's cross-attention to decide how far ahead of the audio it's safe to emit, giving lower latency than LocalAgreement but needing access to attention internals (not a black box). WhisperLiveKit ships both; LocalAgreement is "simpler and more stable … slightly higher latency." ([ufal/SimulStreaming](https://github.com/ufal/SimulStreaming), [WhisperLiveKit repo](https://github.com/QuentinFuxa/WhisperLiveKit))

### whisper.cpp streaming

`stream.cpp` has two modes: **sliding window** (`--step`, `--length`, `--keep`) and **VAD mode**. Sliding window processes fixed `--length` audio every `--step` ms with `--keep` context tokens carried from the last full segment as the next prompt (same rolling-prompt idea). Typical `--step 500 --length 5000`. Practical latency **0.5–2 s** behind live, model-size dependent; naive sliding windows split words mid-boundary (why VAD/LocalAgreement are preferred). ([whisper.cpp stream.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp), [whisper.cpp streaming DeepWiki](https://deepwiki.com/ggml-org/whisper.cpp/3.3-talk-llama))

### Relevance to Daiya

Daiya currently does **one decode per VAD utterance** — effectively "MinChunkSize = utterance length", no re-decode, no LocalAgreement. That gives finals but **no intra-utterance partials** and no principled correction. Two paths:

- **Keep utterance-final architecture** (simplest; fits mux-by-time-overlap) and add **partials** only for long utterances by periodically re-decoding the growing buffer with LocalAgreement-2 while the utterance is still open. Daiya's `utterance_cap_seconds=8.0` bounds the buffer well under 30 s.
- **Adopt LocalAgreement-2 for the correction path**: Daiya already has `apply_correction` and a `CorrectionUpdate` event in the mux, plus a delayed-correction routine (`_apply_delayed_asr_correction`) that re-decodes utterance + left window. LocalAgreement is the principled version: emit longest-common-prefix as confirmed, keep the tail as partial, and only fire a correction event when a later pass changes an *unconfirmed* token. This replaces the current ad-hoc `_too_similar_or_repetitive` heuristic.

**Experiment**: Prototype a LocalAgreement-2 wrapper in `lab/` that, given an utterance's audio, decodes at growing sub-windows (e.g. every +1.0 s) and logs (confirmed prefix, unconfirmed tail, latency-to-confirm per word) vs a single full-utterance decode. Measure added CER from early commitment and the confirm latency distribution. Reuse `--min-chunk-size` semantics (0.5/1.0/2.0 s) to reproduce the paper's tradeoff on Thai-En/Ja-En clips.

---

## 3. Chunk-boundary error handling

**Why mid-word cuts hurt**: Whisper decodes a whole window as one language-model sequence; a word split across the boundary is acoustically incomplete on both sides, so each side hallucinates a plausible completion → doubled/garbled tokens and boundary repetition. Fixed windows "can split a word in the middle." ([ufal README](https://github.com/ufal/whisper_streaming/blob/main/README.md))

Published mitigations:

- **VAD-aligned cutting at quiet points** (WhisperX Cut & Merge): split long speech regions **at the minimally-active (quietest) instant**, and **merge** short neighbors up to ~30 s. This both avoids mid-word cuts and gives Whisper near-30 s context (its sweet spot). Improves transcription WER *and* word-segmentation precision/recall vs naive chunking, and enables ~12× batched throughput. ([WhisperX, arXiv:2303.00747](https://arxiv.org/html/2303.00747v2))
- **Overlapping windows + longest-common-prefix stitching**: overlap adjacent windows and stitch by the LocalAgreement prefix logic so boundary tokens are decided by agreement, not by a hard cut. ([arXiv:2307.14743](https://arxiv.org/html/2307.14743))
- **Left/right context padding**: prepend prior audio (left context) and/or hold a lookahead of following audio (right context) so the decoded region sits *inside* a padded buffer, then clip the output to the region of interest by word timestamps. Daiya already implements left-context prepend + clip (`transcribe_utterance(left_context_samples=...)`, `_clip_segments_to_window`) — the missing half is **right context / lookahead**, which is usually more valuable for boundary completion because it lets the model finish the last word.
- **Dedup of repeated boundary tokens**: when stitching, drop tokens in the new chunk that duplicate the tail of the confirmed text.

### Relevance to Daiya

Daiya's energy segmenter (`EnergyUtteranceSegmenter`) cuts on `trailing_silence_seconds=0.45` or a hard `max_utterance_seconds=8.0` **cap**. The **cap cut is the dangerous one** — it can slice mid-word at 8 s with no silence check. WhisperX's "cut at the quietest interior point" is the direct fix for that cap. The reported regression of Daiya's *left-audio-context* strategy (per project memory) is consistent with the literature only partially: left context helps disambiguation but does **not** fix a *trailing* boundary word — right/lookahead context does. Left context can also *add* prompt-bleed/repetition if the borrowed audio's transcript leaks.

**Experiments**:
1. Add a WhisperX-style "cut at min-energy within a search window before the 8 s cap" to `EnergyUtteranceSegmenter` (and later Silero). Measure CER on the cap-cut subset specifically.
2. Add a **right-context/lookahead** variant to `transcribe_utterance` symmetric to the existing left-context path: decode `[utterance | next ~1–2 s]`, clip to utterance end by word timestamps. Compare against left-context-only and no-context on the 32-sample benchmark, reporting the *boundary-word* error separately from interior error.
3. On stitched/merged utterances, add tail-dedup and verify it removes the doubled boundary tokens.

---

## 4. Short-utterance robustness at runtime

Short clips are Whisper's weak spot: it internally **pads/trims every clip to 30 s**, so a 0.7 s utterance is 29.3 s of silence → high hallucination and repetition risk, worsened by leading silence (train/test silence mismatch: training median initial silence 0.04 s vs test ~1.92 s drives autoregressive cycling). ([arXiv:2501.11378](https://arxiv.org/html/2501.11378v1), [Memo AI](https://memo.ac/blog/whisper-hallucinations))

Runtime knobs and evidence:

- **Temperature fallback ladder**: default `[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`; a failed decode (by compression-ratio/logprob thresholds) retries at the next temperature. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)) Daiya's sparse `[0.0, 0.8, 1.0]` is a deliberate, defensible optimization: intermediate rungs rarely rescue and each rung is a full extra decode (~16 s stall on a 1 s clip per the code comment). Keep it, but note each rung is only entered when a threshold trips (below).
- **`compression_ratio_threshold`** (2.4): gzip ratio above → treat as failed (catches repetition loops). ([faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))
- **`log_prob_threshold`** (−1.0): avg token logprob below → failed → temperature bump. For noisy audio, −1.5 is a more permissive suggestion. ([Whisper API docs](https://whisper-api.com/docs/transcription-options/))
- **`no_speech_threshold`** (0.6): if `no_speech_prob` above **and** avg logprob below `log_prob_threshold` → mark segment silent (suppress). Lower to ~0.4 to be *more* aggressive about calling something silence (helps kill fillers on near-silent clips). ([faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [Whisper API docs](https://whisper-api.com/docs/transcription-options/)) Caveat: hallucinations often have **high** avg logprob and **low** no_speech_prob, so they slip past this filter — thresholds are necessary but not sufficient. ([arXiv:2606.07473](https://arxiv.org/pdf/2606.07473))
- **`hallucination_silence_threshold`** (default `None`, **requires `word_timestamps=True`**): when a hallucination is suspected, skip silent gaps longer than N seconds. Recommended range **2–8 s** (optimal in the middle); too high and it ignores post-silence hallucinations. ([faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [Whisper API docs](https://whisper-api.com/docs/transcription-options/)) **Caveat for Daiya**: this operates *within* a multi-segment long-form decode by skipping interior silence; on Daiya's already-VAD-trimmed short single-utterance clips there is little interior silence to skip, so expected benefit is small. It's more relevant if Daiya ever decodes longer merged windows.
- **Beam size** (default 5): larger beam modestly improves accuracy at latency cost; on very short clips beam offers less benefit and greedy may already loop. Daiya's `asr.py` leaves beam at default (5); `lab/asr_eval.py` exposes `--beam-size`.
- **Minimum-audio padding**: intentionally pad a very short clip with a little **real** left context (not silence) so the model has acoustic runway and less leading silence — this is Daiya's left-context idea; the literature supports *audio* context over silence padding.

### Relevance / experiments

- Daiya doesn't currently pass `compression_ratio_threshold`, `log_prob_threshold`, or `no_speech_threshold` explicitly — it relies on faster-whisper defaults, which is fine, but they are the cheapest hallucination levers to sweep. **Experiment**: in `lab/asr_eval.py`, add sweep flags for these three + `hallucination_silence_threshold`, run on the short-utterance subset (already computed via `short_utterance_seconds`), and report CER *and* a repetition-hallucination count. Hypothesis: lowering `no_speech_threshold` to ~0.45 and keeping `compression_ratio_threshold`=2.4 removes filler hallucinations on <1 s clips without hurting real speech.
- **Filler suppression**: `suppress_tokens`/`suppress_blank` (faster-whisper defaults suppress `-1` set) plus post-filtering of known hallucinated fillers (e.g., repeated "ครับ", "thank you", "うん") on low-confidence short segments.

---

## 5. VAD choices and their interaction with accuracy + diarization

### Silero VAD (upstream defaults)

`threshold=0.5`, `min_speech_duration_ms=250`, `min_silence_duration_ms=100`, `speech_pad_ms=30`, `window_size_samples=512` (for 16 kHz; models trained on 256/512/768 @8k and 512/1024/1536 @16k), `max_speech_duration_s=inf`. ([silero-vad utils_vad.py](https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py), [snakers4/silero-vad #518](https://github.com/snakers4/silero-vad/issues/518))

### faster-whisper built-in VAD (different, more conservative defaults)

`threshold=0.5`, `min_speech_duration_ms=250`, `max_speech_duration_s=inf`, **`min_silence_duration_ms=2000`**, **`window_size_samples=1024`**, **`speech_pad_ms=400`**. I.e. **20× longer min-silence, 2× window, 13× padding** vs upstream Silero. ([SYSTRAN/faster-whisper #477](https://github.com/guillaumekln/faster-whisper/issues/477)) Rationale (implied): fewer, longer, well-padded segments feed Whisper more context and avoid mid-word cuts — at the cost of latency (2 s of trailing silence before a cut). `BatchedInferencePipeline` overrides with `max_speech_duration_s=chunk_length`, `min_silence_duration_ms=160`. ([faster-whisper transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py))

### WhisperX VAD + Cut & Merge

VAD segments are cut/merged into ~30 s chunks with boundaries at minimally-active regions → batched decode, ~12× speedup, better WER + word-segmentation P/R. But WhisperX ASR-VAD and pyannote diarization use **different VAD models → temporal drift** at boundaries when combined. ([WhisperX, arXiv:2303.00747](https://arxiv.org/html/2303.00747v2), [WhisperAlign, arXiv:2603.04809](https://arxiv.org/html/2603.04809v1))

### VAD ↔ accuracy ↔ diarization

- Using external VAD boundaries (instead of Whisper's own timestamp tokens) **avoids repetition loops and hallucination in silent regions** — the core WhisperX result. ([WhisperX](https://arxiv.org/html/2303.00747v2))
- **Padding tension**: `speech_pad_ms` (400 ms in faster-whisper) helps ASR keep word onsets/offsets but **blurs speaker boundaries** — padded audio can pull in the neighboring speaker, hurting diarization at turn changes. This is directly relevant to Daiya's "best segment size that doesn't compromise diarization" open question (AGENTS.md Q6).
- **Segment length**: longer segments = better ASR context, worse latency and coarser diarization turn resolution; shorter = better diarization turn granularity, worse ASR on short clips. The tension is fundamental.

### Relevance / experiments for Daiya

Daiya's `SileroUtteranceSegmenter` is a **stub** (`NotImplementedError`) — the energy segmenter is what actually runs. Wiring real Silero VAD is the biggest single upgrade available and directly answers AGENTS.md Q6.

**Experiments**:
1. Implement `SileroUtteranceSegmenter` and sweep `threshold` (0.4/0.5/0.6), `min_silence_duration_ms` (100/300/500/2000), `speech_pad_ms` (30/200/400). For each, log **ASR CER**, **utterance count**, **median utterance length**, and **diarization boundary error** (against the mux's turns). Expect a knee where more padding stops helping CER but starts hurting diarization overlap accuracy.
2. Compare energy segmenter vs Silero on the same audio for boundary-word CER and false-cut rate.
3. Test faster-whisper's **built-in `vad_filter=True`** (currently `False` in `asr.py`) *inside* each utterance decode as a cheap cleanup of leading/trailing non-speech — but beware it may re-trim already-tight utterances and shift timestamps used by the mux.

---

## 6. Consistency across chunks

The published mechanism for cross-chunk term/spelling consistency is exactly what Daiya's `ASRPromptMemory` does and what Whisper-Streaming formalizes: carry the **last ~200 confirmed words as the `prompt`** so style, terminology, and references stay consistent across windows. ([arXiv:2307.14743](https://arxiv.org/html/2307.14743)) whisper.cpp's `--keep` (tokens from last segment as next prompt) is the same idea. ([whisper.cpp stream.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp))

- **Glossary/term injection**: putting domain/English terms in the prompt biases spelling (Daiya's `Terms:` list). Because prompt is *soft* context, this is best-effort; **hard biasing** is `hotwords` (see §1 mechanism: context tokens before `<|startoftranscript|>`, replacing history) — **deferred to the biasing report**. ([CB-Whisper, arXiv:2309.09552](https://arxiv.org/html/2309.09552v3))
- **Casing/spelling normalization**: prompts influence casing/punctuation style; a deterministic post-normalizer (Daiya already has `normalize_thai_spacing`) should own canonical term casing so the prompt and output don't drift (e.g., force "Kubernetes" not "kubernetes").
- **Risk**: the same memory that gives consistency also carries hallucinations forward (§1). Gate what enters memory on confidence.

**Experiment**: measure term-consistency directly — for a repeated domain term across a conversation, count spelling/casing variants produced with vs without prompt memory, and with terms-last ordering vs terms-first (ties to the §1 token-budget fix). `lab/asr_eval.py` already extracts `english_terms` per row, so a term-consistency metric is a small addition.

---

## 7. Latency / accuracy tradeoffs (runtime knobs)

| Knob | Accuracy effect | Latency/throughput effect | Evidence |
|---|---|---|---|
| **Beam size** (1 vs 5) | larger = modestly better, esp. on ambiguous audio; less benefit on very short clips | larger = slower per decode | faster-whisper default 5 ([transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)) |
| **Temperature ladder** | more rungs = more recovery from failures | each fallback rung is a full extra decode (Daiya notes ~16 s on a 1 s clip) | Daiya `asr.py`; faster-whisper default 6-rung |
| **Batching** (`BatchedInferencePipeline` / WhisperX) | neutral-to-better WER with VAD cut&merge | **~12×** throughput | ([WhisperX](https://arxiv.org/html/2303.00747v2), [mobiusml batched blog](https://mobiusml.github.io/batched_whisper_blog/)) |
| **Quantization** (`int8_float16` etc.) | small WER change; Daiya already ships `int8_float16` CT2 | large speed/VRAM win | Daiya config; faster-whisper compute types |
| **Window / MinChunkSize** | longer = better WER, near-30 s is Whisper's sweet spot | longer = higher latency (0.5 s→3.3 s, 2.0 s→5.5 s) | ([arXiv:2307.14743](https://arxiv.org/html/2307.14743)) |
| **condition_on_previous_text** | consistency ↑ but snowball risk | negligible | ([faster-whisper](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py)) |

**RTF/latency anchors**: Whisper-Streaming ~3.3 s latency (A40); whisper.cpp streaming 0.5–2 s behind live (model-size dependent); WhisperX batched ~12× throughput vs sequential. ([arXiv:2307.14743](https://arxiv.org/html/2307.14743), [whisper.cpp DeepWiki](https://deepwiki.com/ggml-org/whisper.cpp/3.3-talk-llama), [WhisperX](https://arxiv.org/html/2303.00747v2))

### Relevance to Daiya

Daiya is single-utterance, non-batched. If it moves to periodic re-decodes for partials (§2), **batching** the re-decodes (or diarization + ASR) becomes the main lever to keep RTF < 1. The quantized CT2 model already handles the per-decode speed; the risk is *number of decodes* (temperature fallback rungs + re-decode passes + left/right-context retries all multiply cost per utterance). **Experiment**: instrument `transcribe_utterance` to count actual decode passes per utterance (temperature rungs + context retries) and RTF, on the benchmark set, to find where the pass-count blows up (likely short degenerate clips) and cap it.

---

## Priority recommendations for Daiya (runtime-only)

1. **Fix prompt token-budgeting** (§1): order `Terms:` *last*, budget in tokens not chars, gate remembered text on confidence. Cheapest, lowest-risk win; likely improves the mixed-but-promising rolling prompt.
2. **Wire real Silero VAD** and sweep padding/min-silence (§5): unblocks AGENTS.md Q6 and reduces mid-word cap-cuts.
3. **WhisperX-style min-energy cut at the 8 s cap** (§3): removes the worst boundary errors cheaply.
4. **LocalAgreement-2 for partials + corrections** (§2): replaces ad-hoc `_too_similar_or_repetitive` with a principled confirmed/unconfirmed model that already matches Daiya's mux correction API.
5. **Add right-context/lookahead** symmetric to existing left-context (§3/§4): fixes trailing boundary words (left context alone can't).
6. **Sweep hallucination thresholds** on the short-utterance subset (§4): `no_speech_threshold` ↓, keep `compression_ratio_threshold`, test `hallucination_silence_threshold` only if merged/long windows are used.

---

## Sources

- faster-whisper `transcribe.py` (param semantics, defaults, VAD options): https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py
- Whisper prompt token limit (224): https://github.com/openai/whisper/discussions/1386 and https://github.com/openai/whisper/discussions/1824
- Prompt engineering / bleeding in Whisper: https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d
- Whisper hallucination/repetition solutions: https://memo.ac/blog/whisper-hallucinations
- whisper.cpp hallucination/repetition discussions: https://github.com/ggml-org/whisper.cpp/discussions/2286 and https://github.com/ggml-org/whisper.cpp/issues/3744
- Whisper-Streaming (LocalAgreement-2, latency/WER, buffer trimming), Macháček et al. 2023: https://arxiv.org/html/2307.14743
- ufal/whisper_streaming README: https://github.com/ufal/whisper_streaming/blob/main/README.md
- ufal/SimulStreaming (AlignAtt successor): https://github.com/ufal/SimulStreaming
- WhisperLiveKit repo: https://github.com/QuentinFuxa/WhisperLiveKit
- WhisperLiveKit LocalAgreement backend (HypothesisBuffer, confirmed/unconfirmed): https://deepwiki.com/QuentinFuxa/WhisperLiveKit/3.2-localagreement-backend
- whisper.cpp stream.cpp (sliding window --step/--length/--keep): https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp
- whisper.cpp streaming overview: https://deepwiki.com/ggml-org/whisper.cpp/3.3-talk-llama
- WhisperX (VAD Cut & Merge, batching, boundary handling), Bain et al. 2023: https://arxiv.org/html/2303.00747v2
- WhisperAlign (VAD-model drift between ASR and diarization): https://arxiv.org/html/2603.04809v1
- Silero VAD utils/defaults: https://github.com/snakers4/silero-vad/blob/master/src/silero_vad/utils_vad.py ; streaming max_speech_duration: https://github.com/snakers4/silero-vad/issues/518
- faster-whisper vs silero VAD default differences: https://github.com/guillaumekln/faster-whisper/issues/477
- Whisper hallucination via silence mismatch: https://arxiv.org/html/2501.11378v1
- Whisper hallucination detection (high-confidence hallucinations slip filters): https://arxiv.org/pdf/2606.07473
- Whisper transcription options (threshold tuning guidance): https://whisper-api.com/docs/transcription-options/
- CB-Whisper / contextual biasing (hotwords mechanism, deferred to biasing report): https://arxiv.org/html/2309.09552v3
- Batched Whisper throughput: https://mobiusml.github.io/batched_whisper_blog/
