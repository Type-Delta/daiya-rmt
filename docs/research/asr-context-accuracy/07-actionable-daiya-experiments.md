# 07 — Actionable Daiya Experiments (Ranked)

This is the synthesis report. It pulls the concrete, Daiya-specific moves out of
reports 01–06 and ranks them by expected value: **quick win**, **medium
effort**, **high-risk research**. Each item states what to do, why (with the
source report), the exact Daiya files to touch, how to measure it, and the main
risk.

Read the topic reports for evidence and citations:

- [01 — Project landscape](01-project-landscape.md)
- [02 — Training / fine-tune methods](02-training-finetune-methods.md)
- [03 — Runtime context & streaming](03-runtime-context-and-streaming.md)
- [04 — Contextual biasing, hotwords, rescoring](04-contextual-biasing-hotwords-rescoring.md)
- [05 — LLM correction & multi-pass](05-llm-correction-and-multipass.md)
- [06 — Diarization-aware ASR](06-diarization-aware-asr.md)

## Cross-cutting themes (read first)

1. **Measure before you optimize.** Daiya's headline CER (22.9%) is computed
   against clean-read labels with a Thai-spacing-fragile WER. Two of the six
   reports independently warn this likely *inflates* Daiya's error and may
   already misrank models and checkpoints. Fixing evaluation is the cheapest way
   to stop chasing phantom regressions. → Experiments **Q1, Q2**.

2. **Daiya's runtime already has the right skeleton.** The mux's
   `apply_correction → transcript.update` path is idempotent and revision-tracked;
   `_apply_delayed_asr_correction` is already a poor-man's second pass;
   `is_low_confidence_segment` / `low_confidence_words` already exist; word
   timestamps are already decoded. Several high-value wins are *wiring existing
   primitives together*, not new plumbing. → Experiments **Q4, Q5, M4**.

3. **The two biggest accuracy levers are not runtime tweaks.** They are
   (a) **prompt-conditioned training** so the model actually learns to use
   `initial_prompt`/terms (report 02), and (b) **generation-gated checkpoint
   selection** so you stop shipping the wrong checkpoint (report 02). Everything
   in the runtime is downstream of the model you pick.

4. **In the CT2/faster-whisper runtime there is no true logit biasing.**
   `hotwords` and `initial_prompt` are the *same* prompt-injection lever (report
   04, verified from source). Real per-term weighting needs a different runtime
   (HF transformers) or training (B-Whisper). Do not expect a `hotwords` vs
   `Terms:` A/B to move the needle — spend the effort on the phonetic normalizer
   or trained biasing instead.

---

## Quick wins (low effort, high confidence)

### Q1. Generation-gated checkpoint selection ⭐ top priority
**What:** Stop selecting checkpoints by teacher-forced eval loss. Gate on
generation CER (plus a short-utterance / technical-term probe subset).
**Why:** Report 02 confirms the "loss falls while generation WER rises" pattern
is a well-documented Whisper phenomenon, and Daiya already lived it (probes
favored ckpt-400 over the best-loss final). The trainer already has the disabled
path: `--predict-with-generate` flips `metric_for_best_model` to WER.
**Files:** `training/whisper/.../train.py` (flip `predict_with_generate`, or add
an external CER callback that decodes a small probe set every N steps);
`lab/asr_eval.py` already produces the CER/short/term subsets to reuse.
**Measure:** Re-select M2's checkpoint by probe CER; compare shipped artifact vs
ckpt-400 on a *larger* held-out set (already a "Recommended Next Step" in the
model-state doc).
**Risk:** Generation eval is slow; keep the gating probe set small (32–64 clips).

### Q2. Fix the evaluation metric (deepcut WER + reference-style audit)
**What:** Add a Thai-word-tokenized WER (deepcut/newmm) alongside CER in the eval
harness, and audit whether reference labels are clean-read while any comparison
is verbatim-leaning.
**Why:** Report 02 — Thonburian and other Thai-Whisper work report WER on
deepcut tokens; Daiya's char-level CER + phrase-spacing WER is fragile and may be
hiding that M2 is already better than 22.9% implies.
**Files:** `lab/asr_eval.py` (`token_units` / `text_metrics` — add a
deepcut-based tokenizer path behind a flag).
**Measure:** Re-score the M2 vs iter4 comparison with the new metric; see if the
ranking flips.
**Risk:** Adds a Thai-tokenizer dependency to the lab tool only (not runtime).

### Q3. Fix the prompt-memory truncation bug + drop folklore instruction
**What:** In `ASRPromptMemory.build_prompt()`, the parts are ordered
`static → "Terms: …" → "Recent transcript: …"` and then passed through
`_bounded_tail(prompt, max_prompt_chars)`, which keeps the **last** N chars. On a
long Thai transcript tail the `Terms:` glossary at the front gets silently
truncated away. Reorder so terms survive (terms last, or budget terms and tail
separately), and **budget in tokens, not chars** (the real limit is 224 tokens;
Thai is token-dense). Also drop the `"Use this only as context. Do not repeat…"`
sentence — Whisper is not instruction-tuned, so it just burns prompt budget
(reports 03 & 04).
**Why:** Reports 03 and 04 — the glossary is Daiya's main cross-chunk consistency
mechanism today; truncating it defeats the purpose.
**Files:** `daiya/src/daiya/pipeline.py` (`ASRPromptMemory.build_prompt`,
`_bounded_tail`, `pipeline.py:442` instruction sentence).
**Measure:** `lab/asr_eval.py` `rolling_initial_prompt` strategy, technical-term
subset CER before/after.
**Risk:** Low. Pure runtime code; A/B is cheap.

### Q4. Word-level speaker assignment in the mux
**What:** `mux._speaker_for` assigns speakers by whole-**segment** time overlap.
WhisperX / SA-ASR / Sortformer all assign per **word**. Daiya already carries
`WordTimestamp`s through the mux — use them: assign each word (or majority-vote
words) to the max-overlap turn.
**Why:** Report 06 — this is the single biggest boundary-accuracy win available,
and it directly helps the short back-channel problem ("ครับ", "อ๋อ") landing on
the wrong speaker.
**Files:** `daiya/src/daiya/mux.py` (`_speaker_for`,
`_update_overlapping_final_segments`); words already present on
`ASRSegment.words` / `TranscriptSegment.words`.
**Measure:** Needs a diarization-labeled clip; compare speaker-attribution error
at turn boundaries segment-level vs word-level.
**Risk:** Depends on word-timestamp quality — see Q6.

### Q5. Confidence-gated correction scaffold (no LLM yet)
**What:** Replace `NoOpCorrectionStage.review` with a rule that only *flags*
low-confidence spans using the existing `is_low_confidence_segment` /
`low_confidence_words`, and route them through the existing
`mux.apply_correction`. This is the plumbing for M4/H1 and is testable with a
trivial deterministic corrector first.
**Why:** Report 05 — naive full-segment correction usually makes WER *worse*;
every source converges on confidence-gating. Daiya already has the primitives.
**Files:** `daiya/src/daiya/correct.py` (`NoOpCorrectionStage.review`),
`asr.py` (`is_low_confidence_segment`, `low_confidence_words`), `mux.py`
(`apply_correction`).
**Measure:** Unit-level: assert only flagged spans change; end-to-end deferred.
**Risk:** Low (scaffolding).

### Q6. Verify `alignment_heads` survived the LoRA-merge → CT2 conversion
**What:** Whisper word timings come from DTW over model-specific
`alignment_heads` cross-attention. After LoRA-merge + CT2 conversion these can be
wrong/default, silently corrupting word timestamps — which then corrupt Q4's
speaker assignment at exactly the turn boundaries.
**Why:** Reports 02 and 06 both flag this independently.
**Files:** conversion step (`merge.py` / CT2 convert), spot-check word timings in
`lab/asr_eval.py` output vs audio.
**Measure:** Eyeball word-timestamp alignment on a few clips; compare to the base
large-v3.
**Risk:** Low to check; high if left unchecked and you build Q4 on top of it.

### Q7. Tune short-utterance decoding thresholds
**What:** Daiya sets a good sparse temp ladder but leaves default thresholds.
Try `no_speech_threshold` 0.6→0.45, keep `compression_ratio_threshold` 2.4, and
evaluate `hallucination_silence_threshold` (requires `word_timestamps=True`,
already on). Whisper pads every clip to 30s, so short clips hallucinate fillers.
**Why:** Report 03 — cheap levers Daiya isn't currently setting.
**Files:** `daiya/src/daiya/asr.py` (`transcribe_utterance` kwargs).
**Measure:** `lab/asr_eval.py` short-utterance subset CER; count hallucinated
outputs on silence/near-silence clips.
**Risk:** Low; over-loosening `no_speech_threshold` can drop real short words.

---

## Medium effort

### M1. Prompt-conditioned training ⭐ biggest accuracy lever
**What:** Inject the dataset's context/term columns as `<|startofprev|>`
decoder-prompt tokens during training so the model actually learns to use
`initial_prompt` at inference. Optionally use B-Whisper-style shuffled
true+distractor bias-word lists.
**Why:** Report 02 — clairaudience got up to 33% WER reduction on unseen domains;
B-Whisper got 45–60% relative rare/OOV-word WER reduction. Critically, B-Whisper
showed an *un-fine-tuned* prompt slot makes things *worse* — which likely
explains why Daiya's runtime rolling-prompt is only "mixed but promising." The
model was never taught to use the slot. This targets the core
`relation กัน → รีเอชันกัน` instability.
**Files:** `training/whisper/.../train.py` (label construction — currently context
columns are deliberately not injected, per `training/whisper/README.md`); the
labeler already emits `context_before`/`context_after`/`Terms:` columns
(`training/processor/whisper/.../llm.py`).
**Measure:** Held-out CER with vs without prompt at inference; technical-term
recall subset.
**Risk:** Medium-high effort; must retrain. Gate with Q1 or you won't know if it
helped.

### M2. Wire Silero VAD (answers AGENTS.md Q6)
**What:** `SileroUtteranceSegmenter` is a `NotImplementedError` stub. Implement it.
This is called out as the single biggest runtime upgrade and directly answers the
open "best segment size for streaming" question.
**Why:** Report 03 — energy VAD is crude; Silero gives cleaner utterance
boundaries. Note the tension: faster-whisper's VAD defaults (`min_silence 2000ms`,
`speech_pad 400ms`) are far more conservative than Silero's; padding helps ASR but
blurs diarization boundaries — so tune, don't copy defaults.
**Files:** `daiya/src/daiya/asr.py` (`SileroUtteranceSegmenter`,
`create_utterance_segmenter`).
**Measure:** ASR CER *and* diarization boundary error as a function of Silero
`min_silence` / `speech_pad`.
**Risk:** Adds torch to the VAD path (already an optional extra).

### M3. Right-context / lookahead decoding (fix the boundary, not the start)
**What:** Daiya has left-audio context but no right-context/lookahead. Left
context regressed in the benchmark; report 03 argues trailing boundary words are
fixed by *right* context. Add a small lookahead pad before finalizing an
utterance, or a WhisperX-style min-cut at the quietest interior point instead of
the hard 8s `max_utterance_seconds` slice.
**Why:** Reports 03 and 06 — mid-word cuts at the 8s cap corrupt boundary words;
this likely explains the left-context regression.
**Files:** `daiya/src/daiya/asr.py` (`transcribe_utterance` left/right context),
`pipeline.py` (segmenter cap, `_transcribe_with_left_context`), the segmenter's
`max_utterance_seconds`.
**Measure:** `lab/asr_eval.py` — add a `right_audio_context` strategy alongside
the existing ones.
**Risk:** Adds latency (lookahead); keep the pad small (~0.5–1s).

### M4. Confidence-gated LLM correction as pass-2 (behind Q5)
**What:** Put a small LLM behind `NoOpCorrectionStage.review`, correcting only
low-confidence spans, fed the glossary (`ASRPromptMemory.terms()`) + a short
rolling summary, emitting `transcript.update`. Default to **2-pass**, not 3.
**Why:** Report 05 — GER works but only when confidence-gated; the mux update path
is already idempotent/revision-tracked. Model picks: Thai-EN → Typhoon 2 3B/7B;
JA-EN → Qwen2.5-7B.
**Files:** `daiya/src/daiya/correct.py`, `pipeline.py`
(`_apply_delayed_asr_correction` already models the async-correct pattern),
`mux.py` (`apply_correction`).
**Measure:** CER on low-confidence spans before/after; p95 added latency;
flicker rate of update events.
**Risk:** Over-editing / hallucinated corrections; freeze high-confidence words,
enforce minimal edits, preserve code-switch points.

### M5. Post-hoc phonetic glossary normalizer ⭐ targets the core term problem
**What:** After decode, romanize Thai/kana spans, Double-Metaphone +
edit-distance match against Daiya's canonical glossary, replace phoneticized
terms (`รีเอชัน → relation`). Runtime-agnostic; gate on low-confidence spans to
limit false inserts.
**Why:** Report 04 — since CT2 has no logit biasing, this is the highest-ROI way
to enforce cross-chunk term consistency, and it reuses `_english_domain_terms` /
`terms()`.
**Files:** `daiya/src/daiya/asr.py` (post-processing hook after
`_convert_fw_segments`, mirroring `normalize_thai_spacing`), glossary from
`pipeline.py` `ASRPromptMemory`.
**Measure:** technical-term subset CER; count correct vs false term rewrites.
**Risk:** False positives rewriting correctly-Thai words; keep the match strict
and confidence-gated. Straddles quick-win/medium — small code, needs a term list
and tuning.

### M6. Fast first-pass model via distillation (2-vs-3-pass bake-off)
**What:** Evaluate distil-large-v3.5, whisper-large-v3-turbo,
kotoba-whisper-bilingual, and Thonburian distil-whisper-th as pass-1 candidates.
Better: distill Daiya's *own* fine-tuned large-v3 (Kotoba recipe: keep full
encoder, shrink decoder to ~2 layers, ~6× speedup, small CER loss).
**Why:** Reports 01 and 02 — addresses M2's speed regression (2.9× vs 4.8×
realtime) and the open 2-vs-3-pass question.
**Files:** new training recipe under `training/whisper/`; `lab/asr_eval.py` for
the bake-off; `pipeline.py`/`asr.py` if a two-model cascade is wired.
**Measure:** CER vs RTF Pareto on the benchmark set; p95 latency.
**Risk:** A second model doubles serving complexity; only worth it if pass-1
meaningfully cuts latency without wrecking CER.

### M7. Mux hardening (min-overlap floor, collar fallback, hysteresis)
**What:** `_speaker_for` currently trusts any `overlap > 0`. Add a min-overlap
floor, a nearest-turn collar fallback (~0.25s) instead of returning `UNKNOWN`,
and a hysteresis margin in `_update_overlapping_final_segments` to cut
`transcript.update` flicker.
**Why:** Report 06 — Daiya's three-event model + `_commit_horizon` already
implements the correct commit-delay/relabel-the-past pattern; these are the
robustness gaps.
**Files:** `daiya/src/daiya/mux.py` (`_speaker_for`,
`_update_overlapping_final_segments`, `_finalize_ready_segments`).
**Measure:** speaker-attribution error + update-event count on a diarized clip.
**Risk:** Low; tune the collar/hysteresis so you don't reintroduce wrong labels.

### M8. LocalAgreement-2 streaming policy (real partial/final stabilization)
**What:** Replace the ad-hoc `_too_similar_or_repetitive` heuristic with a
proper HypothesisBuffer: committed / buffer / new tokens, confirm the
longest-common-prefix that agrees across two passes, `>0.95` prob = instant
commit. Feed the last ~200 confirmed words as the prompt.
**Why:** Report 03 — this is the mainstream streaming policy (Whisper-Streaming,
Macháček 2023; WhisperLiveKit template) and maps cleanly onto Daiya's existing
mux correction API. Published numbers: MinChunkSize 0.5s → 8.5% WER/3.27s latency.
**Files:** `daiya/src/daiya/pipeline.py` (streaming loop, correction gating),
`mux.py` (partial/final/update events).
**Measure:** WER vs latency curve at several chunk sizes; flicker rate.
**Risk:** Real re-architecture of the streaming loop; do M2 (VAD) first.

---

## High-risk research

### H1. Generative Error Correction (GER) pass-3
Feed Whisper N-best hypotheses to an LLM to produce a corrected transcript
(HyPoradise / Whispering-LLaMA / RobustGER). Big reported gains (up to ~54% rel
WER on noisy English) — but **all English-trained**; Daiya must first build its
own Thai-EN/JA hypothesis→transcription pairs. Needs N-best out of faster-whisper
(beam alternatives + logprobs). High effort, uncertain transfer. (Report 05.)

### H2. B-Whisper-style trained contextual biasing
Fine-tune with shuffled true+distractor bias-word lists (β≈1.1 weighted CE, drop
probs ~0.3/0.2) to get real per-term weighting the CT2 runtime can't do at
decode time. This is the "proper" fix for technical-term instability but couples
training and biasing. (Reports 02, 04.)

### H3. Streaming Sortformer / AOSC for stateful diarization
NVIDIA Streaming Sortformer's Arrival-Order Speaker Cache is a validated version
of Daiya's own "cache embeddings and re-match" design, with DER barely degrading
to 0.32s latency (≤4 speakers). Consider as a replacement or benchmark target for
the pyannote path. Large integration effort. (Reports 01, 06.)

### H4. Gladia-style rollback-and-re-infer on language switch
On mid-utterance LID/confidence drop, roll back and re-decode the tail — a
concrete pattern for Daiya's delayed-correction event on code-switch points.
Gladia claims ~13% code-switch WER via an ensemble of small monolingual models.
Speculative for Daiya's single-model runtime but cheap to prototype behind the
existing delayed-correction knob. (Report 01.)

### H5. HF-transformers second pass with true logit biasing
For the low-confidence spans Daiya already flags, run a targeted second decode in
the HF transformers runtime using `sequence_bias` / a custom `LogitsProcessor` /
trie-constrained decoding for genuine per-term weighting. Slower runtime, so
strictly a gated second pass. (Report 04.)

---

## Suggested sequencing

1. **Week 1 (de-risk + cheap wins):** Q1, Q2, Q3, Q6, Q7 — fix measurement and
   checkpoint selection, patch the prompt bug, verify alignment heads, tune
   thresholds. Nothing here needs a retrain and several change which numbers you
   trust.
2. **Week 2 (structural runtime wins):** Q4 + M7 (word-level speaker assignment +
   mux hardening), M5 (phonetic normalizer), Q5 scaffold.
3. **Weeks 3–4 (the real levers):** M1 (prompt-conditioned training, gated by Q1),
   M2 (Silero VAD), then M3/M6 as latency/quality data comes in.
4. **Research track (parallel, optional):** M4/M8 then H1–H5 as capacity allows.

## Honest gaps

- No source gives **Thai-EN / JA 2-vs-3-pass latency or per-GPU tokens/sec** —
  those must be measured in-house (report 05).
- The **Japanese multi-pass GER CER deltas** were not extractable from the source
  PDF and are flagged unconfirmed (report 05).
- **GER gains are all English** — transfer to Thai-EN/JA is a hypothesis, not a
  result (report 05).
- Several biasing wins (TCPGen, CB-Whisper, trie decoding) are **not usable in
  the CT2 runtime** as-is (report 04) — they gate H2/H5, not the quick wins.
