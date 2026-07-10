# Post-ASR LLM Correction and Multi-Pass Decoding for Daiya

Research report on the two open Daiya questions: **"2 vs 3 pass transcription — which is more efficient?"** and **"best small LLM for verifying/correcting transcription, fast + accurate enough?"** Focused on Generative Error Correction (GER), N-best rescoring, correction robustness, context/terminology-aware correction, audio-capable LLMs, small/local Thai-capable models, and streaming-safe correction.

Daiya today has a **NoOp** correction stage (`daiya/src/daiya/correct.py`) that returns no corrections, a mux (`daiya/src/daiya/mux.py`) that can already emit `transcript.update` correction events, and a pipeline (`daiya/src/daiya/pipeline.py`) that already implements one form of "delayed correction" by **re-decoding audio with left context** (`_apply_delayed_asr_correction`). The `asr.py` module already exposes `is_low_confidence_segment` and `low_confidence_words`. This report maps every technique to those hooks.

> Evidence vs speculation: WER/latency numbers below are quoted from the cited papers. Where a number is not stated in a source, this is flagged explicitly. Relevance-to-Daiya and experiment suggestions are my synthesis, not claims from the papers.

---

## 1. Generative Error Correction (GER): feeding N-best to an LLM

### Mechanism
Instead of the classic ASR pipeline picking the single top beam, GER feeds the **N-best hypothesis list** (the top N decoded candidates) to an LLM and asks it to *generate* the corrected transcript. This differs fundamentally from LM rescoring: rescoring can only re-rank existing candidates, while a generative LLM can produce tokens **absent from every hypothesis** (the "beyond the oracle" property). ([HyPoradise, NeurIPS 2023](https://arxiv.org/abs/2309.15701))

### Evidence
- **HyPoradise** (NeurIPS 2023) is the seminal open benchmark: **334K+ pairs of N-best hypotheses ↔ ground-truth transcriptions** across WSJ, CHiME-4, LibriSpeech, ATIS, CommonVoice, TED-LIUM, SwitchBoard, etc. It shows LLM GER "surpasses the upper bound of traditional re-ranking-based methods" — i.e. beats the N-best oracle — and can recover tokens missing from the N-best list entirely. ([arXiv 2309.15701](https://arxiv.org/abs/2309.15701)) The paper uses a fixed prompt template presenting the ranked hypotheses and asking for the true transcription; the LLM is fine-tuned with LoRA/adapters on the HT pairs.
- **Whispering-LLaMA** (EMNLP 2023) is the most directly relevant to Daiya because it uses **Whisper**: it takes Whisper's N-best hypotheses, and fuses Whisper's **acoustic encoder embeddings** into a LLaMA decoder via a small adapter (**only 7.97M trainable params**). Reported **37.66% relative WER improvement vs the N-best oracle** across diverse datasets. Code + weights open: [github.com/Srijith-rkr/Whispering-LLaMA](https://github.com/Srijith-rkr/Whispering-LLaMA). ([arXiv 2310.06434](https://arxiv.org/abs/2310.06434))
- **RobustGER** (ICLR 2024, "LLMs are Efficient Learners of Noise-Robust Speech Recognition") extends HyPoradise to noisy audio with a **language-space noise embedding** distilled from the audio, reporting **up to 53.9% relative WER reduction** on the Robust HyPoradise set (113K noisy HT pairs from CHiME-4, VoiceBank-DEMAND, NOIZEUS, etc.). Uses LLaMA/LLaMA-2 + LoRA. Code: [github.com/YUCHEN005/RobustGER](https://github.com/YUCHEN005/RobustGER). ([arXiv 2401.10446](https://arxiv.org/abs/2401.10446))
- **Japanese multi-pass GER benchmark** (2024) applies GER to Japanese ASR-LLM setups with an iterative multi-pass correction loop; confirms the paradigm transfers to Japanese orthography (kanji/kana), though the PDF's exact CER deltas were not machine-extractable here (**number not confirmed from source**). ([arXiv 2408.16180](https://arxiv.org/pdf/2408.16180))

### Relevance to Daiya
This is the exact "pass 3" the AGENTS.md plan describes. The strongest signal for Daiya: **GER needs an N-best list, not just the 1-best text.** Daiya currently only keeps the 1-best `ASRSegment.text`. To use real GER you must expose N-best from faster-whisper (see §2). Whispering-LLaMA is the closest published recipe (Whisper + small LLM + fused acoustic embeddings) and would be the reference implementation to study.

### Risks
- Off-the-shelf GER models are **trained on English datasets**. None of HyPoradise/RobustGER cover Thai-English or Japanese-English code-switching; the reported WER deltas **do not transfer** without fine-tuning on Daiya's own HT pairs.
- Fine-tuned GER is another model to train/serve on top of the already-fine-tuned Whisper — real cost for a research prototype.

### Concrete experiment
Build a small Daiya "Robust-HyPoradise-TH/JA" set: run the merged-LoRA Whisper-large-v3 with `beam_size=5` over the existing eval clips, dump the **5-best hypotheses + ground truth** to JSONL. First measure the **oracle WER** (best-of-5 vs 1-best) — this is the free upper bound of any rescoring approach and tells you whether N-best even helps on your data before investing in an LLM. Then try zero-shot GER with a small instruct LLM (Typhoon 2 7B, Qwen2.5-7B) prompting "Here are 5 ASR hypotheses, output the single most likely transcription."

---

## 2. Getting N-best / logprobs out of faster-whisper; second-pass rescoring

### Mechanism
Whisper decodes with beam search (default `beam_size=5`, logprob scoring). The beam *does* hold N candidate sequences internally, but faster-whisper's public API returns only the 1-best plus scores. To get N-best you either (a) raise `best_of`/`beam_size` and patch the decoder to return all finished beams, or (b) run multiple temperature samples. faster-whisper **does** natively expose per-segment `avg_logprob` and, with `word_timestamps=True`, **per-word `probability`** — which Daiya already captures (`WordTimestamp.probability`, `ASRSegment.confidence`). ([faster-whisper issue #1358](https://github.com/SYSTRAN/faster-whisper/issues/1358), [openai/whisper discussion #1619](https://github.com/openai/whisper/discussions/1619))

### Evidence
- Standard Whisper returns a single hypothesis; getting true N-best "requires custom modifications to the standard implementation" — there is no one-flag API. ([openai/whisper #478](https://github.com/openai/whisper/discussions/478), [#1619](https://github.com/openai/whisper/discussions/1619))
- **Two-pass deliberation** (Google): an RNN-T first pass streams fast, a second-pass transformer/LAS **re-decodes attending to both the audio encoder output AND the first-pass hypotheses** (that dual attention is what makes it "deliberation" rather than plain rescoring). Reported **12% relative WER reduction vs LAS rescoring**, and **23% on a proper-noun test set**. ([Transformer deliberation, arXiv 2101.11577](https://arxiv.org/abs/2101.11577); [non-autoregressive deliberation, arXiv 2112.11442](https://arxiv.org/pdf/2112.11442))
- Non-autoregressive / parallel rescoring reduces second-pass latency by scoring tokens in parallel rather than autoregressively. ([Parallel Rescoring, arXiv 2008.13093](https://arxiv.org/pdf/2008.13093))
- Constrained decoding over a **10-best T5** with lattice constraints gave **up to 12% relative WER reduction** vs a strong Conformer-Transducer on LibriSpeech. ([Revisiting ASR EC, arXiv 2405.15216](https://arxiv.org/html/2405.15216v2))

### Relevance to Daiya
The **proper-noun / entity gain (23%)** is the interesting signal: deliberation and N-best rescoring mostly help exactly the tokens Daiya cares about — technical terms and code-switch English words embedded in Thai/Japanese. Daiya's existing `_apply_delayed_asr_correction` is effectively a **poor-man's second pass**: it re-decodes the finalized utterance with extra left context and emits a `transcript.update`. That is deliberation-lite without an LLM.

### Risks / notes
Patching faster-whisper's CT2 beam to emit all beams is non-trivial and couples you to library internals. The pragmatic N-best source is **temperature sampling** (faster-whisper already falls back through `temperature=[0.0, 0.8, 1.0]` — see `asr.py`), or `best_of` sampling — cheaper to wire than a beam patch, though lower-quality candidates.

### Concrete experiment
Add an `ASRSegment.alternatives: tuple[str, ...]` field populated from temperature/`best_of` samples, gated behind a config flag so it costs nothing when off. Feed those alternatives into the correction stage (§1). Measure oracle WER of the alternatives first; if oracle ≈ 1-best, N-best won't help and you should skip GER entirely and rely on re-decode + context prompting.

---

## 3. When LLM correction HELPS vs HURTS (over-correction / hallucination)

This is the single most important robustness finding for Daiya and it is unusually consistent across papers: **naive, unconstrained LLM correction frequently makes WER WORSE than the raw ASR baseline.**

### Evidence
- **Confidence-Guided Error Correction** (2025): "naive LLM correction significantly degraded performance compared to the original ASR baseline." Embedding **word-level confidence into the prompt/training** and correcting only low-confidence regions gave **10% relative WER reduction vs naive LLM correction on spontaneous speech, and 47% on TORGO** (disordered speech). ([arXiv 2509.25048](https://arxiv.org/abs/2509.25048))
- **Interfacing LLMs with ASR via confidence measures** (2024): proposes three confidence filters — correct only when the *sentence* confidence is low, only when the *lowest word* confidence is low, or correct only the specific low-confidence words. Confidence gating prevents the LLM from touching already-correct high-confidence spans. ([arXiv 2407.21414](https://arxiv.org/html/2407.21414v1))
- **Specialized models beat LLMs in the low-error regime** (2405.15216): a compact seq2seq corrector (**15× fewer params than an LLM**) hit 1.5%/3.3% WER on LibriSpeech clean/other and gave "precise corrections in the low-error regime where LLMs struggle." Training the corrector to **abstain (copy the input)** on non-improvable pairs "significantly reduces overcorrection, especially out-of-domain." ([arXiv 2405.15216](https://arxiv.org/html/2405.15216v2))
- **RLLM-CF / three-stage verification** (2025): (1) error pre-detection, (2) chain-of-thought iterative correction, (3) reasoning verification — the CoT sub-tasks exist specifically "to constrain the LLM output space and prevent it from adding or deleting words without basis." ([arXiv 2505.24347](https://arxiv.org/abs/2505.24347))
- Contextual-biasing surveys repeatedly note LLMs "are prone to hallucinations or overcorrection without strict constraints." ([contextual biasing, arXiv 2512.21828](https://arxiv.org/html/2512.21828v1))

### Relevance to Daiya — this is a near-perfect fit
Daiya **already has the gating primitives**: `is_low_confidence_segment(segment, min_avg_logprob, min_word_probability)` and `low_confidence_words(segment, min_word_probability)` in `asr.py`. The literature says: **only invoke the corrector when `is_low_confidence_segment` is True, and only let it rewrite the `low_confidence_words` spans**, keeping high-probability words frozen. That converts the biggest risk (over-editing correct code-switch terms) into a controlled operation.

### Risks
- If the LLM is allowed to rewrite the full segment, it will "fix" correctly-transcribed Thai or English technical jargon it doesn't recognize — the classic over-correction failure. Freeze high-confidence words.
- LLMs may **normalize** valid code-switching (e.g. rewrite an English loanword into Thai script, or vice-versa), which is *wrong* for Daiya's goal of faithful mixed-lingual transcription. The correction prompt must explicitly forbid this.

### Concrete experiment
Replace `NoOpCorrectionStage.review` with a `ConfidenceGatedCorrectionStage`: return `[]` immediately unless `is_low_confidence_segment(segment)`. When triggered, send the LLM the segment text with low-confidence words marked (e.g. `⟦word⟧`) plus an instruction to **only** alter marked spans. Emit the result as a `CorrectionUpdate` (which the mux already turns into `transcript.update`). A/B measure WER **with correction on the whole test set vs on gated segments only** — expect gating to be the difference between improvement and regression.

---

## 4. Context / terminology-aware correction (glossary + rolling summary)

### Mechanism
Give the corrector a **bias list** (glossary of expected technical terms / named entities / participant names) and optionally a **rolling conversation summary** at correction time, so it can fix domain terms and code-switch spelling from context rather than guessing.

### Evidence
- LLM-based contextual biasing with a provided hotword/bias list yields large entity gains: recent systems report **up to 54.3% relative entity-WER reduction and 41.3% overall WER improvement**. ([contextual biasing survey, arXiv 2512.21828](https://arxiv.org/html/2512.21828v1))
- **Contextual Biasing of Named-Entities with LLMs** (2023): injecting entity lists into the LLM prompt improves rare-entity recognition. ([arXiv 2309.00723](https://arxiv.org/pdf/2309.00723))
- **PARCO** (phoneme-augmented) and **LOGIC** (logit-space integration) show prompt-only biasing is fragile at large vocab; phoneme/logit approaches are more robust — relevant if Daiya's glossary grows large. ([PARCO, arXiv 2509.04357](https://arxiv.org/pdf/2509.04357); [LOGIC, arXiv 2601.15397](https://arxiv.org/html/2601.15397v1))
- Retrieval-augmented contextual ASR (retrieve relevant glossary entries per-utterance rather than dumping the whole list) reduces prompt bloat. ([RAG contextual ASR, ACL 2025 Findings](https://aclanthology.org/2025.findings-emnlp.203.pdf))

### Relevance to Daiya — already half-built
Daiya's `ASRPromptMemory` in `pipeline.py` **already mines English domain terms** from the running transcript (`_english_domain_terms`, `_is_useful_english_term`) and builds a `Terms: ...` + `Recent transcript: ...` prompt that is fed to Whisper as `initial_prompt`. This is contextual biasing **at the acoustic decoder**. The natural extension: pass the *same* `terms()` list + a rolling summary into the **LLM correction prompt** (pass 3), so the corrector fixes term spelling the acoustic model missed. Daiya's `match_threshold`/prompt infra means the glossary already exists in memory — reuse it.

### Risks
Large glossaries dumped into a prompt degrade the LLM (attention dilution) and increase latency/cost — hence the retrieval/logit approaches. Keep the per-utterance term list bounded (Daiya already caps at `asr_prompt_max_terms=24`).

### Concrete experiment
Feed `ASRPromptMemory.terms()` into the §3 gated corrector's system prompt as an allow-list ("these terms are known-correct spellings; prefer them for low-confidence spans"). Measure term-level accuracy (NEER) on clips containing known glossary terms, not just overall WER — the gain concentrates on entities.

---

## 5. Multi-pass tradeoffs: 2-pass vs 3-pass, cascade vs single-large

### Mechanism / the actual Daiya question
- **2-pass**: large Whisper (fast enough) → LLM context/term correction.
- **3-pass**: tiny fast Whisper (instant partial) → large Whisper (replace) → LLM correction.
- **Cascade** (small→large) vs **single large**: run a cheap model first for low latency, escalate to the large model only when needed vs always run large.

### Evidence
- Two-pass RNN-T→deliberation: first pass **streams for low latency**, second pass refines; **12% rel WER** improvement, non-autoregressive rescoring keeps second-pass latency low. ([arXiv 2101.11577](https://arxiv.org/abs/2101.11577), [2112.11442](https://arxiv.org/pdf/2112.11442))
- The specialized-corrector paper argues a **small dedicated corrector often beats a big LLM** on both latency and accuracy in the low-error regime — an argument for **skipping the LLM pass** when the large ASR is already good, i.e. 2-pass is often enough. ([arXiv 2405.15216](https://arxiv.org/html/2405.15216v2))
- **Partial Rewriting for Multi-Stage ASR** (2023): a later stage rewrites only the *unstable/changed* portion of an earlier stage's output rather than re-emitting everything — directly relevant to a streaming 3-pass design. ([arXiv 2312.09463](https://arxiv.org/html/2312.09463v1))

*No source gives a Thai-English / Japanese-English 2-vs-3-pass latency table — that number does not exist in the literature and Daiya must measure it in-house.*

### Relevance to Daiya
Given the AGENTS.md note "if the Large model is fast enough, skip the first pass," and given faster-whisper-large-v3 int8_float16 on GPU is typically real-time-capable for short utterances, **2-pass (large ASR → async LLM correction) is the pragmatic default**; add pass-1 tiny only if measured large-model latency breaks the streaming budget. The literature does **not** support adding an LLM pass unconditionally — it should be **confidence-gated and asynchronous** (§3, §8).

### Concrete experiment
Instrument the pipeline: log per-utterance ASR decode wall-time in `_transcribe_utterance`. If p95 decode latency on target hardware < your streaming hop, you have a 2-pass system; the "pass 1 tiny model" is unnecessary complexity (a YAGNI candidate). Only benchmark 3-pass if 2-pass p95 latency fails.

---

## 6. Audio-capable LLMs as a re-listening verification pass

### Mechanism
Instead of (or in addition to) text-only GER, send the **raw audio of a low-confidence span** to a multimodal audio LLM (Gemini, GPT-4o-audio, Qwen2-Audio/Qwen3-Omni) and ask it to transcribe/verify — it can use acoustic evidence the text corrector cannot.

### Evidence
- **Qwen2-Audio / Qwen3-Omni**: open-weight audio LLMs. Qwen3-Omni reports open-source SOTA on 32/36 audio benchmarks, **outperforming Gemini-2.5-Pro and GPT-4o-Transcribe** on many, with streaming **end-to-end latency as low as 234 ms**. ([Qwen3-Omni report, arXiv 2509.17765](https://arxiv.org/html/2509.17765v1); [Qwen2-Audio repo](https://github.com/qwenlm/qwen2-audio))
- Local audio-LLM median latencies observed ~0.9–1.4 s per query (Flamingo 0.92s, Qwen2 1.41s) — **too slow for per-word, acceptable for occasional low-confidence spans**. ([Stanford CS191 audio LLM analysis](https://cs191.stanford.edu/projects/Spring2025/Laya___Iyer_.pdf))
- **Cost** (cloud, order-of-magnitude): for 100K min/month, GPT-4o Realtime audio-in ≈ $1,366/mo, Gemini (Flash tier) ≈ $427/mo. ([WaveSpeed comparison](https://wavespeed.ai/blog/posts/qwen3-5-omni-vs-gpt4o-gemini-2026/)) These are illustrative, not Daiya-specific.

### Relevance to Daiya
This is the highest-accuracy verification option and the most natural "pass 3" for **code-switch spans** where text-only correction is blind to pronunciation. But it re-introduces an audio dependency and network/latency/cost. Best used **sparingly**: only re-listen to segments flagged by `is_low_confidence_segment`, only after the segment is finalized (async), emitting a `transcript.update`.

### Risks
- Cloud audio LLMs = privacy concern for real conversations + recurring cost + network latency (kills the "near-real-time local" property). Qwen2-Audio local is the privacy-preserving option but needs GPU headroom on top of Whisper + pyannote.
- Audio LLMs hallucinate too; still gate + constrain.

### Concrete experiment
Prototype an optional `AudioReListenCorrectionStage`: for segments where `is_low_confidence_segment` is True, slice the utterance audio (Daiya already retains `_audio_history` and has `_audio_between`), send to Qwen2-Audio locally with prompt "transcribe this Thai-English clip; the current guess is: <text>". Compare its output to the ASR text; only emit a `CorrectionUpdate` if they disagree materially (reuse `_too_similar_or_repetitive`). Measure latency per call and WER gain on the flagged subset only.

---

## 7. Small / local LLMs for fast correction (Thai-English capable)

### Evidence
- **Typhoon 2** (SCB 10X, Dec 2024): open-weight Thai+English models at **1B / 3B / 7B / 8B / 70B**, built on Llama 3 and Qwen2.5. The **7B** delivers the highest scores on IFEval-TH, MT-Bench-TH, and explicitly a **Code-Switching Accuracy** metric, **outperforming Qwen2.5-7B**. The paper even defines a code-switching eval measuring unwanted non-Thai characters when following Thai instructions. This is the best-documented small Thai-English-capable open model. ([Typhoon 2, arXiv 2412.13702](https://arxiv.org/html/2412.13702v1); [release blog](https://opentyphoon.ai/blog/en/typhoon-2-release-9dd36e3882c0))
- **Qwen2.5 3B/7B**: strong multilingual base (includes Japanese, Chinese; decent Thai); the substrate under Typhoon. Good default for the **Japanese-English** side.
- **Specialized compact correctors** (§3): 15× smaller than LLMs, faster, and better in the low-error regime — a **non-LLM alternative** worth considering over a 7B model for latency. ([arXiv 2405.15216](https://arxiv.org/html/2405.15216v2))
- No source gives exact tokens/sec for these models on Daiya's specific GPU — **latency must be measured locally** (not stated in sources).

### Relevance to Daiya
For the **Thai-English** pass, **Typhoon 2 3B or 7B** is the standout recommendation because it is the only small open model with a published code-switching metric and Thai-English optimization. For **Japanese-English**, Qwen2.5-7B (or its Japanese-tuned derivatives) is the natural pick. Run quantized (GGUF/llama.cpp or the CT2/AWQ path) to fit alongside Whisper + pyannote.

### Risks
Even a 3B model adds hundreds of ms per correction; that's why gating (§3) + async (§8) matter. A 7B may not co-reside on the same GPU as Whisper-large-v3 + pyannote without VRAM pressure — measure.

### Concrete experiment
Benchmark Typhoon 2 3B (Thai clips) and Qwen2.5 3B/7B (Japanese clips) quantized, on the target GPU, measuring **tokens/sec and time-to-correct a typical 8s-utterance segment**. Compare WER gain vs a specialized T5-small corrector. Pick per-language.

---

## 8. Streaming considerations: correcting already-emitted segments without flicker

### Mechanism
Corrections arrive **after** a segment was already shown to the user. To avoid "flicker" (words visibly changing/jumping), corrections must be applied to **finalized** segments only, be **idempotent**, and be **revision-tracked**.

### Evidence
- **Flickering Reduction with Partial Hypothesis Reranking** (Google): reranking partials toward a stable prefix "roughly halves the amount of flickering with negligible impact on quality and latency." ([Bruguier et al., PDF](https://www.bruguier.com/pub/deflickering.pdf); [IEEE](https://ieeexplore.ieee.org/document/10023016))
- **Analyzing Quality and Stability of Streaming ASR** (Interspeech 2020): defines word/segment-level **instability metrics**; most stability fixes work by **delaying** partials, trading latency for stability. ([arXiv 2006.01416](https://arxiv.org/abs/2006.01416))
- **Revision-Controllable Decoding** (2023): bound how much already-emitted output a later pass is allowed to revise. ([arXiv 2310.04399](https://arxiv.org/pdf/2310.04399))
- **Partial Rewriting for Multi-Stage ASR** (2023): rewrite only the changed portion in later stages. ([arXiv 2312.09463](https://arxiv.org/html/2312.09463v1))

### Relevance to Daiya — architecture already supports this
The mux is **already designed for idempotent, revision-tracked correction**:
- `CorrectionUpdate(segment_id, text, words, speaker_id)` targets a specific segment by id.
- `TranscriptMultiplexer.apply_correction` (`mux.py:140`) looks up the segment, `replace(...)` with `revision=current.revision + 1`, and **returns `[]` if nothing changed** (`if updated == current: return []`) — that is idempotency for free.
- It emits `TranscriptEvent("transcript.update", updated, current, source="correction")` carrying both new and previous state — the frontend can diff.
- `pipeline._apply_delayed_asr_correction` already **only corrects segments within a bounded window** and guards with `_too_similar_or_repetitive` to suppress no-op / repetitive rewrites (anti-flicker).

The literature's guidance — correct finalized segments async, bound revisions, suppress trivial changes — is essentially **already implemented** for the re-decode path. An LLM corrector should reuse the same `apply_correction` → `transcript.update` channel, so all the idempotency/revision machinery is inherited.

### Risks
- Correcting a **still-partial** segment causes visible flicker; gate the LLM corrector on **finalized** segments only (the delayed-correction path already checks the diarization horizon).
- Late corrections that arrive after the user has moved on are jarring; bound correction latency (Daiya's `asr_delayed_correction_window_seconds`) and drop stale ones.
- Ensure `words` timestamps in a `CorrectionUpdate` stay within the original segment window (the pipeline already clips via `_clip_segments_to_window` / `_clip_word`) — an LLM corrector returning text without timestamps should set `words=None` so the mux keeps existing word timings.

### Concrete experiment
Route the §3 gated LLM corrector through `mux.apply_correction` and add a client-side flicker metric (count of words that change after finalization). Confirm the `if updated == current: return []` guard + a similarity guard keep flicker near zero. Only correct segments that are `transcript.final`, never `transcript.partial`.

---

## Bottom-line recommendations for Daiya

1. **2-pass, not 3-pass, as default.** Large Whisper is likely fast enough; add a tiny pass-1 only if measured p95 latency fails the streaming budget. The LLM pass should be pass-2's async refinement, not a mandatory stage. (§5)
2. **Confidence-gate everything.** Naive full-segment LLM correction *regresses* WER in multiple papers. Use the already-present `is_low_confidence_segment` / `low_confidence_words` to correct only low-confidence spans, freeze the rest. This is the highest-leverage, lowest-risk change. (§3)
3. **Reuse the existing context/glossary infra.** `ASRPromptMemory.terms()` already builds a domain-term list for the acoustic prompt; feed the same list into the LLM correction prompt as an allow-list for code-switch spellings. (§4)
4. **Reuse the existing correction channel.** `NoOpCorrectionStage.review` → `CorrectionUpdate` → `mux.apply_correction` → `transcript.update` already gives idempotent, revision-tracked, flicker-safe updates. Drop the LLM in behind `review`; correct **finalized** segments only. (§8)
5. **Model picks:** Thai-English → **Typhoon 2 3B/7B** (only small open model with a code-switching metric). Japanese-English → **Qwen2.5-7B**. Consider a **specialized compact corrector** over a 7B if latency is tight. Consider **Qwen2-Audio** for optional re-listening of low-confidence spans only. (§6, §7)
6. **Before building GER, measure N-best oracle WER.** If best-of-5 ≈ 1-best on Daiya's data, skip N-best GER and rely on re-decode-with-context + gated LLM term correction. (§1, §2)

### Direct file map
| Technique | Daiya hook |
|---|---|
| Gated LLM correction | `correct.py` `NoOpCorrectionStage.review` → `ConfidenceGatedCorrectionStage`, using `asr.is_low_confidence_segment` / `low_confidence_words` |
| Emit correction as flicker-safe update | `mux.apply_correction` / `TranscriptEvent("transcript.update")` (idempotent via revision + `updated == current` guard) |
| Async second-pass / delayed correct | `pipeline._apply_delayed_asr_correction` (already re-decodes with left context; extend to call LLM) |
| Glossary/context for corrector | `pipeline.ASRPromptMemory.terms()` / `build_prompt()` |
| N-best / confidence source | `asr.py` `ASRSegment.confidence`, `WordTimestamp.probability`; add `alternatives` via temperature/`best_of` |
| Audio re-listen | `pipeline._audio_history` / `_audio_between` to slice low-confidence span audio |

---

## Sources

- HyPoradise (NeurIPS 2023): https://arxiv.org/abs/2309.15701
- Whispering-LLaMA (EMNLP 2023): https://arxiv.org/abs/2310.06434 — code: https://github.com/Srijith-rkr/Whispering-LLaMA
- RobustGER / "LLMs are Efficient Learners of Noise-Robust ASR" (ICLR 2024): https://arxiv.org/abs/2401.10446 — code: https://github.com/YUCHEN005/RobustGER
- Japanese multi-pass augmented GER: https://arxiv.org/pdf/2408.16180
- Revisiting ASR Error Correction with Specialized Models: https://arxiv.org/html/2405.15216v2
- Fewer Hallucinations, More Verification (RLLM-CF, 3-stage): https://arxiv.org/abs/2505.24347
- Confidence-Guided Error Correction for Disordered Speech: https://arxiv.org/abs/2509.25048
- Interfacing LLMs with ASR using confidence measures and prompting: https://arxiv.org/html/2407.21414v1
- Transformer-Based Deliberation for Two-Pass ASR: https://arxiv.org/abs/2101.11577
- Deliberation of Streaming RNN-T by Non-autoregressive Decoding: https://arxiv.org/pdf/2112.11442
- Parallel Rescoring with Transformer: https://arxiv.org/pdf/2008.13093
- Contextual Biasing for LLM-Based ASR (hotword retrieval + RL): https://arxiv.org/html/2512.21828v1
- Contextual Biasing of Named-Entities with LLMs: https://arxiv.org/pdf/2309.00723
- PARCO (phoneme-augmented contextual ASR): https://arxiv.org/pdf/2509.04357
- LOGIC (logit-space contextual biasing): https://arxiv.org/html/2601.15397v1
- Retrieval-Augmented Contextual ASR (ACL 2025 Findings): https://aclanthology.org/2025.findings-emnlp.203.pdf
- Qwen3-Omni Technical Report: https://arxiv.org/html/2509.17765v1
- Qwen2-Audio repo: https://github.com/qwenlm/qwen2-audio
- Audio LLM latency analysis (Stanford CS191): https://cs191.stanford.edu/projects/Spring2025/Laya___Iyer_.pdf
- Audio LLM cost comparison (WaveSpeed): https://wavespeed.ai/blog/posts/qwen3-5-omni-vs-gpt4o-gemini-2026/
- Typhoon 2 (Thai/English + code-switching): https://arxiv.org/html/2412.13702v1 — release: https://opentyphoon.ai/blog/en/typhoon-2-release-9dd36e3882c0
- Flickering Reduction with Partial Hypothesis Reranking: https://www.bruguier.com/pub/deflickering.pdf
- Analyzing Quality and Stability of Streaming ASR (Interspeech 2020): https://arxiv.org/abs/2006.01416
- Revision-Controllable Decoding for Simultaneous Translation: https://arxiv.org/pdf/2310.04399
- Partial Rewriting for Multi-Stage ASR: https://arxiv.org/html/2312.09463v1
- faster-whisper sentence-level logprobs (issue #1358): https://github.com/SYSTRAN/faster-whisper/issues/1358
- Whisper N-best hypotheses discussion: https://github.com/openai/whisper/discussions/1619
