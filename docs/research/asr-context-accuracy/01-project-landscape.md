# 01 — Project Landscape: ASR Architectures Similar to Daiya

**Scope.** Survey of open-source projects, research systems, and commercial products whose ASR architecture overlaps with Daiya-RMT (faster-whisper / merged-LoRA CTranslate2 Whisper-large-v3, VAD utterance chunking, separate stateful pyannote diarization, multiplexer time-overlap alignment, rolling `initial_prompt` context + English/domain term biasing, streaming partials/finals with later correction). Emphasis on **what is transferable** to a Thai-English / Japanese-English code-switching, near-real-time, context-aware pipeline.

Evidence is cited inline. Where a claim is a project's own marketing number or an unverified secondary source, it is flagged as such.

---

## 1. WhisperX (m-bain/whisperX)

**Architecture / stack.** A three-stage batch pipeline glued into one CLI + Python API: (1) VAD-based segmentation cuts long silences and produces speech chunks; (2) those chunks are decoded by **batched faster-whisper** (CTranslate2 backend); (3) **wav2vec2 forced phoneme alignment** produces word-level timestamps; optionally (4) **pyannote-audio (pyannote 3.1)** diarization labels each word/segment by clustering. ([GitHub](https://github.com/m-bain/whisperX), [DeepWiki](https://deepwiki.com/m-bain/whisperX))

**ASR model.** Any faster-whisper CT2 model (large-v2/v3 default). No fine-tuning of its own — it is a *pipeline*, not a model.

**Runtime details.** Because OpenAI Whisper does not batch within a single file, WhisperX's key contribution is **VAD-cut + batched inference**, claiming ~70× realtime with large-v2. VAD segmentation also reduces Whisper hallucination on long silences and gives cleaner segment boundaries for alignment. ([GitHub](https://github.com/m-bain/whisperX), [openai/whisper discussion #684](https://github.com/openai/whisper/discussions/684))

**Accuracy tricks.** Forced alignment decouples *timestamp accuracy* from Whisper's notoriously drifty token timestamps — this is the single most reusable idea. Diarization is post-hoc (offline), assigning speakers after full transcription; holds up on ~2–6 speakers without per-mic separation.

**Relevance to Daiya.** WhisperX is essentially the *batch/offline cousin* of Daiya's pipeline (VAD → CT2 Whisper → align → diarize). Two transferable pieces: (a) **wav2vec2 forced alignment** would give Daiya far better word timestamps than Whisper's, tightening the multiplexer's time-overlap alignment with diarization turns; (b) its VAD-batching pattern is what faster-whisper's `BatchedInferencePipeline` now does natively.

**Risks / caveats.** (1) Fully **offline** — diarization runs after the whole file, so nothing here helps streaming *finalization*. (2) wav2vec2 alignment models are **per-language**; a Thai-English code-switched utterance needs either a Thai model or an English model, not both — alignment across a switch point is a known weak spot and may need a phoneme model that covers both scripts or per-language re-alignment. (3) Alignment adds a model + latency to every utterance.

---

## 2. faster-whisper / CTranslate2 (SYSTRAN)

**What it is.** A reimplementation of Whisper on the **CTranslate2** inference engine (Daiya's exact runtime). Current release **v1.2.1 (Oct 2025)**. Claims up to ~4× faster than openai-whisper at equal accuracy with lower memory; further gains from 8-bit quantization. ([GitHub](https://github.com/SYSTRAN/faster-whisper), [PyPI](https://pypi.org/project/faster-whisper/))

**Quantization / compute types.** `float16`, `int8`, `int8_float16` (and `int16`, `float32`, `bfloat16`, `int8_bfloat16`). Guidance: on CPU `int8` is best; on GPU `float16` or `int8_float16`. Reported quality impact of `int8` is small (≈0.1 WER on the small model in one test); aggressive int8 can slightly degrade hard audio. ([whisper-ctranslate2](https://pypi.org/project/whisper-ctranslate2/), [Medium/Balaragavesh](https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204))

**Speed.** ~13-min audio on GPU fp16: openai 2m23s vs faster-whisper 1m03s; with `BatchedInferencePipeline` (batch 8) ~17s. CPU: 2m37s vs 6m58s. ([GitHub](https://github.com/SYSTRAN/faster-whisper))

**Hotword / prompt support — important for Daiya.** faster-whisper exposes **both** `initial_prompt` (Daiya already uses this for rolling context) **and a dedicated `hotwords: Optional[str]` parameter** for contextual biasing. The `WhisperModel` builds the decoder's initial token sequence from special/language/task tokens plus optional hotwords/context. ([transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [DeepWiki WhisperModel](https://deepwiki.com/SYSTRAN/faster-whisper/4.1-whispermodel)) Note: hotwords/prompt are still just **prompt-token biasing** — capped at 224 prompt tokens, and attention weights the *end* of the prompt more heavily, so long term lists are unequal in effect. ([arXiv 2410.18363](https://arxiv.org/abs/2410.18363))

**VAD.** Built-in **Silero VAD** filtering (default: only removes silence >2s). Daiya's planned Silero move can therefore be done *inside* faster-whisper rather than as a separate stage.

**Relevance to Daiya.** This is Daiya's runtime — the actionable finding is that Daiya can consolidate: (a) use the native `hotwords` param for English/domain-term biasing instead of (or alongside) stuffing terms into `initial_prompt`, freeing prompt budget for actual transcript-tail context; (b) adopt `BatchedInferencePipeline` for any non-streaming re-decode / correction pass; (c) use native Silero VAD.

**Risks.** Hotwords ≠ true FST/on-the-fly biasing (see NeMo/sherpa below) — it can't hard-constrain vocabulary and shares the 224-token ceiling with the prompt. Merged-LoRA CT2 models work fine but must be re-converted per checkpoint.

---

## 3. whisper_streaming (ufal) + WhisperLiveKit

**whisper_streaming — LocalAgreement-2.** UFAL's streaming wrapper runs Whisper repeatedly on a growing audio buffer and confirms output using **LocalAgreement-n**: when *n* consecutive updates (each after a new chunk) agree on a transcript prefix, that prefix is committed. **LocalAgreement-2 (n=2)** was found most effective and is the default. ([GitHub](https://github.com/ufal/whisper_streaming), [arXiv 2307.14743 "Turning Whisper into Real-Time"](https://arxiv.org/html/2307.14743))

**Chunk size / latency.** `MinChunkSize` sets the minimum audio processed per iteration and trades latency vs quality; if an update takes longer than `MinChunkSize`, the next one runs immediately on all accumulated audio. Because a word is only confirmed once two consecutive predictions agree, **final-emission latency ≈ 2× chunk size** (e.g. 1.0s chunk → ~2.0s average final latency). Whisper-medium does **not** hit real-time on CPU even at int8, and on GPU struggles below ~0.5–1.0s chunks. ([whisper_streaming README](https://github.com/ufal/whisper_streaming/blob/main/README.md), [arXiv 2307.14743](https://arxiv.org/html/2307.14743))

**SimulStreaming (ufal, 2025).** Successor using **AlignAtt** — an attention-guided policy that decides how much to emit based on encoder-decoder attention alignment, needing only the encoder pass for the policy (cheaper than full re-decode). ([SimulStreaming](https://github.com/ufal/SimulStreaming))

**WhisperLiveKit (QuentinFuxa).** A packaged self-hosted real-time STT+diarization server (FastAPI + WebSocket, sub-second latency). Selectable **backend policy**: `simulstreaming` (AlignAtt) or `localagreement`; keeps context across small buffers so tiny slices are never fed raw to Whisper. For diarization it offers **online** options — **NVIDIA Streaming Sortformer** (see §8) and **Diart** — cutting end-to-end latency vs offline post-hoc diarization. ([DeepWiki](https://deepwiki.com/QuentinFuxa/WhisperLiveKit), [refft writeup](https://refft.com/en/QuentinFuxa_WhisperLiveKit.html))

**Relevance to Daiya.** This is the most directly relevant *streaming-policy* prior art. Daiya's "partials then finals with correction" maps almost 1:1 onto **LocalAgreement-2** (emit unstable partial; commit prefix on 2-way agreement) — a cleaner, well-studied finalization rule than ad-hoc delayed-correction. **AlignAtt/SimulStreaming** is the lower-latency evolution worth watching. WhisperLiveKit is a near-complete reference implementation of Daiya's target (streaming Whisper + online diarization in one server) and pairs Whisper streaming with **Sortformer** — exactly the stateful-diarization pattern Daiya wants.

**Risks.** LocalAgreement re-runs Whisper on a growing buffer → compute grows with buffer length; needs buffer trimming. Repeated decoding of overlapping audio can amplify Whisper hallucination/looping (documented in issues). Latency floor is real: ~2× chunk for finals.

---

## 4. whisper.cpp streaming (ggml-org)

**Architecture.** GGML/GGUF C++ runtime; `examples/stream` does sliding-window near-real-time transcription. Streaming params: `--max-tokens`, `--audio-ctx` (shrinks encoder context for speed), `--keep-context` to carry `prompt_n_tokens` across segments. ([stream.cpp](https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp))

**Context / prompt.** Standard Whisper prompt limits apply: 448-token total context, ~224 usable for input prompt; the prompt normally only seeds the first 30s window and is then overwritten by decoded text unless an implementation deliberately re-applies it each segment. ([Medium/axinc prompt engineering](https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d))

**Hotwords.** **No native hotword/vocabulary biasing** — a long-standing feature request (issue #1979). Bindings (pywhispercpp) expose `prompt_tokens` / `prompt_n_tokens` so biasing is done the same prompt-hack way as everyone else. ([issue #1979](https://github.com/ggml-org/whisper.cpp/issues/1979), [pywhispercpp](https://pypi.org/project/pywhispercpp/))

**Relevance to Daiya.** Mostly a deployment/edge alternative (CPU/Metal, no Python/CUDA). Transferable idea: `--audio-ctx` shrinking to trade accuracy for latency on short utterances, and explicit cross-segment prompt carry (which Daiya already does via rolling `initial_prompt`). Not a runtime change Daiya needs given CT2 is faster on GPU.

**Risks.** Weaker biasing story than faster-whisper (no hotwords param); GGML LoRA/merge workflow differs from CT2.

---

## 5. NVIDIA NeMo — Canary, Parakeet, cache-aware streaming Conformer

**Models.** **Parakeet** (FastConformer CTC/TDT/RNNT, e.g. `parakeet-tdt-0.6b-v2`) — very high throughput English ASR. **Canary** (FastConformer + transformer decoder, multitask ASR+translation). Both sit on the **FastConformer** encoder. ([NeMo ASR models](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html))

**Cache-aware streaming Conformer.** Trained with *limited right context* so train == inference behaviour, and **caches intermediate activations** to avoid recomputation across chunks — true low-latency streaming without re-decoding overlapping audio (contrast with Whisper LocalAgreement, which re-decodes). `stt_en_fastconformer_hybrid_large_streaming_multi` is trained for multiple latencies at once. Newer `nemotron-speech-streaming-en-0.6b` claims accuracy comparable to `parakeet-ctc-1.1b` at lower cost. ([HF fastconformer streaming](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi), [NVIDIA blog: scaling voice agents](https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents), [arXiv 2312.17279 Stateful Conformer](https://arxiv.org/pdf/2312.17279))

**Streaming decoding for Canary.** Wait-K and **AlignAtt** policies added for Canary; buffered + cache-aware transducer streaming supported. ([NVIDIA-NeMo/Speech releases](https://github.com/NVIDIA-NeMo/Speech/releases))

**Biasing.** NeMo/k2 ecosystems support **FST/word-boost contextual biasing** (sherpa hotwords) — a *real* biasing mechanism (weighted graph over the token lattice), stronger than prompt-token hacks. ([sherpa hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html))

**Relevance to Daiya.** Three transferable ideas even if Daiya stays on Whisper: (a) **cache-aware streaming** is the architecturally "correct" way to stream (no re-decode blowup) — a candidate if Whisper streaming latency/compute becomes the bottleneck; (b) **AlignAtt** as a streaming policy is model-agnostic in spirit; (c) **FST word-boosting** is a stronger biasing target than the 224-token prompt for Daiya's English/domain-term problem. Parakeet/Canary themselves are **weak on Thai** (English/European-centric), so not drop-in for Daiya's languages.

**Risks.** No strong Thai support; Japanese limited. Switching off Whisper means losing the fine-tuned mixed-lingual LoRA investment. Heavier framework.

---

## 6. Commercial streaming ASR (Deepgram, AssemblyAI, Speechmatics, Gladia, Azure, Google USM)

Deployed systems, useful as **design references** for finalization, biasing, diarization, and code-switching — internals mostly undisclosed (treat numbers as vendor claims).

- **Deepgram (Nova-3).** Sub-300ms streaming, natural turn/endpoint detection, **Keyterm Prompting** (paid add-on) for biasing, separate diarization add-on. ([Deepgram vs Speechmatics vs AssemblyAI](https://deepgram.com/learn/deepgram-vs-speechmatics-vs-assemblyai))
- **AssemblyAI (Universal-Streaming).** **Native code-switching across 6 languages for streaming** (99+ async); **streaming speaker diarization at sub-300ms**; **full natural-language prompting with dynamic key-terms mid-stream** (upgrade over keyword-only). Add-ons priced per hour. ([AssemblyAI best realtime APIs](https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription))
- **Speechmatics.** Recommends ~1.5s finalization as a starting point for voice agents; sub-second finals require tuning. Keyword-only custom dictionary. ([Speechmatics STT guide](https://www.speechmatics.com/company/articles-and-news/best-speech-to-text-ai-guide-apis-platforms-and-services-compared))
- **Gladia (Solaria-1, Jan 2026).** Most relevant to Daiya's code-switching goal. **Native code-switching across 100+ languages** via a **CPU-friendly ensemble routing between small monolingual streaming Zipformer models**; claims ~13% WER on inter-utterance code-switching (vs Nova-3 ~14%). **Continuous in-pipeline LID**: on a high-confidence language switch it **switches the active ASR model, rolls the transcript back to the switch boundary, and re-infers the buffered audio** for that range. Rich custom vocabulary (per-item value, intensity, pronunciations, language). Partials ~300ms, finals often <600ms. ([Gladia code-switching build](https://www.gladia.io/blog/building-real-time-multilingual-asr-with-code-switching), [Gladia LID vs code-switching](https://www.gladia.io/blog/code-switching-vs-language-identification-whats-the-difference), [Gladia custom vocab](https://www.gladia.io/blog/custom-vocabulary-stt-accuracy))
- **Azure / Google (USM/Chirp).** Continuous LID and multilingual streaming; custom vocabulary / speech-adaptation phrase lists; diarization. (General product capability; internals undisclosed.)

**Relevance to Daiya.** (1) **Gladia's rollback-and-re-infer on detected language switch** is a concrete, borrowable *correction* pattern — Daiya's delayed-correction event could be triggered by a mid-utterance LID/confidence signal, re-decoding just the tail. (2) **Sub-300ms partials + ~1.5s finals** are realistic latency targets to design toward. (3) Every serious product treats **keyword/keyterm boosting** as a first-class, dynamic (mid-stream) input — validates Daiya's rolling-term approach but suggests making it dynamic per utterance. (4) **Streaming diarization is expected to run online** (not post-hoc) — aligns with Daiya's stateful path.

**Risks.** Closed; numbers are marketing. Code-switching support is often "inter-utterance" (switch *between* utterances) rather than the harder **intra-utterance** Thai-English word-level switching Daiya targets.

---

## 7. Thai ASR projects

- **Thonburian Whisper (Biodatlab).** Whisper fine-tuned for Thai (medium/large) on **Common Voice 13, Gowajee, Thai Elderly Speech, Thai Dialect** corpora. The de-facto open Thai Whisper baseline and a common **base for further fine-tunes**. ([GitHub thonburian-whisper](https://github.com/biodatlab/thonburian-whisper))
- **Pathumma-Whisper-Large (NECTEC/ThaiLLM ecosystem).** Strong Thai Whisper-Large; used as a high-confidence reference/fallback in ensemble labeling pipelines. ([Typhoon ASR paper, arXiv 2601.13044](https://arxiv.org/html/2601.13044v1))
- **Typhoon ASR Real-time (SCB 10X / Typhoon).** **FastConformer-Transducer**, causal/streaming-first (no future context), CER ≈ **0.0984**, ~**4097× realtime** throughput, 6× faster than next-best, 15–19× faster than Whisper variants. **Explicitly limited on English loanwords and Thai-English code-switching** (code-switching on the future roadmap). Its **data pipeline uses 3-model majority voting** (Pathumma / Biodatlab-Distill / internal Whisper-Large) to auto-label, defaulting to Pathumma on ties — improving CER >4% absolute on noisy TVSpeech. ([Typhoon ASR release](https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release), [arXiv 2601.13044](https://arxiv.org/html/2601.13044v1))
- **Typhoon Isan ASR.** Whisper fine-tune (on Biodatlab medium) for the Isan dialect — shows the region's pattern of stacking fine-tunes on Whisper. ([Typhoon Isan](https://opentyphoon.ai/blog/en/typhoon-isan-release))

**Relevance to Daiya.** (1) **Typhoon ASR Real-time is Daiya's closest Thai streaming peer** — but its own admitted weakness (Thai-English code-switching + English loanwords) is *exactly Daiya's differentiator*; Daiya's LoRA-on-large-v3 + term biasing directly targets that gap. (2) The **3-model majority-vote auto-labeling** pipeline is a strong, cheap technique to expand/clean Daiya's mixed-lingual training data. (3) **Thonburian/Pathumma** are candidate teachers or warm-starts for Daiya's Thai side; also useful as WER/CER reference baselines. (4) Typhoon's FastConformer-Transducer proves a **non-Whisper streaming option** for Thai if Whisper latency proves limiting — at the cost of code-switching quality.

---

## 8. Japanese ASR projects

- **ReazonSpeech.** Largest Japanese speech-transcription corpus (from TV); the dataset backbone for Japanese Whisper work. ([kotoba-whisper v2.0 card](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0))
- **Kotoba-Whisper (v1/v2).** **Distil-Whisper-style distillation**: teacher = openai/whisper-large-v3; student = **full large-v3 encoder + 2-layer decoder** (init from first/last decoder layers). v1.0 on 1,253h ReazonSpeech-large; v2.0 on all ReazonSpeech (~7.2M clips). **~6.3× faster than large-v3 at comparable CER/WER**, beating large-v3 on the in-domain test set. **kotoba-whisper-bilingual-v1.0** targets JA+EN. ([kotoba-whisper v1.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0), [v2.0](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0), [bilingual](https://huggingface.co/kotoba-tech/kotoba-whisper-bilingual-v1.0))

**Relevance to Daiya.** (1) **Kotoba's distillation recipe (keep full encoder, shrink decoder to 2 layers, ~6× speedup, minimal CER loss)** is a directly transferable path to a *fast first-pass* model for Daiya's 3-pass idea — distill Daiya's own fine-tuned large-v3 into a 2-layer-decoder student for the preliminary pass while keeping the large model for pass 2. (2) **kotoba-whisper-bilingual** is a concrete JA-EN reference/warm-start. (3) ReazonSpeech is a ready Japanese data source; its ">10 WER filtering" is a reusable cleaning heuristic.

---

## 9. Stateful streaming diarization — NVIDIA Streaming Sortformer

Called out separately because it maps onto Daiya's stateful-diarization requirement more directly than pyannote.

**What it is.** Production streaming diarization tracking up to 4 speakers with an **Arrival-Order Speaker Cache (AOSC)**: frame-level acoustic embeddings of previously seen speakers are stored, **ordered by arrival time**, and dynamically refreshed by keeping the highest-scoring frames. Current chunk speakers are matched against the cache so **each person keeps the same label across the whole stream** — i.e., persistent speaker identity across processing turns, exactly Daiya's stated goal. Optimized for English + Mandarin, millisecond-level. Integrated as an online diarizer in WhisperLiveKit. ([arXiv 2507.18446](https://arxiv.org/html/2507.18446v1), [NVIDIA blog](https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/), [HF diar_streaming_sortformer_4spk-v2](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2))

**Relevance to Daiya.** Daiya's memo says it can keep pyannote stateful *by caching speaker vectors and matching across turns* — **Sortformer's AOSC is a published, benchmarked implementation of that exact idea.** Even if Daiya keeps pyannote, AOSC's design (arrival-order index, score-based cache eviction, per-speaker adaptive cache size) is a concrete blueprint for the stateful cache. It also removes the offline post-hoc constraint that WhisperX-style pyannote imposes.

**Risks.** 4-speaker cap; tuned for EN/ZH (Thai/Japanese diarization is acoustic, so likely fine, but unverified); NeMo dependency.

---

## What Daiya should look at next

1. **Adopt LocalAgreement-2 (or AlignAtt) as the explicit streaming finalization policy.** It formalizes Daiya's "partial → commit → correct" behaviour with a studied rule (finals ≈ 2× chunk). Prototype against **WhisperLiveKit** as a reference server. ([whisper_streaming](https://github.com/ufal/whisper_streaming), [WhisperLiveKit](https://deepwiki.com/QuentinFuxa/WhisperLiveKit))
2. **Use faster-whisper's native `hotwords` param** for English/domain-term biasing, separating it from the rolling transcript-tail `initial_prompt` (reclaims the 224-token budget). Evaluate against FST/word-boost biasing (sherpa/NeMo) for a stronger long-term biasing mechanism. ([transcribe.py](https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py), [sherpa hotwords](https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html))
3. **Steal Gladia's rollback-and-re-infer-on-language-switch** as the mechanism for Daiya's delayed-correction event: trigger a tail re-decode when a mid-utterance LID/confidence signal fires. ([Gladia](https://www.gladia.io/blog/building-real-time-multilingual-asr-with-code-switching))
4. **Prototype stateful diarization with Streaming Sortformer / AOSC**, or at minimum port AOSC's cache design onto the pyannote path. ([arXiv 2507.18446](https://arxiv.org/html/2507.18446v1))
5. **Add wav2vec2 forced alignment (WhisperX-style)** to get precise word timestamps → tighter multiplexer overlap with diarization turns; validate on Thai↔English switch points. ([whisperX](https://github.com/m-bain/whisperX))
6. **For the fast first pass (2 vs 3-pass question), distill Daiya's fine-tuned large-v3 with the Kotoba recipe** (full encoder + 2-layer decoder, ~6×) instead of adding a separate small model. ([kotoba-whisper](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0))
7. **Mine Typhoon's 3-model majority-vote auto-labeling** to grow/clean Daiya's Thai-English training data cheaply; use Thonburian/Pathumma as Thai teachers and CER/WER baselines. ([Typhoon ASR paper](https://arxiv.org/html/2601.13044v1))
8. **Watch NeMo cache-aware streaming Conformer / Typhoon FastConformer-Transducer** as the fallback architecture if Whisper's re-decode streaming cost/latency becomes the bottleneck — knowing it trades away code-switching quality. ([NeMo streaming](https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi))

---

## Sources

- WhisperX: https://github.com/m-bain/whisperX · https://deepwiki.com/m-bain/whisperX · https://github.com/openai/whisper/discussions/684
- faster-whisper / CTranslate2: https://github.com/SYSTRAN/faster-whisper · https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py · https://deepwiki.com/SYSTRAN/faster-whisper/4.1-whispermodel · https://pypi.org/project/faster-whisper/ · https://pypi.org/project/whisper-ctranslate2/ · https://medium.com/@balaragavesh/converting-your-fine-tuned-whisper-model-to-faster-whisper-using-ctranslate2-b272063d3204
- whisper_streaming / SimulStreaming / WhisperLiveKit: https://github.com/ufal/whisper_streaming · https://github.com/ufal/whisper_streaming/blob/main/README.md · https://arxiv.org/html/2307.14743 · https://github.com/ufal/SimulStreaming · https://deepwiki.com/QuentinFuxa/WhisperLiveKit · https://refft.com/en/QuentinFuxa_WhisperLiveKit.html
- whisper.cpp: https://github.com/ggml-org/whisper.cpp/blob/master/examples/stream/stream.cpp · https://github.com/ggml-org/whisper.cpp/issues/1979 · https://pypi.org/project/pywhispercpp/ · https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d
- NVIDIA NeMo (Canary/Parakeet/cache-aware): https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html · https://huggingface.co/nvidia/stt_en_fastconformer_hybrid_large_streaming_multi · https://huggingface.co/blog/nvidia/nemotron-speech-asr-scaling-voice-agents · https://github.com/NVIDIA-NeMo/Speech/releases · https://arxiv.org/pdf/2312.17279 · https://k2-fsa.github.io/sherpa/onnx/hotwords/index.html
- Commercial (Deepgram/AssemblyAI/Speechmatics/Gladia): https://deepgram.com/learn/deepgram-vs-speechmatics-vs-assemblyai · https://www.assemblyai.com/blog/best-api-models-for-real-time-speech-recognition-and-transcription · https://www.speechmatics.com/company/articles-and-news/best-speech-to-text-ai-guide-apis-platforms-and-services-compared · https://www.gladia.io/blog/building-real-time-multilingual-asr-with-code-switching · https://www.gladia.io/blog/code-switching-vs-language-identification-whats-the-difference · https://www.gladia.io/blog/custom-vocabulary-stt-accuracy · https://docs.gladia.io/chapters/language/code-switching
- Thai ASR: https://github.com/biodatlab/thonburian-whisper · https://opentyphoon.ai/blog/en/typhoon-asr-realtime-release · https://arxiv.org/html/2601.13044v1 · https://opentyphoon.ai/blog/en/typhoon-isan-release
- Japanese ASR: https://huggingface.co/kotoba-tech/kotoba-whisper-v1.0 · https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0 · https://huggingface.co/kotoba-tech/kotoba-whisper-bilingual-v1.0
- Streaming diarization (Sortformer): https://arxiv.org/html/2507.18446v1 · https://developer.nvidia.com/blog/identify-speakers-in-meetings-calls-and-voice-apps-in-real-time-with-nvidia-streaming-sortformer/ · https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- Contextual biasing research: https://arxiv.org/abs/2309.09552 · https://arxiv.org/abs/2410.18363 · https://arxiv.org/html/2506.21576 (soft prompt tuning for code-switching)
