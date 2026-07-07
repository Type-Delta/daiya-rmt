# 04 — Contextual Biasing, Hotwords & Rescoring for Daiya ASR

**Scope:** methods to make English technical terms transcribe consistently when embedded in
Thai/Japanese speech, so `"relation กัน"` stays `relation` instead of `รีเอชันกัน`, and
`percent` doesn't turn to garbage. Daiya's ASR runtime is **faster-whisper (CTranslate2)** over
a merged-LoRA Whisper-large-v3. That runtime constraint is the single most important filter for
everything below: most "biasing" research modifies the PyTorch/HF decoder and **cannot run on
CT2 as-is**.

Daiya already does prompt-based biasing: `pipeline.py:ASRPromptMemory` (lines 410-460) extracts
English/domain terms via `_english_domain_terms` (line 519) and injects a `Terms: a, b, c` list
into `initial_prompt`. This report evaluates whether that is the right lever and what else is
reachable.

---

## TL;DR for Daiya

| Method | Runtime fit (CT2/fw) | Effort | Expected payoff | Verdict |
|---|---|---|---|---|
| `hotwords=` param (fw) | ✅ native | trivial | moderate, same class as prompt | **Try now**, A/B vs current `Terms:` prompt |
| `initial_prompt` "Terms:" (current) | ✅ native | done | moderate | Keep, but see caps/bleed risks |
| Post-hoc fuzzy/phonetic glossary fix | ✅ pure Python, runtime-agnostic | low | high for *known* term set | **Do this** — cheapest real win |
| CT2 `suppress_sequences` (block phoneticized junk) | ✅ native | low-med | narrow | Situational |
| CT2 `prefix_bias_beta` | ⚠️ partial (prefix-only, not term list) | med | low for this problem | Skip |
| HF `sequence_bias` / logit boost | ❌ needs HF transformers runtime | high (dual runtime) | high, true weighting | Only if prompt+fuzzy insufficient |
| Trie/WFST decode-time biasing (TCPGen, trie-decoding) | ❌ needs custom decoder | very high | high | Research-only |
| KWS/CB-Whisper (encoder KWS → prompt) | ❌ needs HF + training | very high | high, entity recall | Research-only |

**Recommended path:** (1) A/B `hotwords` vs the `Terms:` prompt; (2) add a **post-hoc phonetic
glossary normalizer** — it's the only method that gives true per-term control inside the CT2
runtime with no model surgery; (3) keep true logit/trie biasing as a fallback that requires
switching the hot path to HF transformers.

---

## 1. faster-whisper `hotwords` parameter

**Mechanism (from source).** `hotwords` is **pure prompt injection — there is no logit
boosting.** In `WhisperModel.get_prompt` (`faster_whisper/transcribe.py`, ~line 1341) the string
is tokenized and appended right after the `<|startofprev|>` (sot_prev) token, the same slot
`initial_prompt` / previous-text context uses:

```python
if hotwords and not prefix:
    hotwords_tokens = tokenizer.encode(" " + hotwords.strip())
    if len(hotwords_tokens) >= self.max_length // 2:
        hotwords_tokens = hotwords_tokens[: self.max_length // 2 - 1]
    prompt.extend(hotwords_tokens)
...
if previous_tokens:
    prompt.extend(previous_tokens[-(self.max_length // 2 - 1):])
```

Key facts:
- **Docstring:** *"Hotwords/hint phrases to provide the model with. Has no effect if prefix is
  not None."* So `hotwords` and `prefix` are mutually exclusive — you can't combine hotword
  hints with a forced prefix.
- **Token cap:** hotwords are truncated to `max_length // 2 - 1`. Whisper `max_length = 448`, so
  hotwords are capped at ~**223 tokens**, and previous-context tokens share the *other* half.
  (Note: the widely-quoted "224-token prompt limit" is exactly this half-window.)
- **hotwords vs initial_prompt:** they occupy the same conditioning region. In practice, passing
  `hotwords` and `initial_prompt` together is redundant/competing for the same budget; several
  wrappers treat `hotwords` as a convenience that builds the prompt for you. Community guidance:
  use `hotwords` for a short list of rare named terms; use `initial_prompt` for general
  domain/style priming.
- **Per-segment / dynamic:** yes — it's a `transcribe()` kwarg, evaluated per call, so Daiya can
  vary it per utterance (issue #1313 asks exactly this; the answer is that transcribe-time args
  are re-read each call). This matches Daiya's per-utterance `build_prompt()` flow.
- **prompt_reset_on_temperature:** *"Resets prompt if temperature is above this value. Arg has
  effect only if condition_on_previous_text is True."* Relevant because Daiya uses a temperature
  ladder `[0.0, 0.8, 1.0]` (`asr.py:180`). On fallback rungs the conditioning context can be
  dropped — worth checking whether hotwords survive the reset (they are re-added each transcribe
  call, so per-call hotwords persist; it's the *rolling previous-text* that resets).

**Evidence it helps.** Weaker than vendors imply. There is **no rigorous public WER study of
`hotwords` vs `initial_prompt`** on code-switch data. Because it's prompt-only, it inherits all
prompt limits: no true weighting, "prompt bleeding" (the model can echo the term list verbatim),
and diminishing returns past a handful of terms. faster-whisper-GUI ships a "Prompt And
Hotwords" doc treating them as siblings.

**Relevance to Daiya:** direct, zero-cost to try. It is architecturally the *same lever* as the
existing `Terms:` prompt, so expect similar behavior — but worth A/B testing because the framing
(bare term list vs `"Terms: ..."` sentence) changes tokenization and the model's echo tendency.

**Risks:** prompt bleeding (term list appears in output), budget contention with recent-transcript
context (Daiya packs `Terms:` + `Recent transcript:` into one prompt — that already competes for
the 223-token half-window), over-priming toward English when the speaker actually said the Thai
word.

**Experiment:** hold audio + model fixed; three arms — (a) current `Terms:`-in-`initial_prompt`,
(b) `hotwords=` with the same terms and no `Terms:` line, (c) both. Metric: per-term recall
(did the exact English surface form appear when spoken?) + false-insert rate (term appeared when
*not* spoken) + overall CER. Use the code-switch eval set from `lab/asr_eval.py`.

---

## 2. Prompt-based biasing (OpenAI Whisper / whisper.cpp / current Daiya)

**Mechanism.** `initial_prompt` seeds the decoder as if that text preceded t=0, shifting token
priors. It is a **soft biasing vector, not a constraint** — you cannot weight individual terms,
and the model may ignore or over-copy it.

**Hard limits (all apply to Daiya today):**
- **224-token prompt window** (`max_length // 2`). Daiya's `max_prompt_chars=900` (~pipeline
  line 414) can exceed this once Thai (multi-token) text is included → silent truncation of the
  *oldest* half. Terms placed late may be dropped.
- **Only conditions the current 30s window**; in long-form the rolling `previous_text` overwrites
  the prompt on subsequent segments unless re-injected. Daiya sidesteps this by rebuilding the
  prompt per utterance — good.
- **No weighting / no guarantee** — glossary prompting is "best effort."
- **Prompt bleeding / hallucination** — long or list-like prompts get echoed. Daiya's mitigation
  line *"Use this only as context. Do not repeat text unless it is spoken again."*
  (`pipeline.py:442`) is folklore — Whisper was not instruction-tuned, so this sentence mostly
  just consumes token budget and adds its own words (`context`, `repeat`, `spoken`) to the prior.
  Recommend measuring whether it helps or should be dropped.

**Relevance/verdict:** this is Daiya's current approach; it's the correct *baseline* but is at its
ceiling. The OpenAI Cookbook prompting guide and axinc-ai's writeup both frame prompt biasing as
"nudge, not control." For consistent technical-term rendering you want a mechanism with actual
per-term force — sections 5-8.

**Cheap improvements to the current code without new runtime:**
- Drop the instruction sentence from the prompt (test A/B); Whisper doesn't follow it.
- Put `Terms:` *last* (closest to audio) rather than after `Recent transcript:`, so it's least
  likely to be truncated and most salient — currently order is static→terms→tail (line 433-440).
- Cap Thai tail chars by *token* budget, not char budget, to avoid silent truncation.

---

## 3. General ASR contextual biasing (background — mostly NOT CT2-compatible)

These are the "real" biasing families. Included so Daiya can recognize what a runtime switch buys.

- **LM shallow fusion / on-the-fly rescoring:** add `λ·logP_LM(term-context)` to acoustic scores
  during beam search. Needs access to per-step logits + a beam hook. **CT2 exposes logits
  (`return_logits_vocab`) but not a per-step Python callback in `WhisperModel.generate`**, so
  true shallow fusion isn't available in the fw hot path without patching CT2.
- **WFST / FST class-based biasing (Kaldi-style, "contextual biasing" à la Google):** compile a
  biasing FST, compose with the decoding graph, add boost on arcs. Encoder-decoder Whisper has
  no external FST graph → not directly applicable.
- **CLAS (Contextual Listen-Attend-Spell), neural biasing:** an attention module over a bias-phrase
  list, trained jointly. Requires architecture change + training.
- **TCPGen (tree-constrained pointer generator):** a trie over bias phrases + a neural pointer that
  interpolates the vocab distribution with a copy distribution at each step. Adaptable to Whisper
  (see §4) but needs a small trained component and a custom decode loop.
- **Keyword boosting (commercial, §6):** additive/multiplicative logit boost on keyword tokens
  during decoding — the pragmatic version of shallow fusion.

**Adaptable to encoder-decoder Whisper?** TCPGen, CLAS, and logit-boost/trie decoding all have
Whisper adaptations in the literature (§4), but **every one requires either training or a custom
(non-CT2) decoder.** None drop into faster-whisper.

---

## 4. Whisper-specific biasing research

Ordered by how reachable they are for Daiya.

**(a) "Can Contextual Biasing Remain Effective with Whisper and GPT-2?" (arXiv 2306.01942).**
Adapts **TCPGen** to Whisper + a GPT-2 rescorer via a training scheme that *"dynamically adjusts
the final output without modifying any Whisper model parameters."* Biasing list of **1000 words**;
reports *"considerable reduction in errors on biasing words"* across 3 datasets; more effective on
domain-specific data; preserves generality. **Runtime:** custom decode, PyTorch — not CT2.

**(b) KWS-Whisper / CB-Whisper (arXiv 2309.09552).** Runs **open-vocabulary keyword spotting on
Whisper encoder hidden states** (TTS-generated entity features + cosine-similarity + a 4-layer CNN
detector); detected entities are injected as decoder prompts. Recommended variant keeps Whisper
frozen (plug-in module, no retrain of the base). Reported: entity recall jumps from **6.3-10.1%
→ 88-92%** on Aishell hotword subsets; on internal **code-switched** data recall **79.9-87.3% →
96.9-97.7%**, MER −1.8 to −2.0 pts. **Highly relevant** (explicitly code-switch + entity recall)
**but runtime = HF transformers + a trained KWS head; not CT2.** Reference impl:
`github.com/BriansIDP/WhisperBiasing`.

**(c) Zero-shot trie-based decoding w/ synthetic multi-pronunciation (arXiv 2508.17796).** Builds
a **trie over bias words** and modifies **beam search** to boost/constrain hypotheses containing
them; synthetic multi-pronunciations handle spelling/phonetic variants. **No fine-tuning.** Directly
targets Whisper beam search. Improves WER + F1 on biased terms. **Most promising "no-train"
research option**, but still needs a custom beam-search decoder (CT2 beam search isn't hookable),
so it's HF/openai-whisper territory.

**(d) `hotwords-for-whisper` (jiang-yw, community fork).** Adds hotword support to openai-whisper;
implementation details not documented on the repo card (likely logit/beam biasing in the PyTorch
decoder). Not CT2.

**Prompt-based vs adapter-based vs logit-based takeaway:** prompt-based (§1-2) is the only family
that runs unmodified on CT2. Everything with *true weighting* (logit boost, trie, TCPGen, KWS)
needs a different runtime. For Daiya this is the crux: **you cannot get real per-term boost weights
inside faster-whisper today.**

---

## 5. Decoding-time boosting actually available in the CT2 / faster-whisper runtime

What Daiya *can* touch without leaving CT2:

- **`suppress_tokens` / `suppress_blank`** (exposed by fw): blacklist tokens. Could suppress
  specific junk tokens that produce phoneticized garbage — but it's a blunt global instrument, not
  context-aware; risks collateral damage. Situational at best.
- **`suppress_sequences`** (CT2 `WhisperModel.generate`, also `Generator.generate_tokens`):
  disables generation of specific token *sequences*. Could block a known bad transliteration like
  the tokens for `รีเอชัน`. Narrow, brittle (one spelling), but zero model changes. **Not currently
  surfaced by faster-whisper's `transcribe()`** — would need to call CT2 directly.
- **`prefix` (fw) / `prefix_bias_beta` (CT2):** `prefix_bias_beta ∈ (0,1)` biases beam search
  toward a *given prefix* (stronger as β→1, beam search only). This forces the *start* of the
  output, not a floating term list — wrong shape for "make this term consistent wherever it
  appears." Also disables `hotwords`. **Skip for this problem.**
- **`return_logits_vocab`:** you can *read* logits, but there is **no per-step Python logits
  callback** in fw's transcribe loop, so you can't inject a boost mid-decode without patching CT2.

**Conclusion:** the CT2 runtime gives you *suppression* and *prefix bias*, but **not additive
per-term logit boosting**. True logit boosting requires HF transformers (§7) or a CT2 patch.

---

## 6. Commercial keyword boosting (reference for boost ranges & UX)

| Vendor | Feature | Weighting | Range / limit | Notes |
|---|---|---|---|---|
| **Deepgram** (Nova-2) | Keywords | Yes — exponential **intensifier** | `keyword:intensifier`, decimals ok, no hard min/max; start small | Over-boost → hallucination |
| **Deepgram** (Nova-3/Flux) | **Keyterm Prompting** | **No intensifiers** (model-driven, contextual) | up to **500 tokens (~100 words)**, focus 20-50 | Multilingual; claims KRR up to ~90% |
| **AssemblyAI** | `word_boost` + `boost_param` | 3 levels | `"low" / "default" / "high"` | Docs warn high → false inserts |
| **Google STT** | Speech adaptation / phrase hints | Yes — `boost` float | **0-20** practical; >0 | Explicitly warns of false positives |
| **Azure Custom Speech** | Phrase list + custom pronunciation | Phrase list = soft; lexicon (IPA) for pronunciation | phrase list soft-boost; lexicon exact | Structured-text training for display form |
| **Speechmatics** | Custom dictionary `additional_vocab` | `sounds_like` pronunciations | ≤6 words/entry, ≤4000 chars/word | Output surface form + phonetic variants |

**Lessons for Daiya:**
- The industry consensus boost range where it helps without hallucinating is **modest** (Google
  0-20 but usually low single digits; AssemblyAI's "high" is the aggressive edge). Every vendor
  warns that aggressive boosting **inserts terms that weren't spoken** — exactly Daiya's risk given
  short code-switch utterances.
- Deepgram's Nova-3 pivot *away* from numeric intensifiers toward "contextual, model-driven"
  keyterms mirrors what prompt-biasing already does in Whisper — evidence that for modern
  attention decoders, prompt-style hints are a legitimate design point, not a hack.
- **`sounds_like` (Speechmatics) is the most Daiya-relevant idea:** map known phoneticizations
  (`รีเอชัน`, `เปอร์เซ็นต์`) → canonical English surface (`relation`, `percent`). Whisper has no
  such API, but Daiya can implement it as post-processing (§8).

---

## 7. HF transformers logit processors (feasible only if Daiya adds an HF path)

If Daiya runs a second (non-CT2) decode for hard cases, HF `generate` offers **true weighting**:

- **`sequence_bias`** (`SequenceBiasLogitsProcessor`): `{ (token_ids…): bias_float }`; positive
  bias raises odds of that token sequence, negative lowers. This is the closest thing to real
  keyword boosting — you'd map each glossary term's token sequence to a positive bias.
- **`bad_words_ids`** (`NoBadWordsLogitsProcessor`) now accepts a **float per sequence** for
  penalize/boost (issue #22168): large positive → encourage, large negative → forbid.
- **Custom `LogitsProcessor`:** full control — e.g., boost English-term tokens only when the
  preceding context is Thai/JP script.

**Feasibility for Daiya:** requires loading the merged-LoRA model in HF transformers (not CT2),
which is **slower** — defeats the near-real-time goal on the main path. Realistic use: a **targeted
second pass** only on low-confidence segments (`asr.py:is_low_confidence_segment`,
`low_confidence_words` at lines 210-232 already flag these). That keeps the fast CT2 first pass and
spends HF+logit-boost only where it matters. This dovetails with the project's open "2 vs 3-pass"
question (AGENTS.md).

**Risk:** dual-runtime maintenance; token-sequence boosting mis-fires on multi-token English words
whose first subword is ambiguous. Test boost magnitude carefully (start ~+2 to +5 logits,
analogous to Google's low boost).

---

## 8. Post-hoc term normalization — the cheapest real win (do this)

Because Daiya *already knows the glossary* (`_english_domain_terms`, plus user-supplied terms),
the highest ROI, fully **runtime-agnostic** method is to **fix phoneticized terms after decoding**.

**Mechanism — two complementary matchers:**
1. **Surface fuzzy match:** for each decoded token/word, if edit-distance (Levenshtein) to a
   glossary term is small, replace. Catches near-miss English spellings (`relashion` → `relation`).
2. **Phonetic match across scripts:** the hard case is Thai/JP *transliteration* of an English
   term (`รีเอชัน` → `relation`). Approach: romanize the Thai/kana span (e.g. via a Thai→Latin or
   kana→romaji transliterator), then Double-Metaphone / Soundex the romanization and match against
   metaphone codes of glossary terms. Literature confirms this pattern: Double-Metaphone +
   edit-distance + context scoring is a standard ASR post-correction recipe (EnviousWispr ships a
   6-pass fuzzy pipeline; arXiv 2102.11480 optimizes phonetic-correction contexts; ResearchGate
   220875302 does NER+pronunciation-primitive correction).

**Why it fits Daiya specifically:**
- Runs on the CT2 output — **no runtime change, no model surgery**.
- Uses the exact glossary Daiya already extracts, so precision is controllable (only replace when
  match confidence is high).
- Deterministic and debuggable, unlike prompt priming.
- Handles the *consistency-across-chunks* requirement directly: normalize every chunk's output to
  the canonical surface form, so `relation` renders identically regardless of how the acoustic
  model wobbled.

**Casing / mixed-script normalization:** maintain a canonical form per term (e.g. `Kubernetes`,
`OAuth`, `percent`) and force the output to it — mirrors Azure's "display text" + Speechmatics'
`content` field. Handle word-boundary spacing between Thai and inserted Latin term (Daiya already
has `normalize_thai_spacing` in `asr.py:199`).

**Risks:** false replacement (a real Thai word that romanizes near a glossary term). Mitigate with
(a) a minimum edit-distance/phonetic threshold, (b) only replace inside spans the model already
flagged low-confidence (`low_confidence_words`), (c) require the term to be in the *active* prompt
memory (recently seen), reusing `ASRPromptMemory._terms` counts as a prior.

**Experiment:** build a gold set of code-switch utterances with known phoneticized failures.
Baseline CER + per-term recall from current pipeline. Add the phonetic-normalizer as a new stage
after `_convert_fw_segments`. Sweep the edit-distance/metaphone threshold; measure per-term recall
gain vs false-insert rate. Success = higher term recall with false-insert rate ≤ baseline.

---

## 9. Concrete mapping to Daiya files

- **`daiya/src/daiya/asr.py`**
  - `FasterWhisperASR.transcribe_utterance` (line 147): add a `hotwords` passthrough kwarg to
    `self._model.transcribe(...)` (line 170) for the §1 A/B. Note `hotwords` is ignored if a
    `prefix` is ever set.
  - New post-processing hook after `_convert_fw_segments` (line 235) for the §8 phonetic
    normalizer — operate on `ASRSegment.text` and `WordTimestamp.word`.
  - `is_low_confidence_segment` / `low_confidence_words` (lines 210-232) are the natural gate for
    a targeted HF second pass (§7) and for scoping §8 replacements.
- **`daiya/src/daiya/pipeline.py`**
  - `ASRPromptMemory.build_prompt` (line 432): reorder so `Terms:` is closest to audio; consider
    dropping the instruction sentence (line 442); switch char caps to token-aware caps.
  - `ASRPromptMemory.terms()` (line 446) is the ready-made source for both a `hotwords` string and
    the post-hoc glossary — reuse it, don't build a second term store.
  - `_english_domain_terms` (line 519) already yields canonical surface forms — extend it to hold
    a canonical-casing map for §8 normalization.

---

## 10. Recommendations (priority order)

1. **A/B `hotwords` vs the `Terms:` prompt** (§1). Trivial code, immediate signal. Likely a wash
   since both are prompt injection — but confirms the ceiling of prompt methods.
2. **Ship a post-hoc phonetic-glossary normalizer** (§8). Best ROI, no runtime change, directly
   enforces cross-chunk consistency. This is the recommended primary fix.
3. **Tighten the existing prompt** (§2): terms last, drop the instruction sentence, token-aware
   caps.
4. **If 1-3 leave a residual on truly-hard terms:** add a **targeted HF-transformers second pass
   with `sequence_bias` logit boost** on low-confidence segments only (§7) — real weighting, cost
   contained. Feeds the project's 2-vs-3-pass question.
5. **Research track (not near-term):** trie-based zero-shot decoding (§4c) or WhisperBiasing/TCPGen
   (§4a) — both need a non-CT2 decoder; revisit only if a heavier accuracy push is justified.

**Bottom line:** inside faster-whisper/CT2 you get prompt hints and suppression, not real boost
weights. The pragmatic route to consistent technical terms is **prompt hint + deterministic
post-hoc phonetic normalization against Daiya's own glossary**, reserving true logit/trie biasing
for a separate HF pass on the hard cases.

---

## Sources

- faster-whisper source (`get_prompt`, hotwords, docstrings): https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/transcribe.py and raw https://raw.githubusercontent.com/SYSTRAN/faster-whisper/master/faster_whisper/transcribe.py
- faster-whisper hotwords dynamic loading issue #1313: https://github.com/SYSTRAN/faster-whisper/issues/1313
- faster-whisper finetuned + initial_prompt issue #590: https://github.com/SYSTRAN/faster-whisper/issues/590
- faster-whisper repo / README: https://github.com/SYSTRAN/faster-whisper
- faster-whisper special tokens/sequences (DeepWiki): https://deepwiki.com/SYSTRAN/faster-whisper/6.2-special-tokens-and-sequences
- OpenAI Whisper hotwords discussion #1477: https://github.com/openai/whisper/discussions/1477
- OpenAI Whisper prompt vs prefix #117: https://github.com/openai/whisper/discussions/117
- OpenAI Cookbook Whisper prompting guide: https://cookbook.openai.com/examples/whisper_prompting_guide
- Prompt engineering in Whisper (axinc-ai): https://medium.com/axinc-ai/prompt-engineering-in-whisper-6bb18003562d
- Can Contextual Biasing Remain Effective with Whisper and GPT-2? (TCPGen): https://arxiv.org/abs/2306.01942 / https://arxiv.org/pdf/2306.01942
- KWS-Whisper / CB-Whisper multitask biasing + OV-KWS: https://arxiv.org/abs/2309.09552 / https://arxiv.org/html/2309.09552v3
- WhisperBiasing reference implementation: https://github.com/BriansIDP/WhisperBiasing
- CB-Whisper (TTS-based KWS): https://www.catalyzex.com/paper/cb-whisper-contextual-biasing-whisper-using
- Zero-shot trie-based decoding w/ synthetic multi-pronunciation: https://arxiv.org/pdf/2508.17796
- hotwords-for-whisper community fork: https://github.com/jiang-yw/hotwords-for-whisper
- CTranslate2 Whisper API (suppress_tokens, suppress_sequences, return_logits_vocab, prefix_bias_beta): https://opennmt.net/CTranslate2/python/ctranslate2.models.Whisper.html and https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html
- HF generation LogitsProcessor / sequence_bias / bad_words: https://huggingface.co/docs/transformers/main_classes/text_generation and https://github.com/huggingface/transformers/blob/main/src/transformers/generation/logits_process.py and https://github.com/huggingface/transformers/issues/22168
- Deepgram Keyterm Prompting: https://developers.deepgram.com/docs/keyterm
- Deepgram Keywords (intensifiers): https://developers.deepgram.com/docs/keywords
- Deepgram Nova-3 announcement: https://deepgram.com/learn/introducing-nova-3-speech-to-text-api
- AssemblyAI custom vocabulary / word_boost / boost_param: https://docs.assemblyai.com/guides/boosting-accuracy-for-keywords-or-phrases
- Word boosting tradeoffs (MindStudio): https://www.mindstudio.ai/blog/word-boosting-ai-transcription-custom-vocabulary
- Google Cloud Speech model adaptation / boost 0-20: https://docs.cloud.google.com/speech-to-text/docs/adaptation-model
- Azure improve accuracy with phrase list: https://learn.microsoft.com/en-us/azure/ai-services/speech-service/improve-accuracy-phrase-list
- Azure custom pronunciation (lexicon/IPA): https://learn.microsoft.com/en-us/azure/ai-services/speech-service/customize-pronunciation
- Speechmatics custom dictionary (sounds_like): https://docs.speechmatics.com/speech-to-text/features/custom-dictionary
- ASR post-processing correction (NER + pronunciation primitive): https://www.researchgate.net/publication/220875302_ASR_post-processing_correction_based_on_NER_and_pronunciation_primitive
- Evolutionary optimization of phonetic-correction contexts: https://arxiv.org/pdf/2102.11480
- ASR Error Correction survey (EmergentMind): https://www.emergentmind.com/topics/asr-error-correction-aec
