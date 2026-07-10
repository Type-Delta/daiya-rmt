# 06 — Diarization-Aware ASR and ASR↔Diarization Interaction

Research report for Daiya-RMT. Focus: how ASR and speaker diarization interact,
how to make the multiplexer's speaker-assignment robust, and what the streaming
diarization literature says about keeping speaker identity stable in
near-real-time Thai-English / Japanese-English code-switched conversation.

**How this maps to Daiya code (verified in this repo):**

- `daiya/src/daiya/mux.py` — `TranscriptMultiplexer`. Assigns each *whole* ASR
  segment to a speaker via `_speaker_for()` using `overlap_seconds()` against
  diarization turns; ties broken by `(overlap, turn.confidence)`. Finalizes
  segments only once they fall behind `_commit_horizon()` (= max end of any
  *committed/final* turn). Re-emits `transcript.update` from
  `_update_overlapping_final_segments()` when a committed turn's speaker changes.
- `daiya/src/daiya/pipeline.py` — owns `diarizer` and `mux`; config exposes
  `window_seconds`, `diarization_commit_delay_seconds`, `diarization_profile`
  (`fast`/`balanced`/…), `diarization_backend`. `_handle_diarization_turns()`
  feeds turns into `mux.ingest_diarization_many()`.
- `daiya/src/daiya/asr.py` — calls faster-whisper with `word_timestamps=True`
  (line ~161) and packs per-word `start/end/probability` into
  `WordTimestamp` tuples on each `ASRSegment` (lines ~202-216). **These word
  timestamps are currently carried through the mux but NOT used for speaker
  assignment** — `_speaker_for` only looks at segment-level `start`/`end`.

The single most actionable finding: **Daiya assigns speakers at segment
granularity, but the whole diarization-aware-ASR literature (WhisperX, SA-ASR,
Sortformer) assigns at word granularity.** Word-level assignment is the biggest
available accuracy win for boundary words and back-channels, and Daiya already
has the word timestamps needed to do it.

---

## 1. WhisperX: VAD cut-&-merge + forced alignment + overlap assignment

**Mechanism.** WhisperX ([Bain et al., 2023, arXiv:2303.00747](https://arxiv.org/abs/2303.00747))
is the reference "cascaded" diarization-aware ASR pipeline:

1. **VAD pre-segmentation** with the pyannote segmentation model to find voice
   regions.
2. **Cut & Merge** into ~30 s chunks. A *min-cut* splits long active regions
   "at the point of minimum voice activation score," constrained to lie between
   half and the full Whisper training window (15–30 s); a *merge* then packs
   neighbouring regions up to 30 s to maximize context. Boundaries deliberately
   land on low-speech points, which **reduces Whisper hallucination/repetition**
   and enables batched within-audio inference → reported **11.8× speedup** and
   *better* WER (9.7 vs 10.5 on TED-LIUM).
3. **Forced phoneme alignment.** Whisper's own timestamps are discarded. A
   wav2vec2.0 phoneme recognizer classifies phonemes in each segment; **DTW over
   the phoneme logits** yields per-phoneme timing, and word boundaries are read
   off the constituent phonemes. Word-segmentation precision/recall: 93.2%/65.4%
   on Switchboard, 84.1%/60.3% on AMI.
4. **Speaker assignment** (`assign_word_speakers`): pyannote diarization runs
   separately; each already-timed word is tagged with whichever diarization
   segment its time falls inside — i.e. **time-overlap matching at the word
   level** ([DeepWiki: WhisperX diarization](https://deepwiki.com/m-bain/whisperX/3.4-speaker-diarization)).

**Known limits (stated by WhisperX + issues).**
- **Overlapped speech is explicitly not handled** — the word→speaker step picks
  exactly one speaker per word; both WhisperX and whisper-diarization READMEs say
  overlap is poorly handled ([search corpus](https://novascribe.ai/whisper-diarization)).
- **Boundary words** near a turn change are only as good as the word timestamp;
  a word straddling a turn boundary can be mis-assigned.
- **Multilingual alignment needs a language-specific phoneme model**; coverage
  varies and translation mode disables alignment (audio↔text mismatch).
- `assign_word_speakers` can **omit the `speaker` key** for some words
  ([issue #1072](https://github.com/m-bain/whisperX/issues/1072)) and is an
  O(words×turns) bottleneck (up to 63% of runtime) unless you use an interval
  tree ([issue #1335](https://github.com/m-bain/whisperX/issues/1335)).

**Relevance to Daiya.** WhisperX's *word-level overlap assignment* is exactly
the upgrade path for `mux._speaker_for`. Its **VAD cut-&-merge** philosophy also
matters for Daiya's open question on ASR utterance segmentation: cut on low-VAD
points, not fixed clocks, to avoid slicing a word across chunks. Note Daiya
uses faster-whisper's *native* word timestamps (cross-attention DTW, §3), **not**
wav2vec2 forced alignment — so Daiya's word timings are coarser than WhisperX's
and word-level assignment must be robust to that (min-overlap + tie-breaking).

**Risk.** A separate wav2vec2 aligner per language (Thai + Japanese + English)
is heavy for streaming and Thai/Japanese phoneme models are weaker than English.
Do **not** adopt full WhisperX forced alignment for streaming; adopt only the
*word-level assignment* idea on top of Whisper's existing word timestamps.

**Experiment.** On a held-out Thai-EN / JA-EN diarized clip, compare
segment-level assignment (current `_speaker_for`) vs a word-level variant that
assigns each word by max time-overlap and lets one ASR segment carry mixed
speakers (split on speaker change). Metric: word-level speaker error rate
(fraction of words with wrong speaker) and boundary-word accuracy specifically.

---

## 2. pyannote.audio + streaming: keeping labels consistent across chunks

**Offline pyannote pipeline.** `speaker-diarization-3.1` (and community-1) =
local **segmentation** model (overlap-aware, per-frame speaker activity) →
**speaker-embedding** model on active regions → **clustering** (agglomerative)
to assign global speaker labels ([pyannote-audio repo](https://github.com/pyannote/pyannote-audio),
[hf: speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)).
Offline clustering sees the whole file, so labels are globally consistent by
construction — the hard part is doing that **online**.

**diart (online).** diart is the official impl of Coria et al.,
["Overlap-aware low-latency online speaker diarization based on end-to-end local
segmentation" (arXiv:2109.06483)](https://arxiv.org/abs/2109.06483). It treats
online diarization as **incremental clustering + local diarization on a rolling
buffer updated every 500 ms**. Key devices:
- **Latency adjustable 500 ms – 5 s**; the paper systematically studies the
  latency↔DER curve on AMI/DIHARD/VoxConverse.
- **Modified statistics-pooling** down-weights frames where segmentation predicts
  simultaneous speakers (overlap-aware embeddings).
- **Cannot-link constraints** from the local segmentation stop two locally
  co-active speakers from being wrongly merged during incremental clustering.
- **Consistency across chunks = incremental clustering**: new local speakers are
  matched to existing global centroids rather than re-labeled fresh.

**How to keep labels stable across chunks — two families:**
1. **Centroid/embedding matching** (diart-style, and Daiya's own approach):
   cache a representative embedding per known speaker; for each new chunk,
   match local speakers to the nearest cached centroid (cosine), spawn a new
   speaker only if no centroid is close enough. Cheap, streaming-friendly,
   avoids re-labeling the past. Risk: centroid drift and threshold sensitivity.
2. **Re-clustering** the buffer each step (global AHC over a window). More
   accurate within the window but can *renumber* speakers → **label flicker**
   and forces re-emitting past segments.

**Relevance to Daiya.** Daiya's diarizer already does family (1) (cache speaker
embeddings, re-match in later turns) — this matches diart's incremental-cluster
design and is the right choice for streaming. The mux's
`_update_overlapping_final_segments` + `transcript.update` is exactly the
mechanism you need to *repair* a past segment when a re-identified speaker
changes a committed turn's label — i.e. Daiya already tolerates the "relabel the
past" event that pure family-(1) systems try to avoid, which is a strength.

**Risk / experiment.** Centroid matching with too-tight a threshold splits one
speaker into many (label proliferation); too-loose merges two speakers. Run a
sweep of the embedding-match threshold on Thai-EN/JA-EN clips and report DER +
number-of-speakers error + label-flip count (how often a committed turn's
speaker later changes). Adopt diart's **cannot-link** idea: never merge two
speakers that were simultaneously active in the same chunk.

---

## 3. Word-timestamp quality from Whisper and how it corrupts speaker assignment

**Why native Whisper timestamps are coarse.** Whisper was trained to emit
timestamp *tokens* at ~0.02 s grid but on large-scale noisy data; the timestamps
are known to be unreliable, which is precisely why WhisperX replaced them with
forced alignment. faster-whisper's `word_timestamps=True` (what Daiya uses)
instead derives per-word timing by running **DTW over the cross-attention
weights of designated `alignment_heads`** — a subset of decoder attention heads
empirically found to track audio position. DTW "hardens" the fuzzy attention
into a monotonic token→frame path; word start/end come from that path
([openai/whisper word-timestamp method; whisper-timestamped](https://github.com/linto-ai/whisper-timestamped)).

**Quality caveats that matter for diarization:**
- Accuracy depends entirely on the **quality of the cross-attention** and the
  choice of `alignment_heads` — these are **model-specific** and for a
  **fine-tuned / LoRA-merged large-v3** the default alignment-head set may no
  longer be optimal. This is a real risk for Daiya: LoRA fine-tuning can shift
  which heads carry alignment signal, degrading word timing silently.
- Timings drift most at **segment boundaries** and around **disfluencies /
  code-switch points** — exactly where speaker turns also change. A boundary
  word timed 100–300 ms off can overlap the *wrong* diarization turn.
- Recent work (CrisperWhisper, [arXiv:2408.16589](https://arxiv.org/html/2408.16589v1);
  "Whisper Has an Internal Word Aligner", [arXiv:2509.09987](https://arxiv.org/html/2509.09987v1))
  shows careful alignment-head selection + retokenization substantially tightens
  Whisper word timing without an external aligner — a lighter alternative to
  WhisperX for Daiya.

**How this feeds mux robustness.** Because Daiya's word timings are DTW-coarse,
a naïve word-level assignment could be *noisier* than segment-level. Make it
robust:
- **Word-level overlap, not just segment-level**: assign each word by max overlap
  with diarization turns, then group consecutive same-speaker words into
  sub-segments. Split an ASR segment when the dominant speaker changes mid-word-run.
- **Min-overlap threshold**: require overlap ≥ e.g. 30–50% of the word's
  duration (or an absolute floor like 80–120 ms) before trusting it; below that,
  fall back to the segment-level speaker or nearest-turn. Daiya's current
  `_speaker_for` only requires `overlap > 0`, so a 1-frame graze wins — tighten
  this.
- **Tie-breaking**: current `(overlap, confidence)` is reasonable; add a third
  key = temporal continuity (prefer the speaker of the adjacent word/segment) to
  suppress single-word flips.
- **Collar / nearest-fallback**: for a word with no turn overlap (VAD/diar
  disagreement), assign the *nearest* turn within a small collar (0.25 s is the
  standard diarization non-scoring collar) rather than `UNKNOWN`.

**Experiment.** (a) Re-derive `alignment_heads` for the LoRA-merged large-v3 and
measure word-timing error vs stock large-v3 on a hand-aligned Thai-EN/JA-EN set.
(b) Ablate min-overlap floor {0, 80 ms, 150 ms, 30%, 50%} on boundary-word
speaker accuracy.

---

## 4. Speaker-attributed ASR, TS-ASR, SOT, and joint systems — transferable ideas

Daiya is a **cascade** (independent ASR + diarization joined by the mux). The
end-to-end literature won't be swapped in, but several ideas transfer to keeping
speaker identity stable:

- **Serialized Output Training (SOT)** and **token-level t-SOT**
  ([Kanda et al., arXiv:2203.16685](https://arxiv.org/abs/2203.16685);
  [t-SOT overview](https://www.emergentmind.com/topics/serialized-output-training-t-sot)):
  one decoder emits lexical tokens *and* speaker-change tokens in a single
  arrival-ordered stream, natively handling overlap without a separate diarizer.
  Transferable idea: **arrival-time ordering** as the canonical speaker index —
  stable, permutation-free labels (SPEAKER_00 = first to speak), which is exactly
  what a streaming UI wants.
- **Target-speaker ASR / Diarization-Conditioned Whisper (DiCoW)**
  ([arXiv:2510.03723](https://arxiv.org/abs/2510.03723)): condition the ASR
  encoder on a target speaker's embedding to transcribe *that* speaker even under
  overlap. Transferable but heavy — would require modifying the Whisper encoder;
  out of scope for v0 but the strongest answer to overlap if Daiya ever goes
  end-to-end.
- **NeMo Streaming Sortformer** ([arXiv:2507.18446](https://arxiv.org/html/2507.18446v1),
  [hf model](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2)):
  the most directly relevant. Trained with **Sort Loss** so speakers are *output*
  in arrival order (no permutation step → **no label flicker**). An
  **Arrival-Order Speaker Cache (AOSC)** stores frame-level embeddings of
  previously seen speakers and is concatenated with a FIFO of recent chunks +
  the current buffer, so the transformer re-identifies returning speakers
  itself. Cache size adapts to model confidence (not fixed-length). Latency
  configs: chunk 3 frames ≈ 240 ms (0.32 s latency) up to 124 frames ≈ 10 s;
  frame = 80 ms; cache ≈ 188 frames ≈ 15 s. **DER barely degrades vs offline**:
  DIHARD-III 14.17% offline → 13.67% @1.04 s → 13.43% @0.32 s (with
  post-processing), and streaming even *beats* offline on 4+ speakers.

**Relevance to Daiya.** AOSC is a validated version of Daiya's own
"cache speaker embeddings, re-match across turns" design — evidence the approach
is sound and can be near-real-time without wrecking accuracy. Two concrete
borrowings: (1) **arrival-time speaker ordering** for stable, flicker-free
labels; (2) **confidence-adaptive cache size** per speaker instead of a fixed
embedding. If a drop-in is ever wanted, `diar_streaming_sortformer_4spk-v2` is a
ready streaming diarizer with published low-latency DER and consistent labels —
a strong candidate to benchmark against the current pyannote path.

**Risk.** Sortformer public models cap at **4 speakers**; fine for 2-person
Thai-EN/JA-EN interviews, risky for group meetings.

---

## 5. Diarization-aware transcript correction (and the LLM pass)

**Using speaker turns to fix segmentation.** Split-When-Merged (SWM): detect an
ASR segment that a single speaker was assigned but that a diarization turn
boundary crosses, and split it at the boundary; conversely merge adjacent
same-speaker fragments ([search corpus / SWM](https://arxiv.org/html/2406.04927v1)).
This is the mux's job — currently `_speaker_for` gives one speaker per ASR
segment, so a segment spanning a turn change is *forced* to a single speaker.
Adding word-level split (§3) *is* SWM for Daiya.

**LLM correction, diarization-conditioned.** Evidence LLMs help *when
fine-tuned*: Wang et al., ["LLM-based speaker diarization correction: a
generalizable approach" (arXiv:2406.04927)](https://arxiv.org/html/2406.04927v1)
show a fine-tuned, ASR-agnostic LLM post-processor measurably lowers diarization
error — but **zero-shot LLMs made it *worse* than baseline**. For code-switched
clinical dialogue, ["Doctor or Patient?" (arXiv:2603.06373)](https://arxiv.org/html/2603.06373)
pairs EEND-VC diarization with a **dialogue-level LLM** that applies *minimal,
high-confidence* edits while preserving spoken style, segmentation and the
code-mixed nature — directly analogous to Daiya's Thai-EN/JA-EN correction pass.

**Relevance to Daiya.** When Daiya adds its LLM correction pass (`correct.py`),
feed it **speaker structure**, not just raw text: (a) who is speaking each line,
so the model keeps **speaker-consistent terminology** (one speaker's jargon /
name spellings stay stable); (b) turn boundaries, so it doesn't "correct" across
a speaker change; (c) instruct **minimal high-confidence edits only** — the
evidence says aggressive/zero-shot rewriting hurts. Speaker labels also let the
LLM resolve code-switch homophones by speaker habit.

**Risk.** LLM can "fix" a real disfluency or a genuine code-switch into
monolingual text, violating Daiya's clean-read-but-faithful label policy. Keep
edits conservative and diff-gated. **Experiment**: A/B the correction pass with
vs without speaker/turn context on WER *and* code-switch-point preservation.

---

## 6. Overlapped speech and short back-channels ("ครับ", "อ๋อ", "うん")

**The problem.** A back-channel is a short utterance produced *while the other
speaker holds the floor* → one segment fully contained in another (overlap). Two
failure modes: (a) diarization misses the short/overlapped speaker; (b) the ASR
segment covering the overlap gets one speaker, so the back-channel is attributed
to the floor-holder — or the floor-holder's word is attributed to the
back-channeler. Interactive dialog has frequent back-channels → high rate of
missed speaker-change points ([Frontiers/​PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC10824834/)).

**What the literature does.** Overlap-aware segmentation (pyannote/diart) can
*detect* overlap regions, but cascade word-assignment still emits one speaker per
word (WhisperX limitation, §1). A common heuristic: assign an overlapped segment
to the **nearest speaker in time before/after** the overlap. End-to-end
(SOT/Sortformer/DiCoW) is the principled fix because it can emit two speakers'
words in one region.

**Relevance to Daiya — the tiny-utterance interaction.** This is where §6 meets
Daiya's known tiny-utterance problem. A 200–400 ms "ครับ" / "うん":
- is too short for a **reliable speaker embedding** (see §8 — short context →
  noisy embeddings), so diarization may not even create a distinct turn for it;
- may be swallowed by the floor-holder's ASR segment, so segment-level
  `_speaker_for` can never separate it.

Mitigations: (1) **word-level assignment** (§3) so the back-channel word can get
its own speaker even inside a larger ASR segment; (2) a **short-utterance
exception** — if a word/segment is very short *and* overlaps two turns, prefer
the turn whose speaker differs from the surrounding floor-holder (back-channels
are by definition the non-floor speaker); (3) don't force `UNKNOWN` on
sub-threshold turns — inherit from nearest turn within the 0.25 s collar.

**Risk.** Aggressively splitting for back-channels can invent phantom speakers
on noise. Gate on overlap-detection confidence. **Experiment**: label a set of
Thai/Japanese back-channels; measure back-channel speaker accuracy under
segment-level vs word-level vs word-level-with-back-channel-rule.

---

## 7. Streaming diarization latency vs accuracy; re-labeling and flicker

**Commit delay / look-ahead.** diart trades latency 500 ms–5 s for DER along a
measured curve; Streaming Sortformer uses **right-context lookahead** (e.g. 7
frames ≈ 560 ms at the 1.04 s config) and shows DER barely moves from offline
even at 0.32 s latency. Takeaway: **a small look-ahead (0.5–1 s) buys most of
the accuracy; going lower costs little more** on ≤4-speaker audio
([Sortformer](https://arxiv.org/html/2507.18446v1),
[diart paper](https://arxiv.org/abs/2109.06483)).

**Relevance to Daiya.** This is exactly what `_commit_horizon()` +
`diarization_commit_delay_seconds` encode. A segment is only finalized once it
falls behind the newest *committed* turn end — i.e. Daiya defers commit until
diarization has "caught up," a commit-delay/look-ahead in disguise. The literature
says pick this delay in the **~0.5–1.0 s** band for a good latency/accuracy
knee. Emit `transcript.partial` immediately with the *provisional* speaker, then
`transcript.final` after the horizon, and `transcript.update` if a re-identified
speaker changes a committed turn — Daiya's three-event model already implements
"relabel the past cleanly," the correct answer to flicker (don't hide flips,
*version* them). `_turn_speaker_changed` + `revision`/`version` bumps give the
frontend a stable way to patch.

**Avoiding flicker.** Sortformer's lesson: flicker comes from **permutation**
(re-clustering renumbers speakers). Mitigate by (a) **arrival-time ordering** so
labels are assigned once and never renumbered; (b) centroid matching (§2) rather
than per-step global re-clustering; (c) **hysteresis** — only re-emit a
`transcript.update` when the new speaker's overlap advantage exceeds a margin
over the old (Daiya currently flips on *any* strictly-greater overlap, which can
oscillate). Add a margin to `_update_overlapping_final_segments`.

**Risk / experiment.** Sweep `diarization_commit_delay_seconds` ∈ {0.5, 1, 2, 3}
and measure end-to-end latency, DER, and **update-churn** (number of
`transcript.update` events per minute). Add an overlap-margin hysteresis param
and show it cuts churn without raising DER.

---

## 8. Segment size for streaming that doesn't wreck diarization (Daiya open Q6)

**Core tension (well-documented).** For diarization there are *two* windows:
- **Embedding window**: must be long enough for a discriminative speaker vector;
  short context → noisy embeddings → harder clustering.
- **Segmentation/decision window**: shorter is often *better* for the
  segmentation/EEND stage because fewer speakers appear and less context must be
  tracked.

Empirically these push opposite ways. An MDPI pyannote study found supervised
diarization **improved ~4.5% when the chunk dropped from 2.0 s to 1.0 s**
([MDPI Sensors 23(4):2082](https://www.mdpi.com/1424-8220/23/4/2082)); but the
same short context yields noisier embeddings. ["Dissecting the segmentation
model" (arXiv:2506.11605)](https://arxiv.org/pdf/2506.11605) frames the
trade-off directly: short chunks simplify the local diarization problem but
degrade embedding quality. Sortformer sidesteps it by learning speaker identity
across chunks via the cache rather than relying on one chunk's embedding.

**Relevance to Daiya (this is literally Q6 in AGENTS.md).** Two knobs:
`window_seconds` (diarizer buffer) and the ASR utterance size. Guidance from the
evidence:
- Keep the **diarization decision cadence short** (diart updates every 500 ms;
  Sortformer chunks 0.24–1 s) for low latency and responsive turn detection.
- Keep the **embedding context longer** (accumulate ≥ ~1.5–3 s of a speaker
  before trusting a fresh centroid) so short back-channels don't spawn phantom
  speakers — i.e. **decouple the two windows** rather than using one segment size
  for both. This is the key design recommendation.
- For **ASR**, use VAD cut-&-merge (§1): cut on low-VAD points near a target
  length rather than a fixed clock, so words/back-channels aren't sliced.
- Because Daiya's ASR utterance segmentation is *separate* from diarization
  segmentation, you can tune them independently — a genuine architectural
  advantage over WhisperX's single VAD.

**Experiment for Q6.** Grid `window_seconds` × ASR-target-length on a fixed
Thai-EN/JA-EN diarized set; report DER, speaker-count error, back-channel
accuracy, word-speaker-error, and end-to-end latency. Hypothesis from the
literature: a **short (~0.5–1 s) diarization decision cadence with a longer
(~2–3 s) embedding-accumulation window** is the knee; a single ~1.5–2 s segment
used for both is a decent baseline but not optimal.

---

## Summary of concrete changes for Daiya

1. **`mux._speaker_for` → word-level assignment** using the word timestamps
   already on `ASRSegment.words`; split segments on mid-segment speaker change
   (SWM). Biggest boundary-word / back-channel win. (§1, §3, §6)
2. **Add a min-overlap floor** (absolute ~80–150 ms or ~30–50% of word duration)
   and a **nearest-turn collar fallback** (0.25 s) instead of `overlap>0` /
   `UNKNOWN`. (§3)
3. **Re-derive Whisper `alignment_heads`** for the LoRA-merged large-v3 and
   verify word-timing quality; bad timestamps silently corrupt assignment. (§3)
4. **Hysteresis margin** in `_update_overlapping_final_segments` to cut
   `transcript.update` churn/flicker; consider **arrival-time speaker ordering**
   for stable labels. (§4, §7)
5. **Decouple diarization decision cadence (short) from embedding window
   (longer)**; tune `window_seconds` and ASR length independently — answers Q6.
   (§8)
6. **Feed speaker/turn structure to the LLM correction pass**, minimal
   high-confidence edits only; preserve code-switch points. (§5)
7. **Benchmark `nvidia/diar_streaming_sortformer` (AOSC)** as an alternative
   stateful streaming diarizer — validated low-latency, flicker-free labels for
   ≤4 speakers. (§4)

---

## Sources

- WhisperX (Bain et al., 2023): https://arxiv.org/abs/2303.00747 and https://arxiv.org/html/2303.00747v2
- WhisperX diarization / assign_word_speakers: https://deepwiki.com/m-bain/whisperX/3.4-speaker-diarization
- WhisperX issue #1072 (missing speaker key): https://github.com/m-bain/whisperX/issues/1072
- WhisperX issue #1335 (assign_word_speakers perf): https://github.com/m-bain/whisperX/issues/1335
- pyannote-audio: https://github.com/pyannote/pyannote-audio
- pyannote speaker-diarization-3.1: https://huggingface.co/pyannote/speaker-diarization-3.1
- pyannote speaker-diarization-community-1: https://huggingface.co/pyannote/speaker-diarization-community-1
- diart (repo): https://github.com/juanmc2005/diart and https://diart.readthedocs.io/en/stable/
- Coria et al., overlap-aware low-latency online diarization: https://arxiv.org/abs/2109.06483
- Streaming Sortformer (AOSC, arrival-time ordering): https://arxiv.org/html/2507.18446v1
- NeMo streaming Sortformer model: https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2
- NeMo diarization models docs: https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/speaker_diarization/models.html
- Streaming SA-ASR with token-level speaker embeddings (t-SOT): https://arxiv.org/abs/2203.16685
- t-SOT overview: https://www.emergentmind.com/topics/serialized-output-training-t-sot
- Diarization-Conditioned Whisper (DiCoW) multi-talker ASR: https://arxiv.org/abs/2510.03723
- SA-SOT (speaker-aware SOT): https://arxiv.org/html/2403.02010v1
- whisper-timestamped (DTW/alignment heads, native Whisper word timing): https://github.com/linto-ai/whisper-timestamped
- CrisperWhisper (verbatim timestamps): https://arxiv.org/html/2408.16589v1
- "Whisper Has an Internal Word Aligner": https://arxiv.org/html/2509.09987v1
- LLM-based diarization correction (fine-tuned helps, zero-shot hurts): https://arxiv.org/html/2406.04927v1
- "Doctor or Patient?" code-switched Hinglish diarization + LLM correction: https://arxiv.org/html/2603.06373
- Speaker-turn aware diarization / back-channels: https://pmc.ncbi.nlm.nih.gov/articles/PMC10824834/
- pyannote chunk-size study (MDPI Sensors): https://www.mdpi.com/1424-8220/23/4/2082
- Dissecting the EEND-VC segmentation model (chunk-size trade-off): https://arxiv.org/pdf/2506.11605
- Overlap-aware resegmentation via neural OSD: https://arxiv.org/pdf/1910.11646
