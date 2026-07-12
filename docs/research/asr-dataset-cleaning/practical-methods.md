# Practical ASR dataset cleaning for Thai–English and Japanese–English code-switching

## Scope and position

This note proposes *experiments*, not universal cutoffs. Most cited results are from English or other multilingual settings; transfer to Thai–English (TH–EN) and Japanese–English (JA–EN) must be measured. Cleaning should preserve a frozen evaluation set and produce an auditable manifest containing raw signals, the rule/version, and the action (`keep`, `down-weight`, `repair`, `quarantine`). Prefer ranking and soft weights to deletion until a threshold has demonstrated downstream benefit.

## Reproducible signals

| Signal | Computation and likely use | Can assess without new labels? | Main pitfall |
|---|---|---:|---|
| Normalized sequence confidence | Teacher log probability divided by emitted token/character count; also record mean/min token posterior, entropy, CTC blank mass, beam margin, and LM-free versus LM-assisted score. Rank within language-pair, duration, and source strata. | Yes | Raw scores favor short/easy/majority-language speech and are not probabilities of transcript correctness. An LM can confidently “correct” real code-switches. |
| Calibration | On an existing trusted dev set, map confidence to empirical utterance correctness (e.g., WER/CER below a declared bound), report reliability bins/ECE, and fit temperature or isotonic scaling out of sample. | With an existing trusted set; otherwise only relative ranking | Temperature scaling needs representative clean labels and can fail under domain/language shift. Guo et al. establish the general calibration issue, not ASR-specific validity. |
| Forced alignment / CTC consistency | Align the supplied transcript with an independent CTC acoustic model. Record normalized path score, aligned-token coverage, unaligned prefix/suffix, blank/non-speech spans, token duration outliers, and boundary proximity. Use low coverage plus edge truncation to route segmentation repair. | Yes, as anomaly evidence | Alignment inherits model/tokenizer weaknesses; foreign words, Thai no-space text, kanji readings, filled pauses, and overlap can look “bad” while labels are correct. CTC segmentation was explicitly proposed for extracting utterances from long recordings [Kürzinger et al.]. |
| Timestamp consistency | Compare transcript-derived token spans, ASR timestamp tokens/word spans, VAD speech regions, and clip limits. Flag non-monotonic spans, speech outside labeled bounds, long internal speech with no tokens, or text aligned mainly to silence. | Yes | Timestamp heads are model estimates, not ground truth; do not reject solely on one timestamp source. |
| Multi-view agreement | Decode with at least two meaningfully different views: different architecture/checkpoint, LM-free versus LM-assisted, original versus mild augmentation, or CTC versus seq2seq. Normalize text, then compute pairwise CER plus script/language-tag agreement and boundary agreement. Consensus can select a candidate label; disagreement should quarantine, not majority-vote automatically. | Yes | Correlated models share errors. Models trained on the candidate corpus are not independent. Transliteration and equivalent orthography inflate disagreement. |
| Per-example loss / forgetting | Score with cross-fitting: train on folds excluding the scored utterance; record normalized CTC/NLL across epochs, rank stability, and forgetting/high-loss persistence. Combine with independent alignment/agreement signals. | Yes | High loss also identifies rare names, accents, switches, overlap, and hard-but-valuable examples. In-sample loss rewards memorization of wrong labels. |
| Duration and character rate | Hard integrity checks (empty text/audio, corrupt decode, impossible CTC input/output length), then stratified robust outliers for duration, speech duration, Unicode grapheme count per speech-second, token count per second, and silence ratio. Use median/MAD or quantiles per pair/source/switch class. | Yes | Thai has no obligatory inter-word spaces; Japanese mixes scripts; character rates are not comparable across scripts/tokenizers. Fast speech is not necessarily noise. |
| Language/script consistency | Unicode-script proportions (Thai, Han, Hiragana, Katakana, Latin), transcript LID, acoustic LID, and their agreement. Define an allow-list for digits, punctuation, symbols, Latin acronyms, Japanese romaji, and conventional loanword spellings. Estimate switch count and run lengths after deterministic normalization. | Yes | Script is not language: English can be transliterated; Japanese uses Latin and kanji; named entities and technical terms are exactly the desired tail. Never require one script per utterance. |
| Duplicate/leakage checks | Exact audio hash after canonical decoding; near-audio fingerprint or embedding nearest neighbors; exact and normalized-text hash; combined audio/text clusters. Split duplicate components as a group and compare train candidates against dev/test. | Yes | Same text with different speakers is useful, while same audio with alternate valid transcripts may expose normalization variants. Aggressive embedding thresholds merge formulaic but distinct speech. |
| Segmentation repair | For suspect clips, use VAD plus CTC/token alignment to propose new boundaries with context padding; merge fragments when text aligns across the join; split long clips only at supported silence/alignment gaps. Re-score repaired and original versions and keep provenance. | Yes for proposal and consistency; quality claim needs downstream test | Cutting at VAD alone can remove unvoiced phones or split a code-switch. Overlap and cross-talk may be irreparable without source separation/manual review. |
| Robust label selection | Candidate set = supplied label plus independent model hypotheses and conservative normalization variants. Select only when agreement/alignment improves by a predeclared margin; otherwise retain original with weight or quarantine. Soft targets/n-best alternatives are preferable when the trainer supports them. | Yes for selection heuristics | Replacing human text with the current model's preferred text erases rare switches and creates confirmation bias. Keep immutable originals. |

The weak-supervision pipeline behind Whisper used language/transcript matching, heuristic removal of machine-generated text, fuzzy deduplication, and evaluation-set deduplication at scale, while also reporting that filtering thresholds trade quantity against quality [Radford et al.]. This is useful precedent, not evidence that its exact heuristics fit Daiya.

## Self-training without laundering label errors

Noisy Student ASR iterates teacher pseudo-labeling and student training with SpecAugment, including confidence filtering and balancing [Park et al.]. slimIPL instead continuously refreshes hard LM-free pseudo-labels through a cache [Likhomanenko et al.]. These methods motivate two distinct treatments:

1. **Cleaning labeled data:** use cross-fitted teacher disagreement as one diagnostic; do not silently overwrite the supplied transcript.
2. **Using quarantined data:** retain it as unlabeled/weakly labeled audio and train with a lower unsupervised weight, augmentation, an EMA or independently initialized teacher, and periodically refreshed labels. Compare against simply dropping it.

Confirmation-bias protections should be predeclared: frozen human-labeled dev/test sets; teacher and scorer trained without the scored item; at least one architecturally or data-independent view; no threshold tuning on test; preserve switch-rate, language-pair, speaker/source and duration distributions; cap pseudo-label contribution per stratum; monitor teacher–student error correlation and diversity; retain uncertain samples rather than forcing labels; and audit a small stratified sample if annotation becomes available. Iterative self-training can collapse or reinforce systematic errors, so an improvement in teacher confidence is not evidence of better transcripts.

## Proposed experimental matrix

Run the same base training recipe/seeds and data budget where possible. Thresholds are selected on an existing trusted development set or by fixed retained-data quantiles; never on the final test set.

| Arm | Data action | Signals |
|---|---|---|
| A0 | Baseline, integrity failures only | Decode validity, empty/missing fields, impossible length |
| A1 | Hard heuristic filter | A0 + broad duration/speech-rate/silence bounds, stratified by pair/source |
| A2 | Alignment rank | A0 + bottom 1/5/10% independent CTC alignment consistency removed or down-weighted |
| A3 | Agreement rank | A0 + bottom 1/5/10% multi-view agreement removed or down-weighted |
| A4 | Loss rank | A0 + persistent cross-fitted high loss at matched retained hours |
| A5 | Composite soft weight | Calibrated/percentile ranks from alignment, agreement, loss, and rate; no script veto |
| A6 | Conservative repair | A5 plus CTC+VAD boundary repair; accept only if two independent consistency metrics improve |
| A7 | Robust label selection | A6 plus candidate replacement only above a fixed agreement margin; ambiguous cases down-weighted |
| A8 | Quarantine-as-unlabeled | Best A2–A7 clean subset supervised; rejected audio used by noisy-student/EMA pseudo-labeling |
| Controls | Bias checks | Random removal matched by hours; model-only confidence; LM-free-only; each signal ablated |

Run TH–EN and JA–EN separately and pooled, and report results by dominant language, switch/no-switch, switch density, script pattern, source, duration, speaker (if available), and rare-term bucket. If compute is constrained, screen with A0/A2/A3/A5 and matched random controls, then promote only promising arms to full-seed runs.

## Decision criteria

Adopt a cleaning policy only if it:

- improves or is non-inferior on the frozen, human-trusted evaluation set across repeated seeds, with paired bootstrap confidence intervals over utterances (and speaker-cluster bootstrap when speaker IDs exist);
- improves the primary normalized metric declared for each language pair (report both CER and WER or a documented language-aware word segmentation), without material regression in code-switched and rare-term slices;
- beats matched random removal at the same retained hours and improves data efficiency, not merely training speed;
- does not materially reduce switch rate, minority-script token share, speaker/source coverage, or tail-term coverage relative to the raw corpus;
- has stable rankings/decisions across model seeds or scorer checkpoints and yields an auditable rejection-reason distribution;
- for repairs/relabels, improves independent alignment/agreement and downstream ASR; self-consistency alone is insufficient.

Without new manual labels, one can reproducibly evaluate integrity, duplicate/leakage rates, signal stability, cross-model agreement, alignment/timestamp consistency, distribution preservation, and—if a trusted test set already exists—downstream CER/WER. One cannot directly estimate precision/recall of “bad-label” detection, prove a replacement transcript is correct, or distinguish rare valid code-switches from systematic errors. A small blinded, stratified audit is therefore the strongest next investment if close arms remain tied.

## Primary sources and official references

- Ludwig Kürzinger, Dominik Winkelbauer, Lujun Li, Tobias Watzel, and Gerhard Rigoll. “CTC-Segmentation of Large Corpora for German End-to-End Speech Recognition.” *Proceedings of Interspeech 2020*, pp. 2677–2681. DOI: [10.21437/Interspeech.2020-1392](https://doi.org/10.21437/Interspeech.2020-1392); [paper record](https://www.isca-archive.org/interspeech_2020/kuerzinger20_interspeech.html).
- Daniel S. Park, Yu Zhang, Ye Jia, Wei Han, Chung-Cheng Chiu, Bo Li, Yonghui Wu, and Quoc V. Le. “Improved Noisy Student Training for Automatic Speech Recognition.” *Proceedings of Interspeech 2020*, pp. 2817–2821. DOI: [10.21437/Interspeech.2020-1470](https://doi.org/10.21437/Interspeech.2020-1470); [ISCA record](https://www.isca-archive.org/interspeech_2020/park20d_interspeech.html).
- Tatiana Likhomanenko, Qiantong Xu, Jacob Kahn, Gabriel Synnaeve, Ronan Collobert. “slimIPL: Language-Model-Free Iterative Pseudo-Labeling.” *Proceedings of Interspeech 2021*, pp. 741–745. DOI: [10.21437/Interspeech.2021-1014](https://doi.org/10.21437/Interspeech.2021-1014); [arXiv:2010.11524](https://arxiv.org/abs/2010.11524).
- Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. “Robust Speech Recognition via Large-Scale Weak Supervision.” *Proceedings of the 40th International Conference on Machine Learning*, PMLR 202 (2023), pp. 28492–28518. [Official PMLR record](https://proceedings.mlr.press/v202/radford23a.html).
- Chuan Guo, Geoff Pleiss, Yu Sun, and Kilian Q. Weinberger. “On Calibration of Modern Neural Networks.” *Proceedings of the 34th International Conference on Machine Learning*, PMLR 70 (2017), pp. 1321–1330. [Official PMLR record](https://proceedings.mlr.press/v70/guo17a.html).
- Alex Graves, Santiago Fernández, Faustino Gomez, and Jürgen Schmidhuber. “Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks.” *Proceedings of ICML 2006*, pp. 369–376. DOI: [10.1145/1143844.1143891](https://doi.org/10.1145/1143844.1143891).
- Shubham Toshniwal, Tara N. Sainath, Ron J. Weiss, Bo Li, Pedro J. Moreno, Eugene Weinstein, and Kanishka Rao. “Multilingual Speech Recognition with a Single End-to-End Model.” *2018 IEEE ICASSP*, pp. 4904–4908. DOI: [10.1109/ICASSP.2018.8461972](https://doi.org/10.1109/ICASSP.2018.8461972); [Google Research record](https://research.google/pubs/multilingual-speech-recognition-with-a-single-end-to-end-model/). The paper reports that an explicit language identifier reduced cross-language confusion; this is evidence for tracking language/script consistency, not for rejecting natural switches.
- Unicode Consortium. [Unicode Script Property](https://www.unicode.org/reports/tr24/) (*Unicode Standard Annex #24*) and [Unicode Text Segmentation](https://www.unicode.org/reports/tr29/) (*Unicode Standard Annex #29*). These are the official basis for reproducible script counts and grapheme-cluster length; neither provides language identification.
