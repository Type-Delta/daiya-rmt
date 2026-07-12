# ASR dataset cleaning experiment v1

Date: 2026-07-12

## Conclusion

This is a negative/inconclusive result for automatic label replacement. The
current artifacts support a useful quarantine and review-prioritization pipeline,
but they do not support claiming that the proposed corrected labels are cleaner.
Do not use the `correct` rows from this run as production M3A/M3.1A labels.

## Inputs and validity gates

The audit found 7,091 canonical clips (8.69 hours) and 109 protected human-gold
clips. Exact byte hashing found 82/109 (75.2%) gold clips in the canonical
training audio. All mapped clips belong to one source conversation, so a grouped
dev/test split is blocked: scoring those rows is diagnostic contamination, not a
held-out estimate. The other 27 gold clips have no exact canonical-byte mapping;
there is no retained original-label or model-output mapping for them. No Japanese
script was present in the canonical labels.

The full audit is in [dataset-audit.md](../dataset-audit.md), with aggregate-only
machine-readable evidence in [audit-summary.json](../dataset-audit-fixtures/audit-summary.json).

## Candidate policy

The versioned manifest builder preserves source identity, original labels, content
hashes, provenance, signals, reason codes, and dispositions. It drops empty labels
and exact protected-gold overlaps from a training candidate, routes heuristic
outliers and duplicate-content groups to review, and proposes a correction only when at least two distinct
prediction views reach the same normalized transcript. Consensus confidence is
explicitly marked raw and uncalibrated; disagreement is retained as review evidence.
No lexical correction table or free-form LLM rewrite is used.

On the audited metadata plus the retained primary benchmark predictions, the
generated manifest contained 7,091 records: 6,650 keep, 256 review, 92 drop, and
93 uncalibrated `correct` proposals. Review includes the 178 rows in exact
duplicate-content groups. The latter proposals are candidates for human review,
not accepted labels. The full manifest was written outside the repository to a
temporary path; the schema is represented by
[candidate-manifest-v1.sample.jsonl](candidate-manifest-v1.sample.jsonl).

## Spelling audit rerun

A 2026-07-12 rerun added PyThaiNLP `pn` for Thai spans and SymSpell 6.10.0 with
its packaged English frequency dictionary. No allowlist was derived from gold or
test labels, raw suspicious units remained hashed, and Japanese checking was
skipped because the canonical labels contain no Japanese script. At the
provisional per-language ratio threshold 0.20 with at least one issue, 2,281 of
7,091 rows (32.2%) were spelling-suspect. The candidate manifest became 4,475
keep, 2,453 review, 92 drop, and 71 uncalibrated correct proposals; relative to
the baseline, 2,175 keep rows moved to review.

This is not evidence that all flagged rows are mislabeled. At least one dictionary
issue occurred in 5,034 rows (71.0%), consistent with substantial segmentation,
technical-term, name, transliteration, and dictionary-coverage false positives.
Use the signal for stratified review and threshold calibration, not automatic
deletion or correction. Aggregate results and artifact hashes are in
[spelling-audit-v2.json](spelling-audit-v2.json); full row-level outputs remain
outside Git in the temporary artifact directory.

## Gold comparison

The evaluator scored the original label and available model hypotheses only on
the 82 exact-overlap rows, and marks every such metric contaminated. The original
label diagnostic was micro CER 0.0573 (no-space micro CER 0.0535). The retained
primary benchmark had only 8 mapped rows per model/strategy candidate (7.3% of
the 109 gold rows), so those candidate metrics are not a valid held-out cleaning
comparison. The evaluator reports the exact coverage and contamination status in
[evaluation-summary.json](evaluation-summary.json).

Because no candidate has an uncontaminated, grouped gold comparison with usable
coverage, this experiment cannot establish a quality/retention tradeoff for
filtering or correction. The correct decision is to reject automatic replacement,
not to manufacture a positive result.

## Reproduction commands

Run from the repository root. The commands read source resources from the main
worktree and write only the requested candidate manifest/summary.

```powershell
$env:PYTHONPATH = 'training/dataset_cleaning/src'
$meta = 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/hf_datasets/whisper/metadata.jsonl'
$audio = 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/hf_datasets/whisper'
$gold = 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/manual-label/m2-label-ref/audio'
$pred = 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/whisper/runs/m3.1/benchmarks/primary-v1/details_20260711T154402Z_models-4-ef6d5ef557f4_benchmark_isolated_rolling_initial_prompt.jsonl'

python training/dataset_cleaning/scripts/build_candidate_manifest.py `
  $meta $audio "$env:TEMP/daiya-cleaning-v1-manifest.jsonl" `
  --dataset-version 'hf-whisper-9d56736b055df552fa20b325fce0720b87d4bd29f0b166fd4d1197d87874a028' `
  --protected-gold-dir $gold --predictions $pred `
  --expected-script thai --expected-script latin

python training/dataset_cleaning/scripts/evaluate_gold.py `
  --metadata $meta --audio-root $audio --gold-audio $gold `
  --gold-labels 'C:/JokaMain/ProjectShowRoom/daiya-rmt/training/dataset/manual-label/m2-label-ref/ref_labels.txt' `
  --predictions $pred --output docs/experiments/asr-dataset-cleaning-v1/evaluation-summary.json
```

Focused verification:

```powershell
python -m unittest discover -s training/dataset_cleaning/tests -v
python -m compileall -q training/dataset_cleaning/src training/dataset_cleaning/scripts
```

## Recommendation

Use the manifest builder now for immutable identity, duplicate/protected-gold
quarantine, empty-label removal, and review ranking. Before producing training
labels, create decoded-PCM/perceptual identities, split by source conversation,
generate independent multi-view predictions for every candidate clip, calibrate
scores on a non-overlapping development set, and conduct a small blinded review
stratified by source, duration, script mix, disagreement, and anomaly bucket.
Compare filtered data against matched random-retention controls at equal retained
hours. If that review does not show a clear held-out gain, keep the original label
and use the signal only to prioritize human review.
