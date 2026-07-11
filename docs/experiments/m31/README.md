# M3.1: generation-gated prompt-conditioned Whisper

Status: complete. Final Claude `opus`/`xhigh` QA found no blockers and judged the experiment/tooling PR-ready.

## Objective

M3.1 combines M3's causal `context_before` terms-only Whisper prompt conditioning with generated-text checkpoint selection. The post-hoc generation gate is authoritative. Trainer eval generation is recorded as unprompted and is retained only as an in-run diagnostic/eval-loss ablation.

## Dataset identity and leakage controls

- Metadata SHA-256: `9d56736b055df552fa20b325fce0720b87d4bd29f0b166fd4d1197d87874a028`
- Split manifest: `m31-split-v1.jsonl`
- Split manifest SHA-256: `16d9655b0a57839b4562bb72e3114c949592a641b294b7586b490ffe07d80ad3`
- Generation selector SHA-256: `62837673770ca5c08883185ac5e6188b02a6b95e80be4d1337bf1d9a4099dd28` (128 rows)
- Benchmark manifest SHA-256: `8d13aa10e6863039f3736587947c6d3658f819a59d2bf142296b861dc3af582e` (128 rows)
- Train conversations: 01, 02, 05, 06, 07, 08, 09, 10 (4,696 rows)
- Validation/selector conversations: 03, 04 (744 rows)
- Benchmark quarantine conversation: 11 (1,651 rows)
- Seed: 42
- Future context is disabled. `context_after`, `future_context`, and `right_context` are not prompt inputs.
- Benchmark/test rows are not preprocessed or cached by the trainer.

The deterministic manifest generator is `training/whisper/scripts/build_m31_manifests.py`. It creates:

- `m31-generation-gate-v1.jsonl`: 128 validation rows, 16 per source (03/04) and each short/long × Latin/no-Latin stratum, ranked by SHA-256 with seed `m31-gate-v1`.
- `m31-benchmark-v1.jsonl`: 128 source-11 rows, 32 per stratum, ranked with seed `m31-benchmark-v1` and emitted in source-time order.

## Full training configuration

- Base model: `openai/whisper-large-v3`
- Epochs: 2
- Train/eval batch: 2 / 2
- Gradient accumulation: 8 (effective train batch 16)
- Learning rate: 2e-5; warmup: 50 steps
- Eval/save/log cadence: 100 / 100 / 25 steps
- LoRA: r=16, alpha=32, dropout=0.05
- LoRA modules: `q_proj,k_proj,v_proj,out_proj,fc1,fc2`
- 4-bit base load, fp16, gradient checkpointing
- Prompt: terms-only from `context_before`, maximum 64 body tokens
- Generated trainer metrics enabled; `load_best_model_at_end` enabled
- Full provenance is written to `run_provenance.json`, including resolved config, package/runtime versions, split sample hashes, and git state.

## Checkpoint selection

Every saved checkpoint and the final adapter are scored on all 128 frozen selector rows. A candidate is ineligible if any row is missing or generation fails, or if the generated metric is missing/NaN/infinite. Primary metric is micro CER; tie-break is micro CER then checkpoint name/step. Primary selection uses no eval-loss fallback.

Two generation contexts are recorded:

1. `isolated`
2. `rolling-initial-prompt`, with per-row causal terms plus up to three previous hypotheses / 512 characters, reset at conversation boundaries and source-time gaps above 1.5 seconds

The runtime-aligned rolling selection is the M3.1 delivery checkpoint. Isolated results are reported as an ablation. Eval-loss-best and final checkpoints from the same run are benchmarked where conversion time permits, separating checkpoint-selection gain from the combined training-method result.

## Benchmark protocol

M2, M3, and M3.1 use exactly `m31-benchmark-v1.jsonl` with the same faster-whisper/CTranslate2 version, int8_float16 compute, beam size, language hint, normalization, and two non-pooled strategies:

- isolated
- rolling initial prompt

Both strategies receive the same causal per-row technical-term prompt derived only from `context_before`; rolling additionally receives recent hypotheses. Raw JSONL includes per-sample edit counts, prompt hashes, latency, target/processed-audio RTF, endpoint RAM/process GPU-memory snapshots, model/dataset/manifest fingerprints, and failures. Summary JSON includes micro CER/WER-like, language/mixed/short/source breakdowns, and contiguous-block bootstrap intervals on model deltas. Because the benchmark has one source conversation, intervals describe chunk-level variation and cannot estimate between-conversation uncertainty.

## Known limitation before interpreting results

Legacy M2 and M3 used a seeded row-random 5% validation split over the complete AudioFolder training table, reconstructed from PR #9 commit `8f256d329cbdd65dc491133401d04cb0ab864392` (`dataset["train"].train_test_split(test_size=0.05, seed=42)`). Replaying that exact split on the frozen metadata assigns 1,574 source-11 rows to legacy training and 77 to validation. Source 11 is held out from M3.1 but contaminated for the frozen M2/M3 baselines. Therefore:

- M2/M3 reruns are descriptive runtime baselines, not unbiased training-method controls.
- A positive M3.1 result despite the legacy exposure advantage is encouraging but still narrow.
- A negative M3.1 result is inconclusive.
- A method-level prompt-training claim would require matched control training on this split and preferably multiple seeds.
- No Japanese-English claim is supported by this corpus.

## Outcome

Both selector contexts chose checkpoint 588, but the CT2 deployment benchmark reversed that ranking: eval-loss checkpoint 500 beat checkpoint 588 by 1.48 pp CER isolated and 1.14 pp rolling. M3 checkpoint 800 remained best overall (24.32% isolated, 24.31% rolling), while M3.1 checkpoint 588 scored 26.51% and 25.99%. Do not promote M3.1; retain M3. The implementation and reproducibility infrastructure are suitable to merge as experimental tooling, subject to final QA.
