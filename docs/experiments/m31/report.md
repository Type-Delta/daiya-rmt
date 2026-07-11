# M3.1: generation-gated prompt-conditioned Whisper

Status: experiment in progress. Primary benchmark results will replace the pending table below before review.

## Question and design

M3.1 combines M3 prompt-conditioned LoRA training with generation-metric-gated checkpoint selection. The training method and checkpoint-selection method are evaluated separately: checkpoint 588 is the rolling-prompt generation-gate winner, while checkpoint 500 is the lowest finite scheduled eval-loss checkpoint and serves as the selection ablation.

The frozen split manifest groups by source conversation, so adjacent chunks from a conversation cannot cross M3.1 train, validation, and benchmark partitions. Sources `01,02,05,06,07,08,09,10` are training (4,696 samples), `03,04` are validation (744), and `11` is benchmark quarantine (1,651). The generation selector contains 128 validation samples; the primary benchmark contains 128 source-11 samples in source-time order. Both are deterministic frozen manifests.

Important limitation: the historical M2 and M3 training recipes used a seeded row-random 5% validation split over the complete AudioFolder `train` dataset. This is reconstructed from PR #9 commit `8f256d329cbdd65dc491133401d04cb0ab864392`, where `dataset["train"].train_test_split(test_size=0.05, seed=42)` is the split path; there was no conversation/source grouping. Replaying that exact split on metadata hash `9d567...a028` assigns 1,574 source-11 rows to legacy training and 77 to validation. Their rerun scores on the source-11 benchmark are useful descriptive controls but are not uncontaminated estimates of generalization. M3.1 did not train on source 11. Consequently, this experiment cannot support an unconditional default-model promotion without confirmation on an untouched external conversation set.

## Reproducibility

- Implementation commit used for training: `a6429e58900811afb2b36652e14c3abb67187a74`.
- Corrected probe implementation: `76b5335d3333d7cf01e5e472fca2ea9071e05a52`.
- Base model: `openai/whisper-large-v3`.
- Dataset metadata SHA-256: `9d56736b055df552fa20b325fce0720b87d4bd29f0b166fd4d1197d87874a028`.
- Split manifest SHA-256: `16d9655b0a57839b4562bb72e3114c949592a641b294b7586b490ffe07d80ad3`.
- Generation selector SHA-256: `62837673770ca5c08883185ac5e6188b02a6b95e80be4d1337bf1d9a4099dd28` (128 rows).
- Benchmark manifest SHA-256: `8d13aa10e6863039f3736587947c6d3658f819a59d2bf142296b861dc3af582e` (128 rows).
- Seed: 42.
- Prompt construction: `context_before`, technical terms only, no future context, at most 64 prompt tokens.
- Training: 2 epochs (588 optimizer steps), batch size 2, gradient accumulation 8, learning rate `2e-5`, 50 warmup steps, LoRA rank 16 / alpha 32 / dropout 0.05 on `q_proj,k_proj,v_proj,out_proj,fc1,fc2`, 4-bit base loading, FP16, gradient checkpointing.
- Cadence: evaluate and save every 100 steps; log every 25 steps; final step 588 also saved.
- Runtime: Python 3.12.10, PyTorch 2.11.0+cu128, Transformers 4.57.6, PEFT 0.19.1, CUDA 12.8, NVIDIA GeForce RTX 5060 Laptop GPU.
- Full training runtime: 32,119 seconds (8 h 55 m), exit code 0.

The authoritative generation probe requires every one of the 128 frozen selector samples to produce a fresh, finite metric under the current candidate, dataset, and decoding fingerprints. Missing, partial, stale, NaN, or infinite results are ineligible. The production policy permits eval-loss fallback only when explicitly requested and only with fresh finite losses; this experiment's authoritative probes use `generation_failure_policy=raise`, so no fallback was used.

## Checkpoint selection

The corrected rolling-initial-prompt probe used three previous hypotheses (maximum 512 characters), row technical-term context, metadata language policy, and a 225-token generation cap. All candidates scored all 128 samples successfully.

| Candidate | Eval loss | Selector micro CER | Selector micro WER-like | Short micro CER |
| --- | ---: | ---: | ---: | ---: |
| checkpoint 100 | 0.462268 | 31.77% | 33.31% | 42.02% |
| checkpoint 200 | 0.425305 | 29.05% | 30.19% | 41.00% |
| checkpoint 300 | 0.415354 | 28.28% | 29.27% | 39.42% |
| checkpoint 400 | 0.410348 | 28.44% | 29.91% | 40.50% |
| checkpoint 500 (eval-loss selection) | **0.406548** | 28.17% | 28.96% | 39.61% |
| checkpoint 588 (generation-gate selection) | n/a | **27.18%** | **27.95%** | **39.16%** |

The gate selected checkpoint 588, improving selector micro CER by 1.00 percentage point relative to eval-loss-selected checkpoint 500. This is a checkpoint-selection result within one training run, not evidence for the prompt-training method itself. It is not a clean policy-only ablation: checkpoint 588 is the final-step save, 88 optimizer steps after the last scheduled eval-loss measurement at checkpoint 500, and no same-backend eval loss was recorded for 588. The deployment benchmark therefore reports the comparison as selection-plus-additional-training, not pure checkpoint-selection causality. The final root adapter contains checkpoint-300 weights because Trainer's in-training unprompted generation metric selected checkpoint 300; it is not the authoritative M3.1 artifact.

An earlier diagnostic rolling probe omitted an explicit attention mask while Whisper's pad and EOS token IDs are equal. Its output is superseded and excluded. The corrected probe explicitly supplies the feature-extractor attention mask.

## Primary apples-to-apples benchmark

Pending completion of CT2 conversion and the single-harness rerun. The final table will contain M2, M3, generation-selected M3.1 checkpoint 588, and eval-loss-selected M3.1 checkpoint 500 under both isolated and rolling-initial-prompt decoding. Both strategies use the same causal row-term prompt derived only from `context_before`; rolling additionally carries up to three recent hypotheses and resets when a selected-row source-time gap exceeds 1.5 seconds. Results include micro CER/WER-like, contiguous-block bootstrap intervals, short-utterance and language/mixed-language breakdowns, latency/RTF, and endpoint memory snapshots. With only one benchmark conversation, intervals are descriptive chunk-level uncertainty and do not estimate between-conversation variance.

The selector uses Transformers/PEFT with metadata language policy, while the deployment benchmark uses quantized CT2/faster-whisper with a fixed Thai hint. Selector-backend gains and deployment-backend gains are reported separately; transfer between them is not assumed.

## Historical context

PR #9 documents the prior 32-row M2/M3 comparison. These values used a different sample selection and additional strategies, so they are historical context only and are never pooled with the primary rerun.

| Historical strategy | M2 micro CER | M3 micro CER | M3-M2 |
| --- | ---: | ---: | ---: |
| isolated | 14.24% | 14.59% | +0.34 pp |
| rolling initial prompt | 13.38% | 12.93% | -0.45 pp |
| all four old strategies pooled | 16.17% | 14.22% | -1.95 pp |

The authoritative source is the body of GitHub PR #9, “feat(whisper): Add prompt-conditioned M3 training.” No machine-readable per-row output for that run is packaged here. The shared-harness rerun is the only numerical comparison used for the M3.1 conclusion.

## Recommendation

Pending the primary benchmark and QA. Regardless of its point estimates, source-11 contamination in the legacy controls prevents an unconditional default-model promotion from this experiment alone.

## Artifact locations

Machine-readable selector and benchmark outputs will be committed under `docs/experiments/m31/raw/`. Large adapters, merged models, and CT2 models remain outside Git under the main worktree's `training/whisper/runs/m3.1/` directory.
