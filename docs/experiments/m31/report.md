# M3.1: generation-gated prompt-conditioned Whisper

Status: complete. Final Claude `opus`/`xhigh` QA found no blockers and judged the experiment/tooling PR-ready.

## Question and design

M3.1 combines M3 prompt-conditioned LoRA training with generation-metric-gated checkpoint selection. The training method and checkpoint-selection method are evaluated separately: checkpoint 588 is the rolling-prompt generation-gate winner, while checkpoint 500 is the lowest finite scheduled eval-loss checkpoint and serves as the selection ablation.

The frozen split manifest groups by source conversation, so adjacent chunks from a conversation cannot cross M3.1 train, validation, and benchmark partitions. Sources `01,02,05,06,07,08,09,10` are training (4,696 samples), `03,04` are validation (744), and `11` is benchmark quarantine (1,651). The generation selector contains 128 validation samples; the primary benchmark contains 128 source-11 samples in source-time order. Both are deterministic frozen manifests.

Important limitation: the historical M2 and M3 training recipes used a seeded row-random 5% validation split over the complete AudioFolder `train` dataset. This is reconstructed from PR #9 commit `8f256d329cbdd65dc491133401d04cb0ab864392`, where `dataset["train"].train_test_split(test_size=0.05, seed=42)` is the split path; there was no conversation/source grouping. Replaying that exact split on metadata hash `9d567...a028` assigns 1,574 source-11 rows to legacy training and 77 to validation. Their rerun scores on the source-11 benchmark are useful descriptive controls but are not uncontaminated estimates of generalization. M3.1 did not train on source 11. Consequently, this experiment cannot support an unconditional default-model promotion without confirmation on an untouched external conversation set.

## Reproducibility

- Implementation commit used for training: `a6429e58900811afb2b36652e14c3abb67187a74`.
- Corrected probe implementation: `76b5335d3333d7cf01e5e472fca2ea9071e05a52`.
- Base model: `openai/whisper-large-v3`.
- Base-model revision: not captured (`null` in provenance); exact reconstruction depends on the upstream snapshot remaining available.
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

The isolated probe independently selected checkpoint 588 at 27.19% micro CER, 28.48% micro WER-like, and 40.50% short-utterance micro CER. Agreement between isolated and rolling selector contexts makes the selection reproducible on this validation set, but does not guarantee transfer to the quantized deployment backend.

## Primary apples-to-apples benchmark

The primary run completed 1,024/1,024 model-strategy-sample decodes with no failures in 2,119 seconds (35 m 19 s). All four CT2 `int8_float16` models used the same 128 frozen samples, beam size 5, fixed Thai hint, disabled previous-text conditioning, normalization/metric code, causal row-term prompt, and block-bootstrap seed. Rolling additionally carries up to three recent hypotheses and resets when a selected-row source-time gap exceeds 1.5 seconds.

| Model / selection | Strategy | Micro CER (block 95% interval) | Micro WER-like | Short CER | Mean latency | Mean RTF | Max endpoint RAM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 | isolated | 29.02% (21.18–38.00) | 31.27% | 40.61% | 2.07 s | 0.591 | 1,100 MiB |
| M2 | rolling | 26.63% (20.90–32.47) | 27.52% | 40.38% | 2.14 s | 0.582 | 1,102 MiB |
| M3 checkpoint 800 | isolated | **24.32%** (20.30–27.22) | **24.96%** | **40.78%** | **1.74 s** | **0.426** | 1,106 MiB |
| M3 checkpoint 800 | rolling | **24.31%** (20.04–27.23) | **24.66%** | **41.40%** | **1.85 s** | **0.456** | 1,106 MiB |
| M3.1 checkpoint 500 (eval loss) | isolated | 25.03% (20.81–28.24) | 26.06% | 42.26% | 1.93 s | 0.510 | 1,110 MiB |
| M3.1 checkpoint 500 (eval loss) | rolling | 24.85% (20.85–27.77) | 26.04% | 42.38% | 1.86 s | 0.480 | 1,110 MiB |
| M3.1 checkpoint 588 (generation gate) | isolated | 26.51% (21.27–31.32) | 27.82% | 41.35% | 2.06 s | 0.535 | 1,107 MiB |
| M3.1 checkpoint 588 (generation gate) | rolling | 25.99% (20.96–30.03) | 26.96% | 41.69% | 2.03 s | 0.543 | 1,107 MiB |

Checkpoint 588 does not reproduce its selector advantage after CT2 conversion. Relative to checkpoint 500, generation-selected checkpoint 588 is worse by 1.48 pp CER isolated and 1.14 pp rolling; both moving-block intervals include zero. M3.1 checkpoint 588 is also worse than M3 by 2.19 pp CER isolated (95% delta interval 0.17–5.38 pp) and 1.68 pp rolling (−0.19–4.59 pp). M3 remains the best point estimate for both strategies, WER-like, latency, and RTF. Checkpoint 500 is the better M3.1 deployment artifact, but still trails M3.

The benchmark labels are overwhelmingly Thai: each model-strategy cell has 124 Thai, 2 English, and 2 Thai-English samples. Rolling-context bucket results are shown below; the two-sample English and mixed cells are descriptive only and cannot support language-specific claims.

| Model | Thai CER (n=124) | English CER (n=2) | Thai-English CER (n=2) |
| --- | ---: | ---: | ---: |
| M2 | 25.77% | 52.50% | 92.50% |
| M3 | **23.49%** | **45.00%** | **88.75%** |
| M3.1 checkpoint 500 | 23.93% | 65.00% | **88.75%** |
| M3.1 checkpoint 588 | 25.14% | 47.50% | 92.50% |

GPU process memory was unavailable from `nvidia-smi` under this Windows runtime. RAM values are maxima of before/after endpoint snapshots, not transient inference peaks. With only one benchmark conversation, moving-block intervals describe within-conversation chunk variation and do not estimate between-conversation variance.

The selector uses Transformers/PEFT with metadata language policy, while the deployment benchmark uses quantized CT2/faster-whisper with a fixed Thai hint. The reversal between selector and deployment rankings demonstrates that selector-backend gains did not transfer here; checkpoint selection must be validated on the delivery backend before promotion.

## Historical context

PR #9 documents the prior 32-row M2/M3 comparison. These values used a different sample selection and additional strategies, so they are historical context only and are never pooled with the primary rerun.

| Historical strategy | M2 micro CER | M3 micro CER | M3-M2 |
| --- | ---: | ---: | ---: |
| isolated | 14.24% | 14.59% | +0.34 pp |
| rolling initial prompt | 13.38% | 12.93% | -0.45 pp |
| all four old strategies pooled | 16.17% | 14.22% | -1.95 pp |

The authoritative source is the body of GitHub PR #9, “feat(whisper): Add prompt-conditioned M3 training.” No machine-readable per-row output for that run is packaged here. The shared-harness rerun is the only numerical comparison used for the M3.1 conclusion.

## Recommendation

Do not promote M3.1 checkpoint 588 or replace M3. The generation gate selected a checkpoint that regressed on the delivery backend, and the legacy controls are contaminated in M3.1's benchmark conversation. Merge the reproducibility, leakage-control, strict-gating, and benchmark infrastructure as experimental tooling, but retain M3 checkpoint 800 as the current model. A future promotion experiment should use at least two untouched conversations, matched M3/M3.1 training on the same grouped split, multiple seeds, and delivery-backend generation metrics for selection.

## Artifact locations

Machine-readable training, selector, conversion, and 1,024-row benchmark outputs are committed under `docs/experiments/m31/raw/` with SHA-256 checksums. Large adapters, merged models, and CT2 models remain outside Git under the main worktree's `training/whisper/runs/m3.1/` directory.
