# M3.1 training-harness migration

This replacement extracts the reusable experiment infrastructure needed for
the future M3/M3.1/M3A/M3.1A 2x2 study. It is intentionally not a model
promotion and does not contain M3.1 weights, datasets, raw benchmark output,
or a claim that checkpoint 588 passed a deployment gate.

## Retained and rewritten

| Need | Replacement | Boundary |
| --- | --- | --- |
| Conversation-level splits | `SplitManifest` with immutable assignments, overlap checks, canonical hash, and atomic JSON I/O | Rewritten as a small dependency-free module; PR #14's split artifacts remain historical evidence only |
| Exact run identity | `ProvenanceRecord` and `TrainingRecipe` | Rewritten to pin dataset version, conversion settings, base-model revision, manifest hashes, prompt fields, and evaluation backend |
| Prompt-conditioned training | `PromptTemplate` plus recipe/backend contract | Extracted as an adapter-neutral contract; model-specific training code remains in the existing training package |
| Checkpoint selection | `TopKValidationProtocol` and `select_checkpoint` | Rewritten so Transformers/PEFT scores rank candidates only; CT2/quantized deployment requires matching CT2 evidence for every top-K candidate |
| Benchmarking | `run_benchmark` and compact JSON summaries | Rewritten as fixture-friendly deterministic WER/CER/exact-match evaluation; raw text is hashed, not emitted |

The old and new harnesses must load the same recipe JSON and verify the same
manifest hashes before a run. Full training and CT2 conversion are deliberately
out of scope for this replacement and will happen only after the contract is
reviewed.

## Relationship to open PRs

- **PR #14** (`feature/m3.1-generation-gated-prompt-training`) is preserved as
  historical source material. This replacement supersedes its reusable
  infrastructure portion only after review, and drops its monolithic lab
  script, source-11 comparison, model-ranking claim, raw outputs, and model
  artifacts from the replacement PR. PR #14 is not merged, force-pushed, or
  modified.
- **PR #2** (`feat/generation-gated-checkpoint-selection`) contains the earlier
  generation-probe implementation. This replacement supersedes the generic
  selection-contract portion for future experiments, but does not copy or
  invalidate its model-specific probe. A checkpoint is not promoted here
  without deployment-backend evidence.
- **PR #9** (`feat/m3-prompt-conditioned-whisper`) contains the earlier
  prompt-conditioned M3 training implementation. This replacement reuses the
  idea through a portable recipe/prompt contract and avoids duplicating its
  model-specific training edits. The two efforts can coexist until this
  contract replaces the duplicated configuration surface.

The replacement PR body repeats this boundary and explicitly states that it
contains no valid model-promotion result.
