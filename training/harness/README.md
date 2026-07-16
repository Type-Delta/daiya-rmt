# Daiya training harness

This package contains the reusable, dependency-free contract and validation
layer for the future M3/M3.1/M3A/M3.1A 2x2 experiment. It does not train a
model, download data, or ship model artifacts.

The experiment contract is `TrainingRecipe`. Both an old harness adapter and a
new harness adapter can consume the same JSON contract, which pins:

- conversation-level train/validation/test manifest paths and SHA-256 hashes;
- dataset version, audio conversion settings, and exact base-model revision;
- prompt template and declared fields;
- training/deployment backend identity and evaluation metrics.

`selection.select_checkpoint` treats Transformers/PEFT scores as candidate
ranking only. A CT2 or quantized deployment requires matching CT2 validation
records for every top-K candidate before any checkpoint can be selected.

Run the focused fixture suite from this repository with:

```powershell
$env:PYTHONPATH = (Resolve-Path 'training/harness/src').Path
uv run --no-project --with pytest python -m pytest training/harness/tests -q
```

The repository-wide `uv run --project training/harness` form also works when
all workspace members are present. In a partial checkout, use the isolated
command above because unrelated lab members may reference optional local
packages that are not part of this harness.

See [`docs/experiments/m31-training-harness-migration.md`](../../docs/experiments/m31-training-harness-migration.md)
for the migration boundary from PRs #14, #2, and #9.

No full model training is part of this package.
