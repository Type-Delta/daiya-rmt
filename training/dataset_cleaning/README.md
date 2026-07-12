# Daiya dataset cleaning baseline

This directory contains a dependency-light, audit-friendly baseline for evaluating
ASR dataset metadata. It reads metadata and writes a separate manifest; it never
opens source media for writing or changes source labels.

The package provides:

- frozen manifest records with stable source identity and provenance;
- `keep`, `drop`, `review`, and `correct` dispositions with enum reason codes;
- explicit confidence values that state whether and how they were calibrated;
- Unicode/whitespace normalization and deterministic content/source identities;
- duration, character-rate, and Unicode script-mixture signals without assuming a
  single language;
- conservative multi-signal decisions and extension evidence for later ASR score,
  timestamp, alignment, or model-agreement adapters;
- atomic, UTF-8 JSONL and CSV manifest writers.

Corrections are proposals only. A `correct` record must contain a `ProposedLabel`
with a named method, confidence, and evidence references. The baseline performs no
free-form generation and contains no lexical replacement table.

## CLI

Input is JSONL with `uri`, `label`, and `duration_seconds`; optional `id` is used in
source identity. Missing labels or malformed durations become explicit reasons.

```console
python -m daiya_dataset_cleaning.cli input.jsonl manifest.jsonl
python -m daiya_dataset_cleaning.cli input.jsonl manifest.csv --format csv \
  --expected-script thai --expected-script latin
```

Script names are case-insensitive families currently reported by `script_profile`:
Thai, Han, Hiragana, Katakana, Hangul, Arabic, Hebrew, Cyrillic, Latin, and Other. An
expected-script check requires at least one expected family to be present, allowing
mixed-language examples.

## API sketch

```python
from daiya_dataset_cleaning import SourceIdentity, decide

result = decide(
    SourceIdentity("stable-id", "dataset/audio/example.wav", dataset="train"),
    "ภาษาไทย and English",
    2.4,
    expected_scripts=frozenset({"thai", "latin"}),
)
```

Adapters can append typed `Evidence` and an explicit `Confidence`. A proposed label
should reference evidence names or durable external artifact identifiers so every
change can be reproduced and audited.

Run focused tests without installing dependencies:

```console
python -m unittest discover -s training/dataset_cleaning/tests -v
```
