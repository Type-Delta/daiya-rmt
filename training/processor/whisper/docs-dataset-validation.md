# Daiya dataset validation baseline

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
- script-routed spelling validation for Thai, Japanese, and English, with optional
  PyThaiNLP, SudachiPy, and SymSpell adapters;
- conservative multi-signal decisions and extension evidence for later ASR score,
  timestamp, alignment, or model-agreement adapters;
- atomic, UTF-8 JSONL and CSV manifest writers.

Corrections are proposals only. A `correct` record must contain a `ProposedLabel`
with a named method, confidence, and evidence references. The baseline performs no
free-form generation and contains no lexical replacement table.

Spelling findings are review evidence only. They cannot produce a correction
proposal or overwrite a label. Results record the checker name, checked-unit count,
suspicious-unit ratio, routed scripts, and hashes of suspicious units by default;
raw issue text is opt-in.

## CLI

Input is JSONL with `uri`, `label`, and `duration_seconds`; optional `id` is used in
source identity. Missing labels or malformed durations become explicit reasons.

```console
python -m daiya_dataset_validation.cli input.jsonl manifest.jsonl
python -m daiya_dataset_validation.cli input.jsonl manifest.csv --format csv \
  --expected-script thai --expected-script latin
```

Script names are case-insensitive families currently reported by `script_profile`:
Thai, Han, Hiragana, Katakana, Hangul, Arabic, Hebrew, Cyrillic, Latin, and Other. An
expected-script check requires at least one expected family to be present, allowing
mixed-language examples.

## Spelling validation

Install only the language adapters needed for an experiment:

```console
uv run --directory training/processor/whisper --extra spelling python -c "import daiya_dataset_validation"
```

Generate spelling evidence separately so different checkers and thresholds can be
compared without rebuilding or mutating the source dataset:

```console
uv run --directory training/processor/whisper --extra spelling python training/processor/whisper/scripts/run_spelling_validation.py \
  metadata.jsonl spelling-pn.jsonl \
  --thai-engine pn \
  --japanese-dictionary core \
  --english-dictionary frequency_dictionary_en_82_765.txt \
  --allowlist technical-terms.txt \
  --workers 4 \
  --max-in-flight 8

uv run --directory training/processor/whisper --extra spelling python training/processor/whisper/scripts/build_candidate_manifest.py \
  metadata.jsonl /dataset/root candidate-manifest-v2.jsonl \
  --dataset-version dataset-v2 \
  --spelling-results spelling-pn.jsonl \
  --spelling-review-threshold 0.2 \
  --hash-workers 4
```

Spelling validation is CPU-bound, so `--workers` uses Windows-safe spawned
processes. Each process initializes its own checker, tokenizer, dictionary, and
cache. `--max-in-flight` bounds queued records; results are still emitted in
canonical metadata order. The default is one worker, and raw issue text remains
opt-in with `--include-issue-text`.

Audio hashing is disk-I/O bounded, so `build_candidate_manifest.py` uses a
bounded `ThreadPoolExecutor`. `--hash-workers` controls its conservative thread
count (default `4`); source hashes remain aligned to metadata order and
protected-gold hashes retain precedence over candidate decisions. Metadata audio
paths are resolved and rejected if they escape `audio_root`, including through a
symlink.

Both scripts resolve the output path before doing any work and reject aliases of
their metadata, checker resources, predictions, spelling results, audio sources,
or protected-gold inputs/directories. Existing inputs and outputs therefore remain
unchanged when an unsafe path is supplied.

Run PyThaiNLP engines (`pn`, `symspellpy`, `phunspell`) as separate outputs when
comparing them. Do not combine their issue counts into a single threshold-tuning
run; the CLI rejects multiple Thai engines in one invocation. The review threshold
is evaluated independently per routed language so a minority-script error is not
diluted by a longer span in another language. Sudachi OOV status is a
dictionary-coverage signal, not proof of an error;
technical terms, names, acronyms, transliterations, and code switches belong in a
versioned allowlist and in false-positive reporting.

Freeze allowlists and custom dictionaries from external/train/development sources
before held-out testing; never derive them from protected test labels. Output hashes
suspicious units by default. `--include-issue-text` contains raw label fragments and
must remain a local diagnostic artifact rather than a committed/shared result.

## API sketch

```python
from daiya_dataset_validation import SourceIdentity, decide

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
python -m unittest discover -s training/processor/whisper/tests/dataset_validation -v
```
