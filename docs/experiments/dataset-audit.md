# Dataset audit: ASR cleaning experiment

Audit date: 2026-07-12. The audit was read-only against the current worktree and `C:\JokaMain\ProjectShowRoom\daiya-rmt`; only this report and the compact aggregate fixture were created. `training/dataset/manual-label/m2-label-ref` was treated as protected human gold. Its text was parsed only for structural/count checks and was never used as pseudo-label input. No audio, labels, predictions, model dumps, caches, or secrets are reproduced here.

## Executive findings

- The canonical generated dataset is `training/dataset/hf_datasets/whisper`: 7,091 mono, 16 kHz, 16-bit WAV clips and 7,091 valid JSONL records, with a one-to-one metadata/file join. Total speech is 31,284.544 s (8.690 h); clips range 1.000–24.173 s (median 3.143 s).
- Provenance resolves to 11 raw Thai-English recordings (`Th-En_sample_01` … `11`), totaling 7,091 segments. The metadata contains absolute source paths, but no source checksums, pipeline/version identifier, generation model, prompt, decoding parameters, confidence, or per-record split. Reproduction is therefore incomplete.
- Protected gold contains 109 WAV clips (451.159 s) and 109 label entries. Exact SHA-256 byte identity finds 82/109 gold clips (75.2%) in canonical training data, all among source 11 segments. This is direct train/evaluation contamination if the gold set is used for evaluation. The remaining 27 gold clips have no exact byte match; this does not exclude acoustic overlap after re-encoding, padding, or alternate segmentation.
- Canonical audio has 7,002 unique SHA-256 values: 89 duplicate groups, each a pair (178 rows). Forty duplicate pairs have different text; 42 differ in text or language. Deduplicate/split by content hash before training, and resolve conflicting labels rather than selecting one silently.
- Labels show clear review candidates: 10 empty texts; 9 texts longer than 500 characters; 10 records above 100 characters/s; maximum text length 5,986 and maximum density 1,813 characters/s. These are implausible for their audio and likely context/label leakage or concatenation. There are 126 rows duplicated by exact text (127 after NFKC + casefold + trim), but repeated short utterances may be legitimate.
- Language values are not normalized: 11 forms (`Thai` 6,683; `th` 289; `Thai, English` 36; `en` 34; `th-en` 17; `mixed` 16; `none` 7; `English` 5; `unknown` 2; `ja` 1; `Thai/English` 1). Script inspection finds Thai in 7,025 labels, Latin in 2,152, Thai+Latin in 2,103, and no Japanese script. The lone `ja` value is thus a review candidate, not evidence of Japanese coverage.

## Structure, segmentation, and normalization

Raw input has 11 compressed recordings (2 MP3, 8 M4A, 1 AAC). Canonical metadata has these fields on every row: `file_name`, `text`, `language`, `context_before`, `context_after`, `notes`, `source_file`, `source_start`, `source_end`, and `speech_duration`. All filenames are unique and all resolve under the dataset root. Audio is uniformly mono/16 kHz/16-bit.

Segment counts by source are: 01 386, 02 91, 03 613, 04 131, 05 1,918, 06 149, 07 1,293, 08 291, 09 14, 10 554, 11 1,651. No adjacent timestamp intervals overlap. Large source gaps exist (maximum 333.2 s), which is consistent with speech selection but means source coverage is not continuous. `speech_duration` differs from `source_end - source_start` by more than 10 ms in 2,294 records (maximum 1.6 s); likely padding versus detected speech, but the semantics are undocumented and should not be treated as interchangeable. Adjacent `context_after`/next `context_before` agree for all 7,080 within-source transitions.

Duration percentiles (s) are p01 1.000, p05 1.100, p50 3.143, p95 12.358, p99 16.600. There are 158 exactly 1.000 s clips and four over 20 s. Text has no leading/trailing whitespace, repeated whitespace, invalid replacement characters, or unexpected control characters. NFKC changes the duplicate cardinality by one row, demonstrating at least one Unicode-normalization collision.

## Provenance and protected-gold boundary

The current dataset is documented elsewhere as a relabeled M2 replacement for the deleted pre-M2 dataset. Available logs include `m2-relabel.log`, which reportedly completed despite ALAC decode warnings. Because the old dataset and separate relabel output no longer exist, label lineage cannot be reconstructed from retained artifacts alone. The generated metadata does not identify whether each text came from a model, manual correction, or an LLM pass.

The protected gold label file has four header lines followed by 109 repeated `ID|audio` / label / blank blocks (331 lines total, 109 blank). This audit never copies label content. Exact overlap was established from WAV bytes only. A collision-resistant digest of the sorted 82 overlapping SHA-256 values is recorded in the fixture, allowing future reruns to detect identity-set changes without publishing clip-level mappings.

Policy for the cleaning experiment: exclude all 109 gold identities from pseudo-label generation and training candidate selection. At minimum, quarantine the 82 exact matches by SHA-256. To cover the 27 unmatched clips and possible re-encodes, add decoded-PCM hashes and robust acoustic fingerprints before claiming isolation; neither was available as retained metadata.

## Model/output metadata available

`training/whisper/runs` contains 19 top-level artifact directories, 45 `trainer_state.json` files, adapters/checkpoints, merged models, CT2 conversions, logs, and a roughly 10.9 GB feature cache. Older runs mostly expose eval loss; smoke/probe states include WER, and M3.1 states include CER, micro CER, no-space CER, and WER-like metrics. M3.1 also retains run provenance plus compact benchmark/probe summaries and detailed JSONL outputs. These outputs are useful for model comparison but are not dataset provenance and may contain full predictions, so none are copied to fixtures.

The run name `largev3-m2-iter1` and logs demonstrate prior M2 training/evaluation activity, but names alone cannot prove exactly which records entered a run. Saved trainer states do not provide a content-hash manifest or immutable train/eval membership. Eval loss and generation metrics are not directly comparable across historical runs with different metric sets and undocumented dataset snapshots.

## Deterministic procedure

Commands were run from the worktree in PowerShell 7 with Python 3. The core identity operation was:

```powershell
@'
import hashlib, pathlib
root = pathlib.Path(r'C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset')
train = root / 'hf_datasets/whisper/train'
gold = root / 'manual-label/m2-label-ref/audio'  # identity only; do not read labels
def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()
train_map = {sha256(p): p.name for p in sorted(train.glob('*.wav'))}
gold_map = {sha256(p): p.name for p in sorted(gold.glob('*.wav'))}
overlap = sorted(train_map.keys() & gold_map.keys())
print(len(train_map), len(gold_map), len(overlap))
print(hashlib.sha256(('\n'.join(overlap) + '\n').encode()).hexdigest())
'@ | python -
```

Metadata/file joins used normalized POSIX-relative `file_name` values rooted at `training/dataset/hf_datasets/whisper`; WAV properties came from Python's `wave` module. Label normalization used `unicodedata.normalize('NFKC', text).casefold().strip()`. Repository discovery excluded `.git`, `.venv`, caches, and nested worktrees to avoid double-counting.

Limitations: byte hashes detect exact containers only, not perceptually identical/re-encoded audio; no decoded-PCM or acoustic fingerprints were retained; no raw-source hashes are present; no speaker/session IDs or train/eval split manifest exist; label-error flags are heuristic because audio was not listened to; absolute `source_file` values are machine-specific; the deleted pre-M2 snapshot prevents historical diffing; and Japanese-English coverage is absent from the canonical labels inspected.

## Required gates before the experiment

1. Create an immutable manifest with dataset version, raw-source SHA-256, decoded-PCM SHA-256, clip hash, source interval, split, provenance class, generator/checkpoint/prompt/decoding configuration, and manual-review status.
2. Quarantine protected gold before any pseudo-label call, then assert zero exact PCM/hash and robust-fingerprint overlap across train, validation, test, and gold.
3. Resolve the 89 duplicate pairs, especially 42 metadata conflicts; review empty and extreme-density labels; normalize language to a controlled vocabulary while preserving the original value.
4. Split by source recording/session (and speaker when available), never by clip, to prevent neighboring/context leakage. Keep protected gold as evaluation-only.

