# Short-Utterance Decoding Policy Experiment

Status: a 32-segment Thai-English probe is complete. No experimental policy is
proposed as the default.

## Question and Decision Rule

This experiment asks whether a duration-gated faster-whisper decoding policy
improves short-utterance accuracy without materially regressing the full
benchmark or latency. It does not use transcript content, glossary matches, or
sample-specific rules.

The current `baseline` policy remains the runtime default. A candidate may be
recommended only if the completed run shows:

- a clear overall micro-CER improvement;
- an improvement in short-utterance micro CER;
- no material regression in WER-like or terminology/context probes;
- acceptable offline inference runtime; and
- no concerning regression isolated by the duration buckets.

If the evidence is mixed, the recommendation is to retain `baseline` and keep
the policy as opt-in benchmark tooling.

## Runtime Policies

All policies preserve `word_timestamps=True`, `vad_filter=False`, and the
existing sparse temperature fallback ladder `[0.0, 0.8, 1.0]`.
The benchmark harness now applies that production ladder explicitly; older
artifacts produced before this experiment may instead reflect
faster-whisper's longer default ladder and should not be compared directly.

| Policy | Eligible duration | Additional faster-whisper options | Runtime default |
| --- | --- | --- | --- |
| `baseline` | All utterances | None | Yes |
| `short_beam` | Original duration `<= threshold` | `beam_size=8`, `patience=1.2` | No |
| `short_greedy` | Original duration `<= threshold` | `beam_size=1`, `best_of=1` | No |

The threshold is inclusive and defaults to `3.0` seconds. Longer utterances
always use baseline options. CLI configuration uses
`--asr-decoding-policy` and `--asr-short-utterance-seconds`; environment
configuration uses `DAIYA_ASR_DECODING_POLICY` and
`DAIYA_ASR_SHORT_UTTERANCE_SECONDS`.

## Reproduction Resources

Large resources are intentionally not copied into this worktree or committed.
For the original run, they are expected in the main worktree:

| Resource | Path used for run | Identity |
| --- | --- | --- |
| Dataset | `C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper` | 7,091 metadata rows; `metadata.jsonl` SHA-256 `9d56736b055df552fa20b325fce0720b87d4bd29f0b166fd4d1197d87874a028` |
| CTranslate2 checkpoint | `C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\largev3-m2-iter1-ct2-int8_float16` | `model.bin` SHA-256 `afcca1964a812ada5dd15bf548325c684b2685665205b7ad11d4b5cc81afe8a6`; checkpoint directory 1,556,379,809 bytes |
| Prior benchmark artifacts | `C:\JokaMain\ProjectShowRoom\daiya-rmt\lab\artifacts\asr_eval` | Historical corrected isolated baseline: `summary_20260707T065832Z_...json` |
| New artifacts | Ignored `lab/artifacts/asr_eval` in this worktree | Run IDs `20260709T030100Z`, `030308Z`, `030422Z`, `030543Z`, `030748Z`, and `030854Z` |

Before the run, capture identities without changing the resources:

```powershell
Get-FileHash -Algorithm SHA256 `
  C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper\metadata.jsonl
Get-FileHash -Algorithm SHA256 `
  C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\largev3-m2-iter1-ct2-int8_float16\model.bin
Get-Item `
  C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\largev3-m2-iter1-ct2-int8_float16\model.bin |
  Select-Object FullName,Length,LastWriteTimeUtc
```

## Environment

Record these values with the final results:

| Field | Value |
| --- | --- |
| Git commit | Base `de949d1` plus this branch's uncommitted experiment changes |
| Date/time and timezone | 2026-07-09 10:01-10:09, Asia/Bangkok (UTC+7) |
| OS | Windows |
| Python | 3.12.10 |
| `faster-whisper` | 1.2.1 |
| `ctranslate2` | 4.8.0 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU |
| Device / compute type | `cuda` / `int8_float16` |
| Language hint | `th` |
| Initial prompt | None |
| Condition on previous text | False |
| Sample count | 32, the existing representative prefix benchmark |
| Short-utterance thresholds | 2.0 and 3.0 seconds |
| Duration bucket boundaries | `<2`, `2-<3`, `3-<5`, `5-<10`, `>=10` seconds |

Suggested version capture:

```powershell
git rev-parse HEAD
python --version
uv run --package daiya python -c "import faster_whisper, ctranslate2; print('faster-whisper', faster_whisper.__version__); print('ctranslate2', ctranslate2.__version__)"
nvidia-smi
```

## Exact Command Templates

First reproduce baseline by itself. Replace `<output-dir>` only with an
untracked or ignored location:

```powershell
uv run --package daiya python lab/asr_eval.py `
  --model C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\largev3-m2-iter1-ct2-int8_float16 `
  --dataset-dir C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper `
  --output-dir <output-dir> `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --limit 32 `
  --strategy isolated `
  --decoding-policy baseline `
  --short-utterance-seconds 3.0
```

Then run each serious candidate against exactly the same rows and environment,
substituting `short_beam` and `short_greedy` for `<policy>` and both `2.0`
and `3.0` for `<threshold>`:

```powershell
uv run --package daiya python lab/asr_eval.py `
  --model C:\JokaMain\ProjectShowRoom\daiya-rmt\training\whisper\runs\largev3-m2-iter1-ct2-int8_float16 `
  --dataset-dir C:\JokaMain\ProjectShowRoom\daiya-rmt\training\dataset\hf_datasets\whisper `
  --output-dir <output-dir> `
  --device auto `
  --compute-type int8_float16 `
  --language th `
  --no-condition-on-previous-text `
  --limit 32 `
  --strategy isolated `
  --decoding-policy <policy> `
  --short-utterance-seconds <threshold>
```

These are the literal parameters used, with `<output-dir>` equal to this
worktree's ignored `lab/artifacts/asr_eval`. The 32 rows are consecutive
segments from one Thai-English source recording, not a stratified sample.

## Results

### Aggregate and Probe Metrics

| Policy | Samples | Micro CER | WER-like | Short micro CER | Term-tagged subset | Runtime | Offline RTF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline`, 3.0 s | 32 | 15.10% | 15.59% | 15.97% | 15.10% term-subset CER | 73.19 s | 0.343 |
| `short_beam`, 3.0 s | 32 | 14.83% | 15.50% | 15.97% | 14.83% term-subset CER | 80.23 s | 0.375 |
| `short_greedy`, 3.0 s | 32 | 15.17% | 16.11% | 17.09% | 15.17% term-subset CER | 63.70 s | 0.294 |
| `short_beam`, 2.0 s | 32 | 16.27% | 17.11% | 38.81% (`<2` subset) | 16.27% term-subset CER | 65.30 s | 0.305 |
| `short_greedy`, 2.0 s | 32 | 14.65% | 15.31% | 38.81% (`<2` subset) | 14.65% term-subset CER | 59.25 s | 0.274 |

### Duration Buckets

Use the exact same bucket boundaries and sample membership for every policy.

| Duration bucket | Samples | `baseline` micro CER | `short_beam` micro CER | `short_greedy` micro CER | Regression notes |
| --- | ---: | ---: | ---: | ---: | --- |
| `<2` | 4 | 38.81% | 38.81% | 38.81% | Neither candidate changed this bucket. |
| `2-<3` | 9 | 10.69% | 10.69% | 12.07% | Greedy regressed; beam was unchanged. |
| `3-<5` | 3 | 19.63% | 19.63% | 19.63% | Policies are inactive. |
| `5-<10` | 8 | 8.59% | 8.59% | 8.59% | Policies are inactive. |
| `>=10` | 8 | 17.68% | 17.16% | 17.55% | Inactive-policy variation exposes sampling/fallback run variance. |

### Inspected Regressions

Record the largest candidate-vs-baseline regressions, including sample
identifier, duration, reference, each prediction, metric delta, and whether the
change appears related to decoding or measurement:

| Sample | Duration | Policy | CER delta | Reference / predictions | Disposition |
| --- | ---: | --- | ---: | --- | --- |
| 2-3 s aggregate | 2-<3 s | `short_greedy` | +1.38 pp micro CER | Per-row details in run `20260709T030543Z` | Candidate regression; reject as default. |
| Long inactive buckets | >=10 s | all candidates | varied by up to several pp between runs | Decoder options are identical to baseline in these rows | Treat aggregate changes outside the gated bucket as fallback sampling variance, not policy effect. |

Artifacts:

- Baseline replicate: run `20260709T030100Z` (micro CER 15.03%, WER-like
  15.36%, 71.03 s). Inspection shows this run used the same sparse ladder, so
  it demonstrates baseline-to-baseline fallback sampling variance.
- Production-matched baseline: run `20260709T030308Z`.
- 3.0 s candidates: runs `20260709T030422Z` and `20260709T030543Z`.
- 2.0 s candidates: runs `20260709T030748Z` and `20260709T030854Z`.

## Recommendation

Preserve `baseline` as the default. `short_beam` produced no measurable
short-bucket accuracy change and increased RTF in the 3.0 s run.
`short_greedy` reduced runtime but regressed the 2-3 s bucket, short CER, and
WER-like metric. Aggregate differences in inactive long-duration buckets show
that single-run fallback sampling variance is large enough to make the apparent
overall wins unreliable.

The opt-in policies and richer benchmark output remain useful for larger,
repeated experiments. A future default-selection run should use a stratified,
multi-source Thai-English and Japanese-English manifest rather than 32
consecutive segments from `Th-En_sample_01_f9ff8c3a6d15`, and repeat each
policy (or otherwise control sampling) before attributing changes outside the
gated duration bucket. The current term-tagged subset contains all 32 rows, so
it is identical to the overall metric and supplies no independent terminology
evidence. No context-specific accuracy probe or streaming latency measurement
was available in this harness.

## Validation

Record final commands and outcomes:

| Check | Outcome |
| --- | --- |
| Focused ASR/config tests | Included in backend suite; passed |
| Benchmark-tool tests | 3 deterministic tests passed |
| Backend unit suite | 25 tests passed |
| Backend compile check | `compileall` passed |
| Formatter/linter/type checker | Not configured in this repository |
| Lockfile check | Unavailable: this checkout has no committed `uv.lock` |
| Frontend build | Not run; no frontend files changed |

## Independent Claude QA

Reviewer command/model: Claude Code with `--model opus --effort xhigh`.

Review scope: correctness, benchmark methodology, hidden regressions, test
gaps, and whether the recommendation follows from the measured data.

- Findings: runtime policy implementation was sound. The reviewer found that
  the report misidentified a baseline replicate, overstated the 32 consecutive
  Thai-English segments as representative, omitted `--limit 32` from the
  baseline reproduction command, treated the all-row term subset as an
  independent probe, and called offline RTF latency. It also requested that
  the harness record `--limit`.
- Actionable changes made: corrected all of those claims and commands, recorded
  `limit` in new summary JSON, documented the production-ladder change relative
  to older harness artifacts, moved the sparse-ladder comment to its source,
  and strengthened the future stratified/repeated-run recommendation.
- Checks rerun: backend tests, benchmark-tool tests, compilation, and diff
  whitespace check.
- Final disposition: code is suitable as opt-in experiment tooling; keep
  `baseline` as default and do not treat this small probe as positive evidence
  for a policy change.
