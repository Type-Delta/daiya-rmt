# Spelling validation for mixed-lingual ASR labels

## Intended use

Spelling validation is an anomaly detector, not an automatic transcript editor.
Thai word segmentation, Japanese dictionary coverage, English technical terms,
names, acronyms, and transliteration all create legitimate out-of-vocabulary text.
The cleaning manifest therefore maps spelling findings to `review` and records
checker provenance; spelling alone cannot create a `correct` disposition.

The implementation routes contiguous Thai, Japanese (Han/Hiragana/Katakana), and
Latin spans to separate adapters. A versioned allowlist handles project terminology
without hard-coding lexical replacements in the cleaning logic.
The operational review threshold is applied independently per language, avoiding
dilution of a short Thai or Japanese error inside a much longer English span.

## Candidate checkers

| Span | Candidate | Signal | Main risk |
|---|---|---|---|
| Thai | PyThaiNLP `pn`, `symspellpy`, `phunspell` | Tokenize with `newmm`; compare token with each engine's correction/suggestions | Segmentation and corpus frequency can flag valid compounds, dialect, names, and English transliteration |
| Japanese | SudachiPy + core/full dictionary | Morpheme OOV status and normalized form | OOV is common for names, new technical terms, romaji, and domain compounds; normalization is not necessarily an ASR correction |
| English | SymSpell with an explicit frequency dictionary | Closest dictionary candidates and edit distance | Frequency dictionaries under-cover proper names and technical vocabulary; short acronyms should be exempt |

PyThaiNLP documents `pn`, `phunspell`, and `symspellpy` engines and supports custom
dictionaries in `NorvigSpellChecker`: [PyThaiNLP spelling API](https://pythainlp.org/docs/4.0/api/spell.html).
Its `newmm` word tokenizer is dictionary/TCC based: [PyThaiNLP tokenization API](https://pythainlp.org/docs/2.2/api/tokenize.html).
SudachiPy exposes selectable dictionaries and tokenizer construction:
[SudachiPy API](https://worksapplications.github.io/sudachi.rs/python/api/sudachipy.html),
while Sudachi's normalized-form behavior is documented in the
[official repository](https://github.com/WorksApplications/SudachiPy).
SymSpell requires an explicit frequency dictionary and exposes fast edit-distance
lookup: [symspellpy documentation](https://symspellpy.readthedocs.io/en/latest/index.html)
and [dictionary installation](https://symspellpy.readthedocs.io/en/stable/users/installing.html).

## Experiment design

Create a protected benchmark with untouched clean negatives and deterministic
corruptions that resemble observed ASR label failures:

- Thai character deletion/insertion/transposition, invalid or displaced combining
  marks, and broken word joins;
- Japanese kana substitutions, prolonged-sound/small-kana errors, and safe
  normalization variants, while keeping kanji-reading ambiguity separate;
- English edit errors around technical terms, plus untouched names/acronyms as
  hard negatives;
- code-switch boundary errors and mixed-script technical terms.

Compare each checker independently using detection precision/recall/F1, false
positive rate on untouched labels, suggestion top-1/top-5 accuracy, coverage, and
latency. Report Thai-only, Japanese-only, English-only, and mixed-language slices.
Select thresholds on a development split grouped by source conversation; keep the
test split untouched. Synthetic corruption performance is insufficient by itself:
the winning configuration must also be checked on a small blinded sample of real
labels prioritized by checker disagreement.

Freeze each allowlist and custom/frequency dictionary using external, training, or
development data only. Protected held-out labels must never seed these resources;
also report false positives for terms that did not contribute to the allowlist.
Raw issue text is a local diagnostic only because it can reproduce protected label
substrings; committed machine-readable results should retain the default hashes.
