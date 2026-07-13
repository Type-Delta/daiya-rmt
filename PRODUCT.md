# Product

## Register

product

## Users

Daiya researchers and dataset curators work locally while evaluating Thai-English
and Japanese-English ASR data. They are technically capable, but their attention
is on listening, reading mixed scripts, and making defensible decisions—not on
remembering CLI flags or reconstructing how a label changed.

The realtime testbed serves live-tuning sessions. The manual-labeling workbench
serves a slower, deliberate review workflow: run automatic labeling, inspect the
validator's evidence, listen to a chunk, and record a human decision that can be
traced back to the original dataset.

## Product Purpose

Daiya-RMT is a research system for speaker- and context-aware, mixed-lingual
transcription. Its tools make experiments observable and repeatable: the live
testbed exposes the streaming pipeline, while the manual-labeling workbench turns
auto-labeled audio into explicitly reviewed training data.

Success for the labeling workbench is that a curator can configure the existing
auto-label and validation commands, browse every disposition (including `keep`),
play the exact chunk, and make a correction or confirmation with a versioned
human-review record. Source audio and source labels remain untouched. Review
artifacts are durable, but the active workbench session is not resumable after
the local API process stops.

## Brand Personality

Calm, technical, accountable. These are instruments for research work, not
marketing surfaces. They should feel precise under pressure and quiet enough for
rapid reading of Thai, Japanese, and English text.

## Anti-references

- Marketing landing pages, hero sections, and feature grids.
- SaaS dashboard clichés: hero metrics, gradient accents, and card walls.
- Over-decorated AI chrome or glassmorphism; the transcript and evidence are the
  interface.
- Destructive "cleanup" controls that hide whether an automatic system or a
  person changed a label.

## Design Principles

1. **The work item is the page.** Transcript text, audio, and evidence command
   the visual hierarchy; controls stay compact and nearby.
2. **State is visible, never implied.** Show command status, disposition,
   unsaved edits, source paths, and review progress at a glance.
3. **Human judgment is additive.** A human review is new, append-only
   provenance—not an overwrite of the source or an automatic proposal.
4. **Mixed scripts are first-class.** Thai, Japanese, and English text must
   remain legible at compact working sizes without artificial transliteration.
5. **Local-first tolerance.** Work must be safe on a researcher workstation and
   usable at desktop and narrow laptop widths. Saved review artifacts are
   inspectable later; a stopped workbench starts a fresh session rather than
   claiming durable session or job resumption.

## Accessibility & Inclusion

Meet WCAG AA: body and control text maintain at least 4.5:1 contrast, all
interactive controls have visible focus, labels are programmatically associated,
and status changes are announced. Respect `prefers-reduced-motion`. Never rely on
color alone for a disposition, spelling warning, job state, or save state.
