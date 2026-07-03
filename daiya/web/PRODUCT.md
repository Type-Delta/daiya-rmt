# Product

## Register

product

## Users

The Daiya researchers themselves (and collaborators on the same LAN, often on a
phone) running live tuning sessions against the realtime transcription
pipeline. They are technical, in the middle of an experiment, usually watching
the transcript and the server console at the same time.

## Product Purpose

A real-world testing ground for the Daiya v0 pipeline: stream live audio
(browser mic, server mic, desktop loopback, or a replayed file) through the
ASR + diarization stack, watch speaker-labeled transcript segments arrive,
solidify, and get corrected in place, and turn the pipeline's tuning knobs
without touching a terminal. Success = a tuning session needs only this page.

## Brand Personality

Calm, technical, legible. An instrument, not a product page. The interface
should disappear into the task of reading fast-moving mixed-script text
(Thai / Japanese / English) and judging pipeline behavior.

## Anti-references

- Marketing landing pages, hero sections, feature grids — this is a lab bench.
- SaaS dashboard clichés: hero metrics, gradient accents, card walls.
- Over-decorated "AI product" chrome; the transcript is the interface.

## Design Principles

1. **The transcript is the page.** Everything else (source, status, knobs,
   console) is chrome that stays out of its way.
2. **State is visible, never implied.** Partial vs final, connected vs erring,
   which source is live — always readable at a glance.
3. **Phone-first tolerance.** Tuning happens with a phone propped next to a
   laptop; every control must work at 390px over LAN HTTPS.
4. **Native scripts first.** Thai, Japanese, and English render through system
   fonts tuned for those scripts; no display-font vanity.

## Accessibility & Inclusion

WCAG AA: body text ≥ 4.5:1, meta text ≥ 3.5:1, visible focus rings, labeled
controls, `prefers-reduced-motion` respected, status changes announced via
`aria-live`.
