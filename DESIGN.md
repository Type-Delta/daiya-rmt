---
name: Daiya Research Workbench
description: Calm, accountable interfaces for inspecting mixed-lingual audio and transcription evidence.
colors:
  bg: "oklch(0.16 0.016 220)"
  surface: "oklch(0.205 0.018 220)"
  raised: "oklch(0.25 0.02 220)"
  edge: "oklch(0.32 0.02 220)"
  ink: "oklch(0.93 0.012 220)"
  muted: "oklch(0.7 0.016 220)"
  faint: "oklch(0.56 0.016 220)"
  primary: "oklch(0.78 0.13 205)"
  primaryInk: "oklch(0.17 0.03 210)"
  accent: "oklch(0.74 0.11 240)"
  danger: "oklch(0.68 0.17 25)"
  warning: "oklch(0.8 0.13 85)"
  success: "oklch(0.76 0.14 150)"
typography:
  body:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "14px"
    fontWeight: 400
    lineHeight: 1.5
  label:
    fontFamily: "ui-sans-serif, system-ui, sans-serif"
    fontSize: "12px"
    fontWeight: 600
    lineHeight: 1.2
rounded:
  sm: "6px"
  md: "10px"
spacing:
  compact: "8px"
  standard: "16px"
  roomy: "24px"
components:
  button-primary:
    backgroundColor: "{colors.primary}"
    textColor: "{colors.bg}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
  button-secondary:
    backgroundColor: "{colors.raised}"
    textColor: "{colors.ink}"
    rounded: "{rounded.sm}"
    padding: "8px 12px"
---

# Design System: Daiya Research Workbench

## Overview

**Creative North Star: "The Listening Bench"**

The workbench uses a dark ocean-blue field of view so the waveform, mixed-script
label, and evidence can be read for long stretches without the interface calling
attention to itself. It is dense by design, but never cramped: the current clip
gets the clearest surface and the list remains a stable navigation rail.

It rejects product-tour theatrics, decorative gradients, and false precision.
Every color or motion change carries state: selection, a disposition, an active
job, unsaved text, or a completed human review.

**Key Characteristics:** restrained accent use; tonal surface layering instead of
card walls; native mixed-script typography; compact controls; explicit audit state;
fresh local sessions rather than implied session or job resumption.

## Colors

The palette stays in a narrow ocean-blue neutral range. Turquoise carries the
diamond-inspired signal without making this research instrument feel decorative.

### Primary

- **Signal turquoise** (`oklch(0.78 0.13 205)`): primary action, keyboard focus,
  current selection, and active progress only.

### Secondary

- **Evidence blue** (`oklch(0.74 0.11 240)`): linked evidence and informational
  context; never a competing primary action.

### Neutral

- **Listening field** (`oklch(0.16 0.016 220)`): page background.
- **Bench surfaces** (`oklch(0.205 0.018 220)` and `oklch(0.25 0.02 220)`):
  panels and selected/inset controls.
- **Readable ink** (`oklch(0.93 0.012 220)`): primary labels and transcript text.
- **Quiet metadata** (`oklch(0.7 0.016 220)`): paths, timestamps, and supporting
  copy; it must still remain legible.

**The Signal Rule.** The primary accent marks action and current state, not
decoration. A screen should read as mostly neutral at rest.

## Typography

**Display Font:** System sans-serif
**Body Font:** System sans-serif with platform fallbacks for Thai and Japanese
**Label/Mono Font:** `ui-monospace, SFMono-Regular, Consolas, monospace` for paths,
ids, and command output

**Character:** Native system faces preserve familiar Thai, Japanese, and English
shaping. The working type scale is compact and fixed rather than expressive.

### Hierarchy

- **Headline** (600, 18px, 1.25): page or selected-item titles.
- **Title** (600, 14px, 1.35): panel headings and chunk identifiers.
- **Body** (400, 14px, 1.5): transcript and explanations.
- **Label** (600, 12px, 1.2): controls, table metadata, and compact state.

**The Script Rule.** Never reduce labels to tiny all-caps text; sentence case and
the body size must remain readable for every script.

## Elevation

Depth is structural, not decorative. A faint border separates persistent rails and
tonal layers distinguish active controls from their background. Panels use no
ambient drop shadows; overlays only use a short, defined shadow when they need to
clear a scroll region.

**The Flat-By-Default Rule.** Use borders or tonal layering, never a wide soft
shadow paired with a border on the same resting container.

## Components

### Buttons

- **Shape:** compact rounded rectangle (6px).
- **Primary:** signal turquoise with dark ink; reserved for start, save, and the
  next meaningful step.
- **Hover / Focus:** 150ms color change; a 2px offset focus outline in the primary
  color. Disabled controls retain their label but lose action emphasis.
- **Secondary / Ghost:** raised neutral or transparent controls for filtering,
  navigation, and non-destructive commands.

### Chips

- **Style:** compact, square-ish pills that pair a text label with a status dot or
  count where needed.
- **State:** selected filters use raised surface plus primary text or outline;
  semantic dispositions always have text as well as color.

### Cards / Containers

- **Corner Style:** 10px at most.
- **Background:** background, surface, and raised are used as a shallow hierarchy,
  never as a repeated card grid.
- **Border:** one-pixel edge line for persistent rails and input group boundaries.
- **Internal Padding:** 16px for panels, 8px for dense lists.

### Inputs / Fields

- **Style:** surface background, one-pixel edge, 6px radius.
- **Focus:** primary outline with offset; no color-only error indication.
- **Error / Disabled:** semantic text and icon accompany the color state.

### Navigation

- **Style:** persistent left clip list on wide screens; it becomes a compact
  selector before the main editor on narrow screens. Current item is visibly
  selected with a primary cue and a tonal surface.

### Review Editor

The transcript editor is the signature component: audio playback and original
label provenance sit directly above an editable text area, while save state and
review action remain visible without hiding the evidence.

## Do's and Don'ts

### Do:

- **Do** keep transcript text, audio playback, and the save action in one visible
  working area.
- **Do** use `oklch(0.78 0.13 205)` only for primary action, focus, or current
  selection.
- **Do** make disposition, job, and unsaved states readable without color.
- **Do** preserve native Thai, Japanese, and English strings exactly as entered.

### Don't:

- **Don't** add marketing hero sections, feature grids, or hero metrics.
- **Don't** use gradient accents, gradient text, glassmorphism, or decorative
  "AI product" chrome.
- **Don't** use card walls or a colored side stripe wider than one pixel to create
  hierarchy.
- **Don't** make human edits look like source-label overwrites; provenance is part
  of the interface.

## App Icon and Favicon

**The shared app mark is the Phosphor `WaveformIcon` in bold weight.** It is the
existing Daiya v0 testbed header mark and must be used by both web UIs anywhere an
app-identifying icon is shown. It is not a substitute for semantic, task-specific
icons.

The favicon is a vector rendering of that exact mark, not a new logo. Keep one
identical `favicon.svg` file in the project root, `daiya/web`, and
`training/processor/whisper/web`; each UI's `index.html` must reference its local
copy. The favicon uses the signal-turquoise foreground on the listening-field
background so it remains recognisable in browser tabs and matches the app mark.
