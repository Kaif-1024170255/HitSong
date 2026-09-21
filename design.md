# design.md — Hit Song Predictor

Binding visual specification for `web/`. An agent implementing this should treat every
token and rule below as fixed. Where this file is silent, choose the quieter option.

---

## 1. The brief

**Subject:** a classifier that reads a song's measured audio properties and returns the
probability it charts. **Audience:** an examiner, and an A&R person who is not technical.
**Primary job:** let someone move a few sliders and immediately understand both the
prediction and what drove it.

**The design's one idea:** the page should feel like a sheet of manuscript paper on a
desk — quiet, ruled, analytical. Not a streaming app, not a dashboard. The music is
present as *notation*, not as neon.

**Deliberately excluded:** dark hero sections, album artwork, gradient washes, equalizer
animations, glassmorphism, card grids with identical shadows, purple/green "Spotify"
signalling. The frontend is a thin, well-made surface over the real work, and it should
look like it knows that.

---

## 2. Colour

Cream and black carry the page. One cool accent, one warm accent, both used sparingly.

| Token | Hex | Role |
|---|---|---|
| `--paper` | `#FBF8F1` | page background — warm cream, the manuscript sheet |
| `--card` | `#FFFFFF` | raised surfaces: the predictor panel, the result panel |
| `--ink` | `#000000` | headings, numerals, primary text |
| `--ink-2` | `#5A544B` | secondary text, labels, captions |
| `--rule` | `#E6DFD1` | hairlines, staff lines, borders, slider tracks |
| `--indigo` | `#26407A` | the accent: positive prediction, slider fill, focus ring, links |
| `--brass` | `#A8842C` | the threshold marker and tick marks only. Never body text. |

**Why these:** cream sits near hue 40°. `--indigo` at hue 218° is its near-complement, so
the accent separates cleanly from the paper without shouting; `--brass` is analogous to the
cream, so it reads as part of the paper rather than as a second accent competing with the
indigo. The result is a split-complementary scheme with black and cream doing 90% of the
work. Contrast: `--ink-2` on `--paper` ≈ 6.3:1, `--indigo` on `--paper` ≈ 9:1 — both pass
AA. `--brass` is below 4.5:1 and is therefore restricted to graphic marks, never to text.

**Semantics of the verdict.** "Likely to chart" uses `--indigo`. "Unlikely to chart" uses
`--ink-2` — grey, not red. A low probability is a finding, not an error, and colouring it
red would misrepresent the model.

**Dark mode.** Provide a restrained inversion so the page doesn't break in a dark viewer:
`--paper: #14120E`, `--card: #1C1915`, `--ink: #FBF8F1`, `--ink-2: #A9A296`,
`--rule: #332E26`, `--indigo: #8FA9DE`, `--brass: #D6B463`. Same layout, same weights.

---

## 3. Type

Two families, clearly distinct in role.

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Sans:wght@400;500;600&display=swap" rel="stylesheet">
```

- **Fraunces** — the page title, section headings, and the large probability numeral.
  Use the optical-size axis: `font-variation-settings: "opsz" 120` on the big numeral,
  `"opsz" 36` on headings. Its slightly engraved, old-style character is what keeps the
  page from reading as a generic SaaS panel.
- **IBM Plex Sans** — every label, value, button, table cell, and paragraph.

Fallbacks: `Fraunces, "Iowan Old Style", Georgia, serif` and
`"IBM Plex Sans", "Segoe UI", system-ui, sans-serif`.

### Scale

| Element | Size / line-height | Family, weight |
|---|---|---|
| Probability numeral | `clamp(4rem, 12vw, 7rem)` / 0.9 | Fraunces 600, tabular figures |
| Page title | `clamp(1.75rem, 4vw, 2.5rem)` / 1.15 | Fraunces 600 |
| Section heading | `1.25rem` / 1.3 | Fraunces 600 |
| Verdict line | `1.125rem` / 1.4 | Plex Sans 600 |
| Body / description | `0.9375rem` / 1.6 | Plex Sans 400 |
| Control label | `0.875rem` / 1.4 | Plex Sans 500 |
| Live slider value | `0.875rem` / 1 | Plex Sans 600, `font-variant-numeric: tabular-nums` |
| Caption / footnote | `0.8125rem` / 1.5 | Plex Sans 400, `--ink-2` |

**Rules.** Sentence case everywhere — no all-caps labels, no tracked-out eyebrows above
headings. Do not accent a single word of a heading in a different colour or weight. Body
text stays under 70 characters per line. Every number that updates live uses tabular
figures so the layout doesn't jitter.

---

## 4. The musical motif

Three uses only. Anything beyond these is decoration and should be cut.

1. **Staff rule.** A five-line stave, `1px` lines in `--rule`, 7px apart, spanning the
   full content width. It appears exactly twice: directly beneath the masthead, and above
   the footer. Render it as an inline SVG or a repeating-linear-gradient block ~29px tall.
   It is a divider, not a background pattern.
2. **Barline separators.** Section dividers are a `1px --rule` horizontal line with one
   short vertical tick (`1px × 10px`, `--rule`) sitting on it at the left edge — a barline.
   That single detail does all the "this is music" work the page needs.
3. **The threshold marker.** On the probability meter, the decision threshold is a
   `2px` vertical `--brass` line with a small filled triangle above it, labelled
   `threshold 0.45` in caption type. This is the one place brass appears.

The page background is flat `--paper`. No waveforms, no notes, no vinyl grooves, no
repeating pattern behind content.

---

## 5. Layout

12-column grid, `max-width: 1080px`, `--gutter: 24px`, centred. Spacing scale:
`4 / 8 / 12 / 16 / 24 / 40 / 64px`. Radius: `4px` on inputs and buttons, `8px` on the two
panels, `0` on everything else. One shadow only, on the two panels:
`0 1px 2px rgba(0,0,0,.04), 0 8px 24px rgba(0,0,0,.05)`.

Content is **left-aligned** throughout — including the probability numeral, which anchors
the left edge of the result panel. Centred text would make this feel like a marketing page.

```
┌──────────────────────────────────────────────────────────────┐
│  Hit Song Predictor                              ● model ok  │
│  Move the sliders to see a song's chance of charting.        │
│  ════════════════════════════  (stave)                       │
├──────────────────────────────────┬───────────────────────────┤
│  Song features                   │  Prediction               │
│                                  │                           │
│  Tempo            118 BPM        │       64.2%               │
│  ▬▬▬▬▬▬▬▬▬●────────────          │       ┌──── the numeral   │
│  Energy            0.78          │  Likely to chart          │
│  ▬▬▬▬▬▬▬▬▬▬▬▬●──────             │                           │
│  Danceability      0.71          │  0 ──────┃────●──── 1     │
│  ▬▬▬▬▬▬▬▬▬▬▬●───────             │       threshold           │
│  ... 6 more ...                  │                           │
│                                  │  What moved it            │
│  Genre  [ pop        ▾ ]         │  Danceability   ███▶      │
│                                  │  Energy          ██▶      │
│  Load an example:                │  Acousticness   ◀███      │
│  [ pop ] [ ballad ] [ ambient ]  │  Speechiness     ◀█       │
├──────────────────────────────────┴───────────────────────────┤
│ │ Model comparison                                           │
│   (table: 7 rows × 6 metrics, from /api/insights)            │
├──────────────────────────────────────────────────────────────┤
│ │ What predicts a hit                                        │
│   (feature importance figure)                                │
│  ════════════════════════════  (stave)                       │
│  Trained on N songs · Billboard Hot 100 2000–2023 · not      │
│  investment advice for your record label                     │
└──────────────────────────────────────────────────────────────┘
```

- Predictor panel: 7 columns. Result panel: 5 columns, `position: sticky; top: 24px`.
- Below 900px: stack, result panel first so the number stays visible while scrolling to
  the controls. Below 420px: single column, sliders full width, meter labels wrap.

---

## 6. Components

**Slider row.** Label on the left, live value right-aligned on the same line, track
beneath spanning full width. Custom-styled `input[type=range]`: track `4px`, `--rule`,
radius 2; filled portion `--indigo`; thumb `16px` circle, `--card` fill, `2px --ink`
border, no shadow. Thumb grows to `18px` on `:active`. Under the track, place the
min and max values in caption type at each end — nothing else.

**Genre select.** Native `<select>`, `1px --rule` border, `--card` background, `36px`
tall, custom chevron in `--ink-2`.

**Example buttons.** Three text buttons in a row, `1px --rule` border, `--card` fill,
`--ink` label. On selection, border becomes `--indigo` and the label goes Plex Sans 600.
No fill change, no icons. Label them "Pop single", "Acoustic ballad", "Ambient
instrumental" — say what they are, not "Preset 1".

**Probability numeral.** Fraunces 600, `--ink` always — not `--indigo`. The colour lives
in the meter and the verdict line, so the number itself stays neutral and authoritative.
One decimal place. `aria-live="polite"` on its container.

**Meter.** Full-width `10px` track in `--rule`, radius 5. Fill from 0 to the probability
in `--indigo`. Threshold marker per §4.3. A `14px` circular knob in `--ink` sits at the
probability position. Endpoints labelled `0` and `1` in caption type.

**Contributions.** Up to five rows, each a label and a bar diverging from a shared centre
axis: right in `--indigo` for features pushing the probability up, left in `--ink-2` for
features pushing it down. A `1px --rule` vertical line marks the axis. Bars share one
scale, normalised to the largest absolute contribution. Caption beneath, once:
"Relative pull on this prediction, not a general ranking."

**Comparison table.** No vertical rules, no zebra striping, no card. `1px --rule`
horizontal line under the header row and under the final row only. Numeric cells
right-aligned with tabular figures. The selected final model's row is marked with a
`2px --indigo` left border and its name in Plex Sans 600 — the only emphasis in the table.

---

## 7. Motion

One orchestrated moment, and one response to action.

- **On load:** the meter fill and the probability numeral count up from 0 to the initial
  prediction over 600ms, `cubic-bezier(.2,.7,.2,1)`. This happens once, on first paint only.
- **On slider change:** the meter fill and numeral transition over 200ms. The contribution
  bars re-scale over 200ms. Nothing else moves.

No entrance animations on sections. No hover transforms on cards or table rows. Buttons
change border colour on hover in 120ms and that is all. Wrap every transition in
`@media (prefers-reduced-motion: no-preference)`; under reduced motion, values snap.

---

## 8. States and copy

Copy is sentence case, active voice, no exclamation marks, no apologies.

| State | What to show |
|---|---|
| Loading schema | Skeleton slider rows in `--rule`, and `—` in place of the numeral |
| Waiting on a prediction | Numeral dims to 40% opacity; the meter holds its last position |
| API unreachable | In the result panel: "The model isn't responding. Start the server with `uvicorn app.main:app --reload` and reload this page." |
| Out-of-range input | Under the offending slider: "Tempo must be between 40 and 220 BPM." Name the field and the range. |
| Prediction below threshold | Verdict: "Unlikely to chart" in `--ink-2`, with the caption "Below the 0.45 decision threshold." |
| Prediction above threshold | Verdict: "Likely to chart" in `--indigo`, caption "Above the 0.45 decision threshold." |

Footer carries the honest caveat in caption type: the model reflects patterns in
Billboard 2000–2023 and describes correlation, not causation.

---

## 9. Accessibility floor

Every slider has a visible `<label>` bound by `for`, plus `aria-valuetext` carrying the
value with its unit. Focus ring is `2px --indigo` with `2px` offset, visible on every
interactive element, never removed. The result container is `aria-live="polite"` and
`aria-atomic="true"`. Colour is never the only signal — the verdict is always stated in
words and the contribution direction is conveyed by bar side as well as hue. Target size
minimum 44px on touch. Page is usable at 200% zoom and at 380px width.
