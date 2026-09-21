# Hit Song Predictor — Frontend Verification Walkthrough

This document records the visual and functional verification of the Hit Song Predictor web application (`web/index.html`, `web/style.css`, `web/app.js`), executed using Chrome automation via Playwright against the live FastAPI server at `http://127.0.0.1:8000`.

---

## 1. Initial Load Verification (1440px Desktop & 390px Mobile)

The application adheres strictly to the manuscript paper aesthetic defined in `design.md`:
- Flat warm cream background (`#FBF8F1`), white card panels with subtle shadows, and Fraunces serif typography.
- Five-line staff dividers beneath the masthead and above the footer.
- Live `● model ok` indicator confirming that `/api/health` responded with HTTP 200.
- Dynamic sliders loaded from `/api/schema` with exact dataset boundaries and default values.
- Initial prediction rendered: **97.1%** probability ("Likely to chart") with the brass threshold marker at 0.48 and diverging contribution bars.

### Desktop View (1440px Width)
![Initial Load Desktop 1440px](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/initial_load_1440.png)

### Mobile View (390px Width)
At 390px width, the 12-column grid collapses into a responsive single-column layout. The prediction panel is positioned first (`order: -1`), ensuring the probability numeral and meter remain immediately visible above the fold without requiring initial scrolling. The comparison table supports horizontal scroll overflow.

![Initial Load Mobile 390px](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/initial_load_390.png)

---

## 2. Real-Time Slider Interaction & Inference

Interactive slider adjustment was tested by modifying three continuous audio features simultaneously:
- **Danceability**: Reduced from `0.75` to `0.25`
- **Loudness**: Decreased from `-5.00 dB` to `-20.00 dB`
- **Instrumentalness**: Increased from `0.00` to `0.85`

### Before Adjustment
- **Probability**: `97.1%`
- **Verdict**: `Likely to chart` (Above the 0.48 decision threshold)
- **Top Contributions**: Genre (`+3.87`), Instrumentalness (`+1.00`), Loudness (`+0.87`), Danceability (`+0.57`), Energy (`-0.38`)

### After Adjustment
- **Probability**: Dropped to `0.1%`
- **Verdict**: Shifted to `Unlikely to chart` in muted `--ink-2` grey (Below the 0.48 decision threshold)
- **Meter**: Fill retracted to 0% and knob moved to zero endpoint.
- **Top Contributions**: Re-scaled dynamically to reflect negative pull: Instrumentalness (`-3.95`), Loudness (`-3.21`), Danceability (`-1.01`), while Genre remained positive (`+3.87`).

![Sliders After Adjustment](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/sliders_after.png)

---

## 3. Preset Examples & Sanity Checks

All three presets were evaluated sequentially by clicking their respective preset controls:

| Preset Name | Genre | Tempo | Loudness | Danceability | Instrumentalness | Output Probability | Verdict |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---|
| **Pop single** | `pop` | 122 BPM | -5.0 dB | 0.75 | 0.00 | **97.1%** | Likely to chart |
| **Acoustic ballad** | `singer-songwriter` | 85 BPM | -10.5 dB | 0.45 | 0.00 | **46.0%** | Unlikely to chart |
| **Ambient instrumental** | `ambient` | 95 BPM | -18.0 dB | 0.22 | 0.82 | **0.0%** | Unlikely to chart |

### Sanity-Check Findings:
$$\text{Pop single } (97.1\%) > \text{Acoustic ballad } (46.0\%) > \text{Ambient instrumental } (0.0\%)$$
- **Domain Consistency**: Pop music on the Billboard Hot 100 is characterized by high loudness and danceability, which carry positive model coefficients ($\beta_{\text{loudness}} = +1.349$, $\beta_{\text{danceability}} = +0.552$).
- **Instrumental Music**: High instrumentalness has a strong negative coefficient ($\beta_{\text{instrumentalness}} = -1.868$) and low loudness pushes the log-odds heavily negative, yielding $\approx 0.0\%$.
- **Conclusion**: The ordering matches music industry domain expectations and statistical model behavior. **No bug detected.**

---

## 4. Input Validation & Error Handling

To test error resilience, an invalid payload with `tempo: 999.0` (exceeding the training maximum of $217.282\text{ BPM}$) was submitted via `window.fetchPrediction()`.

The application handled the 422 Unprocessable Entity response cleanly:
1. Identified the offending parameter (`tempo`).
2. Highlighted the specific constraint under the slider: `"Field 'tempo' must be between 30.2 and 217.282 (received 999.0)"`.
3. Displayed instructions in the result panel instead of an unhandled crash or stack trace.

![Out-of-Range Error State](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/error_state.png)

---

## 5. Keyboard Accessibility & Focus States

Sequential Tab navigation was verified across all interactive controls. Focus cycles logically in document order without trapping:
1. `slider-tempo` $\rightarrow$ 2. `slider-energy` $\rightarrow$ 3. `slider-danceability` $\rightarrow$ 4. `slider-valence` $\rightarrow$ 5. `slider-loudness` $\rightarrow$ 6. `slider-acousticness` $\rightarrow$ 7. `slider-instrumentalness` $\rightarrow$ 8. `slider-speechiness` $\rightarrow$ 9. `slider-duration_sec` $\rightarrow$ 10. `genre-select` $\rightarrow$ 11. `Pop single` $\rightarrow$ 12. `Acoustic ballad` $\rightarrow$ 13. `Ambient instrumental`.

Every interactive element renders a prominent `2px solid var(--indigo)` focus ring:

### Preset Button Focus State
![Preset Focus Ring](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/focus_preset_group.png)

### Genre Select Focus State
![Genre Focus Ring](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/focus_genre.png)

### Slider Track Focus State
![Slider Track Focus Ring](C:/Users/Kaif Ahmad/.gemini/antigravity/brain/0451bf9d-63e8-44b9-ba7c-36d9d28e9457/screenshots/focus_slider_tempo_row.png)

---

## 6. Bugs Identified & Fixed

| Item | Finding | Resolution |
|:---|:---|:---|
| **Static Figure Route** | Initial server configuration did not mount `/reports/figures`, preventing `feature_importance_rf.png` from loading in the insights section. | Mounted `reports/figures` as a static directory in `app/main.py` before the root mount. Verified with HTTP 200 OK. |
| **422 Error Handling** | Initial `fetchPrediction` in `web/app.js` treated 422 HTTP responses as generic server failures ("model offline"). | Added explicit 422 response inspection to extract field-level validation messages and render the error both under the offending slider and in the result panel. |
| **Range Slider Focus Rings** | Native range inputs in Chrome lacked sufficient outline clearance when focused. | Added `outline-offset: 4px` on `.slider-input:focus` and `.slider-input:focus-visible` to ensure focus indicators are visible on keyboard navigation. |
