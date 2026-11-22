# Audio Feature Extraction Validation Report (Experimental Feature)

**Dataset:** Held-out test set sample ($N = 20$ tracks)  
**Date:** November 22, 2026  
**Artifact:** `reports/audio_validation.csv`  
**Overall Verdict:** **FAIL** (Integrated as an **Experimental** feature with explicit UI disclaimers)

---

## 1. Executive Summary & Gate Verdict

The audio feature extraction pipeline was evaluated against ground-truth Spotify features across 20 MP3 files corresponding to songs in the held-out test split of the Hit Song dataset.

| Gate Criterion | Requirement | Observed Value | Result |
| :--- | :--- | :--- | :--- |
| **Automatic Feature Count** | $\ge 6$ of 9 features with Spearman $\rho \ge 0.60$ | **1 / 9 features** (`duration_sec`) | **FAIL** |
| **Model Label Agreement** | $\ge 80.0\%$ | **55.0%** (11 / 20) | **FAIL** |
| **Mean Absolute Probability Difference ($|\Delta p|$)** | $\le 0.1500$ | **0.3451** | **FAIL** |
| **Hit Share Discrepancy** | $\le 15.0$ percentage points | **45.0 pp** (Spotify: 65.0%, Extracted: 20.0%) | **FAIL** |
| **Overall Gate Verdict** | All criteria must pass | **0 of 4 criteria passed** | **FAIL** |

> [!CAUTION]
> **Validation Gate Failed:** The audio extraction pipeline does not achieve equivalence with Spotify's proprietary features. Only **duration_sec** satisfies the correlation threshold ($\rho = 0.7188, p < 0.001$). Extracted `tempo`, `energy`, `loudness`, and `danceability` exhibit weak rank correlation with Spotify features, and 4 features (`valence`, `acousticness`, `instrumentalness`, `speechiness`) are intentionally not extracted. The feature is enabled strictly as an experimental convenience, clearly marked in the UI.

---

## 2. Per-Feature Accuracy Metrics ($N = 20$)

| Feature | Extraction Method / Tier | Spotify Mean | Extracted Mean (Raw) | Extracted Mean (Clamped) | MAE (Clamped) | Spearman $\rho$ | p-value | Clamped Rows | Mode |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **duration_sec** | Direct sample length ($N / f_s$) | 229.95 s | 240.00 s | 240.00 s | 20.21 s | **+0.7188** | 0.000356 | 0 / 20 | **Automatic** |
| **tempo** | Onset autocorrelation + octave fold | 126.24 BPM | 119.03 BPM | 119.03 BPM | 16.95 BPM | **+0.2413** | 0.3054 | 0 / 20 | **Manual** |
| **energy** | RMS amplitude normalized proxy | 0.6232 | 0.8967 | 0.8967 | 0.2992 | **+0.2028** | 0.3911 | 0 / 20 | **Manual** |
| **loudness** | ITU-R BS.1770-4 LUFS (`pyloudnorm`) | -7.31 dB | -14.22 LUFS | -14.22 LUFS | 7.34 dB | **+0.1850** | 0.4348 | 0 / 20 | **Manual** |
| **danceability**| Autocorrelation pulse clarity proxy | 0.6132 | 0.7413 | 0.7413 | 0.1717 | **-0.1369** | 0.5649 | 0 / 20 | **Manual** |
| **valence** | Manual only (omitted) | 0.5869 | N/A | N/A | N/A | N/A | N/A | 0 / 20 | **Manual** |
| **acousticness**| Manual only (omitted) | 0.2954 | N/A | N/A | N/A | N/A | N/A | 0 / 20 | **Manual** |
| **instrumentalness**| Manual only (omitted) | 0.0217 | N/A | N/A | N/A | N/A | N/A | 0 / 20 | **Manual** |
| **speechiness** | Manual only (omitted) | 0.0570 | N/A | N/A | N/A | N/A | N/A | 0 / 20 | **Manual** |

---

## 3. End-to-End Prediction Pipeline Impact

Evaluating the frozen model (`models/final/model.joblib`) at operational threshold $\tau = 0.48$:
- **(a) Spotify Ground Truth:** 13 of 20 predicted hit (65.0%)
- **(b) Extracted Audio Input:** 4 of 20 predicted hit (20.0%)
- **Label Agreement:** 11 of 20 tracks agree (55.0%)
- **Overall Mean Absolute Probability Difference:** **0.3451** ($34.51$ percentage points)

---

## 4. Known Limitations & Technical Context

1. **Physical Broadcast LUFS vs. Spotify Proprietary Decibels:** Standard ITU-R BS.1770-4 integrated loudness averages $-14.22$ LUFS on unmastered/local files, compared to Spotify's ingestion values averaging $-7.31$ dB. In our logistic regression model, this negative shift depresses charting log-odds.
2. **Audio Cut & Version Discrepancies:** Outliers such as *Daddy Come Home* (extended compilation track, $\Delta t = +156.26$ s) and *My Way* (video version with skit outro, $\Delta t = +63.27$ s) account for most duration error.
3. **Danceability Fix:** The previous tempogram division bug that saturated at $1.00$ has been resolved with an autocorrelation pulse clarity proxy across plausible musical beat lags ($0.3$s to $1.2$s), yielding continuous values with zero clamping ($n = 0 / 20$).
4. **Experimental Disclaimer:** The UI prominently warns users that predictions based on uploaded audio are experimental and that all sliders remain manually adjustable.
