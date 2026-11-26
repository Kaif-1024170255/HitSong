"""Detailed PDF guide generator for the Hit Song Predictor audio extraction pipeline.

Documents every signal processing, mathematical, API, clamping, and validation
detail of src/audio_extractor.py and its FastAPI/UI integration.
Runs standalone from root: python -m src.generate_audio_guide_pdf
"""

import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, HRFlowable
)
from src.generate_pdf_report import (
    NumberedCanvas, get_report_styles, sec_header, make_callout, make_table
)


def build_audio_cover(styles):
    """Build cover page for the Audio Feature Extraction technical guide."""
    return [
        Spacer(1, 36),
        Paragraph("AUDIO FEATURE EXTRACTION", styles["CoverTitle"]),
        Spacer(1, 6),
        Paragraph(
            "Complete Technical, Mathematical, and Architectural Guide to Local MP3 "
            "Signal Processing, Bounds Clamping, API Integration, and Validation",
            styles["CoverSubtitle"]
        ),
        Spacer(1, 20),
        HRFlowable(width="100%", thickness=2, color=colors.HexColor("#1A2B4C"), spaceAfter=16),
        Paragraph("<b>Module:</b> src/audio_extractor.py | <b>Endpoint:</b> POST /api/extract", styles["CoverMeta"]),
        Paragraph("<b>Libraries:</b> librosa, soundfile, pyloudnorm, numpy (100% Free & Local)", styles["CoverMeta"]),
        Paragraph("<b>Feature Flag:</b> ENABLE_AUDIO_UPLOAD (default: false; experimental mode)", styles["CoverMeta"]),
        Paragraph("<b>Frozen ML Pipeline:</b> models/final/model.joblib (Logistic Regression, \u03c4 = 0.48)", styles["CoverMeta"]),
        Spacer(1, 22),
        make_callout(
            "This document explains in complete detail how the Hit Song Predictor extracts audio features "
            "from user-uploaded MP3 files, why specific DSP formulas and libraries were chosen, why four "
            "semantic features remain manual, how estimates are clamped to training bounds, and what the "
            "empirical 20-track held-out validation audit revealed.",
            title="GUIDE SCOPE", border_hex="#1A2B4C", bg_hex="#F0F4F8", styles=styles
        ),
        PageBreak(),
    ]


def build_architecture_flow(styles):
    """Section 1: End-to-End Upload & Extraction Architecture."""
    flow = []
    flow.extend(sec_header("1. End-to-End Upload & Extraction Architecture", styles))
    p1 = (
        "When ENABLE_AUDIO_UPLOAD=true, the web interface exposes an MP3 dropzone. Uploading a file triggers "
        "POST /api/extract (aliased at /api/estimate-audio) in app/main.py. To guarantee privacy and free "
        "local execution without cloud APIs, the backend enforces strict resource and lifecycle controls:"
    )
    p2 = (
        "1. Extension & Size Validation (save_upload_to_temp): Rejects non-.mp3 filenames, empty 0-byte uploads, "
        "and files exceeding 20 MB (20 * 1024 * 1024 bytes) with clean HTTP 400 JSON errors.<br/>"
        "2. Ephemeral Temp Processing: Writes bytes to tempfile.NamedTemporaryFile(suffix='.mp3', delete=False), "
        "runs extract_audio_features(tmp_path, BOUNDS), and deletes the temporary file unconditionally inside a "
        "finally block—nothing is ever stored on disk or sent to third-party services.<br/>"
        "3. Frontend Integration (web/app.js): Populates only the 5 returned sliders tagged 'Estimated' (or "
        "'Estimated (clamped)'), leaves the remaining 4 sliders and genre dropdown tagged 'Manual', and triggers "
        "POST /api/predict against the frozen Logistic Regression pipeline."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"])])
    return flow


def build_decoding_slicing(styles):
    """Section 2: Audio Decoding, Resampling, and Windowing."""
    flow = []
    flow.extend(sec_header("2. Stage 1: Decoding, Resampling & Middle Slicing", styles))
    p1 = (
        "<b>Mono Resampling (load_audio):</b> Audio is decoded via soundfile.read(file_path, dtype='float32'), "
        "with automatic fallback to librosa.load(..., sr=22050, mono=True) for MP3 codecs soundfile cannot parse "
        "directly. Multi-channel stereo waveforms are averaged to mono (np.mean(data, axis=1)) and resampled to "
        "target_sr = 22,050 Hz. By the Nyquist theorem, 22,050 Hz captures frequencies up to 11,025 Hz—covering "
        "all musical fundamentals, vocal formants, and drum transients while halving memory and FFT cost. Empty "
        "or unreadable streams raise ValueError (mapped to HTTP 400)."
    )
    p2 = (
        "<b>Full-Track vs. 45-Second Middle Segment (get_middle_segment):</b> Duration is measured over the full "
        "waveform y_full. All spectral, loudness, and rhythmic estimators operate on a 45.0-second middle slice "
        "y_mid (if total_sec > 45.0, start_sec = (total_sec - 45.0) / 2.0). Analyzing the middle 45 seconds "
        "avoids quiet intros, spoken skits, and long fade-outs while bounding CPU latency to ~1–2 seconds per song."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"])])
    return flow


def build_feature_duration_tempo(styles):
    """Section 3.1 & 3.2: Duration and Tempo calculations."""
    flow = []
    flow.extend(sec_header("3. Stage 2: Exact Mathematical Breakdown (5 Features)", styles))
    p1 = (
        "<b>3.1 Duration (duration_sec — Method: audio_length_direct):</b><br/>"
        "Computed directly from the full decoded audio array: duration_sec = len(y_full) / sr. "
        "This is an exact physical measurement of the MP3 container audio length. On 20 held-out test songs, "
        "it achieved Spearman \u03c1 = +0.7188 (p = 0.000356), qualifying as Automatic. Its 20.21 s MAE was driven "
        "by track-version mismatches in local MP3s (e.g., Daddy Come Home: 298.7 s extended cut vs. 142.4 s Spotify single)."
    )
    p2 = (
        "<b>3.2 Tempo (tempo — Method: librosa_onset_autocorrelation):</b><br/>"
        "1. Computes spectral flux onset strength: onset_env = librosa.onset.onset_strength(y=y_mid, sr=sr) "
        "(hop_length=512, ~43.07 frames/sec), detecting sudden frame-to-frame energy rises from drum hits and note attacks.<br/>"
        "2. Estimates dominant global periodicity via librosa.feature.tempo(onset_envelope=onset_env, sr=sr).<br/>"
        "3. Applies octave folding into standard commercial bounds: doubles values < 55.0 BPM (while * 2.0 <= 217.0) "
        "and halves values > 185.0 BPM (while / 2.0 >= 30.2). Rounded to 2 decimals. Limitation: 3:2 triplet/hemiola "
        "ambiguities (e.g., Hummingbird: 162.34 BPM Spotify vs. 107.67 BPM extracted) reduced correlation to \u03c1 = +0.2413."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"])])
    return flow


def build_feature_loudness_energy_dance(styles):
    """Section 3.3, 3.4 & 3.5: Loudness, Energy, and Danceability calculations."""
    flow = []
    p1 = (
        "<b>3.3 Loudness (loudness — Method: pyloudnorm_itu_r_bs1770_lufs):</b><br/>"
        "Measured via pyln.Meter(sr).integrated_loudness(y_mid) following the international broadcast standard "
        "<b>ITU-R BS.1770-4</b> in LUFS (Loudness Units relative to Full Scale). The meter applies a two-stage "
        "K-weighting filter (high-shelf +4 dB above 1.5 kHz + high-pass at 38 Hz) to model human ear sensitivity, "
        "computes mean-square power over 400 ms overlapping blocks (75% overlap), and applies dual gating (-70 LKFS "
        "absolute gate, -10 LU relative gate) to ignore silence. Silent/NaN tracks fall back to -70.0 LUFS. "
        "Note: Standard LUFS averages -14.22 LUFS on local tracks vs. -7.31 dB for Spotify's proprietary Echo Nest "
        "loudness (\u0394 = -6.91 \u00b1 4.10 dB); no arbitrary fixed offset is added."
    )
    p2 = (
        "<b>3.4 Energy Proxy (energy — Method: rms_amplitude_proxy):</b><br/>"
        "Computed as normalized time-domain Root-Mean-Square amplitude: RMS = sqrt(mean(y_mid**2)), "
        "energy = clip(RMS / 0.20, 0.0, 1.0). Because commercial mastering limiters push waveform RMS near 0.18–0.30 "
        "even on acoustic ballads, 14/20 test tracks hit 1.000 (\u03c1 = +0.2028, MAE = 0.2992)."
    )
    p3 = (
        "<b>3.5 Danceability Proxy (danceability — Method: autocorrelation_pulse_clarity_proxy):</b><br/>"
        "v1 originally divided a 3-bin tempogram slice mean by a 384-bin global mean, yielding ~3.21 and saturating "
        "all 20 tracks at 1.000 (clamped to 0.980). The fixed v2 estimator computes normalized onset autocorrelation "
        "ac_norm = ac / ac[0] up to 1.5 s lag, then takes max(ac_norm[lag_min:lag_max]) across musical beat periods "
        "[0.3 s, 1.2 s] (50–200 BPM), clipped to [0.05, 0.95]. Result: 0/20 clamped tracks, mean 0.7413, MAE 0.1717."
    )
    flow.extend([Paragraph(p1, styles["Body"]), Paragraph(p2, styles["Body"]), Paragraph(p3, styles["Body"])])
    return flow


def build_omitted_features_section(styles):
    """Section 4: Why 4 semantic features and genre are kept manual."""
    flow = []
    flow.extend(sec_header("4. Why Four Semantic Features & Genre Remain Manual", styles))
    p1 = (
        "Unlike duration, tempo, or LUFS, <b>valence</b>, <b>acousticness</b>, <b>instrumentalness</b>, and "
        "<b>speechiness</b> have no closed-form physical DSP formula. In Spotify's dataset, those numbers are "
        "probabilities from proprietary Echo Nest deep-learning classifiers trained on millions of labeled songs. "
        "Running open-source deep audio models locally (e.g., Essentia-TensorFlow, Demucs, Silero VAD) would add "
        "1–3 GB of weights, exceed Render's 512 MB free-tier RAM, and introduce calibration mismatch against Spotify's "
        "training distribution. Genre is never inferred from audio. Keeping these 5 inputs strictly tagged 'Manual' "
        "preserves scientific honesty and zero-latency reliability."
    )
    flow.append(Paragraph(p1, styles["Body"]))
    return flow


def build_clamping_section(styles):
    """Section 5: Metadata Boundary Clamping and API Contract."""
    flow = []
    flow.extend(sec_header("5. Metadata Boundary Clamping & Prediction Integration", styles))
    p1 = (
        "Because Pydantic validation in /api/predict rejects out-of-range inputs with HTTP 422, every extracted "
        "estimate passes through clamp_feature(raw_val, bounds) using exact training bounds from "
        "models/final/metadata.json. If raw_val falls outside [min, max], it is clamped and flagged with "
        "clamped: true so the UI displays 'Estimated (clamped)'. Both raw_value and clamped value are preserved:"
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Feature", "Train Min", "Train Max", "Method Tag Returned by /api/extract", "Default"]
    rows = [
        ["duration_sec", "60.678 s", "555.419 s", "audio_length_direct", "215.226"],
        ["tempo", "30.2 BPM", "217.282 BPM", "librosa_onset_autocorrelation", "121.989"],
        ["loudness", "-38.615 dB", "0.91 dB", "pyloudnorm_itu_r_bs1770_lufs", "-6.941"],
        ["energy", "0.002", "1.0", "rms_amplitude_proxy", "0.688"],
        ["danceability", "0.061", "0.98", "autocorrelation_pulse_clarity_proxy", "0.586"],
    ]
    flow.append(make_table(headers, rows, [80, 70, 70, 210, 66], styles, align_right_cols=[1, 2, 4]))
    flow.append(Spacer(1, 6))
    return flow


def build_validation_section(styles):
    """Section 6: Empirical 20-Track Validation & Unit Tests."""
    flow = []
    flow.extend(sec_header("6. Empirical 20-Track Validation Audit & Unit Tests", styles))
    p1 = (
        "Evaluated on 20 held-out test MP3s (reports/audio_validation.csv) against frozen Logistic Regression (\u03c4 = 0.48). "
        "The equivalence gate failed (0/4 criteria passed): 1/9 automatic features (\u2265 6 required), 55.0% label agreement "
        "(\u2265 80% required), mean |\u0394p| = 0.3451 (\u2264 0.15 required), and hit share discrepancy = 45.0 pp (Spotify 65.0% vs. "
        "Extracted 20.0%). Consequently, the UI displays an explicit experimental disclaimer."
    )
    flow.append(Paragraph(p1, styles["Body"]))

    headers = ["Feature", "Spotify Mean", "Extracted Mean", "MAE", "Spearman \u03c1", "p-value", "Clamped", "Gate Status"]
    rows = [
        ["duration_sec", "229.95 s", "240.00 s", "20.21 s", "+0.7188", "0.000356", "0 / 20", "Automatic"],
        ["tempo", "126.24 BPM", "119.03 BPM", "16.95 BPM", "+0.2413", "0.3054", "0 / 20", "Manual"],
        ["energy", "0.6232", "0.8967", "0.2992", "+0.2028", "0.3911", "0 / 20", "Manual"],
        ["loudness", "-7.31 dB", "-14.22 LUFS", "7.34 dB", "+0.1850", "0.4348", "0 / 20", "Manual"],
        ["danceability", "0.6132", "0.7413", "0.1717", "-0.1369", "0.5649", "0 / 20", "Manual"],
    ]
    flow.append(make_table(headers, rows, [72, 66, 72, 58, 62, 56, 48, 62], styles, align_right_cols=[1, 2, 3, 4, 5, 6]))
    flow.append(Spacer(1, 8))
    flow.append(make_callout(
        "Verified via 17 dedicated audio unit/API tests (tests/test_audio_extractor.py & tests/test_audio_api.py): "
        "120 BPM click track (\u00b13 BPM), exact 5.0 s duration, louder copy LUFS ordering, proxy bounds [0, 1], "
        "clamping flags, 20 MB oversize rejection, invalid extension rejection, corrupt MP3 handling, and flag gating.",
        title="AUTOMATED TEST COVERAGE", border_hex="#2B6CB0", bg_hex="#EDF2F7", styles=styles
    ))
    return flow


def generate_audio_guide(output_path="reports/HitSong_Audio_Feature_Extraction_Guide.pdf"):
    """Compile and save the detailed audio feature extraction PDF guide."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=54, rightMargin=54, topMargin=54, bottomMargin=54
    )
    styles = get_report_styles()
    story = []
    story.extend(build_audio_cover(styles))
    story.extend(build_architecture_flow(styles))
    story.extend(build_decoding_slicing(styles))
    story.append(PageBreak())
    story.extend(build_feature_duration_tempo(styles))
    story.extend(build_feature_loudness_energy_dance(styles))
    story.extend(build_omitted_features_section(styles))
    story.append(PageBreak())
    story.extend(build_clamping_section(styles))
    story.extend(build_validation_section(styles))
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Audio Guide PDF successfully generated: {output_path}")


if __name__ == "__main__":
    out_file = sys.argv[1] if len(sys.argv) > 1 else "reports/HitSong_Audio_Feature_Extraction_Guide.pdf"
    generate_audio_guide(out_file)
