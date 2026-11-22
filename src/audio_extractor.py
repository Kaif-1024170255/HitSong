"""Audio feature extraction module for local MP3 audio files.

Extracts Tier 1 audio features (duration, tempo, loudness via pyloudnorm)
and documented proxies (energy, danceability) without paid or cloud APIs.
Audio is processed from temporary files and never permanently stored.
"""

from typing import Any
import numpy as np
import soundfile as sf
import librosa
import pyloudnorm as pyln


def load_audio(file_path: str, target_sr: int = 22050) -> tuple[np.ndarray, int]:
    """Load audio file into mono float32 array; raise ValueError on corruption."""
    try:
        data, sr = sf.read(file_path, dtype="float32")
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        if sr != target_sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        if len(data) == 0:
            raise ValueError("Audio file contains zero audio samples.")
        return data, sr
    except Exception as sf_err:
        try:
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            if len(y) == 0:
                raise ValueError("Audio file contains zero audio samples.")
            return y, sr
        except Exception as lib_err:
            raise ValueError(f"Corrupt or unreadable audio file: {sf_err or lib_err}")


def get_middle_segment(y: np.ndarray, sr: int, seg_sec: float = 45.0) -> np.ndarray:
    """Extract a representative 30-60 second segment from the middle of the track."""
    total_sec = len(y) / sr
    if total_sec <= seg_sec:
        return y
    start_sec = max(0.0, (total_sec - seg_sec) / 2.0)
    end_sec = start_sec + seg_sec
    return y[int(start_sec * sr) : int(end_sec * sr)]


def extract_duration(y: np.ndarray, sr: int) -> float:
    """Calculate exact duration in seconds from full audio file."""
    return float(len(y) / sr)


def extract_tempo(y: np.ndarray, sr: int) -> float:
    """Estimate tempo (BPM) via beat tracking, folding octave into [50, 180]."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo_arr = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
    tempo_val = float(tempo_arr[0]) if hasattr(tempo_arr, "__len__") else float(tempo_arr)

    # Fold half / double tempo into standard commercial range [50, 180]
    while tempo_val < 55.0 and (tempo_val * 2.0) <= 217.0:
        tempo_val *= 2.0
    while tempo_val > 185.0 and (tempo_val / 2.0) >= 30.2:
        tempo_val /= 2.0

    return round(tempo_val, 2)


def extract_loudness(y: np.ndarray, sr: int) -> float:
    """Measure integrated loudness in LUFS using ITU-R BS.1770-4.

    Note: Distinct from Spotify's proprietary echo-nest loudness feature,
    standardized ITU-R BS.1770-4 LUFS acts as an open physical loudness metric.
    """
    meter = pyln.Meter(sr)
    lufs = meter.integrated_loudness(y)
    if np.isneginf(lufs) or np.isnan(lufs):
        return -70.0
    return round(float(lufs), 2)


def estimate_energy_proxy(y: np.ndarray) -> float:
    """Estimate perceptual energy proxy from normalized frame-wise RMS amplitude.

    Formula:
        RMS = sqrt(mean(y[n]^2))
        energy_proxy = clip(RMS / 0.20, 0.0, 1.0)
    Mastered tracks typically have RMS in [0.05, 0.25].
    """
    rms = float(np.sqrt(np.mean(y**2)))
    score = min(max(rms / 0.20, 0.0), 1.0)
    return round(score, 3)


def estimate_danceability_proxy(y: np.ndarray, sr: int) -> float:
    """Estimate danceability proxy from rhythmic regularity and pulse clarity.

    Formula:
        Onset envelope autocorrelation peak prominence across musical beat
        lags [0.3s, 1.2s] (corresponding to 50-200 BPM).
    """
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    if len(onset_env) == 0 or np.all(onset_env == 0):
        return 0.50
    max_lag = int(sr * 1.5 / 512)
    ac = librosa.autocorrelate(onset_env, max_size=max_lag)
    if len(ac) < 20 or ac[0] <= 0:
        return 0.50
    ac_norm = ac / ac[0]
    lag_min = int(0.3 * sr / 512)
    lag_max = min(len(ac_norm) - 1, int(1.2 * sr / 512))
    if lag_min >= lag_max:
        return 0.50
    peak = float(np.max(ac_norm[lag_min:lag_max]))
    return round(float(np.clip(peak, 0.05, 0.95)), 3)


def clamp_feature(val: float, bounds: dict[str, Any]) -> tuple[float, bool]:
    """Clamp estimated feature value to training metadata bounds."""
    b_min = float(bounds.get("min", val))
    b_max = float(bounds.get("max", val))
    clamped = val < b_min or val > b_max
    clamped_val = min(max(val, b_min), b_max)
    return round(clamped_val, 3), clamped


def extract_features(file_path: str, bounds_meta: dict[str, Any] | None = None) -> dict[str, Any]:
    """Extract Tier 1 audio features and proxies, clamping to metadata bounds if provided."""
    bounds = bounds_meta or {}
    y_full, sr = load_audio(file_path)
    y_mid = get_middle_segment(y_full, sr, seg_sec=45.0)

    raw_estimates = {
        "duration_sec": (
            extract_duration(y_full, sr),
            "audio_length_direct",
            "Exact audio sample duration in seconds (from full track).",
        ),
        "tempo": (
            extract_tempo(y_mid, sr),
            "librosa_onset_autocorrelation",
            "Tempo estimated via onset envelope autocorrelation with octave folding.",
        ),
        "loudness": (
            extract_loudness(y_mid, sr),
            "pyloudnorm_itu_r_bs1770_lufs",
            "Integrated loudness in LUFS (ITU-R BS.1770-4; open proxy for Spotify loudness).",
        ),
        "energy": (
            estimate_energy_proxy(y_mid),
            "rms_amplitude_proxy",
            "Documented proxy based on normalized frame-wise RMS amplitude.",
        ),
        "danceability": (
            estimate_danceability_proxy(y_mid, sr),
            "autocorrelation_pulse_clarity_proxy",
            "Documented proxy based on onset beat strength autocorrelation prominence.",
        ),
    }

    result = {}
    for feat, (raw_val, method, desc) in raw_estimates.items():
        fbounds = bounds.get(feat, {})
        val, is_clamped = clamp_feature(raw_val, fbounds)
        result[feat] = {
            "value": val,
            "raw_value": raw_val,
            "clamped": is_clamped,
            "method": method,
            "description": desc,
            "bounds": {"min": fbounds.get("min"), "max": fbounds.get("max")},
        }

    return result


# Backward compatible alias
extract_audio_features = extract_features


if __name__ == "__main__":
    import json
    with open("models/final/metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    print("Audio Extractor Module Loaded. Supported Tier 1 features:")
    for k in ["duration_sec", "tempo", "loudness", "energy", "danceability"]:
        print(f"  - {k}: bounds {meta.get('per_feature_min_max', {}).get(k)}")
