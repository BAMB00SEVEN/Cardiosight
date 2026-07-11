"""
feature_extraction.py
----------------------
Extracts heart-rate and beat-level features from a filtered ECG signal.
These features feed the ML classifier that flags disease indications
(arrhythmia types, tachycardia/bradycardia, irregular rhythm, etc.)

Techniques used:
  - R-peak detection (simplified Pan-Tompkins style, via scipy.find_peaks)
  - RR-interval based heart rate calculation
  - Heart Rate Variability (HRV) features: SDNN, RMSSD
  - Per-beat morphological features (amplitude, QRS width proxy)
"""

import numpy as np
from scipy.signal import find_peaks


def detect_r_peaks(signal: np.ndarray, fs: float, min_rr_sec: float = 0.3):
    """
    Detect R-peaks on a filtered ECG signal.
    min_rr_sec limits detection to physiologically plausible heart rates
    (0.3s min RR interval -> caps at 200 bpm).
    """
    distance = int(min_rr_sec * fs)
    height = np.mean(signal) + 0.6 * np.std(signal)
    peaks, props = find_peaks(signal, distance=distance, height=height)
    return peaks, props


def compute_heart_rate(r_peaks: np.ndarray, fs: float) -> float:
    """Average heart rate (bpm) from R-peak indices."""
    if len(r_peaks) < 2:
        return 0.0
    rr_intervals = np.diff(r_peaks) / fs  # seconds
    mean_rr = np.mean(rr_intervals)
    return 60.0 / mean_rr if mean_rr > 0 else 0.0


def compute_hrv_features(r_peaks: np.ndarray, fs: float) -> dict:
    """
    Basic time-domain HRV features:
      SDNN  - standard deviation of NN (RR) intervals -> overall variability
      RMSSD - root mean square of successive RR differences -> short-term variability
    """
    if len(r_peaks) < 3:
        return {"sdnn_ms": 0.0, "rmssd_ms": 0.0}

    rr_ms = np.diff(r_peaks) / fs * 1000.0
    sdnn = np.std(rr_ms)
    rmssd = np.sqrt(np.mean(np.diff(rr_ms) ** 2))
    return {"sdnn_ms": float(sdnn), "rmssd_ms": float(rmssd)}


def extract_beat_window(signal: np.ndarray, r_peak_idx: int, fs: float,
                         pre_sec: float = 0.25, post_sec: float = 0.4) -> np.ndarray:
    """Extract a single-beat window centered on an R-peak (for per-beat classification)."""
    pre = int(pre_sec * fs)
    post = int(post_sec * fs)
    start = max(0, r_peak_idx - pre)
    end = min(len(signal), r_peak_idx + post)
    return signal[start:end]


def extract_feature_vector(signal: np.ndarray, fs: float) -> dict:
    """
    Build the full feature dictionary for one ECG segment.
    This is what gets passed into the trained classifier in train_model.py
    / inference_server.py.
    """
    r_peaks, _ = detect_r_peaks(signal, fs)
    heart_rate = compute_heart_rate(r_peaks, fs)
    hrv = compute_hrv_features(r_peaks, fs)

    if len(r_peaks) > 0:
        r_amplitudes = signal[r_peaks]
        mean_r_amp = float(np.mean(r_amplitudes))
        std_r_amp = float(np.std(r_amplitudes))
    else:
        mean_r_amp, std_r_amp = 0.0, 0.0

    features = {
        "heart_rate_bpm": heart_rate,
        "num_beats": len(r_peaks),
        "sdnn_ms": hrv["sdnn_ms"],
        "rmssd_ms": hrv["rmssd_ms"],
        "mean_r_amplitude": mean_r_amp,
        "std_r_amplitude": std_r_amp,
        "signal_std": float(np.std(signal)),
    }
    return features


if __name__ == "__main__":
    fs = 500
    t = np.linspace(0, 6, fs * 6)
    synthetic = np.sin(2 * np.pi * 1.25 * t) + 0.1 * np.random.randn(len(t))
    feats = extract_feature_vector(synthetic, fs)
    print("Extracted features:", feats)
