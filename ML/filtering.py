"""
filtering.py
------------
Signal-conditioning stage of the CardioSight pipeline.

Raw ECG captured via AD8232 / ADS1292R is contaminated with:
  - Baseline wander (respiration, electrode motion)   -> low frequency drift
  - Powerline interference (50/60 Hz mains)             -> narrowband spike
  - Muscle (EMG) noise                                   -> high frequency

This module applies a Butterworth bandpass filter followed by a notch
filter to produce a clean signal before feature extraction / ML inference.
This filtering step is what the project report should point to when
describing "the filtering algorithm used to get the best accuracy of
signals" before disease-indication classification.
"""

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, medfilt


def bandpass_filter(signal: np.ndarray, fs: float, low: float = 0.5,
                     high: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter (removes baseline wander + high-freq noise)."""
    nyq = fs / 2.0
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def notch_filter(signal: np.ndarray, fs: float, freq: float = 50.0, q: float = 30.0) -> np.ndarray:
    """Notch filter to remove mains interference (use freq=60.0 for 60Hz regions)."""
    nyq = fs / 2.0
    b, a = iirnotch(freq / nyq, q)
    return filtfilt(b, a, signal)


def remove_baseline_wander(signal: np.ndarray, kernel_size: int = 201) -> np.ndarray:
    """Optional extra baseline removal using a median filter (kernel_size must be odd)."""
    if kernel_size % 2 == 0:
        kernel_size += 1
    baseline = medfilt(signal, kernel_size=kernel_size)
    return signal - baseline


def normalize(signal: np.ndarray) -> np.ndarray:
    """Z-score normalize a signal segment (recommended before ML feature extraction)."""
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-8)


def clean_ecg(raw_signal: np.ndarray, fs: float = 500.0, mains_freq: float = 50.0) -> np.ndarray:
    """
    Full filtering pipeline used across the project:
    raw -> bandpass (0.5-40Hz) -> notch (mains freq) -> normalized clean signal
    """
    stage1 = bandpass_filter(raw_signal, fs=fs)
    stage2 = notch_filter(stage1, fs=fs, freq=mains_freq)
    return normalize(stage2)


if __name__ == "__main__":
    # Quick self-test with synthetic noisy sine to confirm filters run without error
    fs = 500
    t = np.linspace(0, 4, fs * 4)
    test_signal = np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 50 * t) + 0.05 * np.random.randn(len(t))
    cleaned = clean_ecg(test_signal, fs=fs)
    print("Filtering self-test OK. Output shape:", cleaned.shape)
