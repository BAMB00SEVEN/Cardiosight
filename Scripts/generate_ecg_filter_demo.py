import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch

np.random.seed(42)

fs = 500  # Hz sampling rate (typical for AD8232 style acquisition)
duration = 6  # seconds
t = np.linspace(0, duration, int(fs*duration), endpoint=False)

def gaussian(t, center, amp, width):
    return amp * np.exp(-((t - center) ** 2) / (2 * width ** 2))

def synth_beat(t0):
    # P, Q, R, S, T waves as sum of gaussians, roughly physiological proportions
    p = gaussian(t, t0 - 0.20, 0.10, 0.015)
    q = gaussian(t, t0 - 0.045, -0.10, 0.006)
    r = gaussian(t, t0, 1.0, 0.008)
    s = gaussian(t, t0 + 0.045, -0.20, 0.008)
    tw = gaussian(t, t0 + 0.28, 0.20, 0.035)
    return p + q + r + s + tw

hr_bpm = 75
rr = 60.0 / hr_bpm
beat_times = np.arange(0.3, duration, rr)

clean = np.zeros_like(t)
for bt in beat_times:
    clean += synth_beat(bt)

# --- Add realistic noise sources ---
baseline_wander = 0.15 * np.sin(2 * np.pi * 0.3 * t)          # respiration-linked drift
powerline = 0.08 * np.sin(2 * np.pi * 50 * t)                  # 50Hz mains interference
muscle_noise = 0.04 * np.random.randn(len(t))                  # EMG/high-freq noise
raw = clean + baseline_wander + powerline + muscle_noise

# --- Filtering pipeline: bandpass (0.5-40Hz) + 50Hz notch ---
def bandpass_filter(sig, low=0.5, high=40.0, fs=500, order=4):
    nyq = fs / 2
    b, a = butter(order, [low/nyq, high/nyq], btype="band")
    return filtfilt(b, a, sig)

def notch_filter(sig, freq=50.0, fs=500, q=30):
    nyq = fs / 2
    b, a = iirnotch(freq/nyq, q)
    return filtfilt(b, a, sig)

stage1 = bandpass_filter(raw)
filtered = notch_filter(stage1)

fig, axes = plt.subplots(2, 1, figsize=(11, 7), dpi=150, sharex=True)

axes[0].plot(t, raw, color="#d1495b", linewidth=1)
axes[0].set_title("Raw ECG Signal (with baseline wander + 50Hz mains + muscle noise)", fontsize=11.5, fontweight="bold")
axes[0].set_ylabel("Amplitude (mV)")
axes[0].grid(alpha=0.3)

axes[1].plot(t, filtered, color="#2a9d8f", linewidth=1.2)
axes[1].set_title("Filtered ECG Signal (0.5-40Hz Butterworth Bandpass + 50Hz Notch)", fontsize=11.5, fontweight="bold")
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("Amplitude (mV)")
axes[1].grid(alpha=0.3)

fig.suptitle("CardioSight — Signal Filtering Stage (Sample/Synthetic Demo)", fontsize=13.5, fontweight="bold")
fig.text(0.5, -0.01, "Synthetic demo signal for illustration — replace with your own AD8232/ADS1292R logged data",
          ha="center", fontsize=8.5, style="italic", color="#555555")

plt.tight_layout()
plt.savefig("/home/claude/CardioSight/assets/raw_vs_filtered_ecg.png", bbox_inches="tight", facecolor="white")
print("saved raw_vs_filtered_ecg.png")
