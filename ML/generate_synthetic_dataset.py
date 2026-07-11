"""
generate_synthetic_dataset.py
-------------------------------
Creates a SMALL synthetic, labeled feature dataset purely so the rest of
the pipeline (train_model.py, inference_server.py) can be run and tested
end-to-end before you plug in real, PhysioNet-derived data.

DO NOT use this synthetic dataset's accuracy numbers in your project
report — it exists only to prove the pipeline wiring works. Replace
data/sample_features_labeled.csv with features extracted from a real
dataset (see data/DATASET_README.md) before reporting results.
"""

import numpy as np
import pandas as pd

np.random.seed(0)

CLASSES = ["Normal", "PVC", "APB", "AFib", "Other"]
N_PER_CLASS = 150

rows = []
# Rough, illustrative feature distributions per class (NOT clinically validated numbers)
class_profiles = {
    "Normal": dict(hr=(60, 90), sdnn=(30, 60), rmssd=(20, 45), r_amp=(0.9, 1.1)),
    "PVC":    dict(hr=(55, 95), sdnn=(40, 80), rmssd=(35, 70), r_amp=(1.1, 1.5)),
    "APB":    dict(hr=(65, 100), sdnn=(35, 65), rmssd=(25, 55), r_amp=(0.8, 1.05)),
    "AFib":   dict(hr=(90, 160), sdnn=(80, 160), rmssd=(60, 140), r_amp=(0.7, 1.0)),
    "Other":  dict(hr=(50, 130), sdnn=(20, 100), rmssd=(15, 90), r_amp=(0.6, 1.3)),
}

for label in CLASSES:
    prof = class_profiles[label]
    for _ in range(N_PER_CLASS):
        hr = np.random.uniform(*prof["hr"])
        sdnn = np.random.uniform(*prof["sdnn"])
        rmssd = np.random.uniform(*prof["rmssd"])
        r_amp = np.random.uniform(*prof["r_amp"])
        rows.append({
            "heart_rate_bpm": hr,
            "num_beats": int(np.random.uniform(5, 9)),
            "sdnn_ms": sdnn,
            "rmssd_ms": rmssd,
            "mean_r_amplitude": r_amp,
            "std_r_amplitude": np.random.uniform(0.05, 0.2),
            "signal_std": np.random.uniform(0.2, 0.6),
            "label": label,
        })

df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
df.to_csv("../data/sample_features_labeled.csv", index=False)
print(f"Saved {len(df)} synthetic rows to ../data/sample_features_labeled.csv")
