"""
train_model.py
----------------
Trains the ECG disease-indication classifier used by CardioSight.

Default: Random Forest on hand-crafted features (heart rate, HRV, R-peak
morphology) extracted via feature_extraction.py. This is a strong,
lightweight baseline that is easy to explain in a college project report
and cheap enough to run inference with on a laptop feeding results back
to the microcontroller in real time.

Alternative (see "Future Scope" in README): a 1D-CNN or LSTM trained
directly on raw/filtered beat windows can capture morphology the
hand-crafted features miss, at the cost of needing more data + compute.

USAGE:
    1. (Recommended) Build data/features_labeled.csv from a real dataset
       such as MIT-BIH Arrhythmia Database (see data/DATASET_README.md)
    2. Or, for a quick pipeline test: run generate_synthetic_dataset.py
       first to create data/sample_features_labeled.csv
    3. python3 train_model.py --data ../data/sample_features_labeled.csv
"""

import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "heart_rate_bpm", "num_beats", "sdnn_ms", "rmssd_ms",
    "mean_r_amplitude", "std_r_amplitude", "signal_std",
]


def main(data_path: str, model_out: str):
    df = pd.read_csv(data_path)

    X = df[FEATURE_COLUMNS].values
    y_raw = df["label"].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, class_weight="balanced"
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\nTest Accuracy: {acc*100:.2f}%\n")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    print("Confusion Matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump({"model": clf, "label_encoder": le, "feature_columns": FEATURE_COLUMNS}, model_out)
    print(f"\nSaved trained model bundle to {model_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="../data/sample_features_labeled.csv",
                         help="Path to labeled feature CSV")
    parser.add_argument("--out", default="../models/cardiosight_model.joblib",
                         help="Output path for trained model bundle")
    args = parser.parse_args()
    main(args.data, args.out)
