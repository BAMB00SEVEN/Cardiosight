"""
inference_server.py
---------------------
Bridges the ESP32 (raw ECG acquisition) and the trained ML model.

Flow:
  1. Receive a rolling buffer of raw ECG samples from the ESP32
     (via Serial USB, or WiFi HTTP POST — pick one mode below)
  2. Filter the buffer with filtering.clean_ecg()
  3. Extract features with feature_extraction.extract_feature_vector()
  4. Predict a class with the trained RandomForest model
  5. Send the result back to the ESP32 so it can update the OLED/buzzer

Two transport modes are provided — use whichever matches your firmware:

  MODE "serial": ESP32 streams raw samples over USB serial as
                 comma-separated floats, one buffer's worth per line.
                 Prediction is written back over the same serial port.

  MODE "wifi":   ESP32 POSTs a JSON buffer to this machine's Flask
                 endpoint; response JSON contains the prediction.
                 (Requires `pip install flask`)

Run:
    python3 inference_server.py --mode serial --port /dev/ttyUSB0
    python3 inference_server.py --mode wifi
"""

import argparse
import json
import time

import joblib
import numpy as np

from filtering import clean_ecg
from feature_extraction import extract_feature_vector

FS = 500  # must match the ESP32 sampling rate
MODEL_PATH = "../models/cardiosight_model.joblib"


def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["model"], bundle["label_encoder"], bundle["feature_columns"]


def predict_from_buffer(raw_samples: np.ndarray, model, label_encoder, feature_columns):
    filtered = clean_ecg(raw_samples, fs=FS)
    feats = extract_feature_vector(filtered, FS)
    x = np.array([[feats[col] for col in feature_columns]])
    pred_idx = model.predict(x)[0]
    pred_label = label_encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(model.predict_proba(x)))
    return {
        "label": pred_label,
        "confidence": round(confidence, 3),
        "heart_rate_bpm": round(feats["heart_rate_bpm"], 1),
    }


def run_serial(port: str, baud: int = 115200):
    import serial  # pyserial

    model, le, cols = load_model()
    ser = serial.Serial(port, baud, timeout=2)
    print(f"Listening on {port} @ {baud} baud...")

    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            raw_samples = np.array([float(v) for v in line.split(",")])
        except ValueError:
            continue  # skip malformed lines

        if len(raw_samples) < FS * 2:  # need at least ~2s of data for reliable R-peak detection
            continue

        result = predict_from_buffer(raw_samples, model, le, cols)
        print(result)
        ser.write((json.dumps(result) + "\n").encode())


def run_wifi(host: str = "0.0.0.0", port: int = 5000):
    from flask import Flask, request, jsonify

    model, le, cols = load_model()
    app = Flask(__name__)

    @app.route("/predict", methods=["POST"])
    def predict():
        payload = request.get_json(force=True)
        raw_samples = np.array(payload["samples"], dtype=float)
        if len(raw_samples) < FS * 2:
            return jsonify({"error": "need at least 2s of samples"}), 400
        result = predict_from_buffer(raw_samples, model, le, cols)
        return jsonify(result)

    print(f"CardioSight inference server running on http://{host}:{port}/predict")
    app.run(host=host, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["serial", "wifi"], default="wifi")
    parser.add_argument("--port", default="/dev/ttyUSB0", help="Serial port (serial mode only)")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--http_port", type=int, default=5000)
    args = parser.parse_args()

    if args.mode == "serial":
        run_serial(args.port, args.baud)
    else:
        run_wifi(args.host, args.http_port)
