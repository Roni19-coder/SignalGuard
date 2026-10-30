
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from .dataset import extract_features
from .impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "signalguard_rf.joblib"
RESULTS_DIR = PROJECT_ROOT / "results"

SNR_LEVELS = [-10, -5, 0, 5, 10, 15, 20]
SAMPLES_PER_SNR = 250
SIGNAL_LENGTH = 1000

CLASS_NAMES = [
    "Healthy",
    "AWGN Noise",
    "Rayleigh Fading",
    "Tone Interference",
]


def make_signal(rng):
    bits = rng.integers(0, 2, SIGNAL_LENGTH)
    return (2 * bits - 1).astype(float)


def get_features(clean, received, feature_names):
    features = extract_features(clean, received)
    return [float(features[name]) for name in feature_names]


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first: py -m signalguard.train"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    # Fresh random generator for each execution.
    rng = np.random.default_rng()

    results = []
    all_true = []
    all_pred = []

    print("\n===== SIGNALGUARD AWGN ROBUSTNESS =====")
    print(f"Samples per SNR: {SAMPLES_PER_SNR}\n")

    for snr_db in SNR_LEVELS:
        X = []

        for _ in range(SAMPLES_PER_SNR):
            clean = make_signal(rng)
            received = add_awgn(clean, snr_db=snr_db, seed=None)
            X.append(get_features(clean, received, feature_names))

        predictions = model.predict(np.asarray(X, dtype=float))
        correct_awgn = int(np.sum(predictions == 1))
        detection_rate = 100 * correct_awgn / SAMPLES_PER_SNR

        results.append({
            "snr_db": snr_db,
            "samples": SAMPLES_PER_SNR,
            "correctly_detected_as_awgn": correct_awgn,
            "awgn_detection_rate_percent": detection_rate,
        })

        all_true.extend([1] * SAMPLES_PER_SNR)
        all_pred.extend(predictions.astype(int).tolist())

        print(
            f"SNR {snr_db:>3} dB: "
            f"AWGN detection = {detection_rate:6.2f}% "
            f"({correct_awgn}/{SAMPLES_PER_SNR})"
        )

    csv_path = RESULTS_DIR / "awgn_robustness.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    snrs = [r["snr_db"] for r in results]
    rates = [r["awgn_detection_rate_percent"] for r in results]

    plt.figure(figsize=(9, 5))
    plt.plot(snrs, rates, marker="o")
    plt.xlabel("AWGN SNR (dB)")
    plt.ylabel("Correct AWGN Detection Rate (%)")
    plt.title("SignalGuard: AWGN Robustness")
    plt.xticks(snrs)
    plt.ylim(0, 105)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "awgn_robustness.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    # This matrix is specifically for AWGN samples across the SNR sweep.
    cm = confusion_matrix(all_true, all_pred, labels=[0, 1, 2, 3])
    print("\nConfusion matrix for AWGN-only samples:")
    print(cm)

    evaluated_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "evaluated_at_utc": evaluated_at,
        "samples_per_snr": SAMPLES_PER_SNR,
        "snr_levels_db": SNR_LEVELS,
        "total_awgn_samples": len(all_true),
        "overall_awgn_detection_rate_percent": float(
            100 * np.mean(np.asarray(all_pred) == 1)
        ),
        "note": (
            "AWGN-only evaluation. This is not four-class overall accuracy."
        ),
    }

    metadata_path = RESULTS_DIR / "awgn_robustness_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"\nCSV: {csv_path}")
    print(f"Graph: {plot_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Evaluation time (UTC): {evaluated_at}")


if __name__ == "__main__":
    main()
