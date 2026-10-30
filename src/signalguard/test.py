
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from signalguard.dataset import extract_features
from signalguard.impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "signalguard_rf.joblib"

CLASS_NAMES = [
    "Healthy",
    "AWGN Noise",
    "Rayleigh Fading",
    "Tone Interference",
]

SAMPLES_PER_CLASS = 250
SIGNAL_LENGTH = 1000


def generate_clean_signal(rng):
    # Random BPSK symbols: -1 or +1
    bits = rng.integers(0, 2, size=SIGNAL_LENGTH)
    return (2 * bits - 1).astype(float)


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    rng = np.random.default_rng(2026)
    X_test = []
    y_test = []

    for label in range(4):
        for _ in range(SAMPLES_PER_CLASS):
            clean = generate_clean_signal(rng)

            if label == 0:
                # Healthy signal
                received = clean.copy()

            elif label == 1:
                # Test different AWGN SNR levels
                snr_db = rng.uniform(-5, 20)
                received = add_awgn(
                    clean, snr_db=snr_db, seed=None
                )

            elif label == 2:
                # Fresh Rayleigh fading realization
                received = apply_rayleigh_fading(
                    clean, seed=None
                )

            else:
                # Test varied tone frequencies and amplitudes
                frequency = rng.uniform(0.02, 0.20)
                amplitude = rng.uniform(0.2, 0.8)
                received = add_tone_interference(
                    clean,
                    frequency_cycles_per_sample=frequency,
                    amplitude=amplitude,
                )

            features = extract_features(clean, received)

            X_test.append([
                float(features[name])
                for name in feature_names
            ])
            y_test.append(label)

    y_pred = model.predict(np.asarray(X_test))

    accuracy = accuracy_score(y_test, y_pred)
    matrix = confusion_matrix(
        y_test, y_pred, labels=[0, 1, 2, 3]
    )

    print("\n===== GENERALIZATION TEST =====")
    print(f"Test samples: {len(y_test)}")
    print(f"Accuracy: {accuracy:.2%}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        zero_division=0,
    ))

    print("===== CONFUSION MATRIX =====")
    print("Rows = actual classes")
    print("Columns = predicted classes")
    print(matrix)


if __name__ == "__main__":
    main()
