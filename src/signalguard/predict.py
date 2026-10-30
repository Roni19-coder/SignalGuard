
import csv
from pathlib import Path

import joblib
import numpy as np

from .impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)
from .metrics import calculate_snr_db
from .dataset import extract_features


MODEL_PATH = Path("models/signalguard_rf.joblib")

LABEL_NAMES = {
    0: "Healthy",
    1: "AWGN Noise",
    2: "Rayleigh Fading",
    3: "Tone Interference",
}


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model not found. Run py -m signalguard.train first."
        )

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    rng = np.random.default_rng()
    bits = rng.integers(0, 2, 1000)
    clean = 2 * bits - 1

    # Change this value to test another condition:
    condition = "rayleigh"

    if condition == "healthy":
        received = clean.copy()
    elif condition == "awgn":
        received = add_awgn(clean, snr_db=5, seed=None)
    elif condition == "rayleigh":
        received = apply_rayleigh_fading(clean, seed=None)
    elif condition == "tone":
        received = add_tone_interference(
            clean,
            frequency_cycles_per_sample=0.05,
            amplitude=0.5,
        )
    else:
        raise ValueError("Unknown condition")

    features = extract_features(clean, received)
    X = np.array([[features[name] for name in feature_names]], dtype=float)

    prediction = int(model.predict(X)[0])
    probabilities = model.predict_proba(X)[0]

    print("Actual condition:", condition)
    print("Predicted condition:", LABEL_NAMES[prediction])
    print("\nClass probabilities:")

    for label, probability in zip(model.classes_, probabilities):
        print(f"{LABEL_NAMES[int(label)]}: {probability:.2%}")


if __name__ == "__main__":
    main()
