
import csv
import json
from datetime import datetime, timezone
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
RESULTS_DIR = ROOT / "results"

CLASS_NAMES = [
    "Healthy",
    "AWGN",
    "Rayleigh fading",
    "Tone interference",
]

SAMPLES_PER_CLASS = 500


def make_test_sample(label, rng):
    # Use a fresh random BPSK signal for each sample.
    bits = rng.integers(0, 2, 1000)
    clean = (2 * bits - 1).astype(float)

    if label == 0:
        received = clean

    elif label == 1:
        snr_db = rng.uniform(-5, 20)
        received = add_awgn(clean, snr_db=snr_db, seed=None)

    elif label == 2:
        faded = apply_rayleigh_fading(clean, seed=None)
        received = add_awgn(
            faded, snr_db=rng.uniform(5, 25), seed=None
        )

    else:
        frequency = rng.uniform(0.02, 0.20)
        amplitude = rng.uniform(0.2, 0.8)
        received = add_tone_interference(
            clean,
            frequency_cycles_per_sample=frequency,
            amplitude=amplitude,
        )

    features = extract_features(clean, received)
    return features


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train the model first using: py -m signalguard.train"
        )

    saved_model = joblib.load(MODEL_PATH)

    # The training script saves a dictionary containing the model.
    model = (
        saved_model["model"]
        if isinstance(saved_model, dict)
        else saved_model
    )

    feature_names = (
        saved_model.get("feature_names")
        if isinstance(saved_model, dict)
        else None
    )

    if not feature_names:
        raise ValueError(
            "Feature names were not found in the saved model. "
            "Check the structure saved by train.py."
        )

    rng = np.random.default_rng()

    X = []
    y = []

    print("Generating fresh four-class test data...")

    for label in range(len(CLASS_NAMES)):
        for _ in range(SAMPLES_PER_CLASS):
            features = make_test_sample(label, rng)
            X.append([features[name] for name in feature_names])
            y.append(label)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)

    print(f"Total test samples: {len(y)}")
    print("Running predictions...")

    predictions = model.predict(X)

    labels = list(range(len(CLASS_NAMES)))
    cm = confusion_matrix(y, predictions, labels=labels)
    accuracy = accuracy_score(y, predictions)

    report = classification_report(
        y,
        predictions,
        labels=labels,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    print(f"\nOverall accuracy: {accuracy:.2%}")
    print("\nClassification report:")
    print(
        classification_report(
            y,
            predictions,
            labels=labels,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix (rows = actual, columns = predicted):")
    print("Class order:", CLASS_NAMES)
    print(cm)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "full_evaluation_confusion_matrix.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + CLASS_NAMES)

        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name] + row.tolist())

    summary = {
        "evaluation_time_utc": datetime.now(timezone.utc).isoformat(),
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_samples": int(len(y)),
        "class_names": CLASS_NAMES,
        "accuracy": float(accuracy),
        "macro_precision": float(report["macro avg"]["precision"]),
        "macro_recall": float(report["macro avg"]["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "note": (
            "Fresh synthetic test signals. Results measure performance "
            "on this simulation distribution, not real-world signals."
        ),
    }

    with (RESULTS_DIR / "full_evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nSaved results:")
    print(RESULTS_DIR / "full_evaluation.json")
    print(RESULTS_DIR / "full_evaluation_confusion_matrix.csv")


if __name__ == "__main__":
    main()
