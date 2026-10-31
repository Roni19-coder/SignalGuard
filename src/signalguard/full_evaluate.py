
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

# Additional robustness experiment
SNR_LEVELS_DB = [-5, 0, 5, 10, 15, 20]
ROBUSTNESS_SAMPLES_PER_CLASS = 200


def make_test_sample(label, rng, snr_db=None):
    """Generate one synthetic signal and extract its features."""
    bits = rng.integers(0, 2, 1000)
    clean = (2 * bits - 1).astype(float)

    if label == 0:
        received = clean

    elif label == 1:
        current_snr = (
            rng.uniform(-5, 20) if snr_db is None else snr_db
        )
        received = add_awgn(
            clean, snr_db=current_snr, seed=None
        )

    elif label == 2:
        faded = apply_rayleigh_fading(clean, seed=None)

        current_snr = (
            rng.uniform(5, 25) if snr_db is None else snr_db
        )
        received = add_awgn(
            faded, snr_db=current_snr, seed=None
        )

    else:
        frequency = rng.uniform(0.02, 0.20)
        amplitude = rng.uniform(0.2, 0.8)

        received = add_tone_interference(
            clean,
            frequency_cycles_per_sample=frequency,
            amplitude=amplitude,
        )

    return extract_features(clean, received)


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train it first using: py -m signalguard.train"
        )

    saved_model = joblib.load(MODEL_PATH)

    if isinstance(saved_model, dict):
        model = saved_model["model"]
        feature_names = saved_model.get("feature_names")
    else:
        model = saved_model
        feature_names = None

    if not feature_names:
        raise ValueError(
            "Feature names were not found in the saved model. "
            "Check the structure saved by train.py."
        )

    return model, feature_names


def build_dataset(rng, feature_names, samples_per_class,
                  snr_db=None):
    """Build a balanced four-class dataset."""
    X = []
    y = []

    for label in range(len(CLASS_NAMES)):
        for _ in range(samples_per_class):
            features = make_test_sample(
                label, rng, snr_db=snr_db
            )

            X.append([
                features[name] for name in feature_names
            ])
            y.append(label)

    return np.asarray(X, dtype=float), np.asarray(y, dtype=int)


def evaluate_full(model, feature_names, rng):
    """Run the original randomized four-class evaluation."""
    print("\nGenerating fresh four-class test data...")

    X, y = build_dataset(
        rng, feature_names, SAMPLES_PER_CLASS
    )

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

    with (
        RESULTS_DIR / "full_evaluation_confusion_matrix.csv"
    ).open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Actual / Predicted"] + CLASS_NAMES)

        for name, row in zip(CLASS_NAMES, cm):
            writer.writerow([name] + row.tolist())

    summary = {
        "evaluation_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_type": "randomized_four_class",
        "samples_per_class": SAMPLES_PER_CLASS,
        "total_samples": int(len(y)),
        "class_names": CLASS_NAMES,
        "accuracy": float(accuracy),
        "macro_precision": float(
            report["macro avg"]["precision"]
        ),
        "macro_recall": float(
            report["macro avg"]["recall"]
        ),
        "macro_f1": float(
            report["macro avg"]["f1-score"]
        ),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "note": (
            "Fresh synthetic test signals. Healthy signals are "
            "clean; AWGN and Rayleigh noise levels are randomized. "
            "Results measure simulated, not real-world, performance."
        ),
    }

    with (RESULTS_DIR / "full_evaluation.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nSaved original evaluation:")
    print(RESULTS_DIR / "full_evaluation.json")
    print(RESULTS_DIR / "full_evaluation_confusion_matrix.csv")


def evaluate_snr_robustness(model, feature_names, rng):
    """Measure classification performance at fixed SNR levels."""
    print("\nStarting SNR robustness evaluation...")

    results = []

    for snr_db in SNR_LEVELS_DB:
        print(f"\nEvaluating at SNR = {snr_db} dB")

        X, y = build_dataset(
            rng,
            feature_names,
            ROBUSTNESS_SAMPLES_PER_CLASS,
            snr_db=snr_db,
        )

        predictions = model.predict(X)

        report = classification_report(
            y,
            predictions,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        accuracy = accuracy_score(y, predictions)

        row = {
            "snr_db": snr_db,
            "accuracy": float(accuracy),
            "macro_precision": float(
                report["macro avg"]["precision"]
            ),
            "macro_recall": float(
                report["macro avg"]["recall"]
            ),
            "macro_f1": float(
                report["macro avg"]["f1-score"]
            ),
        }

        # Recall for each class is its per-class accuracy.
        for class_name in CLASS_NAMES:
            row[f"{class_name}_recall"] = float(
                report[class_name]["recall"]
            )

        results.append(row)

        print(f"Accuracy: {accuracy:.2%}")
        print(f"Macro F1: {row['macro_f1']:.4f}")

    # Save results as CSV for graphing and dashboard integration.
    csv_path = RESULTS_DIR / "snr_robustness.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(results[0].keys())
        )
        writer.writeheader()
        writer.writerows(results)

    # Save structured results as JSON.
    summary = {
        "evaluation_time_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "evaluation_type": "fixed_snr_robustness",
        "snr_levels_db": SNR_LEVELS_DB,
        "samples_per_class_per_snr": ROBUSTNESS_SAMPLES_PER_CLASS,
        "class_names": CLASS_NAMES,
        "results": results,
        "note": (
            "AWGN and Rayleigh classes use the specified SNR. "
            "Healthy and tone-interference classes are unchanged "
            "by the SNR setting. Each SNR uses fresh synthetic data."
        ),
    }

    json_path = RESULTS_DIR / "snr_robustness.json"

    with json_path.open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nSNR robustness summary:")
    print(
        f"{'SNR (dB)':>10} {'Accuracy':>12} {'Macro F1':>12}"
    )

    for row in results:
        print(
            f"{row['snr_db']:>10} "
            f"{row['accuracy']:>11.2%} "
            f"{row['macro_f1']:>12.4f}"
        )

    print("\nSaved robustness results:")
    print(csv_path)
    print(json_path)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model, feature_names = load_model()

    # Unseeded generator gives fresh data on each run.
    rng = np.random.default_rng()

    evaluate_full(model, feature_names, rng)

    evaluate_snr_robustness(
        model, feature_names, rng
    )

    print("\nAll evaluations completed successfully.")


if __name__ == "__main__":
    main()
