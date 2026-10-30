
from pathlib import Path
from datetime import datetime, timezone
import csv
import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score

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
    "AWGN Noise",
    "Rayleigh Fading",
    "Tone Interference",
]

SAMPLES_PER_CLASS = 250
SIGNAL_LENGTH = 1000


def generate_clean_signal(rng):
    bits = rng.integers(0, 2, size=SIGNAL_LENGTH)
    return (2 * bits - 1).astype(float)


def build_test_data(feature_names, rng):
    X_test = []
    y_test = []

    for label in range(4):
        for _ in range(SAMPLES_PER_CLASS):
            clean = generate_clean_signal(rng)

            if label == 0:
                received = clean.copy()

            elif label == 1:
                snr_db = rng.uniform(-5, 20)
                received = add_awgn(
                    clean, snr_db=snr_db, seed=None
                )

            elif label == 2:
                received = apply_rayleigh_fading(
                    clean, seed=None
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
            X_test.append(
                [float(features[name]) for name in feature_names]
            )
            y_test.append(label)

    return np.asarray(X_test), np.asarray(y_test)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Train first using: py -m signalguard.train"
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # No fixed seed: generate a new evaluation set each run.
    rng = np.random.default_rng()

    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    print("Generating new synthetic test signals...")
    X_test, y_test = build_test_data(feature_names, rng)

    predictions = model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, predictions)

    print(f"Test samples: {len(y_test)}")
    print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    print("Calculating permutation importance...")

    # No fixed permutation seed: shuffle differently each run.
    result = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="accuracy",
        n_repeats=15,
        random_state=None,
        n_jobs=-1,
    )

    order = np.argsort(result.importances_mean)[::-1]

    print("\n===== PERMUTATION IMPORTANCE =====")
    print(f"{'Feature':<23} {'Accuracy drop':>14} {'Std dev':>12}")

    rows = []
    for index in order:
        name = feature_names[index]
        mean_drop = float(result.importances_mean[index])
        std_drop = float(result.importances_std[index])

        print(f"{name:<23} {mean_drop:>14.4f} {std_drop:>12.4f}")
        rows.append([name, mean_drop, std_drop])

    evaluated_at = datetime.now(timezone.utc).isoformat()

    csv_path = RESULTS_DIR / "permutation_importance.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "feature",
            "mean_accuracy_drop",
            "standard_deviation",
        ])
        writer.writerows(rows)

    # Plot the most influential features.
    plot_order = order[:10][::-1]
    names = [feature_names[i] for i in plot_order]
    means = result.importances_mean[plot_order]
    errors = result.importances_std[plot_order]

    plt.figure(figsize=(10, 6))
    plt.barh(names, means, xerr=errors)
    plt.axvline(0, linewidth=1)
    plt.xlabel("Decrease in accuracy after shuffling")
    plt.title(
        f"Permutation Importance | Accuracy: {baseline_accuracy:.1%}"
    )
    plt.tight_layout()

    plot_path = RESULTS_DIR / "permutation_importance.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    metadata_path = RESULTS_DIR / "permutation_evaluation.json"
    metadata_path.write_text(
        json.dumps(
            {
                "evaluated_at_utc": evaluated_at,
                "test_samples": int(len(y_test)),
                "baseline_accuracy": float(baseline_accuracy),
                "samples_per_class": SAMPLES_PER_CLASS,
                "permutation_repeats": 15,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\nUpdated CSV: {csv_path}")
    print(f"Updated graph: {plot_path}")
    print(f"Evaluation metadata: {metadata_path}")
    print(f"Evaluation time (UTC): {evaluated_at}")
    print(
        "\nNote: results may still be similar if the model "
        "classifies the synthetic test signals consistently."
    )


if __name__ == "__main__":
    main()
