
from pathlib import Path
import csv
import json

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

DATASET_PATH = ROOT / "data" / "signal_dataset.csv"
MODEL_PATH = ROOT / "models" / "signalguard_rf.joblib"
METRICS_PATH = ROOT / "models" / "evaluation_metrics.json"

CLASS_NAMES = [
    "Healthy",
    "AWGN Noise",
    "Rayleigh Fading",
    "Tone Interference",
]

LABEL_COLUMN = "label"


def main():
    # Load the dataset without Pandas
    with open(DATASET_PATH, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        columns = reader.fieldnames

    if not rows or not columns:
        raise ValueError("Dataset is empty or has no column headers.")

    if LABEL_COLUMN not in columns:
        raise ValueError(
            f"Expected a '{LABEL_COLUMN}' column. "
            f"Found columns: {columns}"
        )

    feature_names = [
        column for column in columns if column != LABEL_COLUMN
    ]

    X = [
        [float(row[name]) for name in feature_names]
        for row in rows
    ]
    y = [int(row[LABEL_COLUMN]) for row in rows]

    # Split before training to keep test data unseen
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # Evaluate on unseen test data
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=[0, 1, 2, 3],
    )

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(report["macro avg"]["precision"]),
        "recall": float(report["macro avg"]["recall"]),
        "f1_score": float(report["macro avg"]["f1-score"]),
        "class_report": report,
        "confusion_matrix": matrix.tolist(),
        "test_samples": len(y_test),
    }

    # Save the model and its evaluation metrics
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    with open(METRICS_PATH, "w") as file:
        json.dump(metrics, file, indent=4)

    # Display results
    print("\n===== SIGNALGUARD MODEL EVALUATION =====")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Accuracy         : {accuracy:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"F1-score         : {metrics['f1_score']:.4f}")

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[0, 1, 2, 3],
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("===== CONFUSION MATRIX =====")
    print("Rows = actual classes; columns = predicted classes")
    print(matrix)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
