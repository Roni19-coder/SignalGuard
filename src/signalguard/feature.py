
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "signalguard_rf.joblib"
OUTPUT_PATH = ROOT / "results" / "feature_importance.png"


def main():
    saved = joblib.load(MODEL_PATH)
    model = saved["model"]
    feature_names = saved["feature_names"]

    importances = model.feature_importances_
    order = np.argsort(importances)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\n===== FEATURE IMPORTANCE =====")
    for index in order[::-1]:
        print(
            f"{feature_names[index]:<22} "
            f"{importances[index]:.4f}"
        )

    plt.figure(figsize=(9, 6))
    plt.barh(
        [feature_names[i] for i in order],
        importances[order],
    )
    plt.xlabel("Random Forest Feature Importance")
    plt.title("SignalGuard Feature Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200)
    plt.close()

    print(f"\nGraph saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
