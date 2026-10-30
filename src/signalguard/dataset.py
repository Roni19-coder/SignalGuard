
import numpy as np
import csv
from collections import Counter
from pathlib import Path

from .impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)
from .metrics import calculate_snr_db


def extract_features(clean, received):
    clean = np.asarray(clean)
    received = np.asarray(received)

    real_received = np.real(received)
    centered = received - np.mean(received)

    power = np.mean(np.abs(received) ** 2)
    std = np.sqrt(np.mean(np.abs(centered) ** 2))
    rms = np.sqrt(power)
    peak = np.max(np.abs(received))

    snr_db = calculate_snr_db(clean, received)
    if not np.isfinite(snr_db):
        snr_db = 60.0

    # Frequency-domain features
    centered_real = real_received - np.mean(real_received)
    spectrum = np.abs(np.fft.rfft(centered_real))
    spectrum_power = spectrum ** 2

    if len(spectrum) > 1:
        spectrum_power[0] = 0

    total_spectrum_power = np.sum(spectrum_power) + 1e-12
    normalized_power = spectrum_power / total_spectrum_power

    spectral_entropy = -np.sum(
        normalized_power * np.log(normalized_power + 1e-12)
    )

    dominant_bin = (
        int(np.argmax(spectrum[1:]) + 1)
        if len(spectrum) > 1
        else 0
    )

    dominant_frequency = dominant_bin / max(len(real_received), 1)

    dominant_ratio = (
        spectrum_power[dominant_bin] / total_spectrum_power
        if len(spectrum) > 1
        else 0.0
    )

    real_std = np.std(real_received)

    return {
        "mean": float(np.mean(real_received)),
        "std": float(std),
        "rms": float(rms),
        "peak": float(peak),
        "crest_factor": float(peak / (rms + 1e-12)),
        "skewness": float(
            np.mean(np.real(centered) ** 3)
            / (real_std ** 3 + 1e-12)
        ),
        "kurtosis": float(
            np.mean(np.abs(centered) ** 4)
            / (std ** 4 + 1e-12)
        ),
        "snr_db": float(min(snr_db, 60.0)),
        "spectral_entropy": float(spectral_entropy),
        "dominant_frequency": float(dominant_frequency),
        "dominant_ratio": float(dominant_ratio),
    }


def generate_dataset(samples_per_class=1000, signal_length=1000):
    rows = []
    rng = np.random.default_rng(42)

    for label in range(4):
        for _ in range(samples_per_class):
            bits = rng.integers(0, 2, signal_length)
            clean = (2 * bits - 1).astype(float)
            received = clean.copy()

            if label == 1:
                # Randomize AWGN SNR during training
                snr_db = rng.uniform(-5, 20)
                received = add_awgn(
                    clean,
                    snr_db=snr_db,
                    seed=None,
                )

            elif label == 2:
                received = apply_rayleigh_fading(
                    clean,
                    seed=None,
                )

            elif label == 3:
                # Randomize tone strength and frequency
                frequency = rng.uniform(0.02, 0.20)
                amplitude = rng.uniform(0.2, 0.8)

                received = add_tone_interference(
                    clean,
                    frequency_cycles_per_sample=frequency,
                    amplitude=amplitude,
                )

            features = extract_features(clean, received)
            features["label"] = label
            rows.append(features)

    return rows


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    output_path = root / "data" / "signal_dataset.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = generate_dataset()

    with output_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    print("Samples per class:")
    for label, count in sorted(
        Counter(row["label"] for row in rows).items()
    ):
        print(f"Label {label}: {count}")

    print(f"\nDataset saved to: {output_path}")
    print(f"Dataset shape: ({len(rows)}, {len(rows[0])})")
