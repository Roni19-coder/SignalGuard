"""Signal quality metrics."""
from __future__ import annotations

import numpy as np


def calculate_snr_db(clean: np.ndarray, received: np.ndarray) -> float:
    """Calculate SNR in dB from aligned clean and received signals."""
    x = np.asarray(clean)
    y = np.asarray(received)
    if x.shape != y.shape:
        raise ValueError("clean and received signals must have the same shape")
    noise = y - x
    signal_power = float(np.mean(np.abs(x) ** 2))
    noise_power = float(np.mean(np.abs(noise) ** 2))
    if signal_power == 0:
        raise ValueError("clean signal must have non-zero power")
    if noise_power == 0:
        return float("inf")
    return float(10 * np.log10(signal_power / noise_power))


def calculate_ber(transmitted_bits: np.ndarray,
                  received_bits: np.ndarray) -> float:
    """Calculate bit error rate for two aligned 0/1 bit arrays."""
    tx = np.asarray(transmitted_bits).astype(int)
    rx = np.asarray(received_bits).astype(int)
    if tx.shape != rx.shape:
        raise ValueError("transmitted_bits and received_bits must have the same shape")
    if tx.size == 0:
        raise ValueError("bit arrays must not be empty")
    return float(np.mean(tx != rx))


def recover_bpsk_bits(received_signal: np.ndarray,
                      samples_per_symbol: int) -> np.ndarray:
    """Hard-decision BPSK demodulation by sampling once per symbol."""
    if samples_per_symbol <= 0:
        raise ValueError("samples_per_symbol must be positive")
    x = np.asarray(received_signal)
    if x.size % samples_per_symbol != 0:
        raise ValueError("signal length must be divisible by samples_per_symbol")
    # Midpoint sampling avoids relying on the first sample of each symbol.
    sample_indices = np.arange(samples_per_symbol // 2, x.size, samples_per_symbol)
    return (np.real(x[sample_indices]) >= 0).astype(int)
