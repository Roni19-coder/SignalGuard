"""Signal generation utilities for SignalGuard."""
from __future__ import annotations

import numpy as np


def generate_bpsk(num_bits: int = 1000, samples_per_symbol: int = 8,
                  seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a rectangular-pulse BPSK baseband signal.

    Returns:
        bits: Array containing 0/1 bits.
        signal: Real-valued oversampled BPSK waveform with amplitudes -1/+1.
    """
    if num_bits <= 0 or samples_per_symbol <= 0:
        raise ValueError("num_bits and samples_per_symbol must be positive")

    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=num_bits)
    symbols = 2 * bits - 1  # 0 -> -1, 1 -> +1
    signal = np.repeat(symbols.astype(float), samples_per_symbol)
    return bits, signal


def generate_qpsk(num_symbols: int = 500, samples_per_symbol: int = 8,
                   seed: int | None = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate a simple oversampled QPSK complex baseband waveform."""
    if num_symbols <= 0 or samples_per_symbol <= 0:
        raise ValueError("num_symbols and samples_per_symbol must be positive")

    rng = np.random.default_rng(seed)
    bit_pairs = rng.integers(0, 2, size=(num_symbols, 2))
    i_component = 2 * bit_pairs[:, 0] - 1
    q_component = 2 * bit_pairs[:, 1] - 1
    symbols = (i_component + 1j * q_component) / np.sqrt(2)
    signal = np.repeat(symbols, samples_per_symbol)
    return bit_pairs, signal
