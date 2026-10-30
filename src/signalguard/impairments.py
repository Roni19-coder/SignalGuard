"""Channel impairments used in SignalGuard simulations."""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt


def add_awgn(signal: np.ndarray, snr_db: float,
             seed: int | None = 42) -> np.ndarray:
    """Add real or complex AWGN at a requested measured signal-to-noise ratio.

    SNR is based on average signal power. For complex signals, noise power is
    split equally between the real and imaginary components.
    """
    x = np.asarray(signal)
    if x.size == 0:
        raise ValueError("signal must not be empty")

    rng = np.random.default_rng(seed)
    signal_power = float(np.mean(np.abs(x) ** 2))
    if signal_power == 0:
        raise ValueError("cannot set SNR for a zero-power signal")

    noise_power = signal_power / (10 ** (snr_db / 10))
    if np.iscomplexobj(x):
        noise = np.sqrt(noise_power / 2) * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )
    else:
        noise = np.sqrt(noise_power) * rng.standard_normal(x.shape)
    return x + noise


def apply_rayleigh_fading(signal: np.ndarray,
                          seed: int | None = 42) -> np.ndarray:
    """Apply a simple flat Rayleigh fading envelope.

    This is a simplified baseband model: each sample gets an independent
    complex Gaussian fading coefficient. It is useful for an initial demo,
    but is not a substitute for a time-correlated physical channel model.
    """
    x = np.asarray(signal)
    if not np.iscomplexobj(x):
        x = x.astype(complex)
    rng = np.random.default_rng(seed)
    fading = (
        rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
    ) / np.sqrt(2)
    return x * fading


def add_tone_interference(signal: np.ndarray, frequency_cycles_per_sample: float = 0.08,
                          amplitude: float = 0.5) -> np.ndarray:
    """Add a sinusoidal interferer to a real-valued signal."""
    x = np.asarray(signal)
    if np.iscomplexobj(x):
        raise ValueError("tone interference currently expects a real-valued signal")
    if not 0 <= frequency_cycles_per_sample <= 0.5:
        raise ValueError("frequency must be between 0 and 0.5 cycles/sample")
    n = np.arange(x.size)
    interference = amplitude * np.cos(2 * np.pi * frequency_cycles_per_sample * n)
    return x + interference


def lowpass_filter(signal: np.ndarray, cutoff_hz: float,
                   sampling_rate_hz: float, order: int = 4) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter."""
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    if not 0 < cutoff_hz < sampling_rate_hz / 2:
        raise ValueError("cutoff_hz must be between 0 and Nyquist frequency")
    if order < 1:
        raise ValueError("order must be at least 1")

    sos = butter(order, cutoff_hz, btype="lowpass",
                 fs=sampling_rate_hz, output="sos")
    return sosfiltfilt(sos, np.asarray(signal))
