"""Plotting helpers for SignalGuard."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt


def plot_signal_comparison(clean: np.ndarray, received: np.ndarray,
                           samples_to_show: int = 400,
                           title: str = "SignalGuard: clean vs received signal") -> None:
    """Plot clean and received time-domain signals."""
    clean = np.asarray(clean)
    received = np.asarray(received)
    if clean.shape != received.shape:
        raise ValueError("clean and received signals must have the same shape")
    n = min(samples_to_show, clean.size)
    if n <= 0:
        raise ValueError("signals must not be empty")

    plt.figure(figsize=(12, 5))
    plt.plot(np.real(clean[:n]), label="Clean signal", linewidth=1.5)
    plt.plot(np.real(received[:n]), label="Received signal", alpha=0.75, linewidth=1)
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_spectrum(signal: np.ndarray, sampling_rate_hz: float = 1.0,
                  title: str = "Signal spectrum") -> None:
    """Plot a normalized single-sided spectrum for a real-valued signal."""
    x = np.asarray(signal)
    if x.size == 0 or sampling_rate_hz <= 0:
        raise ValueError("signal must be non-empty and sampling rate positive")

    spectrum = np.fft.rfft(np.real(x))
    frequencies = np.fft.rfftfreq(x.size, d=1 / sampling_rate_hz)
    magnitude_db = 20 * np.log10(np.maximum(np.abs(spectrum) / x.size, 1e-12))

    plt.figure(figsize=(12, 4))
    plt.plot(frequencies, magnitude_db)
    plt.title(title)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude (dB)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
