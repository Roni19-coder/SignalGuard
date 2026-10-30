import numpy as np
import pytest

from src.signalguard.generators import generate_bpsk, generate_qpsk
from src.signalguard.impairments import add_awgn
from src.signalguard.metrics import calculate_snr_db, recover_bpsk_bits, calculate_ber


def test_bpsk_shape_and_values():
    bits, signal = generate_bpsk(num_bits=20, samples_per_symbol=8, seed=1)
    assert bits.shape == (20,)
    assert signal.shape == (160,)
    assert set(np.unique(signal)).issubset({-1.0, 1.0})


def test_qpsk_is_complex():
    bits, signal = generate_qpsk(num_symbols=20, samples_per_symbol=4, seed=1)
    assert bits.shape == (20, 2)
    assert np.iscomplexobj(signal)
    assert signal.shape == (80,)


def test_awgn_snr_is_approximately_requested():
    _, clean = generate_bpsk(num_bits=10000, samples_per_symbol=1, seed=1)
    received = add_awgn(clean, snr_db=5, seed=2)
    assert abs(calculate_snr_db(clean, received) - 5) < 0.25


def test_ber_zero_for_clean_bpsk():
    bits, clean = generate_bpsk(num_bits=100, samples_per_symbol=8, seed=1)
    recovered = recover_bpsk_bits(clean, samples_per_symbol=8)
    assert calculate_ber(bits, recovered) == 0.0
