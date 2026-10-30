"""Run a first SignalGuard DSP demonstration from the repository root."""
from src.signalguard.generators import generate_bpsk
from src.signalguard.impairments import add_awgn, add_tone_interference
from src.signalguard.metrics import calculate_snr_db, recover_bpsk_bits, calculate_ber
from src.signalguard.visualization import plot_signal_comparison, plot_spectrum


def main() -> None:
    samples_per_symbol = 8
    bits, clean = generate_bpsk(num_bits=1000,
                                samples_per_symbol=samples_per_symbol,
                                seed=7)

    noisy = add_awgn(clean, snr_db=5, seed=10)
    impaired = add_tone_interference(noisy, frequency_cycles_per_sample=0.08,
                                     amplitude=0.35)

    # BER here is a simple hard-decision baseline on the noisy signal.
    recovered_bits = recover_bpsk_bits(noisy, samples_per_symbol)
    measured_snr = calculate_snr_db(clean, noisy)
    ber = calculate_ber(bits, recovered_bits)

    print("=== SignalGuard DSP Demo ===")
    print(f"Samples: {clean.size}")
    print(f"Target AWGN SNR: 5.00 dB")
    print(f"Measured AWGN SNR: {measured_snr:.2f} dB")
    print(f"BPSK BER (AWGN only): {ber:.4f}")

    plot_signal_comparison(clean, impaired,
                           title="BPSK signal: clean vs noisy + interference")
    plot_spectrum(impaired, sampling_rate_hz=1.0,
                  title="Spectrum of impaired BPSK signal")


if __name__ == "__main__":
    main()
