# SignalGuard

SignalGuard is an educational communication-signal analysis project. It starts with a reproducible DSP simulation engine and will later add ML-based fault classification and a web dashboard.

## Current features

- BPSK and QPSK baseband signal generation
- AWGN, simple Rayleigh fading, and tone interference
- SNR and BER metrics
- BPSK hard-decision demodulation
- Time-domain and spectrum plots
- Basic automated tests

## Setup (Windows PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python run_demo.py
```

If PowerShell blocks activation, run this in the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run tests

Install pytest if needed:

```powershell
pip install pytest
python -m pytest
```

## Notes

- The current Rayleigh fading function is a simplified independent flat-fading demonstration, not a full time-correlated channel simulator.
- The demo computes BER on the AWGN-only signal and plots the signal with added tone interference separately.
- This is a simulation project, not a hardware receiver or validated communications product.
