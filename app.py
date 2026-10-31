
from __future__ import annotations

import base64
import csv
import io
import json
import sqlite3
from datetime import datetime
from pathlib import Path
import sys

# Make the src-layout package importable when running: py app.py
ROOT = Path(__file__).resolve().parent
SRC_PATH = ROOT / "src"
if SRC_PATH.exists() and str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, request, render_template_string, redirect, Response, url_for

from signalguard.dataset import extract_features
from signalguard.impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)
from signalguard.metrics import calculate_snr_db

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload limit

# SQLite database for persistent prediction history.
HISTORY_DB = ROOT / "results" / "prediction_history.db"
HISTORY_DB.parent.mkdir(parents=True, exist_ok=True)


def init_history_db():
    """Create the prediction-history table if it does not already exist."""
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                analyzed_at TEXT NOT NULL,
                source_name TEXT NOT NULL,
                predicted_class TEXT NOT NULL,
                confidence REAL NOT NULL,
                measured_snr_db TEXT NOT NULL,
                sample_count INTEGER NOT NULL
            )
            """
        )
        connection.commit()


def save_prediction_history(source_name, predicted_class, confidence,
                            measured_snr_db, sample_count):
    """Save one successful analysis; called only after prediction succeeds."""
    init_history_db()
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute(
            """
            INSERT INTO prediction_history
            (analyzed_at, source_name, predicted_class, confidence,
             measured_snr_db, sample_count)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().astimezone().isoformat(timespec="seconds"),
                source_name,
                predicted_class,
                float(confidence),
                str(measured_snr_db),
                int(sample_count),
            ),
        )
        connection.commit()


init_history_db()

MODEL_PATH = ROOT / "models" / "signalguard_rf.joblib"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}. Train it first using "
        "py -m signalguard.train"
    )

saved_model = joblib.load(MODEL_PATH)
model = saved_model["model"]
feature_names = saved_model["feature_names"]

CONDITIONS = {
    "healthy": "Healthy",
    "awgn": "AWGN Noise",
    "rayleigh": "Rayleigh Fading",
    "tone": "Tone Interference",
}

CLASS_NAMES = [
    "Healthy",
    "AWGN Noise",
    "Rayleigh Fading",
    "Tone Interference",
]

PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">

<title>SignalGuard | Signal Intelligence</title>
<link rel="icon" href="{{ url_for('static', filename='favicon.jpg') }}" type="image/x-icon">

<style>
:root {
    --bg: #080e1b;
    --panel: #111b2e;
    --text: #e8eef9;
    --muted: #94a3b8;
    --blue: #60a5fa;
    --cyan: #22d3ee;
    --border: #26364f;
    --green: #34d399;
}

* {
    box-sizing: border-box;
}

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 15px;
    line-height: 1.6;
}

.wrap {
    width: min(1240px, 100%);
    margin: auto;
    padding: 28px 24px 42px;
}

/* HEADER */

.top {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 18px;
    margin-bottom: 28px;
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.logo {
    width: 48px;
    height: 48px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    color: white;
    font-size: 25px;
    font-weight: 800;
    background: linear-gradient(135deg, #1959a6, #10b9bd);
}

.brand h1 {
    margin: 0;
    font-family: "Trebuchet MS", "Segoe UI", sans-serif;
    font-size: 34px;
    font-weight: 900;
    letter-spacing: -1.3px;
    line-height: 1.2;
    color: #67e8f9;
}

.brand p {
    margin: 3px 0 0;
    color: var(--muted);
    font-size: 13px;
}

.top-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 12px;
    flex-wrap: wrap;
}
.nav-link {
    display: inline-flex;
    align-items: center;
    text-decoration: none;
    border: 1px solid #354763;
    background: #111b2e;
    color: #dce7f8;
    border-radius: 999px;
    padding: 8px 13px;
    font-size: 12px;
    font-weight: 750;
}
.nav-link:hover { border-color: var(--cyan); color: var(--cyan); }

.status {
    border: 1px solid #1c6258;
    background: #102c2b;
    color: #6ee7b7;
    border-radius: 999px;
    padding: 7px 12px;
    font-size: 11px;
    font-weight: 750;
    white-space: nowrap;
}

/* INTRO */

.hero {
    margin: 20px 0 26px;
}

.eyebrow {
    color: var(--blue);
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h2 {
    font-size: clamp(27px, 4vw, 38px);
    line-height: 1.2;
    letter-spacing: -1px;
    margin: 9px 0;
}

.hero p {
    max-width: 720px;
    color: var(--muted);
    margin: 0;
}

/* VERTICAL SECTIONS */

.layout {
    display: flex;
    flex-direction: column;
    gap: 22px;
}

.panel {
    min-width: 0;
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 26px;
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.2);
}

.panel h3 {
    font-size: 21px;
    margin: 0 0 19px;
    letter-spacing: -0.3px;
}

/* CENTERED SECTION 01 */

.layout > .panel:first-child > h3 {
    text-align: center;
    color: var(--cyan);
    font-size: 24px;
    margin-bottom: 24px;
}

/* FORM */

.upload-divider {
    height: 1px;
    background: var(--border);
    margin: 26px 0 22px;
}

.upload-heading {
    text-align: center;
    color: var(--cyan);
    font-size: 19px;
    margin: 0 0 18px;
}

.field input[type="file"] {
    padding: 10px;
    font-size: 13px;
}

.upload-btn {
    background: linear-gradient(100deg, #087f8c, #0e9f9a);
}

.field {
    margin-bottom: 18px;
}

.field label {
    display: block;
    font-size: 14px;
    font-weight: 650;
    margin-bottom: 7px;
    color: #dce7f8;
}

.field select,
.field input {
    width: 100%;
    padding: 12px 14px;
    border-radius: 9px;
    border: 1px solid #354763;
    background: #0b1425;
    color: var(--text);
    font: inherit;
    outline: none;
}

.field select:focus,
.field input:focus {
    border-color: var(--cyan);
    box-shadow: 0 0 0 3px rgba(34, 211, 238, 0.12);
}

.hint,
.fine {
    color: var(--muted);
    font-size: 12px;
    margin-top: 7px;
}

.btn {
    width: 100%;
    border: none;
    border-radius: 10px;
    padding: 14px 18px;
    background: linear-gradient(100deg, #1959a6, #2678cf);
    color: white;
    font-size: 14px;
    font-weight: 750;
    cursor: pointer;
    transition: transform 0.15s, filter 0.15s;
}

.btn:hover {
    filter: brightness(1.12);
    transform: translateY(-1px);
}

/* DETECTION RESULT */

.result-head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 15px;
}

.result-head h3 {
    margin-bottom: 4px;
}

.sub {
    color: var(--muted);
    font-size: 13px;
}

.pill {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    border: 1px solid #1c6258;
    background: #102c2b;
    color: #6ee7b7;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 700;
    white-space: nowrap;
}

.dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: #20a56f;
}

.prediction {
    font-size: clamp(24px, 3vw, 32px);
    font-weight: 800;
    letter-spacing: -0.8px;
    margin: 19px 0 4px;
    color: var(--cyan);
}

.conf {
    color: var(--muted);
    font-size: 13px;
}

.meter,
.prob-track {
    height: 8px;
    background: #26364f;
    border-radius: 99px;
    overflow: hidden;
}

.meter {
    margin-top: 12px;
}

.meter span,
.prob-track span {
    display: block;
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #2678cf, #10b9bd);
}

.stats {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 13px;
    margin-top: 23px;
}

.stat {
    background: #17243a;
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 15px;
    min-width: 0;
}

.stat .label {
    color: var(--muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.6px;
}

.stat .value {
    font-size: 21px;
    font-weight: 800;
    margin-top: 5px;
    overflow-wrap: anywhere;
}

.stat .unit {
    font-size: 11px;
    color: var(--muted);
    font-weight: 500;
}

/* CHARTS */

.main {
    display: grid;
    gap: 22px;
}

.chart-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 22px;
}

.chart {
    min-width: 0;
}

.chart h3 {
    margin-bottom: 5px;
}

.chart p {
    color: var(--muted);
    font-size: 12px;
    margin: 0 0 14px;
}

.plot {
    display: block;
    width: 100%;
    height: auto;
    border-radius: 10px;
    background: #111b2e;
}

/* CLASS PROBABILITIES */

.prob-list {
    display: grid;
    gap: 17px;
    margin-top: 10px;
}

.prob-top {
    display: flex;
    justify-content: space-between;
    gap: 12px;
    font-size: 13px;
    margin-bottom: 7px;
    color: var(--text);
}

.prob-top span:last-child {
    font-weight: 800;
}

.error {
    border: 1px solid #7f3340;
    background: #351923;
    color: #fda4af;
    border-radius: 12px;
    padding: 13px 15px;
    margin-bottom: 18px;
    overflow-wrap: anywhere;
}

.footer {
    color: #64748b;
    font-size: 11px;
    text-align: center;
    margin-top: 26px;
}

/* MODEL EVALUATION DASHBOARD */
.evaluation-section { margin-top: 24px; }
.eval-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }
.eval-card { background: #17243a; border: 1px solid var(--border); border-radius: 12px; padding: 17px; min-width: 0; }
.eval-card .label { color: var(--muted); font-size: 11px; text-transform: uppercase; letter-spacing: .6px; }
.eval-card .value { font-size: 27px; font-weight: 850; margin-top: 5px; }
.eval-chart-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 22px; margin-top: 22px; }
.eval-chart { min-width: 0; }
.eval-chart h4 { margin: 0 0 5px; font-size: 16px; }
.eval-chart p { color: var(--muted); font-size: 12px; margin: 0 0 13px; }
.eval-chart .plot { width: 100%; height: auto; display: block; border-radius: 10px; background: #111b2e; }
.eval-empty { border: 1px dashed #354763; border-radius: 12px; padding: 18px; color: var(--muted); font-size: 13px; }
@media (max-width: 760px) { .eval-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } .eval-chart-grid { grid-template-columns: 1fr; } }

/* RESPONSIVE */

@media (max-width: 760px) {
    .wrap {
        padding: 20px 13px 30px;
    }

    .panel {
        padding: 19px;
    }

    .chart-grid {
        grid-template-columns: 1fr;
    }

    .top {
        align-items: flex-start;
    }

    .brand h1 {
        font-size: 29px;
    }

    .status {
        font-size: 9px;
        padding: 6px 8px;
    }

    .stats {
        grid-template-columns: 1fr 1fr;
    }

    .result-head {
        flex-direction: column;
    }
}

@media (max-width: 420px) {
    .stats {
        grid-template-columns: 1fr;
    }
}

.evaluation-section > summary {
    cursor: pointer;
    font-weight: 800;
    color: var(--text);
    list-style-position: inside;
    padding: 2px 0;
}
.evaluation-section[open] > summary {
    margin-bottom: 18px;
}
</style>
</head>

<body>
<div class="wrap">

<header class="top">
    <div class="brand">
        <div class="logo">S</div>
        <div>
            <h1>SignalGuard</h1>
            <p>Communication signal intelligence</p>
        </div>
    </div>
    <div class="top-actions">
        <a class="nav-link" href="/">⌂ Dashboard</a>
        <a class="nav-link" href="/history">↺ History</a>
        <div class="status">● ML MODEL LOADED</div>
    </div>
</header>

<section class="hero">
    <div class="eyebrow">Machine Learning · DSP</div>
    <h2>Detect signal impairments.<br>Understand what changed.</h2>
    <p>
        Simulate a channel condition, inspect the received waveform and
        frequency spectrum, and let the trained Random Forest classifier
        identify the likely impairment.
    </p>
</section>

{% if error %}
<div class="error">{{ error }}</div>
{% endif %}

<div class="layout">

    <!-- 01: SIGNAL SETUP -->
    <section class="panel">
        <h3>01 / Signal setup</h3>

        <form method="post">
            <div class="field">
                <label for="condition">Channel condition</label>
                <select id="condition" name="condition">
                    {% for key, label in conditions.items() %}
                    <option value="{{ key }}"
                        {% if selected == key %}selected{% endif %}>
                        {{ label }}
                    </option>
                    {% endfor %}
                </select>
            </div>

            <div class="field">
                <label for="snr">Target AWGN SNR (dB)</label>
                <input id="snr" name="snr" type="number"
                    min="-10" max="30" step="1" value="{{ snr }}">
                <div class="hint">
                    Used when AWGN Noise is selected.
                </div>
            </div>

            <button class="btn" type="submit">
                ↗ &nbsp; Generate and analyze signal
            </button>

            <p class="fine">
                Each run generates a new 1,000-symbol BPSK sample.
                Results are simulation-based, not measurements from hardware.
            </p>
        </form>

        <div class="upload-divider"></div>
        <h3 class="upload-heading">Analyze your own signal</h3>
        <form method="post" enctype="multipart/form-data">
            <input type="hidden" name="mode" value="upload">
            <div class="field">
                <label for="signal_file">Signal CSV file</label>
                <input id="signal_file" name="signal_file" type="file"
                    accept=".csv,text/csv" required>
                <div class="hint">
                    Upload a CSV with one column (received BPSK signal) or two columns
                    (clean reference, received signal). Use numeric samples only, with an optional header.
                </div>
            </div>
            <button class="btn upload-btn" type="submit">
                ↑ &nbsp; Upload and analyze signal
            </button>
            <p class="fine">
                CSV is processed in memory and is not saved to disk. Maximum file size: 5 MB.
            </p>
        </form>
    </section>

    <!-- RESULTS BELOW SETUP -->
    <div class="main">

        {% if result %}

        <!-- 02: DETECTION RESULT -->
        <section class="panel">
            <div class="result-head">
                <div>
                    <h3>02 / Detection result</h3>
                    <div class="sub">
                        {{ result_source }}
                    </div>
                </div>
                <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap;">
                    <span class="pill">
                        <span class="dot"></span> Analysis complete
                    </span>
                    <a class="btn upload-btn" href="{{ url_for('download_result') }}"
                       style="text-decoration:none; display:inline-flex; align-items:center; justify-content:center;">
                        ↓ &nbsp; Download result (CSV)
                    </a>
                </div>
            </div>

            <div class="prediction">{{ prediction }}</div>

            <div class="conf">
                Top-class model probability:
                {{ '%.1f'|format(confidence) }}%
            </div>

            <div class="meter">
                <span style="width: {{ confidence }}%"></span>
            </div>

            <div class="stats">
                <div class="stat">
                    <div class="label">Measured SNR</div>
                    <div class="value">
                        {{ snr_display }}
                        <span class="unit">dB</span>
                    </div>
                </div>

                <div class="stat">
                    <div class="label">Samples</div>
                    <div class="value">{{ sample_count }}</div>
                </div>

                <div class="stat">
                    <div class="label">Predicted class</div>
                    <div class="value" style="font-size: 15px">
                        {{ prediction }}
                    </div>
                </div>
            </div>
        </section>

        <!-- 03 AND 04: CHARTS -->
        <div class="chart-grid">

            <section class="panel chart">
                <h3>03 / Received waveform</h3>
                <p>
                    First 200 samples · real component shown for complex signals
                </p>
                <img class="plot"
                    src="data:image/png;base64,{{ wave_plot }}"
                    alt="Received signal waveform">
            </section>

            <section class="panel chart">
                <h3>04 / Frequency spectrum</h3>
                <p>FFT magnitude · cycles per sample</p>
                <img class="plot"
                    src="data:image/png;base64,{{ fft_plot }}"
                    alt="Received signal frequency spectrum">
            </section>

        </div>

        <!-- 05: CLASS PROBABILITIES -->
        <section class="panel">
            <h3>05 / Class probabilities</h3>

            <div class="prob-list">
                {% for name, prob in probabilities %}
                <div>
                    <div class="prob-top">
                        <span>{{ name }}</span>
                        <span>{{ '%.1f'|format(prob) }}%</span>
                    </div>
                    <div class="prob-track">
                        <span style="width: {{ prob }}%"></span>
                    </div>
                </div>
                {% endfor %}
            </div>

            <p class="fine">
                Probabilities are model outputs, not guarantees of correctness.
                Use held-out test data to estimate classification performance.
            </p>
        </section>

        {% else %}

        <section class="panel">
            <h3>02 / Detection result</h3>
            <p class="sub">
                Choose a channel condition and run an analysis.
                Your prediction, signal plots, and class probabilities
                will appear here.
            </p>
        </section>

        {% endif %}

    </div>
</div>

<!-- 06: MODEL EVALUATION (collapsed by default) -->
<details class="panel evaluation-section">
    <summary>06 / Model evaluation</summary>
    <p class="sub" style="margin-top:-10px; margin-bottom:18px;">{{ evaluation_source }}{% if evaluation_time %} · Evaluated {{ evaluation_time }}{% endif %}. Metrics are loaded from saved SignalGuard results.</p>
    {% if evaluation_available %}
    <div class="eval-grid">
        <div class="eval-card"><div class="label">Accuracy</div><div class="value">{{ evaluation.accuracy }}</div></div>
        <div class="eval-card"><div class="label">Precision · Macro</div><div class="value">{{ evaluation.precision }}</div></div>
        <div class="eval-card"><div class="label">Recall · Macro</div><div class="value">{{ evaluation.recall }}</div></div>
        <div class="eval-card"><div class="label">F1-score · Macro</div><div class="value">{{ evaluation.f1 }}</div></div>
    </div>
    {% else %}<div class="eval-empty">Evaluation metrics not found. Run <code>py -m signalguard.train</code> to create models/evaluation_metrics.json.</div>{% endif %}
    <div class="eval-chart-grid">
        <div class="eval-chart"><h4>AWGN robustness</h4><p>Correct AWGN detection rate across tested SNR levels.</p>
        {% if robustness_plot %}<img class="plot" src="data:image/png;base64,{{ robustness_plot }}" alt="AWGN detection rate versus SNR">{% else %}<div class="eval-empty">Run <code>py -m signalguard.robustness</code> to generate results/awgn_robustness.csv.</div>{% endif %}</div>
        <div class="eval-chart"><h4>SNR robustness: accuracy and F1</h4><p>Overall four-class accuracy and macro F1 across fixed SNR levels.</p>
        {% if snr_robustness_plot %}<img class="plot" src="data:image/png;base64,{{ snr_robustness_plot }}" alt="Accuracy and macro F1 versus SNR">{% else %}<div class="eval-empty">Run <code>py -m signalguard.full_evaluate</code> to generate results/snr_robustness.csv.</div>{% endif %}</div>
        <div class="eval-chart"><h4>Per-class recall across SNR</h4><p>Recall for each simulated class at each tested SNR level.</p>
        {% if class_recall_plot %}<img class="plot" src="data:image/png;base64,{{ class_recall_plot }}" alt="Per-class recall versus SNR">{% else %}<div class="eval-empty">Run <code>py -m signalguard.full_evaluate</code> to generate results/snr_robustness.csv.</div>{% endif %}</div>
        <div class="eval-chart"><h4>Confusion matrix</h4><p>Rows are actual classes; columns are predicted classes.</p>
        {% if confusion_plot %}<img class="plot" src="data:image/png;base64,{{ confusion_plot }}" alt="Model confusion matrix">{% else %}<div class="eval-empty">A saved four-class confusion matrix was not found. Run <code>py -m signalguard.full_evaluate</code> to generate it.</div>{% endif %}</div>
        <div class="eval-chart"><h4>Permutation feature importance</h4><p>Accuracy decrease after shuffling a feature on the held-out simulation test set.</p>
        {% if permutation_plot %}<img class="plot" src="data:image/png;base64,{{ permutation_plot }}" alt="Permutation feature importance">{% else %}<div class="eval-empty">Run <code>py -m signalguard.permutation_test</code> to generate results/permutation_importance.csv.</div>{% endif %}</div>
        <div class="eval-chart"><h4>Evaluation notes</h4><div class="eval-empty">Metrics are based on simulated signals. The AWGN chart reports AWGN detection rate, not overall four-class accuracy. Model probability is not a guarantee of correctness.</div></div>
    </div>
</details>

<div class="footer">
    SignalGuard · Simulation-based signal impairment classification
</div>

</div>
</body>
</html>
"""


def plot_to_base64(fig):
    buffer = io.BytesIO()

    fig.savefig(
        buffer,
        format="png",
        dpi=130,
        bbox_inches="tight",
        facecolor="#111b2e",
    )

    plt.close(fig)
    buffer.seek(0)

    return base64.b64encode(buffer.read()).decode("ascii")


def style_dark_plot(ax, fig):
    """Apply SignalGuard's dark theme to Matplotlib charts."""
    fig.patch.set_facecolor("#111b2e")
    ax.set_facecolor("#111b2e")

    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#e8eef9")

    for spine in ax.spines.values():
        spine.set_color("#354763")

    ax.grid(True, color="#354763", alpha=0.65)


def make_plots(received):
    shown = np.real(received)

    # Waveform plot
    fig, ax = plt.subplots(figsize=(7, 3.1))

    ax.plot(
        np.arange(min(200, len(shown))),
        shown[:200],
        linewidth=1.25,
        color="#22d3ee",
    )

    ax.set_title("Received signal — first 200 samples")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")

    style_dark_plot(ax, fig)

    wave_plot = plot_to_base64(fig)

    # Frequency spectrum
    spectrum = np.fft.rfft(shown - np.mean(shown))
    freqs = np.fft.rfftfreq(len(shown), d=1.0)

    magnitude = 20 * np.log10(np.abs(spectrum) + 1e-12)

    fig, ax = plt.subplots(figsize=(7, 3.1))

    ax.plot(
        freqs,
        magnitude,
        linewidth=1.15,
        color="#60a5fa",
    )

    ax.set_title("Magnitude spectrum")
    ax.set_xlabel("Frequency (cycles/sample)")
    ax.set_ylabel("Magnitude (dB)")

    style_dark_plot(ax, fig)

    fft_plot = plot_to_base64(fig)

    return wave_plot, fft_plot


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, bbox_inches="tight", facecolor="#111b2e")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def dashboard_plot_style(ax, fig):
    fig.patch.set_facecolor("#111b2e")
    ax.set_facecolor("#111b2e")
    ax.tick_params(colors="#cbd5e1")
    ax.xaxis.label.set_color("#cbd5e1")
    ax.yaxis.label.set_color("#cbd5e1")
    ax.title.set_color("#e8eef9")
    for spine in ax.spines.values():
        spine.set_color("#354763")
    ax.grid(True, color="#354763", alpha=0.65)


def load_evaluation_metrics():
    """Prefer the latest fresh four-class evaluation, then fall back to training metrics."""
    full_eval_path = ROOT / "results" / "full_evaluation.json"
    metrics = {}

    if full_eval_path.exists():
        try:
            with full_eval_path.open("r", encoding="utf-8") as file:
                metrics = json.load(file)
        except (OSError, json.JSONDecodeError):
            metrics = {}

    # Backward-compatible fallback if the fresh evaluation has not been run yet.
    if not metrics:
        metrics_path = ROOT / "models" / "evaluation_metrics.json"
        if metrics_path.exists():
            try:
                with metrics_path.open("r", encoding="utf-8") as file:
                    metrics = json.load(file)
            except (OSError, json.JSONDecodeError):
                metrics = {}

    if not metrics and isinstance(saved_model.get("metrics"), dict):
        metrics = saved_model["metrics"]

    report = metrics.get("classification_report", {}) if isinstance(metrics, dict) else {}

    def find_metric(*keys):
        for key in keys:
            value = metrics.get(key) if isinstance(metrics, dict) else None
            if isinstance(value, (int, float)) and np.isfinite(value):
                return float(value)
        return None

    precision = find_metric("macro_precision", "precision")
    recall = find_metric("macro_recall", "recall")
    f1 = find_metric("macro_f1", "f1", "f1_score", "macro_f1_score")

    # Handle older evaluation JSON files that store class averages only in the report.
    macro = report.get("macro avg", {}) if isinstance(report, dict) else {}
    if precision is None and isinstance(macro, dict):
        precision = macro.get("precision")
    if recall is None and isinstance(macro, dict):
        recall = macro.get("recall")
    if f1 is None and isinstance(macro, dict):
        f1 = macro.get("f1-score")

    values = {
        "accuracy": find_metric("accuracy", "test_accuracy"),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    formatted = {}
    for key, value in values.items():
        if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
            formatted[key] = "—"
        else:
            percent = value * 100 if value <= 1 else value
            formatted[key] = f"{percent:.2f}%"

    matrix = metrics.get("confusion_matrix") or metrics.get("confusion")
    if isinstance(matrix, dict):
        matrix = matrix.get("matrix")

    evaluation_time = metrics.get("evaluation_time_utc", "") if isinstance(metrics, dict) else ""
    evaluation_source = "Fresh four-class test evaluation" if full_eval_path.exists() and metrics else "Saved training evaluation"
    return formatted, matrix, any(value != "—" for value in formatted.values()), evaluation_time, evaluation_source


def load_dashboard_context():
    evaluation, matrix, evaluation_available, evaluation_time, evaluation_source = load_evaluation_metrics()
    context = {
        "evaluation": evaluation,
        "evaluation_available": evaluation_available,
        "evaluation_time": evaluation_time,
        "evaluation_source": evaluation_source,
        "robustness_plot": None,
        "snr_robustness_plot": None,
        "class_recall_plot": None,
        "confusion_plot": None,
        "permutation_plot": None,
    }

    # Load the fixed-SNR evaluation saved by full_evaluate.py.
    # This only reads existing results; it never runs model evaluation.
    snr_robustness_path = ROOT / "results" / "snr_robustness.csv"
    if snr_robustness_path.exists():
        try:
            with snr_robustness_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))

            if rows:
                xs = [float(row["snr_db"]) for row in rows]

                def as_percent(value):
                    number = float(value)
                    return number * 100 if number <= 1 else number

                accuracy_values = [as_percent(row["accuracy"]) for row in rows]
                f1_values = [as_percent(row["macro_f1"]) for row in rows]

                fig, ax = plt.subplots(figsize=(7, 3.6))
                ax.plot(xs, accuracy_values, marker="o", linewidth=2,
                        label="Accuracy", color="#22d3ee")
                ax.plot(xs, f1_values, marker="s", linewidth=2,
                        label="Macro F1", color="#60a5fa")
                ax.set_xlabel("SNR (dB)")
                ax.set_ylabel("Score (%)")
                ax.set_title("Four-class performance versus SNR")
                ax.set_ylim(0, 105)
                ax.set_xticks(xs)
                ax.legend(facecolor="#17243a", edgecolor="#354763",
                          labelcolor="#e8eef9")
                dashboard_plot_style(ax, fig)
                context["snr_robustness_plot"] = figure_to_base64(fig)

                recall_fields = [
                    ("Healthy_recall", "Healthy", "#22d3ee"),
                    ("AWGN_recall", "AWGN", "#60a5fa"),
                    ("Rayleigh fading_recall", "Rayleigh fading", "#34d399"),
                    ("Tone interference_recall", "Tone interference", "#fbbf24"),
                ]
                fig, ax = plt.subplots(figsize=(7, 3.8))
                plotted = False
                for field, label, color in recall_fields:
                    if field in rows[0]:
                        values = [as_percent(row[field]) for row in rows]
                        ax.plot(xs, values, marker="o", linewidth=1.8,
                                label=label, color=color)
                        plotted = True

                if plotted:
                    ax.set_xlabel("SNR (dB)")
                    ax.set_ylabel("Recall (%)")
                    ax.set_title("Per-class recall versus SNR")
                    ax.set_ylim(0, 105)
                    ax.set_xticks(xs)
                    ax.legend(facecolor="#17243a", edgecolor="#354763",
                              labelcolor="#e8eef9", fontsize=8)
                    dashboard_plot_style(ax, fig)
                    context["class_recall_plot"] = figure_to_base64(fig)
                else:
                    plt.close(fig)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    robustness_path = ROOT / "results" / "awgn_robustness.csv"
    if robustness_path.exists():
        try:
            with robustness_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            xs = [float(row["snr_db"]) for row in rows]
            ys = [float(row["awgn_detection_rate_percent"]) for row in rows]
            if xs and len(xs) == len(ys):
                fig, ax = plt.subplots(figsize=(7, 3.6))
                ax.plot(xs, ys, marker="o", linewidth=2, color="#22d3ee")
                ax.set_xlabel("AWGN SNR (dB)")
                ax.set_ylabel("Detection rate (%)")
                ax.set_title("AWGN detection versus SNR")
                ax.set_ylim(0, 105)
                ax.set_xticks(xs)
                dashboard_plot_style(ax, fig)
                context["robustness_plot"] = figure_to_base64(fig)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    try:
        arr = np.asarray(matrix, dtype=float)
        if arr.shape == (4, 4):
            fig, ax = plt.subplots(figsize=(6.5, 4.6))
            fig.patch.set_facecolor("#111b2e")
            ax.set_facecolor("#111b2e")
            image = ax.imshow(arr, cmap="Blues")
            short_names = ["Healthy", "AWGN", "Rayleigh", "Tone"]
            ax.set_xticks(range(4), short_names, rotation=25, ha="right")
            ax.set_yticks(range(4), short_names)
            ax.set_xlabel("Predicted class", color="#cbd5e1")
            ax.set_ylabel("Actual class", color="#cbd5e1")
            ax.set_title("Confusion matrix", color="#e8eef9")
            ax.tick_params(colors="#cbd5e1")
            threshold = arr.max() / 2 if arr.size else 0
            for row in range(4):
                for col in range(4):
                    ax.text(col, row, f"{int(arr[row, col])}", ha="center", va="center",
                            color="white" if arr[row, col] > threshold else "#0f172a", fontsize=10)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            context["confusion_plot"] = figure_to_base64(fig)
    except (ValueError, TypeError):
        pass

    permutation_path = ROOT / "results" / "permutation_importance.csv"
    if permutation_path.exists():
        try:
            with permutation_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            rows = sorted(rows, key=lambda row: float(row["mean_accuracy_drop"]))[-10:]
            if rows:
                names = [row["feature"] for row in rows]
                means = [float(row["mean_accuracy_drop"]) * 100 for row in rows]
                errors = [float(row.get("standard_deviation", 0) or 0) * 100 for row in rows]
                fig, ax = plt.subplots(figsize=(7, 3.8))
                positions = np.arange(len(names))
                ax.barh(positions, means, xerr=errors, color="#60a5fa", alpha=0.9)
                ax.set_yticks(positions, names)
                ax.set_xlabel("Accuracy decrease (percentage points)")
                ax.set_title("Permutation feature importance")
                dashboard_plot_style(ax, fig)
                context["permutation_plot"] = figure_to_base64(fig)
        except (OSError, ValueError, KeyError, TypeError):
            pass

    return context


def parse_uploaded_signal(file_storage):
    """Parse a numeric CSV containing received samples or clean/received pairs."""
    if file_storage is None or not file_storage.filename:
        raise ValueError("Choose a CSV file before uploading.")
    if Path(file_storage.filename).suffix.lower() != ".csv":
        raise ValueError("Only .csv files are supported.")

    raw = file_storage.read()
    if not raw:
        raise ValueError("The uploaded CSV file is empty.")
    try:
        content = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("The CSV must be encoded as UTF-8.") from exc

    rows = []
    header_skipped = False
    for line_number, row in enumerate(csv.reader(io.StringIO(content)), start=1):
        if not row or all(not cell.strip() for cell in row):
            continue
        values = []
        try:
            values = [float(cell.strip()) for cell in row if cell.strip()]
            if len(values) != len(row):
                raise ValueError
        except ValueError:
            # Permit exactly one optional header row. Reject malformed data rows.
            if not rows and not header_skipped:
                header_skipped = True
                continue
            raise ValueError(f"Row {line_number} contains a non-numeric or empty value.")
        if len(values) not in (1, 2):
            raise ValueError(f"Row {line_number} must contain one or two numeric columns.")
        rows.append(values)
        if len(rows) > 100000:
            raise ValueError("The CSV may contain at most 100,000 samples.")

    if len(rows) < 32:
        raise ValueError("The CSV must contain at least 32 signal samples.")
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError("Every CSV row must have the same number of columns.")

    data = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(data)):
        raise ValueError("The CSV contains NaN or infinite values.")

    if data.shape[1] == 2:
        clean = data[:, 0]
        received = data[:, 1]
        source = "Uploaded signal · clean reference and received samples"
    else:
        received = data[:, 0]
        # The model was trained on BPSK-like samples with a clean reference.
        # For a single-column upload, estimate that reference using hard decisions.
        clean = np.where(np.real(received) >= 0, 1.0, -1.0)
        source = "Uploaded signal · reference estimated from BPSK hard decisions"

    return clean, received, source


PENDING_RESULT = None
LATEST_RESULT = None


@app.route("/", methods=["GET", "POST"])
def index():
    global PENDING_RESULT, LATEST_RESULT
    selected = "awgn"
    snr = 5

    context = {
        "conditions": CONDITIONS,
        "selected": selected,
        "snr": snr,
        "result": False,
        "error": None,
        "result_source": "",
        "sample_count": "—",
    }

    if request.method == "POST":
        mode = request.form.get("mode", "simulate")
        selected = request.form.get("condition", "awgn")

        source_name = ""
        try:
            if mode == "upload":
                uploaded_file = request.files.get("signal_file")
                source_name = Path(uploaded_file.filename).name if uploaded_file and uploaded_file.filename else "Uploaded CSV"
                clean, received, result_source = parse_uploaded_signal(
                    uploaded_file
                )
                snr = 5
            else:
                snr = int(request.form.get("snr", "5"))
                if selected not in CONDITIONS:
                    raise ValueError("Select a valid channel condition.")
                if not -10 <= snr <= 30:
                    raise ValueError("SNR must be between -10 and 30 dB.")

                rng = np.random.default_rng()
                clean = rng.choice(np.array([-1.0, 1.0]), size=1000)

                if selected == "healthy":
                    received = clean.copy()
                elif selected == "awgn":
                    received = add_awgn(clean, snr_db=snr, seed=None)
                elif selected == "rayleigh":
                    received = apply_rayleigh_fading(clean, seed=None)
                else:
                    received = add_tone_interference(
                        clean,
                        frequency_cycles_per_sample=0.05,
                        amplitude=0.5,
                    )
                result_source = f"Simulated condition: {CONDITIONS[selected]}"
                source_name = f"Simulation — {CONDITIONS[selected]}"

            if len(clean) != len(received):
                raise ValueError("Clean reference and received signal must have the same number of samples.")

            # Extract the same features used during model training.
            features = extract_features(clean, received)
            x = np.array(
                [[float(features[name]) for name in feature_names]],
                dtype=float,
            )

            pred_id = int(model.predict(x)[0])
            probs = model.predict_proba(x)[0]
            class_ids = list(model.classes_)
            prob_by_id = {
                int(class_id): float(prob) * 100
                for class_id, prob in zip(class_ids, probs)
            }

            prediction = (
                CLASS_NAMES[pred_id]
                if 0 <= pred_id < len(CLASS_NAMES)
                else str(pred_id)
            )
            confidence = prob_by_id.get(pred_id, 0.0)
            probabilities = [
                (name, prob_by_id.get(i, 0.0))
                for i, name in enumerate(CLASS_NAMES)
            ]

            measured_snr = calculate_snr_db(clean, received)
            snr_display = "∞" if not np.isfinite(measured_snr) else f"{measured_snr:.2f}"
            wave_plot, fft_plot = make_plots(received)

            context.update({
                "selected": selected,
                "snr": snr,
                "result": True,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": probabilities,
                "snr_display": snr_display,
                "wave_plot": wave_plot,
                "fft_plot": fft_plot,
                "result_source": result_source,
                "sample_count": f"{len(received):,}",
            })

            # Record successful analyses once, at POST time—not on refresh.
            save_prediction_history(
                source_name=source_name or result_source,
                predicted_class=prediction,
                confidence=confidence,
                measured_snr_db=snr_display,
                sample_count=len(received),
            )

            # Keep the latest successful analysis available for CSV download.
            LATEST_RESULT = {
                "analyzed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
                "source": source_name or result_source,
                "result_source": result_source,
                "predicted_class": prediction,
                "confidence_percent": round(float(confidence), 2),
                "measured_snr_db": snr_display,
                "sample_count": len(received),
                "probabilities": [
                    {"class": name, "probability_percent": round(float(prob), 2)}
                    for name, prob in probabilities
                ],
                "features": {
                    str(name): (
                        float(value) if np.isfinite(float(value)) else str(value)
                    )
                    for name, value in features.items()
                },
            }

        except Exception as exc:
            context.update({
                "selected": selected,
                "snr": snr,
                "error": str(exc),
            })

        # Post/Redirect/Get: results appear after analysis, but a browser
        # refresh returns to the clean default page instead of resubmitting.
        if context.get("result"):
            PENDING_RESULT = {
                key: context[key]
                for key in (
                    "selected", "snr", "result", "prediction", "confidence",
                    "probabilities", "snr_display", "wave_plot", "fft_plot",
                    "result_source", "sample_count"
                )
            }
            return redirect("/")

    elif PENDING_RESULT is not None:
        context.update(PENDING_RESULT)
        PENDING_RESULT = None

    context.update(load_dashboard_context())
    response = app.make_response(render_template_string(PAGE, **context))

    # Prevent the browser from restoring a cached prediction page on refresh.
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"

    return response



HISTORY_PAGE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SignalGuard | Prediction History</title>
<style>
:root { --bg:#080e1b; --panel:#111b2e; --text:#e8eef9; --muted:#94a3b8; --cyan:#22d3ee; --blue:#60a5fa; --border:#26364f; }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--text); font:15px/1.6 "Segoe UI",Arial,sans-serif; }
.wrap { width:min(1240px,100%); margin:auto; padding:28px 24px 42px; }
.top { display:flex; justify-content:space-between; align-items:center; gap:18px; flex-wrap:wrap; margin-bottom:30px; }
.brand { display:flex; align-items:center; gap:13px; }
.logo { width:48px; height:48px; display:grid; place-items:center; border-radius:14px; color:white; font-size:25px; font-weight:800; background:linear-gradient(135deg,#1959a6,#10b9bd); }
h1 { margin:0; color:#67e8f9; font-size:32px; line-height:1.2; }
.brand p { margin:3px 0 0; color:var(--muted); font-size:13px; }
.actions { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
.link,.btn { display:inline-flex; align-items:center; justify-content:center; text-decoration:none; border:1px solid #354763; border-radius:9px; padding:10px 14px; background:#17243a; color:var(--text); font:inherit; font-size:13px; font-weight:750; cursor:pointer; }
.link:hover { border-color:var(--cyan); color:var(--cyan); }
.btn-primary { border:0; background:linear-gradient(100deg,#1959a6,#2678cf); color:white; }
.btn-danger { border:1px solid #7f3340; background:#351923; color:#fda4af; }
.hero { margin:20px 0 25px; }
.eyebrow { color:var(--blue); font-size:11px; font-weight:800; letter-spacing:2px; text-transform:uppercase; }
.hero h2 { font-size:clamp(27px,4vw,38px); line-height:1.2; margin:9px 0; letter-spacing:-1px; }
.hero p,.muted { color:var(--muted); }
.panel { background:var(--panel); border:1px solid var(--border); border-radius:18px; padding:22px; box-shadow:0 8px 30px rgba(0,0,0,.2); margin-bottom:20px; }
.stats { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:13px; margin-bottom:20px; }
.stat { background:#17243a; border:1px solid var(--border); border-radius:12px; padding:15px; }
.stat span { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; }
.stat strong { display:block; font-size:25px; margin-top:4px; }
.filters { display:grid; grid-template-columns:2fr 1fr auto; gap:12px; align-items:end; }
label { display:block; margin-bottom:6px; font-weight:650; font-size:13px; }
input,select { width:100%; padding:11px 12px; border-radius:9px; border:1px solid #354763; background:#0b1425; color:var(--text); font:inherit; }
.table-wrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; min-width:820px; }
th,td { padding:13px 12px; border-bottom:1px solid var(--border); text-align:left; vertical-align:top; }
th { color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.6px; white-space:nowrap; }
td { font-size:13px; }
.class-pill { display:inline-block; border:1px solid #235a69; background:#102c3a; color:#67e8f9; border-radius:999px; padding:4px 9px; font-size:11px; font-weight:750; white-space:nowrap; }
.empty { padding:35px 16px; text-align:center; color:var(--muted); border:1px dashed #354763; border-radius:12px; }
.table-actions { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; margin-bottom:16px; }
.footer { color:#64748b; font-size:11px; text-align:center; margin-top:26px; }
@media(max-width:700px) { .wrap{padding:20px 13px 30px}.filters{grid-template-columns:1fr}.stats{grid-template-columns:1fr}.panel{padding:16px}.top{align-items:flex-start}h1{font-size:27px} }
</style>
</head>
<body>
<div class="wrap">
<header class="top">
  <div class="brand"><div class="logo">S</div><div><h1>SignalGuard</h1><p>Prediction history</p></div></div>
  <div class="actions"><a class="link" href="/">⌂ Dashboard</a><a class="link" href="/history">↺ History</a></div>
</header>
<section class="hero">
  <div class="eyebrow">Analysis records</div>
  <h2>Your signal analysis history.</h2>
  <p>Successful simulations and CSV analyses are saved locally in SQLite, including after restarting the Flask app.</p>
</section>
<div class="stats">
  <div class="stat"><span>Total records</span><strong>{{ total_count }}</strong></div>
  <div class="stat"><span>Filtered results</span><strong>{{ records|length }}</strong></div>
  <div class="stat"><span>Most recent class</span><strong style="font-size:17px">{{ latest_class }}</strong></div>
</div>
<section class="panel">
  <form method="get" action="/history" class="filters">
    <div><label for="q">Search filename or predicted class</label><input id="q" name="q" value="{{ q }}" placeholder="e.g. signalguard_test_awgn"></div>
    <div><label for="condition">Filter by condition</label><select id="condition" name="condition">
      <option value="">All conditions</option>
      {% for item in class_names %}<option value="{{ item }}" {% if condition == item %}selected{% endif %}>{{ item }}</option>{% endfor %}
    </select></div>
    <button class="btn btn-primary" type="submit">Apply filters</button>
  </form>
</section>
<section class="panel">
  <div class="table-actions">
    <div><strong>Previous analyses</strong><div class="muted" style="font-size:12px">Newest records appear first.</div></div>
    <div class="actions">
      <a class="link" href="/history/export">↓ Export CSV</a>
      <form method="post" action="/history/clear" onsubmit="return confirm('Delete all prediction history? This cannot be undone.');">
        <button class="btn btn-danger" type="submit">Clear history</button>
      </form>
    </div>
  </div>
  {% if records %}
  <div class="table-wrap"><table>
    <thead><tr><th>ID</th><th>Analyzed at</th><th>Source / filename</th><th>Predicted condition</th><th>Confidence</th><th>Measured SNR</th><th>Samples</th></tr></thead>
    <tbody>{% for row in records %}<tr>
      <td>{{ row["id"] }}</td><td>{{ row["analyzed_at"] }}</td><td>{{ row["source_name"] }}</td>
      <td><span class="class-pill">{{ row["predicted_class"] }}</span></td>
      <td>{{ "%.1f"|format(row["confidence"]) }}%</td><td>{{ row["measured_snr_db"] }}{% if row["measured_snr_db"] != "∞" %} dB{% endif %}</td><td>{{ "{:,}".format(row["sample_count"]) }}</td>
    </tr>{% endfor %}</tbody>
  </table></div>
  {% else %}
  <div class="empty">No matching history records yet.<br>Analyze a simulated signal or upload a CSV from the dashboard to create your first record.</div>
  {% endif %}
</section>
<div class="footer">SignalGuard · Local prediction history · Stored on this machine</div>
</div>
<script>
  // If the History page is refreshed, return to the main Dashboard.
  // Opening History normally still works; only a browser reload redirects.
  (function () {
    const navigation = performance.getEntriesByType("navigation")[0];
    if (navigation && navigation.type === "reload") {
      window.location.replace("/");
    }
  })();
</script>
</body>
</html>
"""


@app.route("/download-result", methods=["GET"])
def download_result():
    """Download the latest successful signal analysis as a CSV report."""
    if LATEST_RESULT is None:
        return (
            "No analysis result is available yet. Analyze a signal first.",
            404,
            {"Content-Type": "text/plain; charset=utf-8"},
        )

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["SignalGuard Analysis Result"])
    writer.writerow(["Analyzed at", LATEST_RESULT["analyzed_at"]])
    writer.writerow(["Source", LATEST_RESULT["source"]])
    writer.writerow(["Analysis details", LATEST_RESULT["result_source"]])
    writer.writerow(["Predicted class", LATEST_RESULT["predicted_class"]])
    writer.writerow(["Confidence (%)", LATEST_RESULT["confidence_percent"]])
    writer.writerow(["Measured SNR (dB)", LATEST_RESULT["measured_snr_db"]])
    writer.writerow(["Sample count", LATEST_RESULT["sample_count"]])
    writer.writerow([])
    writer.writerow(["Class probabilities"])
    writer.writerow(["Class", "Probability (%)"])
    for item in LATEST_RESULT["probabilities"]:
        writer.writerow([item["class"], item["probability_percent"]])
    writer.writerow([])
    writer.writerow(["Extracted features"])
    writer.writerow(["Feature", "Value"])
    for name, value in LATEST_RESULT["features"].items():
        writer.writerow([name, value])

    response = Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="signalguard_analysis_result.csv"'
        },
    )
    return response


@app.route("/history", methods=["GET"])
def history():
    q = request.args.get("q", "").strip()
    condition = request.args.get("condition", "").strip()
    where = []
    params = []
    if q:
        where.append("(source_name LIKE ? OR predicted_class LIKE ?)")
        params.extend([f"%{q}%", f"%{q}%"])
    if condition in CLASS_NAMES:
        where.append("predicted_class = ?")
        params.append(condition)

    query = "SELECT * FROM prediction_history"
    if where:
        query += " WHERE " + " AND ".join(where)
    query += " ORDER BY id DESC"

    with sqlite3.connect(HISTORY_DB) as connection:
        connection.row_factory = sqlite3.Row
        records = [dict(row) for row in connection.execute(query, params).fetchall()]
        total_count = connection.execute(
            "SELECT COUNT(*) FROM prediction_history"
        ).fetchone()[0]
        latest = connection.execute(
            "SELECT predicted_class FROM prediction_history ORDER BY id DESC LIMIT 1"
        ).fetchone()

    response = app.make_response(render_template_string(
        HISTORY_PAGE,
        records=records,
        q=q,
        condition=condition,
        class_names=CLASS_NAMES,
        total_count=total_count,
        latest_class=latest[0] if latest else "—",
    ))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    return response


@app.route("/history/export", methods=["GET"])
def export_history():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "id", "analyzed_at", "source_name", "predicted_class",
        "confidence_percent", "measured_snr_db", "sample_count"
    ])
    with sqlite3.connect(HISTORY_DB) as connection:
        rows = connection.execute(
            """
            SELECT id, analyzed_at, source_name, predicted_class,
                   confidence, measured_snr_db, sample_count
            FROM prediction_history ORDER BY id DESC
            """
        ).fetchall()
    writer.writerows(rows)
    return Response(
        output.getvalue(),
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=signalguard_prediction_history.csv"},
    )


@app.route("/history/clear", methods=["POST"])
def clear_history():
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute("DELETE FROM prediction_history")
        connection.commit()
    return redirect(url_for("history"))


@app.errorhandler(413)
def upload_too_large(_error):
    context = {
        "conditions": CONDITIONS,
        "selected": "awgn",
        "snr": 5,
        "result": False,
        "error": "Upload too large. Please choose a CSV smaller than 5 MB.",
    }
    context.update(load_dashboard_context())
    response = app.make_response(render_template_string(PAGE, **context))
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    return response


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
