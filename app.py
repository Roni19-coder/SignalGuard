
from __future__ import annotations

import base64
import csv
import io
import json
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
from flask import Flask, request, render_template_string, redirect

from signalguard.dataset import extract_features
from signalguard.impairments import (
    add_awgn,
    apply_rayleigh_fading,
    add_tone_interference,
)
from signalguard.metrics import calculate_snr_db

app = Flask(__name__)

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
    <div class="status">● ML MODEL LOADED</div>
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
                        Simulated condition: {{ conditions[selected] }}
                    </div>
                </div>
                <span class="pill">
                    <span class="dot"></span> Analysis complete
                </span>
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
                    <div class="value">1,000</div>
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
        "confusion_plot": None,
        "permutation_plot": None,
    }

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


PENDING_RESULT = None


@app.route("/", methods=["GET", "POST"])
def index():
    global PENDING_RESULT
    selected = "awgn"
    snr = 5

    context = {
        "conditions": CONDITIONS,
        "selected": selected,
        "snr": snr,
        "result": False,
        "error": None,
    }

    if request.method == "POST":
        selected = request.form.get("condition", "awgn")

        try:
            snr = int(request.form.get("snr", "5"))

            if selected not in CONDITIONS:
                raise ValueError("Select a valid channel condition.")

            if not -10 <= snr <= 30:
                raise ValueError("SNR must be between -10 and 30 dB.")

            rng = np.random.default_rng()
            clean = rng.choice(
                np.array([-1.0, 1.0]),
                size=1000,
            )

            if selected == "healthy":
                received = clean.copy()

            elif selected == "awgn":
                received = add_awgn(
                    clean,
                    snr_db=snr,
                    seed=None,
                )

            elif selected == "rayleigh":
                received = apply_rayleigh_fading(
                    clean,
                    seed=None,
                )

            else:
                received = add_tone_interference(
                    clean,
                    frequency_cycles_per_sample=0.05,
                    amplitude=0.5,
                )

            # Extract the same features used during training
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

            if not np.isfinite(measured_snr):
                snr_display = "∞"
            else:
                snr_display = f"{measured_snr:.2f}"

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
            })

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
                    "probabilities", "snr_display", "wave_plot", "fft_plot"
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


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
    )
