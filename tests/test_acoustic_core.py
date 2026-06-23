"""
tests/test_acoustic_core.py
=============================
Tests básicos para src/acoustic_core.py usando señales sintéticas
(no dependen de datos de medición reales).

Ejecutar con: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.acoustic_core import (
    aformat_to_bformat,
    detect_onset,
    octave_band_filter,
    compute_LF_band,
    normalize_aformat_keys,
)


def _synthetic_aformat(fs=48000, duration_s=0.5, onset_sample=1000):
    """Genera 4 canales A-format sintéticos: ruido blanco tras un onset."""
    n = int(fs * duration_s)
    rng = np.random.default_rng(seed=42)
    base = np.zeros(n)
    base[onset_sample:] = rng.normal(0, 1, n - onset_sample)
    # Pequeñas variaciones entre canales para simular asimetría direccional
    return {
        "LF": base * 1.0,
        "RF": base * 0.9,
        "LB": base * 0.8,
        "RB": base * 0.85,
    }, fs


def test_normalize_aformat_keys_sp200():
    raw = {"LeftFront": [1], "RightFront": [2], "LeftBack": [3], "RightBack": [4]}
    norm = normalize_aformat_keys(raw)
    assert set(norm.keys()) == {"LF", "RF", "LB", "RB"}


def test_normalize_aformat_keys_soyuz():
    raw = {
        "FRONT L UP": [1], "FRONT R DOWN": [2],
        "BACK L DOWN": [3], "BACK R UP": [4],
    }
    norm = normalize_aformat_keys(raw)
    assert set(norm.keys()) == {"LF", "RF", "LB", "RB"}


def test_aformat_to_bformat_shapes():
    signals, fs = _synthetic_aformat()
    bf = aformat_to_bformat(signals, fs)
    assert len(bf.W) == len(signals["LF"])
    assert bf.fs == fs


def test_detect_onset_synthetic():
    signals, fs = _synthetic_aformat(onset_sample=1000)
    bf = aformat_to_bformat(signals, fs)
    onset = detect_onset(bf.W)
    # Tolerancia: el onset detectado debe estar cerca del impuesto
    assert abs(onset - 1000) < 50


def test_octave_band_filter_runs():
    signals, fs = _synthetic_aformat()
    bf = aformat_to_bformat(signals, fs)
    filtered = octave_band_filter(bf.W, 1000, fs)
    assert len(filtered) == len(bf.W)


def test_compute_LF_band_range():
    signals, fs = _synthetic_aformat()
    bf = aformat_to_bformat(signals, fs)
    onset = detect_onset(bf.W)
    W_band = octave_band_filter(bf.W, 1000, fs)
    Y_band = octave_band_filter(bf.Y, 1000, fs)
    lf = compute_LF_band(W_band, Y_band, fs, onset)
    # LF es una fracción de energía; debe ser finita y no negativa
    assert lf >= 0.0
    assert np.isfinite(lf)
