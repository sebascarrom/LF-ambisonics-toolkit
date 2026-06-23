"""
process_J12_aligned.py
========================
Procesa J12 (Fuente F3, Posición P4) con alineación de 2 etapas antes de
la conversión A→B, corrigiendo el desfase temporal entre cápsulas.

Desfases detectados (threshold=20dB):
    RF: +0 ms (más temprana)
    LB: +183 ms
    LF: +276 ms
    RB: +374 ms

NOTA: threshold_db=20 (no 15 como en G6_SS1) porque J12 tiene pre-ringing
de ~50-68ms que dispara el detector con umbral más bajo. Con threshold=15,
el onset se detecta hasta 68ms antes del pico real, corrompiendo la alineación.

Uso (desde la raíz del proyecto):
    python debug/process_J12_aligned.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

import numpy as np

from src.acoustic_core import (
    load_aformat_mono_files,
    align_aformat_channels,
    fine_align_channels,
    aformat_to_bformat,
    octave_band_filter,
    compute_LF_band,
    OCTAVE_BANDS_HZ,
)

LABEL = "J12 (Fuente F3, Posición P4)"
PATHS = {
    "LF": "data/raw/usina_del_arte/J12/F3 P4 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/J12/F3 P4 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/J12/F3 P4 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/J12/F3 P4 RB SF 4_SS1.wav",
}

print(f"\n{'='*60}\nREPROCESANDO CON ALINEACIÓN: {LABEL}\n{'='*60}")

signals, fs = load_aformat_mono_files(PATHS)

# ── Paso 1: alineación gruesa ───────────────────────────────────────────
aligned, onsets, common_onset = align_aformat_channels(
    signals, fs, noise_window_ms=50, threshold_db=20
)

print("Onsets detectados por canal:")
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: {onsets[ch]/fs*1000:.2f} ms")
print(f"\nDesfase respecto al más temprano:")
earliest = min(onsets.values())
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: +{(onsets[ch]-earliest)/fs*1000:.2f} ms")

# ── Paso 2: alineación fina por correlación cruzada ────────────────────
aligned_fine, fine_lags = fine_align_channels(
    aligned, fs, common_onset, reference_channel="LF",
    search_radius_ms=5.0, correlation_window_ms=30.0
)
print(f"\nAjuste fino (muestras, sobre la alineación gruesa):")
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: {fine_lags[ch]:+d} muestras ({fine_lags[ch]/fs*1000:+.3f} ms)")

# ── Paso 3: A→B ────────────────────────────────────────────────────────
bformat = aformat_to_bformat(aligned_fine, fs)

# ── Paso 4: LF por banda ───────────────────────────────────────────────
print(f"\nOnset común usado para ventaneo: {common_onset/fs*1000:.2f} ms\n")
print(f"{'Banda (Hz)':<12} {'LF':>8}")
print("-" * 22)
lf_vals = {}
for fc in OCTAVE_BANDS_HZ:
    Wf = octave_band_filter(bformat.W, fc, fs)
    Yf = octave_band_filter(bformat.Y, fc, fs)
    lf = compute_LF_band(Wf, Yf, fs, onset=common_onset)
    lf_vals[fc] = lf
    flag = "  <- mid" if fc in (500, 1000) else ""
    print(f"{fc:<12} {lf:>8.4f}{flag}")

mean_lf = np.nanmean(list(lf_vals.values()))
print("-" * 22)
print(f"{'Mean LF':<12} {mean_lf:>8.4f}")
print(f"{'='*60}\n")
