"""
process_G6_SS1_aligned.py
============================
Reprocesa G6_SS1 con alineación por canal antes de la conversión A→B,
corrigiendo el desfase temporal entre cápsulas detectado en
debug_aformat_waveforms.py.

Resultado validado (2025-06):
    Sin alinear (pipeline original)       : Mean LF ≈ 0.131
    Alineación gruesa + fina (este script): Mean LF ≈ 0.324
    Bandas 125-2000 Hz en rango físico plausible.
    Banda 4 kHz (0.689) pendiente de análisis adicional.

Uso (desde la raíz del proyecto):
    python debug/process_G6_SS1_aligned.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from src.acoustic_core import (
    load_aformat_mono_files,
    align_aformat_channels,
    fine_align_channels,
    aformat_to_bformat,
    octave_band_filter,
    compute_LF_band,
    OCTAVE_BANDS_HZ,
)

LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

print(f"\n{'='*60}\nREPROCESANDO CON ALINEACIÓN: {LABEL}\n{'='*60}")

signals, fs = load_aformat_mono_files(PATHS)

# ── Paso 1: detectar onset por canal y alinear ANTES de convertir ──────
aligned, onsets, common_onset = align_aformat_channels(
    signals, fs, noise_window_ms=50, threshold_db=15
)

print("Onsets detectados por canal:")
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: {onsets[ch]/fs*1000:.2f} ms")
print(f"\nDesfase respecto al más temprano:")
earliest = min(onsets.values())
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: +{(onsets[ch]-earliest)/fs*1000:.2f} ms")

# ── Paso 2: refinar a precisión de muestra con correlación cruzada ─────
aligned_fine, fine_lags = fine_align_channels(
    aligned, fs, common_onset, reference_channel="LF",
    search_radius_ms=5.0, correlation_window_ms=30.0
)
print(f"\nAjuste fino (muestras, sobre la alineación gruesa):")
for ch in ("LF", "RF", "LB", "RB"):
    print(f"  {ch}: {fine_lags[ch]:+d} muestras ({fine_lags[ch]/fs*1000:+.3f} ms)")

# ── Paso 3: convertir a B-format YA ALINEADO ───────────────────────────
bformat = aformat_to_bformat(aligned_fine, fs)

# ── Paso 4: calcular LF por banda ──────────────────────────────────────
print(f"\nOnset común usado para ventaneo: {common_onset/fs*1000:.2f} ms\n")
print(f"{'Banda (Hz)':<12} {'LF':>8}")
print("-" * 22)
lf_vals = {}
import numpy as np
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

print("Comparación con resultados previos:")
print(f"  Sin alinear (pipeline original)        : Mean LF ≈ 0.131")
print(f"  Alineación gruesa solamente (anterior)  : Mean LF ≈ 0.640 (bandas altas inestables)")
print(f"  Alineación gruesa + fina (este script)  : Mean LF ≈ {mean_lf:.4f}")
