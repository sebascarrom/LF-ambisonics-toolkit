"""
debug_LF_compare.py
======================
Ablación metodológica: calcula LF con 4 variantes distintas sobre los
mismos archivos, para aislar cuánto aporta cada diferencia metodológica
encontrada entre acoustic_core.py y los scripts de Fede.

  A) Legacy banda ancha   — LF_processing_fede.py (onset=0, denom 0-80ms,
                             sin filtrado por octava).
  B) Legacy por banda     — LF_bandas_fede.py (onset=0, denom 5-80ms
                             ⚠ mismo bug de ventana, bandas sin 125Hz/con 8kHz,
                             filtro sosfilt causal orden 4).
  C) ISO-correcto, onset=0 forzado — usa octave_band_filter (fase cero) y
                             compute_LF_band (denom 0-80ms) de acoustic_core,
                             pero forzando onset=0 para aislar el efecto de
                             SOLO la ventana+filtro+bandas, sin onset detection.
  D) Pipeline completo     — acoustic_core.py tal cual (onset detectado +
                             ventana ISO correcta + filtro fase cero + bandas
                             ISO estándar). Lo que ya corriste.

Comparar A vs C aísla el efecto de [ventana del denominador + tipo de
filtro + conjunto de bandas] manteniendo onset=0 fijo en ambos.
Comparar C vs D aísla el efecto puro de la detección de onset.

Uso (desde la raíz del proyecto):
    python debug/debug_LF_compare.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

import numpy as np
from scipy.signal import butter, sosfilt

from src.acoustic_core import (
    load_aformat_mono_files,
    aformat_to_bformat,
    detect_onset,
    octave_band_filter,
    compute_LF_band,
    OCTAVE_BANDS_HZ,
)

# ── Ajustar con la posición a debuggear ────────────────────────────────
LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

FEDE_BANDS = [250, 500, 1000, 2000, 4000, 8000]   # bandas_octava en LF_bandas_fede.py
ISO_BANDS  = OCTAVE_BANDS_HZ                       # [125,250,500,1000,2000,4000]

# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}\nABLACIÓN METODOLÓGICA: {LABEL}\n{'='*64}")

signals, fs = load_aformat_mono_files(PATHS)
bformat = aformat_to_bformat(signals, fs)
W, Y = bformat.W, bformat.Y
print(f"Duración: {len(W)/fs*1000:.1f} ms  |  fs={fs} Hz")

onset_detected = detect_onset(W, threshold_db=-20.0)
print(f"Onset detectado por acoustic_core: {onset_detected/fs*1000:.2f} ms "
      f"(sample {onset_detected})")

# ── A) Legacy banda ancha (LF_processing_fede.py) ──────────────────────
i1_5  = int(0.005 * fs)
i2_80 = int(0.080 * fs)
LF_A = np.sum(Y[i1_5:i2_80] ** 2) / np.sum(W[0:i2_80] ** 2)

# ── B) Legacy por banda (LF_bandas_fede.py, con su propio bug y bandas) ─
def fede_octave_filter(signal, fs_, center_freq, order=4):
    nyq = fs_ / 2
    low, high = center_freq / np.sqrt(2), center_freq * np.sqrt(2)
    sos = butter(order, [low / nyq, high / nyq], btype="band", output="sos")
    return sosfilt(sos, signal)   # causal, como en el original

lf_bandas_B = []
for fc in FEDE_BANDS:
    Wf = fede_octave_filter(W, fs, fc)
    Yf = fede_octave_filter(Y, fs, fc)
    Ew = np.sum(Wf[i1_5:i2_80] ** 2)   # ⚠ mismo bug: denom también 5-80ms
    Ey = np.sum(Yf[i1_5:i2_80] ** 2)
    lf_bandas_B.append(Ey / Ew if Ew != 0 else np.nan)
LF_B = float(np.nanmean(lf_bandas_B))

# ── C) ISO-correcto (acoustic_core), onset FORZADO a 0 ──────────────────
lf_bandas_C = []
for fc in ISO_BANDS:
    Wf = octave_band_filter(W, fc, fs)
    Yf = octave_band_filter(Y, fc, fs)
    lf_bandas_C.append(compute_LF_band(Wf, Yf, fs, onset=0))
LF_C = float(np.nanmean(lf_bandas_C))

# ── D) Pipeline completo: onset detectado + todo correcto ──────────────
lf_bandas_D = []
for fc in ISO_BANDS:
    Wf = octave_band_filter(W, fc, fs)
    Yf = octave_band_filter(Y, fc, fs)
    lf_bandas_D.append(compute_LF_band(Wf, Yf, fs, onset=onset_detected))
LF_D = float(np.nanmean(lf_bandas_D))

# ════════════════════════════════════════════════════════════════════
print(f"\n{'─'*64}")
print(f"  A) Legacy banda ancha (onset=0, denom 0-80ms)         : {LF_A:.4f}")
print(f"  B) Legacy por banda   (onset=0, denom 5-80ms ⚠bug)    : {LF_B:.4f}")
print(f"  C) ISO-correcto, onset=0 forzado                       : {LF_C:.4f}")
print(f"  D) Pipeline completo (onset detectado: {onset_detected/fs*1000:.0f} ms)  : {LF_D:.4f}")
print(f"{'─'*64}")
print(f"  A vs C (mismo onset=0): aísla bandas+filtro+ventana    : "
      f"{'+' if LF_C>LF_A else ''}{(LF_C-LF_A):.4f}")
print(f"  C vs D (mismo método):  aísla SOLO la detección onset  : "
      f"{'+' if LF_D>LF_C else ''}{(LF_D-LF_C):.4f}")
print(f"{'='*64}\n")

print("Bandas detalladas:")
print(f"  {'Banda':<8} {'B (Fede)':>10} {'C (onset=0)':>12} {'D (onset det.)':>15}")
for i, fc in enumerate(ISO_BANDS):
    b_val = lf_bandas_B[FEDE_BANDS.index(fc)] if fc in FEDE_BANDS else float('nan')
    print(f"  {fc:<8} {b_val:>10.4f} {lf_bandas_C[i]:>12.4f} {lf_bandas_D[i]:>15.4f}")
if 8000 in FEDE_BANDS:
    print(f"  {8000:<8} {lf_bandas_B[FEDE_BANDS.index(8000)]:>10.4f} "
          f"{'(no en ISO)':>12} {'(no en ISO)':>15}")
