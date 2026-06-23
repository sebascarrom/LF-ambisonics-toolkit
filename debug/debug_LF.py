"""
debug_LF.py
============
Script de diagnóstico para comparar nuestro pipeline (acoustic_core.py)
contra el método original de Fede (sin detección de onset, banda ancha)
y para visualizar si la detección de onset cae correctamente sobre el
sonido directo.

Uso: ajustar PATHS abajo con los 4 archivos de UNA posición, y correr
DESDE LA RAÍZ DEL PROYECTO:
    python debug/debug_LF.py

Requiere matplotlib:
    pip install matplotlib
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
_DEBUG_DIR = _Path(__file__).resolve().parent

import numpy as np
import matplotlib.pyplot as plt

from src.acoustic_core import (
    load_aformat_mono_files,
    aformat_to_bformat,
    detect_onset,
    detect_onset_noise_floor,
    octave_band_filter,
    compute_LF_band,
)

# ── Ajustar con la posición a debuggear ────────────────────────────────
LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}\nDEBUG: {LABEL}\n{'='*60}")

signals, fs = load_aformat_mono_files(PATHS)
bformat = aformat_to_bformat(signals, fs)
W, Y = bformat.W, bformat.Y
n = len(W)
print(f"Duración total: {n/fs*1000:.1f} ms ({n} samples @ {fs} Hz)")

# ── 1. Método LEGACY (Fede): onset=0, banda ancha ──────────────────────
i1_legacy = int(0.005 * fs)
i2_legacy = int(0.080 * fs)
energy_Y_legacy = np.sum(Y[i1_legacy:i2_legacy] ** 2)
energy_W_legacy = np.sum(W[0:i2_legacy] ** 2)
LF_legacy = energy_Y_legacy / energy_W_legacy if energy_W_legacy != 0 else np.nan

print(f"\n--- Método LEGACY (onset=0, banda ancha) ---")
print(f"LF = {LF_legacy:.4f}")

# ── 2. Nuestro pipeline: onset detectado + banda ancha (sin filtrar) ──
onset = detect_onset(W, threshold_db=-20.0)
onset_ms = onset / fs * 1000
i1_new = onset + int(0.005 * fs)
i2_new = onset + int(0.080 * fs)
energy_Y_new = np.sum(Y[i1_new:i2_new] ** 2)
energy_W_new = np.sum(W[onset:i2_new] ** 2)
LF_new_broadband = energy_Y_new / energy_W_new if energy_W_new != 0 else np.nan

print(f"\n--- Nuestro pipeline (onset detectado, banda ancha) ---")
print(f"Onset detectado: {onset_ms:.2f} ms (sample {onset})")
print(f"LF = {LF_new_broadband:.4f}")

# ── 3. Sensibilidad del umbral de detección de onset ───────────────────
print(f"\n--- Sensibilidad del umbral de onset ---")
for thr in [-10, -15, -20, -25, -30]:
    try:
        o = detect_onset(W, threshold_db=thr)
        print(f"  threshold={thr:>4} dB  ->  onset = {o/fs*1000:8.2f} ms (sample {o})")
    except ValueError as e:
        print(f"  threshold={thr:>4} dB  ->  ERROR: {e}")

# ── 4. Detección robusta de onset (piso de ruido, no pico global) ─────
print(f"\n--- Detección ROBUSTA de onset (piso de ruido) ---")
try:
    onset_robust = detect_onset_noise_floor(
        W, fs, noise_window_ms=50, threshold_db=15,
        frame_ms=1, min_consecutive_frames=3
    )
    onset_robust_ms = onset_robust / fs * 1000
    print(f"Onset (piso de ruido)         : {onset_robust_ms:.2f} ms (sample {onset_robust})")
    print(f"Onset (pico global, anterior) : {onset_ms:.2f} ms (sample {onset})")
    diff_ms = onset_ms - onset_robust_ms
    print(f"Diferencia: {diff_ms:.2f} ms")
    if abs(diff_ms) > 20:
        print("  ⚠ Difieren significativamente: el detector viejo probablemente")
        print("    se enganchó en un evento más fuerte que el sonido directo real.")

    i1_r = onset_robust + int(0.005 * fs)
    i2_r = onset_robust + int(0.080 * fs)
    energy_Y_r = np.sum(Y[i1_r:i2_r] ** 2)
    energy_W_r = np.sum(W[onset_robust:i2_r] ** 2)
    LF_robust_broadband = energy_Y_r / energy_W_r if energy_W_r != 0 else np.nan
    print(f"LF (onset robusto, banda ancha) = {LF_robust_broadband:.4f}")
except ValueError as e:
    onset_robust = None
    onset_robust_ms = None
    LF_robust_broadband = float("nan")
    print(f"  ERROR: {e}")

# ── 5. Pico global de W ────────────────────────────────────────────────
peak_sample = int(np.argmax(W ** 2))
peak_ms = peak_sample / fs * 1000
print(f"\n--- Pico de energía global de W ---")
print(f"Pico en: {peak_ms:.2f} ms (sample {peak_sample})")
print(f"Onset detectado en: {onset_ms:.2f} ms (sample {onset})")
print(f"Diferencia: {peak_ms - onset_ms:.2f} ms")
if abs(peak_ms - onset_ms) > 5:
    print("  ⚠ El pico global NO coincide con el onset detectado.")
    print("    Esto sugiere un transitorio más fuerte que el sonido directo.")

# ── 6. Visualización ───────────────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(12, 7))
t_ms = np.arange(n) / fs * 1000

axes[0].plot(t_ms, W, linewidth=0.5, color="steelblue")
axes[0].axvline(onset_ms, color="red", linestyle="--", label=f"Onset (pico global, {onset_ms:.1f} ms)")
if onset_robust_ms is not None:
    axes[0].axvline(onset_robust_ms, color="lime", linestyle="--", label=f"Onset (piso de ruido, {onset_robust_ms:.1f} ms)")
axes[0].axvline(peak_ms, color="orange", linestyle=":", label=f"Pico global ({peak_ms:.1f} ms)")
axes[0].set_title(f"Canal W completo — {LABEL}")
axes[0].set_xlabel("Tiempo (ms)")
axes[0].set_ylabel("Amplitud")
axes[0].legend()

zoom_center_ms = onset_robust_ms if onset_robust_ms is not None else onset_ms
zoom_start_ms = max(0, zoom_center_ms - 100)
zoom_end_ms = zoom_center_ms + 200
zoom_start = int(zoom_start_ms / 1000 * fs)
zoom_end = int(zoom_end_ms / 1000 * fs)

axes[1].plot(t_ms[zoom_start:zoom_end], W[zoom_start:zoom_end],
             linewidth=0.8, color="steelblue", label="W")
axes[1].plot(t_ms[zoom_start:zoom_end], Y[zoom_start:zoom_end],
             linewidth=0.8, color="seagreen", alpha=0.7, label="Y")
axes[1].axvline(onset_ms, color="red", linestyle="--", label="Onset (pico global)")
if onset_robust_ms is not None:
    axes[1].axvline(onset_robust_ms, color="lime", linestyle="--", label="Onset (piso de ruido)")
axes[1].set_title("Zoom alrededor del onset")
axes[1].set_xlabel("Tiempo (ms)")
axes[1].set_ylabel("Amplitud")
axes[1].legend()

plt.tight_layout()
output_png = str(_DEBUG_DIR / "debug_onset_plot.png")
plt.savefig(output_png, dpi=120)
print(f"\nGráfico guardado en: {output_png}")

# ── 7. Resumen comparativo ─────────────────────────────────────────────
print(f"\n{'='*60}")
print("RESUMEN COMPARATIVO")
print(f"{'='*60}")
print(f"  LF (legacy, onset=0)                    : {LF_legacy:.4f}")
print(f"  LF (onset pico global, banda ancha)     : {LF_new_broadband:.4f}")
print(f"  LF (onset piso de ruido, banda ancha)   : {LF_robust_broadband:.4f}")
print(f"{'='*60}\n")
