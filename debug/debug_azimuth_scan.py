"""
debug_azimuth_scan.py
========================
Barrido de rotación azimutal sobre una posición A-format, para diagnosticar
si una desalineación entre la orientación del micrófono y la fuente explica
un LF bajo o inusual.

IMPORTANTE — qué SÍ y qué NO puede confirmar este script:
Si las bandas de octava convergen en un mismo ángulo óptimo, hay indicio de
una desalineación geométrica real (afecta a todas las frecuencias por igual).
Si los ángulos óptimos por banda están dispersos sin patrón, la rotación NO
es una buena explicación — conviene revisar mapeo de canales u otra causa.

Uso (desde la raíz del proyecto):
    python debug/debug_azimuth_scan.py

Requiere matplotlib.
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
    detect_onset_noise_floor,
    scan_azimuth_rotation,
    OCTAVE_BANDS_HZ,
)

LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

# ════════════════════════════════════════════════════════════════════
print(f"\n{'='*64}\nBARRIDO DE ROTACIÓN AZIMUTAL: {LABEL}\n{'='*64}")

signals, fs = load_aformat_mono_files(PATHS)
bformat = aformat_to_bformat(signals, fs)

onset = detect_onset_noise_floor(bformat.W, fs, noise_window_ms=50, threshold_db=15)
print(f"Onset usado (piso de ruido): {onset/fs*1000:.2f} ms")

angles, mean_lf, per_band = scan_azimuth_rotation(
    bformat, onset=onset, bands_hz=OCTAVE_BANDS_HZ,
    angles_deg=np.arange(0, 360, 5)
)

best_idx = int(np.nanargmax(mean_lf))
print(f"\nLF en φ=0° (sin corrección, pipeline actual)  : {mean_lf[0]:.4f}")
print(f"Ángulo que maximiza LF medio                   : {angles[best_idx]}°  ->  LF = {mean_lf[best_idx]:.4f}")
print(f"Ángulo equivalente (180° de diferencia, misma energía): "
      f"{(angles[best_idx] + 180) % 360}°")

print(f"\n--- Ángulo óptimo por banda individual ---")
best_angles_per_band = []
for fc in OCTAVE_BANDS_HZ:
    band_vals = per_band[fc]
    best_band_idx = int(np.nanargmax(band_vals))
    best_angles_per_band.append(angles[best_band_idx])
    print(f"  {fc:>5} Hz  ->  óptimo en {angles[best_band_idx]:>3}°  "
          f"(LF={band_vals[best_band_idx]:.4f})")

reliable_angles = best_angles_per_band[1:]  # excluye 125 Hz
spread = np.std(reliable_angles)
print(f"\nDesvío estándar de ángulos óptimos (250-4000 Hz): {spread:.1f}°")
if spread < 20:
    print("  -> Bandas CONVERGEN: indicio de desalineación geométrica real.")
else:
    print("  -> Bandas DISPERSAS: la rotación probablemente NO explica el LF.")
    print("     Conviene revisar mapeo de canales u otra causa.")

# ── Visualización ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(angles, mean_lf, linewidth=2, color="black", label="LF medio (todas las bandas)")
for fc in OCTAVE_BANDS_HZ:
    ax.plot(angles, per_band[fc], linewidth=0.8, alpha=0.5, label=f"{fc} Hz")
ax.axvline(angles[best_idx], color="green", linestyle=":", alpha=0.7,
           label=f"Óptimo medio ({angles[best_idx]}°)")
ax.set_xlabel("Ángulo de rotación φ (grados)")
ax.set_ylabel("LF")
ax.set_title(f"LF vs. rotación azimutal — {LABEL}")
ax.legend(loc="upper right", fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
output_png = str(_DEBUG_DIR / "debug_azimuth_scan.png")
plt.savefig(output_png, dpi=120)
print(f"\nGráfico guardado en: {output_png}")
print(f"{'='*64}\n")
