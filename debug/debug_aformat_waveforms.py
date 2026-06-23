"""
debug_aformat_waveforms.py
=============================
Grafica los 4 canales A-format crudos (LF, RF, LB, RB) de una posición en la
MISMA escala vertical absoluta (sin normalizar por canal), para verificar
objetivamente si hay desbalance de nivel o desincronización entre cápsulas.

Uso (desde la raíz del proyecto):
    python debug/debug_aformat_waveforms.py

Requiere matplotlib.
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
_DEBUG_DIR = _Path(__file__).resolve().parent

import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

print(f"\n{'='*64}\nVERIFICACIÓN OBJETIVA DE NIVEL — {LABEL}\n{'='*64}")

import warnings
warnings.filterwarnings("ignore")

raw = {}
fs = None
for ch, path in PATHS.items():
    fs_i, data = wav.read(path)
    fs = fs_i
    raw[ch] = data.astype(np.float64)
    if data.ndim > 1:
        print(f"  ⚠ {ch}: el archivo tiene {data.shape[1]} canales, se esperaba mono")

sample_fs, sample_raw = wav.read(PATHS["LF"])
print(f"dtype original del WAV: {sample_raw.dtype}")
print(f"fs: {fs} Hz\n")

print(f"{'Canal':<6} {'Pico absoluto':>15} {'RMS absoluto':>15} {'Pico (dBFS, ref 32768)':>24}")
print("-" * 64)
for ch in ("LF", "RF", "LB", "RB"):
    peak = np.max(np.abs(raw[ch]))
    rms = np.sqrt(np.mean(raw[ch] ** 2))
    dbfs = 20 * np.log10(peak / 32768.0) if peak > 0 else float("-inf")
    print(f"{ch:<6} {peak:>15.2f} {rms:>15.4f} {dbfs:>24.2f}")

# ── Gráfico ─────────────────────────────────────────────────────────────
fig, axes = plt.subplots(4, 1, figsize=(12, 9), sharex=True, sharey=True)
n = len(raw["LF"])
t_ms = np.arange(n) / fs * 1000

for ax, ch in zip(axes, ("LF", "RF", "LB", "RB")):
    ax.plot(t_ms, raw[ch], linewidth=0.5, color="steelblue")
    ax.set_ylabel(ch)
    ax.grid(alpha=0.3)

axes[-1].set_xlabel("Tiempo (ms)")
fig.suptitle(f"Canales A-format crudos, MISMA escala vertical — {LABEL}")
plt.tight_layout()
output_png = str(_DEBUG_DIR / "debug_aformat_waveforms.png")
plt.savefig(output_png, dpi=120)
print(f"\nGráfico guardado en: {output_png}")
print("Si LB/RB se ven planos comparados con LF/RF, el desbalance es real.")
print(f"{'='*64}\n")
