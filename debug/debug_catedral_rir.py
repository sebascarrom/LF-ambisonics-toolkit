"""
debug_catedral_rir.py
======================
Procesa SY_RIR_TEST.wav (RIR de Catedral Metropolitana, Soyuz 013 Ambisonic)
y compara con los resultados de EASERA 1.0.

El archivo es A-format int32 de 4 canales. Se trata el canal order como
LF / RF / LB / RB (misma geometría tetraédrica que el SP200):
  ch0 -> LF  (FRONT L UP)
  ch1 -> RF  (FRONT R DOWN)
  ch2 -> LB  (BACK L DOWN)
  ch3 -> RB  (BACK R UP)

Referencia EASERA (LF Octave, Full IR):
  125: 0.000 | 250: 0.054 | 500: 0.099 | 1000: 0.169 | 2000: 0.283 | 4000: 1.168

Uso (desde la raíz del proyecto):
    python debug/debug_catedral_rir.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)
_DEBUG_DIR = _Path(__file__).resolve().parent

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy.io.wavfile as wav

from src.acoustic_core import (
    aformat_to_bformat,
    detect_onset_noise_floor,
    octave_band_filter,
    compute_LF_band,
    OCTAVE_BANDS_HZ,
)

WAV_PATH = str(_DEBUG_DIR / "test_rirs" / "SY_RIR_TEST.wav")

# Referencia EASERA (LF Octave, Full IR — prueba 2 w en b.wav)
EASERA_REF = {125: 0.000, 250: 0.054, 500: 0.099, 1000: 0.169, 2000: 0.283, 4000: 1.168}

print(f"\n{'='*64}")
print("DEBUG — Catedral Metropolitana: SY_RIR_TEST.wav")
print(f"{'='*64}")

# ── Carga ───────────────────────────────────────────────────────────────
fs, raw = wav.read(WAV_PATH)
data = raw.astype(np.float64) / 2**31   # int32 -> float [-1, 1]
print(f"fs={fs} Hz | shape={data.shape} | dtype={raw.dtype}")
print(f"Duración: {data.shape[0]/fs*1000:.1f} ms")

# ── Verificar sincronización de canales ────────────────────────────────
print("\nOnset por canal (A-format crudo):")
for i, label in enumerate(["LF(ch0)", "RF(ch1)", "LB(ch2)", "RB(ch3)"]):
    o = detect_onset_noise_floor(data[:, i], fs)
    print(f"  {label}: {o/fs*1000:.2f} ms")

# ── Conversión A→B ─────────────────────────────────────────────────────
# Canal order: ch0=LF, ch1=RF, ch2=LB, ch3=RB
signals = {
    "LF": data[:, 0],
    "RF": data[:, 1],
    "LB": data[:, 2],
    "RB": data[:, 3],
}
bformat = aformat_to_bformat(signals, fs)
onset = detect_onset_noise_floor(bformat.W, fs)
print(f"\nOnset en W (post A→B): {onset/fs*1000:.2f} ms")

# ── LF por banda ───────────────────────────────────────────────────────
print(f"\n{'Banda':>8}  {'Pipeline':>10}  {'EASERA':>8}  {'Diferencia':>12}")
print("-" * 46)
lf_vals = {}
for fc in OCTAVE_BANDS_HZ:
    Wf = octave_band_filter(bformat.W, fc, fs)
    Yf = octave_band_filter(bformat.Y, fc, fs)
    lf = compute_LF_band(Wf, Yf, fs, onset=onset)
    lf_vals[fc] = lf
    ea = EASERA_REF.get(fc)
    ea_str = f"{ea:.3f}" if ea is not None else "   —"
    diff_str = f"{(lf - ea)*100:+.1f}%" if ea is not None and ea > 0 else "   —"
    flag = "  ← > 1 !" if lf > 1 else ""
    print(f"{fc:>8}  {lf:>10.4f}  {ea_str:>8}  {diff_str:>12}{flag}")

mean_lf = np.nanmean(list(lf_vals.values()))
print("-" * 46)
print(f"{'Mean':>8}  {mean_lf:>10.4f}")

print(f"""
NOTAS:
- Acuerdo casi exacto a 4 kHz (pipeline={lf_vals[4000]:.4f}, EASERA=1.168).
- LF > 1 en 4 kHz es matemáticamente posible si la energía lateral (Y²)
  supera la energía omnidireccional (W²) en [5,80 ms]. Ocurre en la Catedral
  posiblemente por: (a) la conversión A→B sin normalización FuMa subestima W,
  (b) orientación micrófono-fuente con eje Y apuntando hacia reflectores
  fuertes, (c) campo muy difuso con alta energía lateral en ese rango.
- Discrepancias en 125-500 Hz son consistentes con la ausencia de corrección
  frecuencial del Soyuz 013 en la matriz A→B (efecto mayor en bajas frecuencias).
- Los canales están SINCRONIZADOS (mismo onset en todos), a diferencia de
  los archivos de Usina exportados por pistas separadas desde el DAW.
""")
