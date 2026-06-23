"""
debug_channel_balance.py
===========================
Chequea el balance de nivel entre los 4 canales A-format crudos de una
posición, para descartar (o confirmar) si hay un problema de ganancia de
hardware en alguna cápsula.

Uso (desde la raíz del proyecto):
    python debug/debug_channel_balance.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from src.acoustic_core import (
    load_aformat_mono_files,
    aformat_to_bformat,
    detect_onset_noise_floor,
    check_aformat_channel_balance,
)

LABEL = "G6_SS1 (Fuente F2, Posición P3)"
PATHS = {
    "LF": "data/raw/usina_del_arte/G6_SS1/F2 P3 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/G6_SS1/F2 P3 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/G6_SS1/F2 P3 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/G6_SS1/F2 P3 RB SF 4_SS1.wav",
}

print(f"\n{'='*60}\nBALANCE DE CANALES A-FORMAT: {LABEL}\n{'='*60}")

signals, fs = load_aformat_mono_files(PATHS)
bformat = aformat_to_bformat(signals, fs)
onset = detect_onset_noise_floor(bformat.W, fs, noise_window_ms=50, threshold_db=15)
print(f"Onset usado: {onset/fs*1000:.2f} ms\n")

balance = check_aformat_channel_balance(signals, fs, onset, window_ms=80.0)
