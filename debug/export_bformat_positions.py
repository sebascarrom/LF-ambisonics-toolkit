"""
export_bformat_positions.py
==============================
Convierte J12 (Fuente F3, Posición P4) y G6_SS1 (Fuente F2, Posición P3)
de A-format a B-format SIN alineación de canales y exporta cada una como
WAV de 4 canales (orden W,X,Y,Z), para comparar contra EASERA u otro software.

ADVERTENCIA: el B-format de G6_SS1 (y probablemente J12) está corrompido
por desincronización entre cápsulas. Ver process_G6_SS1_aligned.py para
el pipeline corregido antes de usar estos exports para análisis.

Uso (desde la raíz del proyecto):
    python debug/export_bformat_positions.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from pathlib import Path

from src.acoustic_core import (
    load_aformat_mono_files,
    aformat_to_bformat,
    export_bformat_wav,
)

RAW_DIR = Path("data/raw/usina_del_arte")
PROCESSED_DIR = Path("data/processed/usina_del_arte")

POSITIONS = {
    "J12": {
        "fuente": "F3",
        "posicion": "P4",
        "files": {
            "LF": RAW_DIR / "J12" / "F3 P4 LF SF 1_SS1.wav",
            "RF": RAW_DIR / "J12" / "F3 P4 RF SF 2_SS1.wav",
            "LB": RAW_DIR / "J12" / "F3 P4 LB SF 3_SS1.wav",
            "RB": RAW_DIR / "J12" / "F3 P4 RB SF 4_SS1.wav",
        },
    },
    "G6_SS1": {
        "fuente": "F2",
        "posicion": "P3",
        "files": {
            "LF": RAW_DIR / "G6_SS1" / "F2 P3 LF SF 1_SS1.wav",
            "RF": RAW_DIR / "G6_SS1" / "F2 P3 RF SF 2_SS1.wav",
            "LB": RAW_DIR / "G6_SS1" / "F2 P3 LB SF 3_SS1.wav",
            "RB": RAW_DIR / "G6_SS1" / "F2 P3 RB SF 4_SS1.wav",
        },
    },
}

for session_folder, meta in POSITIONS.items():
    print(f"\n=== {session_folder} (Fuente {meta['fuente']}, Posición {meta['posicion']}) ===")

    signals, fs = load_aformat_mono_files(
        {k: str(v) for k, v in meta["files"].items()}
    )
    bformat = aformat_to_bformat(signals, fs)
    print(f"  B-format generado: fs={fs} Hz, {len(bformat.W)} samples "
          f"({len(bformat.W) / fs * 1000:.1f} ms)")

    out_path = export_bformat_wav(
        bformat,
        PROCESSED_DIR / f"{session_folder}_{meta['fuente']}_{meta['posicion']}_Bformat.wav",
        layout="interleaved",
        order="WYZX",
    )
    print(f"  Exportado (interleaved WYZX / ACN): {out_path}")

print("\nListo. Recordá la nota sobre convención FuMa vs AmbiX antes de "
      "comparar valores de LF absolutos contra EASERA (ver acoustic_core.py).")
