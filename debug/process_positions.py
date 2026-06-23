"""
process_positions.py
======================
Procesa dos posiciones de medición (Usina del Arte) SIN alineación de
canales — versión legacy, previa al descubrimiento de la desincronización
entre cápsulas.

Útil como referencia comparativa: los resultados aquí son INCORRECTOS para
G6_SS1 (y probablemente también para J12) por la desincronización del DAW.
Ver process_G6_SS1_aligned.py para el pipeline corregido.

Carpetas DAW → identidad acústica:
    J12      → Fuente F3, Posición P4
    G6_SS1   → Fuente F2, Posición P3

Uso (desde la raíz del proyecto):
    python debug/process_positions.py
"""

from pathlib import Path as _Path
import sys as _sys, os as _os
_ROOT = _Path(__file__).resolve().parent.parent
_sys.path.insert(0, str(_ROOT))
_os.chdir(_ROOT)

from pathlib import Path

from src.acoustic_core import analyze_rir
from src.io_utils import lfresult_to_dict, export_results_csv

RAW_DIR = Path("data/raw/usina_del_arte")

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

results_table = []

for session_folder, meta in POSITIONS.items():
    print(f"\n=== Procesando {session_folder} "
          f"(Fuente {meta['fuente']}, Posición {meta['posicion']}) ===")

    result = analyze_rir({k: str(v) for k, v in meta["files"].items()})
    print(result.summary())

    row = lfresult_to_dict(result, metadata={
        "session_folder": session_folder,
        "fuente": meta["fuente"],
        "posicion": meta["posicion"],
    })
    results_table.append(row)

output_csv = "data/results/usina_del_arte/LF_results_J12_G6SS1.csv"
export_results_csv(results_table, output_csv)
print(f"\nResultados exportados a {output_csv}")
