"""
io_utils.py
============
Utilidades de exportación/importación compartidas entre módulos.

A diferencia de room_loader.py, spatial_metrics.py y stats_comparison.py
(que dependen de datos aún no disponibles), este módulo es funcional
desde ya: exporta resultados de acoustic_core.LFResult a formatos
estándar para análisis posterior (Excel, estadística, paper).
"""

import csv
import json
from pathlib import Path
from typing import Union

# Import relativo dentro del paquete src/
from .acoustic_core import LFResult


def lfresult_to_dict(result: LFResult, metadata: dict = None) -> dict:
    """
    Convierte un LFResult a un dict plano, listo para serializar.

    Parameters
    ----------
    result   : LFResult devuelto por acoustic_core.analyze_rir().
    metadata : dict opcional con info adicional (recinto, posición,
               fuente, micrófono, fecha de medición, etc.) que se
               antepone a los campos de resultado.

    Returns
    -------
    dict con una clave por banda de octava ('LF_125', 'LF_250', ...),
    'LF_mean', 'onset_ms', 'fs', y los campos de metadata si se proveen.
    """
    out = dict(metadata) if metadata else {}
    out["onset_ms"] = round(result.onset_ms, 3)
    out["fs"] = result.fs
    for fc in result.bands_hz:
        out[f"LF_{fc}"] = result.LF_per_band.get(fc, float("nan"))
    out["LF_mean"] = result.LF_mean
    return out


def export_results_csv(rows: list[dict], path: Union[str, Path]) -> None:
    """
    Exporta una lista de resultados (ver lfresult_to_dict) a un CSV.

    Parameters
    ----------
    rows : lista de dicts, todos con las mismas claves (usar
           lfresult_to_dict para generarlos de forma consistente).
    path : ruta de salida del archivo .csv.
    """
    if not rows:
        raise ValueError("No hay resultados para exportar (lista vacía).")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_results_json(rows: list[dict], path: Union[str, Path]) -> None:
    """
    Exporta una lista de resultados a JSON (útil para la GUI web).

    Parameters
    ----------
    rows : lista de dicts (ver lfresult_to_dict).
    path : ruta de salida del archivo .json.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
