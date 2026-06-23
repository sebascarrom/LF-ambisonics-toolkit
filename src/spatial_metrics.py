"""
spatial_metrics.py
===================
[FASE 2 — pendiente de implementación]

Extiende el análisis más allá de LF hacia parámetros direccionales y
espaciales complementarios, derivados del campo B-format completo
(W, X, Y, Z).

Parámetros previstos
---------------------
- LF_E / LF_L          : fracción lateral temprana/tardía (variantes ISO).
- LFC                   : Lateral Fraction Coefficient (ponderado por
                          coseno, alternativa a LF clásico — Hidaka &
                          Beranek 1995).
- IACC_early / IACC_late: Interaural Cross-Correlation, requiere síntesis
                          de señales biaurales desde B-format o medición
                          biaural complementaria.
- Mapas de energía direccional: distribución de energía en función de
  azimut/elevación, integrando sobre ventanas temporales (early/late),
  a partir de un beamformer de primer orden sobre W,X,Y,Z.
- EDC por banda (Schroeder backward integration), insumo para T30/EDT
  y para verificar la calidad de la RIR antes del cálculo de LF.

Dependencias previstas
------------------------
- acoustic_core.BFormatSignals como entrada común.
- Reutiliza detect_onset() y octave_band_filter() de acoustic_core.

Estado: NO IMPLEMENTADO.
"""

# from src.acoustic_core import BFormatSignals, detect_onset, octave_band_filter

raise NotImplementedError(
    "spatial_metrics.py es un stub. Implementar en Fase 2, luego de "
    "validar el cálculo de LF contra literatura de referencia."
)
