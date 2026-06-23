"""
stats_comparison.py
=====================
[FASE 3 — pendiente de implementación]

Comparación espacial y estadística entre posiciones dentro de un mismo
recinto, y entre los dos recintos (Usina del Arte vs. Catedral
Metropolitana).

Análisis previstos
--------------------
- Estadística descriptiva por banda y por recinto (media, desvío
  estándar, intervalo de confianza) sobre LF y demás parámetros.
- Comparación entre posiciones dentro de un mismo recinto (dispersión
  espacial del campo lateral — relevante para la discusión de
  uniformidad acústica).
- Comparación entre recintos: pruebas de hipótesis (t-test / Mann-Whitney
  según normalidad) por banda de octava, con corrección por comparaciones
  múltiples (Bonferroni o FDR) dado que se testean 6 bandas.
- Nota metodológica obligatoria en el paper: los dos recintos fueron
  capturados con micrófonos ambisónicos distintos (SoundField SP200 vs.
  Soyuz 013), lo cual introduce una fuente de variabilidad no atribuible
  a la acústica del recinto — debe declararse como limitación y, de ser
  posible, cuantificarse (ver discusión sobre corrección frecuencial en
  el módulo de conversión A→B).

Dependencias previstas
------------------------
- Resultados tabulares producidos por room_loader.py
- scipy.stats para pruebas de hipótesis
- pandas para organización de resultados multi-recinto/multi-posición

Estado: NO IMPLEMENTADO.
"""

# import pandas as pd
# from scipy import stats

raise NotImplementedError(
    "stats_comparison.py es un stub. Implementar en Fase 3, luego de "
    "tener resultados de LF organizados por recinto/posición/fuente."
)
