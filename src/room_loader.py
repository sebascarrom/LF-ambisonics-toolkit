"""
room_loader.py
==============
[FASE 2 — pendiente de implementación]

Organiza mediciones de RIR por recinto, posición de medición y posición
de fuente, leyendo la configuración correspondiente desde config/*.yaml.

Responsabilidades previstas
----------------------------
- Leer config/usina_config.yaml o config/catedral_config.yaml.
- Resolver, para cada posición de medición (P1, P2, ...) y cada fuente
  (SS1, SS2, ...), las rutas a los 4 archivos A-format correspondientes.
- Invocar acoustic_core.analyze_rir() para cada combinación posición/fuente.
- Promediar resultados entre fuentes cuando corresponda (ISO 3382-1
  recomienda promediar sobre al menos 2 posiciones de fuente).
- Devolver una estructura tabular (lista de dicts o DataFrame) lista
  para exportar vía io_utils.export_results().

Estructura de datos esperada (config YAML)
-------------------------------------------
room_name: "usina_del_arte"
microphone: "SoundField SP200"
channel_order: ["LeftFront", "RightFront", "LeftBack", "RightBack"]
positions:
  P1:
    sources:
      SS1:
        LeftFront:  "data/raw/usina_del_arte/P1/P1_SS1_LeftFront.wav"
        RightFront: "data/raw/usina_del_arte/P1/P1_SS1_RightFront.wav"
        LeftBack:   "data/raw/usina_del_arte/P1/P1_SS1_LeftBack.wav"
        RightBack:  "data/raw/usina_del_arte/P1/P1_SS1_RightBack.wav"
      SS2:
        ...

Estado: NO IMPLEMENTADO. Placeholder para cuando los archivos de
ambos recintos estén nomenclados y organizados en data/raw/.
"""

# import yaml
# from pathlib import Path
# from src.acoustic_core import analyze_rir

raise NotImplementedError(
    "room_loader.py es un stub. Implementar cuando los archivos de "
    "Usina y Catedral estén nomenclados y la estructura de config/*.yaml "
    "esté definida."
)
