# LF Ambisonics Toolkit

Pipeline de procesamiento de respuestas al impulso (RIR) ambisónicas
para el cálculo del parámetro **Lateral Fraction (LF)** y métricas
espaciales asociadas, según **ISO 3382-1**.

Desarrollado para la comparación entre dos tipologías arquitectónicas
contrastantes:

- **Usina del Arte** (auditorio moderno) — micrófono SoundField SP200.
- **Catedral Metropolitana de Buenos Aires** (recinto histórico colonial)
  — micrófono Soyuz 013 Ambisonic.

Proyecto vinculado a la cátedra de Instrumentos y Mediciones Acústicas
(IMA) y orientado a publicación en FIA2026.

## Estado del proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Conversión A→B, filtrado por octavas, cálculo de LF | ✅ Implementado (`src/acoustic_core.py`) |
| 2 | Carga multi-recinto, parámetros espaciales extendidos (IACC, mapas direccionales) | 🔲 Pendiente |
| 3 | Comparación estadística entre recintos | 🔲 Pendiente |
| 4 | GUI web interactiva | 🔲 Pendiente |

## Estructura del repositorio

```
LF-ambisonics-toolkit/
├── config/             # Mapeo de canales y posiciones por recinto (YAML)
├── src/                # Código de procesamiento
│   ├── acoustic_core.py    ✅ A→B, filtrado octava, LF (ISO 3382-1)
│   ├── room_loader.py       🔲 Organización multi-posición/fuente
│   ├── spatial_metrics.py   🔲 IACC, mapas direccionales, EDC
│   ├── stats_comparison.py  🔲 Comparación estadística entre recintos
│   └── io_utils.py         ✅ Exportación CSV/JSON
├── data/
│   ├── raw/             # RIRs A-format crudas por recinto/posición
│   ├── processed/       # B-format cacheado (opcional)
│   └── results/         # Resultados exportados
├── notebooks/           # Exploración y validación
├── gui/                 # Interfaz web (Fase 4)
├── docs/
│   ├── papers/           # Referencias bibliográficas
│   └── metodologia.md    # Documentación metodológica (sincronizada con el paper)
└── tests/                # Tests automáticos (señales sintéticas)
```

## Instalación

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Uso rápido (Fase 1)

```python
from src.acoustic_core import analyze_rir

# Desde un WAV de 4 canales:
result = analyze_rir("data/raw/usina_del_arte/P1/P1_4ch.wav")
print(result.summary())

# Desde 4 archivos mono (nomenclatura Usina):
result = analyze_rir({
    "LeftFront":  "data/raw/usina_del_arte/P1/P1_LeftFront.wav",
    "RightFront": "data/raw/usina_del_arte/P1/P1_RightFront.wav",
    "LeftBack":   "data/raw/usina_del_arte/P1/P1_LeftBack.wav",
    "RightBack":  "data/raw/usina_del_arte/P1/P1_RightBack.wav",
})
print(result.summary())
```

```bash
# O desde terminal:
python src/acoustic_core.py data/raw/usina_del_arte/P1/P1_4ch.wav
```

## Tests

```bash
pytest tests/
```

## Próximos pasos

1. Nomenclar y organizar los archivos exportados del DAW para ambos
   recintos en `data/raw/`.
2. Completar `config/usina_config.yaml` y `config/catedral_config.yaml`
   con las rutas reales de posiciones y fuentes.
3. Implementar `room_loader.py` para automatizar el procesamiento
   multi-posición.
4. Validar `LF_mean` contra valores de referencia de literatura
   (Barron, Beranek, Hidaka).

## Referencias

Ver `docs/papers/` y `docs/metodologia.md` para el detalle de normativa
(ISO 3382-1) y literatura citada en las decisiones metodológicas del
código.
