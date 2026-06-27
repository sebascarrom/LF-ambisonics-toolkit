# LF Ambisonics Toolkit

Pipeline de procesamiento de respuestas al impulso (RIR) ambisónicas
para el cálculo del parámetro **Lateral Fraction (LF)** y métricas
espaciales asociadas, según **ISO 3382-1**.

Desarrollado para la comparación entre dos tipologías arquitectónicas:

- **Usina del Arte** (auditorio moderno) — micrófono SoundField SP200.
- **Catedral Metropolitana de Buenos Aires** (recinto histórico) — micrófono Soyuz 013 Ambisonic.

Proyecto vinculado a la cátedra de Instrumentos y Mediciones Acústicas (IMA),
orientado a publicación en **FIA2026**.

---

## Estado del proyecto

| Fase | Contenido | Estado |
|------|-----------|--------|
| 1 | Conversión A→B, filtrado por octavas, cálculo LF ISO 3382-1 | ✅ Implementado |
| 1b | Validación contra EASERA Aurora, corrección bugs B-format | ✅ Completado |
| 2 | `room_loader.py`: multi-posición/fuente, IACC, EDC | 🔲 Pendiente |
| 3 | Comparación estadística Usina vs. Catedral | 🔲 Pendiente |
| 4 | GUI de escritorio (CustomTkinter + matplotlib) | ✅ Implementado |

---

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## GUI

```bash
python gui/app.py
```

La GUI permite importar RIRs en tres modos:

- **A-format multicanal (4ch):** un WAV de 4 canales en orden LF/RF/LB/RB.
- **4 archivos mono:** cada cápsula por separado.
- **B-format (4ch):** importación directa de un archivo ya convertido, con selector de orden de canales (WYZX / WXYZ). Útil para comparar contra exports de plugins comerciales.

Parámetros de onset, ventana de integración y alineación temporal ajustables desde la interfaz. Exporta a B-format WAV en orden WYZX (ACN, compatible con EASERA Aurora).

---

## Uso desde código

```python
from src.acoustic_core import (
    load_aformat_mono_files,
    align_aformat_channels,
    fine_align_channels,
    aformat_to_bformat,
    detect_onset_noise_floor,
    octave_band_filter,
    compute_LF_band,
    OCTAVE_BANDS_HZ,
)
import numpy as np

# Cargar 4 canales A-format (exportados desde DAW)
signals, fs = load_aformat_mono_files({
    "LF": "data/raw/usina_del_arte/J12/F3 P4 LF SF 1_SS1.wav",
    "RF": "data/raw/usina_del_arte/J12/F3 P4 RF SF 2_SS1.wav",
    "LB": "data/raw/usina_del_arte/J12/F3 P4 LB SF 3_SS1.wav",
    "RB": "data/raw/usina_del_arte/J12/F3 P4 RB SF 4_SS1.wav",
})

# Alineación temporal (necesaria para archivos exportados por pistas desde DAW)
aligned, _, onset = align_aformat_channels(signals, fs, noise_window_ms=50, threshold_db=20)
aligned_fine, _ = fine_align_channels(aligned, fs, onset, reference_channel="LF")

# Conversión A→B y cálculo de LF
bformat = aformat_to_bformat(aligned_fine, fs)
onset_sample = detect_onset_noise_floor(bformat.W, fs)

lf_vals = {}
for fc in OCTAVE_BANDS_HZ:
    Wf = octave_band_filter(bformat.W, fc, fs)
    Yf = octave_band_filter(bformat.Y, fc, fs)
    lf_vals[fc] = compute_LF_band(Wf, Yf, fs, onset=onset_sample)

print({fc: f"{lf:.3f}" for fc, lf in lf_vals.items()})
```

---

## Estructura del repositorio

```
LF-ambisonics-toolkit/
├── src/
│   ├── acoustic_core.py     # ✅ Módulo principal: A→B, LF, alineación, I/O
│   └── io_utils.py          # ✅ Exportación CSV/JSON
├── gui/
│   └── app.py               # ✅ GUI CustomTkinter
├── debug/                   # Scripts de diagnóstico y validación por posición
├── data/
│   ├── raw/                 # RIRs A-format originales
│   ├── processed/           # B-format exportado (WYZX/ACN)
│   └── results/             # Resultados LF en CSV
├── config/                  # usina_config.yaml, catedral_config.yaml
├── tests/                   # Tests automáticos
└── requirements.txt
```

---

## Convenciones B-format

El export usa orden **WYZX (ACN/AmbiX)**, compatible con EASERA Aurora:

| Canal | Señal | EASERA Aurora |
|-------|-------|---------------|
| ch0 | W (omnidireccional) | denominador LF |
| ch1 | Y (lateral izq-der) | numerador LF ← |
| ch2 | Z (vertical) | — |
| ch3 | X (frente-atrás) | — |

> **Nota:** usar orden WXYZ (FuMa) con EASERA hace que EASERA tome X como canal lateral → LF incorrecto.

---

## Validación

Pipeline validado contra EASERA Aurora en:

| Dataset | Diferencia pipeline vs. EASERA |
|---------|-------------------------------|
| Catedral Metropolitana (SY_RIR_TEST) | < 5% en 250–2000 Hz |
| Pori Promenadikeskus (s1_r1_sf) | < 5% en 125–2000 Hz |
| Usina del Arte — G6_SS1 alineado | < 15% en todas las bandas |
| Usina del Arte — J12 alineado | < 14% en todas las bandas |

La diferencia vs. plugins comerciales de conversión A→B es inherente a la
matriz tetraédrica estándar (sin calibración de cápsula), no a un error del
pipeline. Ver `RESUMEN_PROYECTO.md` sección 14 para el análisis completo.

---

## Tests

```bash
pytest tests/
```

---

## Referencias

`docs/papers/` y `docs/metodologia.md` para normativa ISO 3382-1 y literatura citada.
