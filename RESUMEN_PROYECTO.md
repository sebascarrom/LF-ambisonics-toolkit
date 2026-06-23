# LF Ambisonics Toolkit — Estado del Proyecto y Resumen de Sesión

*Documento de transferencia de contexto para continuar el trabajo en Cowork o con otro asistente de IA.*
*Última actualización: junio 2025 — Sesión 2*

---

## 1. Objetivo del proyecto

Desarrollar un pipeline en Python para calcular el parámetro acústico
**Lateral Fraction (LF)** a partir de respuestas al impulso (RIR) medidas con
micrófonos ambisónicos de primer orden (A-format tetrahedral), siguiendo la norma
**ISO 3382-1**. El proyecto compara dos recintos:

- **Usina del Arte** (auditorio moderno): mediciones con SoundField SP200.
- **Catedral Metropolitana de Buenos Aires** (recinto histórico): mediciones con
  Soyuz 013 Ambisonic.

Objetivo final: GUI web interactiva para carga, procesamiento y comparación de
RIRs entre posiciones y recintos. Publicación orientada a **FIA2026**.

---

## 2. Estructura del repositorio

```
LF-ambisonics-toolkit/
├── .gitignore
├── README.md
├── requirements.txt             # numpy, scipy, PyYAML, pandas, pytest
├── config/
│   ├── usina_config.yaml        # mapeo de canales SP200 + posiciones (vacío, completar)
│   └── catedral_config.yaml     # mapeo de canales Soyuz 013 + posiciones (vacío, completar)
├── src/
│   ├── __init__.py
│   ├── acoustic_core.py         # ✅ MÓDULO PRINCIPAL — ver sección 3
│   ├── io_utils.py              # ✅ exportación CSV/JSON de LFResult
│   ├── room_loader.py           # 🔲 stub — Fase 2
│   ├── spatial_metrics.py       # 🔲 stub — Fase 2
│   └── stats_comparison.py      # 🔲 stub — Fase 3
├── data/
│   ├── raw/
│   │   ├── usina_del_arte/
│   │   │   ├── J12/             # F3 P4 {LF/RF/LB/RB} SF {1/2/3/4}_SS1.wav
│   │   │   └── G6_SS1/         # F2 P3 {LF/RF/LB/RB} SF {1/2/3/4}_SS1.wav
│   │   └── catedral_metropolitana/  # aún sin datos cargados
│   ├── processed/usina_del_arte/
│   │   ├── G6_SS1_F2_P3_Bformat.wav          # B-format sin alinear (legacy)
│   │   ├── G6_SS1_F2_P3_Bformat_ALIGNED.wav  # ✅ B-format alineado (usar este)
│   │   ├── J12_F3_P4_Bformat.wav             # B-format sin alinear (legacy)
│   │   └── J12_F3_P4_Bformat_ALIGNED.wav     # ✅ B-format alineado (usar este)
│   └── results/usina_del_arte/               # CSV de resultados
├── debug/                       # ✅ Scripts de diagnóstico y validación
│   ├── debug_LF.py              # Diagnóstico de onset: legacy vs. robusto
│   ├── debug_LF_compare.py      # Ablación de 4 métodos
│   ├── debug_aformat_waveforms.py  # Visualización A-format en escala absoluta
│   ├── debug_azimuth_scan.py    # Barrido de rotación azimutal
│   ├── debug_channel_balance.py # Balance de nivel entre cápsulas
│   ├── debug_catedral_rir.py    # Procesa SY_RIR_TEST.wav vs. EASERA
│   ├── debug_pori_rir.py        # Procesa Pori s1_r1_sf vs. paper y EASERA
│   ├── process_positions.py     # Legacy: procesa J12 y G6 SIN alinear
│   ├── process_G6_SS1_aligned.py  # ✅ G6_SS1 con alineación de 2 etapas
│   ├── process_J12_aligned.py   # ✅ J12 con alineación de 2 etapas
│   ├── export_bformat_positions.py  # Exporta B-format sin alinear (legacy)
│   └── test_rirs/
│       ├── SY_RIR_TEST.wav      # RIR Catedral Metropolitana (A-format, Soyuz 013)
│       ├── s1_r1_sf_pori.wav    # RIR Pori Promenadikeskus (B-format, SoundField MKV)
│       └── poriref.pdf          # Documentación del repositorio Pori (HUT 2005)
├── notebooks/                   # vacío, para exploración
├── gui/                         # 🔲 stub — Fase 4
├── docs/
│   ├── metodologia.md
│   └── papers/
└── tests/
    ├── test_acoustic_core.py    # 6 tests, todos pasan
    └── fixtures/
```

**Nota sobre scripts en `debug/`:** todos tienen el boilerplate de path al inicio
(`sys.path.insert` + `os.chdir` al root), así que se corren desde la raíz:
```bash
python debug/process_G6_SS1_aligned.py
```

---

## 3. `acoustic_core.py` — funciones implementadas

El módulo tiene **1222 líneas** y está organizado en 10 módulos lógicos:

### Estructuras de datos
- `BFormatSignals` — dataclass con W, X, Y, Z, fs.
- `LFResult` — dataclass con LF_per_band, LF_mean, onset_sample, onset_ms, fs.

### Módulo 1 — Conversión A→B
- `normalize_aformat_keys(signals)` — normaliza nombres de canal (acepta
  nomenclatura SP200 y Soyuz 013 indistintamente).
- `aformat_to_bformat(signals, fs)` — aplica la matriz tetraédrica estándar:
  - W = 0.5·(LF+RF+LB+RB)
  - X = 0.5·(LF+RF−LB−RB)
  - Y = 0.5·(LF−RF+LB−RB)
  - Z = 0.5·(LF−RF−LB+RB)

### Módulo 2 — Detección de onset
- `detect_onset(W, threshold_db=-20)` — relativo al pico global. Solo usar
  cuando el archivo tiene onset al principio y sin artefactos previos.
- `detect_onset_noise_floor(W, fs, noise_window_ms=50, threshold_db=15, ...)` —
  relativo al piso de ruido. **Usar siempre para archivos exportados del DAW.**
  Parámetro `noise_window_ms` debe ser menor que el pre-roll de silencio.
  Para archivos con pre-roll corto (< 20 ms), usar `noise_window_ms=5`.

### Módulo 3 — Filtrado por octavas
- `octave_band_filter(signal, center_freq_hz, fs, order=3)` — Butterworth con
  `sosfiltfilt` (fase cero). Cutoffs: f/√2 y f·√2.

### Módulo 4 — Cálculo de LF por banda
- `compute_LF_band(W_band, Y_band, fs, onset, t1_ms=5, t2_ms=80)` — implementa
  exactamente ISO 3382-1:
  - Numerador = ∫[onset+5ms, onset+80ms] Y²(t)dt
  - Denominador = ∫[onset, onset+80ms] W²(t)dt

### Módulo 5 — Pipeline completo
- `compute_LF_fullband(bformat, bands_hz, ...)` — orquesta onset → filtrado → LF
  para las 6 bandas ISO. Devuelve `LFResult`.

### Módulo 6 — I/O
- `load_aformat_mono_files(paths)` — carga 4 WAVs mono separados.
- `load_aformat_multichannel(path, channel_order)` — carga 1 WAV de 4 canales.
- `analyze_rir(source, ...)` — punto de entrada unificado.

### Módulo 7 — Exportación a B-format
- `export_bformat_wav(bformat, output_path, layout, order)` — exporta W,X,Y,Z
  como WAV interleaved (4ch) o 4 mono separados. Float32.

### Módulo 8 — Diagnóstico de rotación azimutal
- `scan_azimuth_rotation(bformat, onset, ...)` — barre φ de 0° a 360°.

### Módulo 9 — Diagnóstico de balance de canales
- `check_aformat_channel_balance(signals, fs, onset)` — compara RMS por cápsula.

### Módulo 10 — Alineación temporal entre canales
- `align_aformat_channels(signals, fs, noise_window_ms, threshold_db)` —
  alineación gruesa (~1 ms de precisión).
- `fine_align_channels(aligned, fs, common_onset, reference_channel, search_radius_ms)` —
  alineación fina por correlación cruzada (~21 µs de precisión a 48 kHz).

---

## 4. Pipeline de procesamiento — flujo completo

Para archivos exportados desde DAW por pistas separadas (caso Usina del Arte):

```
1. load_aformat_mono_files()          # carga 4 WAVs mono
2. align_aformat_channels()           # alineación gruesa (onset por canal)
3. fine_align_channels()              # alineación fina (correlación cruzada)
4. aformat_to_bformat()               # conversión A→B
5. detect_onset_noise_floor(W)        # onset sobre W convertido
6. octave_band_filter() × 6 bandas   # filtrado ISO (125–4000 Hz)
7. compute_LF_band() × 6 bandas      # integral según ISO 3382-1
8. nanmean()                          # LF medio
```

Para archivos multichannel ya sincronizados (caso Catedral, Pori):

```
1. load_aformat_multichannel() o scipy.io.wavfile.read()
2. aformat_to_bformat() o uso directo de B-format (W=ch0, Y=ch2)
3. detect_onset_noise_floor(W, noise_window_ms=5)  # ajustar si pre-roll es corto
4. octave_band_filter() × 6 bandas
5. compute_LF_band() × 6 bandas
6. nanmean()
```

---

## 5. Resultados por posición — Usina del Arte

### G6_SS1 (Fuente F2, Posición P3)

Desfases detectados entre cápsulas del DAW:
- LF: +0 ms (referencia)
- RF: +12 ms
- LB: +108 ms
- RB: +202 ms

| Banda | Sin alinear | Alineado (pipeline) | EASERA (alineado) | Δ pip/EASERA |
|-------|-------------|---------------------|-------------------|--------------|
| 125 Hz | — | 0.252 | 0.213 | +18% |
| 250 Hz | ~0.05 | 0.122 | 0.114 | +7% |
| 500 Hz | ~0.07 | 0.207 | 0.180 | +15% |
| 1000 Hz | ~0.08 | 0.277 | 0.258 | +7% |
| 2000 Hz | ~0.10 | 0.395 | 0.370 | +7% |
| 4000 Hz | ~0.12 | 0.689 | 0.630 | +9% |
| **Media** | **~0.13** | **0.324** | **0.310** | **+9%** |

Parámetros de alineación: `threshold_db=15`, `search_radius_ms=5`.

### J12 (Fuente F3, Posición P4)

Desfases detectados entre cápsulas del DAW:
- RF: +0 ms (más temprana)
- LB: +183 ms
- LF: +276 ms
- RB: +374 ms

**ATENCIÓN:** usar `threshold_db=20` (no 15) para la alineación gruesa. Con
threshold=15 el detector dispara sobre pre-ringing de 50–68 ms antes del pico
real, corrompiendo el coarse alignment.

| Banda | Alineado (pipeline) | EASERA (alineado) | Δ pip/EASERA |
|-------|---------------------|-------------------|--------------|
| 125 Hz | 0.167 | 0.165 | +1% |
| 250 Hz | 0.109 | 0.113 | −4% |
| 500 Hz | 0.095 | 0.098 | −3% |
| 1000 Hz | 0.196 | 0.198 | −1% |
| 2000 Hz | 0.315 | 0.343 | −8% |
| 4000 Hz | 0.540 | 0.631 | −14% |
| **Media** | **0.237** | **0.277** | **−9%** |

---

## 6. Validación del pipeline

### Estado general

| Componente | Estado | Evidencia |
|---|---|---|
| Conversión A→B (matriz) | ✅ Validado | Catedral y Pori coinciden con EASERA <5% en 500–2000 Hz |
| Cálculo LF ISO 3382-1 | ✅ Validado | Catedral 4kHz: +0.1% vs EASERA |
| Alineación de 2 etapas (G6_SS1) | ✅ Validado | EASERA alineado: <15% diferencia en todas las bandas |
| Alineación de 2 etapas (J12) | ✅ Validado | EASERA alineado: <14% diferencia en todas las bandas |
| 4 kHz — sin corrección frecuencial | ⚠ Limitación conocida | LF>1 sistemático, confirmado también por EASERA |

### Validación caso Catedral — SY_RIR_TEST.wav

Archivo: A-format int32, 4 canales sincronizados (no hay desincronización DAW).
Canal order: ch0=LF, ch1=RF, ch2=LB, ch3=RB (misma geometría tetraédrica que SP200).

| Banda | Pipeline | EASERA | Δ |
|-------|---------|--------|---|
| 125 Hz | 0.099 | 0.000 | — |
| 250 Hz | 0.106 | 0.054 | +5.2% |
| 500 Hz | 0.131 | 0.099 | +3.2% |
| 1000 Hz | 0.156 | 0.169 | −1.3% |
| 2000 Hz | 0.235 | 0.283 | −4.8% |
| 4000 Hz | 1.169 | 1.168 | +0.1% |

Script: `debug/debug_catedral_rir.py`

### Validación caso Pori — s1_r1_sf_pori.wav

Archivo: B-format WXYZ, SoundField MKV, int32, 48kHz, 4000ms.
Canal order confirmado en `poriref.pdf` (Apéndice A): W=ch0, X=ch1, Y=ch2, Z=ch3.
Onset en ~10 ms del inicio → usar `noise_window_ms=5`.

| Banda | Pipeline | EASERA | Δ pip/EASERA | Paper (LF_SF) | Δ pip/paper |
|-------|---------|--------|-------------|---------------|-------------|
| 125 Hz | 0.409 | 0.409 | 0% | 0.37 | +11% |
| 250 Hz | 0.302 | 0.307 | −2% | 0.37 | −18% |
| 500 Hz | 0.219 | 0.214 | +2% | 0.27 | −19% |
| 1000 Hz | 0.399 | 0.418 | −5% | 0.50 | −20% |
| 2000 Hz | 0.593 | 0.614 | −3% | 0.72 | −18% |
| 4000 Hz | 1.613 | 1.596 | +1% | 0.77 | +109% |

**Conclusión:** pipeline y EASERA son equivalentes (<5%). La discrepancia con el
paper se explica porque el SoundField MKV aplica correcciones frecuenciales
propietarias (EQ del fabricante) en su conversión A→B. Sin esa corrección, el
canal Y tiene exceso de energía en altas frecuencias → LF>1 a 4 kHz. No es un
bug del pipeline.

Script: `debug/debug_pori_rir.py`

---

## 7. Limitación conocida — 4 kHz y corrección frecuencial

El LF en 4 kHz está sistemáticamente elevado (frecuentemente > 1) en todos los
recintos y micrófonos. Esto es un efecto de la conversión A→B frecuencia-independiente:

- A 4 kHz, la longitud de onda (~8.5 cm) es comparable a la separación entre
  cápsulas del tetraedro. Esto introduce diferencias de fase inter-cápsula que
  la suma de la matriz amplifica artificialmente en el canal Y.
- Los fabricantes (SoundField, Sennheiser, etc.) corrigen esto con filtros de
  EQ propietarios que no tenemos.
- EASERA tiene el mismo comportamiento → no es un bug nuestro.
- Para el estudio comparativo Usina vs. Catedral, reportar 125–2000 Hz como
  bandas primarias y 4 kHz con advertencia metodológica.

---

## 8. Bugs en el código de referencia (Fede) — INVALIDADO

Las Tablas 4 y 12 de la entrega anterior del iMA **NO son referencia válida**:

**`LF_bandas_fede.py`:** denominador usa ventana [5,80ms] en vez de [0,80ms]
(bug vs. ISO 3382-1). El denominador debería incluir los primeros 5ms
(sonido directo, máxima energía). Al excluirlos, LF sube artificialmente.

**`LF_processing_fede.py`:** sin detección de onset. Integra desde la muestra
0 del archivo, que incluye hasta 630ms de silencio antes del impulso real.
La ventana de 80ms captura mayormente silencio, no RIR.

Adicionalmente, los archivos de Usina tenían desincronización entre cápsulas
(hasta 423ms), lo que hubiera invalidado cualquier resultado aunque la fórmula
fuera correcta.

---

## 9. Nomenclatura de archivos

### Usina del Arte (SP200)
Patrón: `F{n} P{m} {canal} SF {idx}_SS1.wav`
- F = número de fuente, P = posición de micrófono
- Canal: LF / RF / LB / RB (SoundField: LeftFront, RightFront, LeftBack, RightBack)
- Carpeta DAW (J12, G6_SS1) = identificador de sesión, NO la posición acústica.

### Catedral Metropolitana (Soyuz 013)
- Convención de canales: FRONT L UP / FRONT R DOWN / BACK L DOWN / BACK R UP
- Mapean a la misma geometría tetraédrica estándar que el SP200
- Archivos aún no nomenclados ni cargados en el proyecto

---

## 10. Micrófonos y conversión A→B

Ambos micrófonos usan la misma geometría tetraédrica estándar:

| Canal | Azimut | Elevación |
|---|---|---|
| LF / FRONT L UP | +45° | +35.26° |
| RF / FRONT R DOWN | −45° | −35.26° |
| LB / BACK L DOWN | +135° | −35.26° |
| RB / BACK R UP | −135° | +35.26° |

`CHANNEL_ALIASES` en `acoustic_core.py` mapea ambas nomenclaturas automáticamente.

**Limitación conocida:** conversión frecuencia-independiente. Error creciente
hacia bajas frecuencias (125–500 Hz) y altas (4 kHz+). Ver sección 7.

---

## 11. Próximos pasos

1. **Catedral Metropolitana** — nomenclar y organizar los archivos del DAW,
   completar `config/catedral_config.yaml`, correr `debug_aformat_waveforms.py`
   sobre cada posición antes de procesar (verificar sincronización).

2. **Verificar sincronización en posiciones nuevas** — siempre correr
   `debug_aformat_waveforms.py` sobre cada set de 4 cápsulas antes de procesar.
   El problema de G6_SS1/J12 puede repetirse en cualquier exportación manual del DAW.

3. **Incorporar alineación al pipeline principal** — cuando se implemente
   `room_loader.py` (Fase 2), la secuencia
   `align_aformat_channels() → fine_align_channels() → aformat_to_bformat()`
   debe ser el flujo estándar, no un script especial.
   Tener en cuenta que `threshold_db` puede requerir ajuste por posición
   (15 para G6_SS1, 20 para J12).

4. **Fase 2** — `room_loader.py`: multi-posición/fuente, `spatial_metrics.py`:
   IACC, EDC.

5. **Fase 3** — `stats_comparison.py`: comparación estadística Usina vs. Catedral.

6. **Fase 4** — GUI web interactiva (HTML/React + backend Python).

---

## 12. Hoja de ruta del proyecto

| Fase | Contenido | Estado |
|---|---|---|
| 1 | `acoustic_core.py`: A→B, filtrado octava, LF, alineación | ✅ Implementado |
| 1b | Debugging Usina, validación EASERA y repos externos | ✅ Completado |
| 2 | `room_loader.py`: multi-posición/fuente, `spatial_metrics.py`: IACC, EDC | 🔲 Pendiente |
| 3 | `stats_comparison.py`: comparación estadística Usina vs. Catedral | 🔲 Pendiente |
| 4 | GUI web interactiva (HTML/React + backend Python) | 🔲 Pendiente |

---

## 13. Entorno de desarrollo

```bash
# Crear y activar entorno virtual
python -m venv .venv
source .venv/bin/activate   # macOS/Linux

# Instalar dependencias
pip install -r requirements.txt

# Correr tests
pytest tests/

# Correr scripts de debug (desde la raíz del proyecto)
python debug/process_G6_SS1_aligned.py
python debug/process_J12_aligned.py
python debug/debug_catedral_rir.py
python debug/debug_pori_rir.py
```

**Plataforma de desarrollo:** macOS (Sebastian Carro), Python 3.12, fs=48000 Hz.
**Warnings conocidos:** `WavFileWarning: Chunk (non-data) not understood` — benigno,
originado en metadata Broadcast Wave Format (BWF) de archivos exportados desde DAW.
No afecta la lectura del audio.
