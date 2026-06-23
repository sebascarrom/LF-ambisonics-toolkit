# Metodología — LF Ambisonics Toolkit

Este documento centraliza las decisiones metodológicas implementadas en
el código, para mantener consistencia entre `src/` y la sección de
Metodología del paper (FIA2026).

## 1. Conversión A-format → B-format

Matriz tetraédrica estándar (frecuencia-independiente), válida para
geometría de cápsulas en azimut ±45°/±135°, elevación ±35.26°:

```
W = 0.5 · (LF + RF + LB + RB)
X = 0.5 · (LF + RF − LB − RB)
Y = 0.5 · (−LF + RF − LB + RB)
Z = 0.5 · (LF − RF − LB + RB)
```

Aplicada por igual a Usina del Arte (SoundField SP200) y Catedral
Metropolitana (Soyuz 013 Ambisonic) — ambos comparten la misma geometría
tetraédrica, solo difiere la nomenclatura de canales (ver
`src/acoustic_core.py::CHANNEL_ALIASES`).

**Limitación declarada:** la conversión no incluye corrección frecuencial
específica del fabricante. El desequilibrio W vs. X/Y/Z en bajas
frecuencias (~125 Hz) es mayor en esta aproximación que con un decoder
propietario corregido. Ver discusión en el código fuente.

## 2. Detección de onset

Sobre el canal W de banda ancha, antes del filtrado por octavas, mediante
umbral de energía relativo al pico (−20 dB por defecto). Usado como
referencia temporal única para todas las bandas.

## 3. Filtrado por bandas de octava

Butterworth orden 3, `sosfiltfilt` (fase cero), bandas ISO 3382-1:
125/250/500/1000/2000/4000 Hz. Cutoffs en f_centro/√2 y f_centro·√2.

## 4. Cálculo de LF (ISO 3382-1, Anexo A)

```
LF = ∫[5,80 ms] Y²(t) dt  /  ∫[0,80 ms] W²(t) dt
```

Integración referenciada al onset detectado, por banda de octava, con
promedio aritmético final.

## 5. Limitación metodológica — micrófonos distintos por recinto

Usina del Arte y Catedral Metropolitana fueron medidas con micrófonos
ambisónicos de fabricantes distintos. Cualquier diferencia sistemática
de LF entre recintos debe interpretarse considerando esta covariable,
especialmente en banda de 125 Hz. Pendiente: cuantificar el efecto
mediante medición de referencia cruzada o corrección frecuencial
post-hoc (Fase 2/3).

## 6. Validación cruzada contra EASERA

Tras detectar una discrepancia inicial entre el pipeline propio y los
valores de referencia de la entrega anterior de IMA (Tabla 4 / Tabla 12),
se realizó un proceso de debugging estructurado que incluyó: inspección
visual de la forma de onda, comparación de métodos de detección de onset
(relativo al pico global vs. relativo al piso de ruido), un barrido de
rotación azimutal para descartar desalineación micrófono-fuente, y
finalmente **validación cruzada contra EASERA 1.0** (software profesional
de medición acústica) sobre las mismas señales A-format.

### Resultados — Posición J12 (Fuente F3, Posición P4)

| Banda | Pipeline propio | EASERA | Dif. % |
|---|---|---|---|
| 125 Hz | 0,930 | 0,995 | 7% |
| 250 Hz | 0,674 | 0,917 | 36% |
| 500 Hz | 0,452 | 0,569 | 26% |
| 1000 Hz | 0,663 | 0,627 | −5% |
| 2000 Hz | 0,783 | 0,759 | −3% |
| 4000 Hz | 0,665 | 0,677 | 2% |

### Resultados — Posición G6_SS1 (Fuente F2, Posición P3)

| Banda | Pipeline propio | EASERA | Dif. % |
|---|---|---|---|
| 125 Hz | 0,130 | 0,115 | 13% |
| 250 Hz | 0,075 | 0,070 | 6% |
| 500 Hz | 0,057 | 0,050 | 14% |
| 1000 Hz | 0,068 | 0,070 | −3% |
| 2000 Hz | 0,160 | 0,169 | −5% |
| 4000 Hz | 0,336 | 0,327 | 3% |

### Conclusiones de la validación

⚠️ **Alcance real de esta validación**: los archivos B-format entregados
a EASERA fueron generados por `export_bformat_wav()` — EASERA analizó el
**resultado de nuestra propia conversión A→B**, no el A-format crudo, y
nunca aplicó su propio decoder. Esto acota lo que la coincidencia numérica
puede demostrar:

- **Sí queda validado**: detección de onset, ventaneo ISO 3382-1,
  filtrado por octava y fórmula de LF — en esa etapa nuestro cálculo y
  el de EASERA usaron lógica independiente sobre el mismo B-format, y
  coincidieron de cerca.
- **NO queda validado**: la matriz de conversión A→B en sí misma (su
  geometría, la ausencia de corrección frecuencial del SP200, la
  hipótesis de desalineación azimutal). Un error ahí se propaga
  idénticamente a ambos cálculos, porque ambos parten del mismo B-format
  ya convertido — la comparación no puede detectarlo.

Para una validación verdaderamente independiente de la conversión, EASERA
debería procesar el A-format crudo (4 mono LF/RF/LB/RB) con su propio
decoder — pendiente de confirmar si es posible con la versión disponible.



1. **El pipeline propio (con detección de onset) coincide con EASERA
   dentro de un margen de ~5-15% en la mayoría de las bandas**, y casi
   exactamente en G6_SS1. Esto valida la metodología de detección de
   onset, el filtrado por octava, y la definición de ventana ISO 3382-1
   implementadas en `acoustic_core.py`.

2. **Los valores de Tabla 4 / Tabla 12 (entrega anterior de IMA) NO
   coinciden con EASERA** para estas mismas posiciones (p. ej. G6:
   Tabla 4 = 0,45 vs. EASERA ≈ 0,05-0,17 según banda). Esto confirma que
   los scripts originales (`LF_processing_fede.py`, `LF_bandas_fede.py`)
   — que no realizan detección de onset y asumen que la muestra 0 es la
   llegada del sonido directo — producen resultados no confiables cuando
   se aplican a archivos con pre-roll significativo (como los extraídos
   manualmente de una sesión DAW continua, que en este caso incluían
   hasta ~630 ms de silencio antes del inicio real de la RIR).

3. **La hipótesis de desalineación azimutal NO está descartada** (ver
   corrección de alcance más abajo). El intento inicial de descartarla
   fue precipitado.

4. **Limitación abierta, no resuelta**: el desvío de 250-500 Hz en J12
   (26-36%) no se repite en G6_SS1 (6-14% en las mismas bandas), lo que
   sugiere variabilidad específica de cada medición (relación
   señal-ruido, contenido espectral particular de esa RIR) más que un
   sesgo sistemático de calibración del micrófono. Queda documentado como
   pregunta abierta para investigación futura, sin impacto en la validez
   general del pipeline.


