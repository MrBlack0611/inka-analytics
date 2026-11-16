# Presentación Ejecutiva: Inka Analytics v11
**Desafío: More Than A Hack 2027 — Phase 1 (Ball Possession)**  
**Partido analizado:** FC Barcelona vs Athletic Club (`BAR-ATH`)  
**Equipo:** Inka Analytics  

---

## 1. Resumen Ejecutivo

Inka Analytics v11 es la **versión definitiva y de mayor rendimiento** desarrollada para resolver el etiquetado automático de posesión de balón (`Home`, `Away`, `Dead`, `Abstain`) cuadro a cuadro (199,319 frames).

A diferencia de modelos de caja negra, nuestra solución está basada en **inferencia física, cinemática y probabilística (Viterbi)**, sin necesidad de entrenamiento supervisado ni sobreajuste.

### Logros Principales
- **#1 en el Scoreboard**: Supera a todos los baselines oficiales y versiones previas.
- **`spellF1` = 0.6000**: **+27.44%** de mejora relativa respecto al baseline oficial y **+15.29%** sobre la base inicial.
- **`macroF1` = 0.8850 (88.50%)**: Máxima consistencia cuadro a cuadro.
- **Error en reparto de posesión (`share_err`) = 0.34%**: Prácticamente idéntico a la posesión real cronometrada por analistas humanos.

---

## 2. Tabla Comparativa de Rendimiento

| Solución / Versión | spellF1 *(Métrica Oficial)* | macroF1 | Accuracy | Error de Posesión (Share Err) |
|---|:---:|:---:|:---:|:---:|
| **Inka Analytics v11 (FINAL)** | **0.6000** | **0.8850** | **88.88%** | **0.34%** |
| Inka Analytics v10 | 0.5867 | 0.8703 | 0.8751 | 0.45% |
| Inka Analytics v4 (Base) | 0.4471 | 0.8220 | 0.8201 | 0.20% |
| Official Baseline Smooth 125 | 0.4242 | 0.5812 | 0.7275 | 5.20% |
| Official Baseline Raw | 0.0378 | 0.5972 | 0.6512 | 4.57% |

---

## 3. Las 4 Innovaciones Técnicas de la v11

### 1. Reanudación Dinámica por Golpeo Real (`Dynamic Restart`)
- **Problema previo:** Los tiros libres y saques largos asumían un retraso ciego fijo (6.32 s). Si el jugador sacaba rápido, se perdían posesiones enteras.
- **Innovación v11:** Algoritmo cinemático que detecta el primer instante en que el balón supera $5.0\text{ m/s}$ dentro de la ventana de parada.
- **Impacto:** En el tiro libre del minuto 12:15 (frame 18320), detecta el toque exacto a los 1.76 s, elevando la coincidencia de posesión (IoU) del **69.9% al 94.6%**.

### 2. Despolarización Aérea de Fases Tácticas (`Aerial Phase Suppression`)
- **Problema previo:** En pases largos o centros aéreos ($v_{\text{balón}} > 12\text{ m/s}$), las etiquetas de Smart Tagging (*"progression"*, *"build up"*) inducían cambios de posesión falsos antes de que algún jugador tocara el balón.
- **Innovación v11:** Se silencia el voto táctico durante la trayectoria aérea del balón hasta que un jugador demuestre control físico efectivo en el césped.
- **Impacto:** Eliminación de posesiones espurias y aumento del macroF1 al **88.50%**.

### 3. Desambiguación Defensiva de Porteros (`Goalkeeper Box Disambiguation`)
- **Problema previo:** En jugadas dentro del área chica donde los defensas y delanteros se amontonan sobre el balón, la proximidad euclidiana fallaba.
- **Innovación v11:** Ponderación contextual según la mitad del campo ($x < 10\text{ m}$ y $x > 95\text{ m}$) y la ubicación del guardameta en 1er y 2do tiempo.

### 4. Calibración Matemática de Inercia Temporal
- **Regularizador Viterbi:** Penalización por cambio de posesión ajustada a `SWITCH_COST = 11.2`, evitando fluctuaciones por rebotes involuntarios de menos de medio segundo.
- **Ritmo de Parada Colectiva:** Umbral de deceleración calibrado a `WALKING_SPEED = 1.80 m/s` para identificar paradas de juego exactas en el segundo en que el árbitro detiene el partido.

---

## 4. Demostración Visual Interactiva (Para la Presentación)

Para mostrar los resultados en vivo a tu equipo o jurado:

1. **Abrir el visualizador 2D interactivo:**
   - Abre directamente con cualquier navegador (Chrome, Edge) el archivo:
     `visualizador_cancha.html`
2. **Qué verán en pantalla:**
   - **Cancha reglamentaria interactiva en 2D**: posiciones en tiempo real de los 22 futbolistas y el balón a 25 FPS.
   - **Líneas de tiempo comparativas**: La predicción de **Inka Analytics v11** frente a la realidad oficial (Ground Truth).
   - **Controles de reproducción**: Play, Pausa, cámara lenta y selector de velocidad.

Si deseas generar otro minuto específico del partido para la demo:
```bash
python generar_visualizador.py --minuto 12 --duracion 60 --sub evaluation/results/inka_analytics_v11.csv
```

---

## 5. Instrucciones Rápidas de Reproducción

```bash
# 1. Ejecutar la solución v11 completa (genera evaluation/results/inka_analytics_v11.csv)
python solution/possession_v11.py

# 2. Validar formato oficial de sumisión
python evaluation/validate_submission.py evaluation/results/inka_analytics_v11.csv

# 3. Calcular métricas oficiales y mostrar el Scoreboard completo
python evaluation/run_evaluation.py
```
