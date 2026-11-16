# Possession solution — Inka Analytics (Versión Final: v11)

Rule-based possession labelling for the BAR-ATH match. It uses no trained model and no labels.
Method, rules, innovations and benchmark results are documented below and in [inka-analytics.md](inka-analytics.md).

## Benchmark Results (vs Official Baselines)

| Model / Version | spellF1 (Primary Metric) | macroF1 | Accuracy | Possession Share Error |
|---|---|---|---|---|
| **Inka Analytics v11 (Final)** | **0.6000** | **0.8850** | **88.88%** | **0.34%** |
| Inka Analytics v10 | 0.5867 | 0.8703 | 0.8751 | 0.45% |
| Inka Analytics v4 (Base) | 0.4471 | 0.8220 | 0.8201 | 0.20% |
| Official Baseline Smooth 125 | 0.4242 | 0.5812 | 0.7275 | 5.20% |
| Official Baseline Raw | 0.0378 | 0.5972 | 0.6512 | 4.57% |

## Requirements

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
```

## Data

The match data is placed in `data/` as described in [DATA.md](../DATA.md), or set via `MTAH_DATA_DIR`.
The submission pipeline requires only the tactical tracking (`.txt`), metadata (`.xml`), and Smart Tagging (`.xml`).

## Reproduce the Final Version (v11)

To run the final solution and generate the submission file:

```bash
python solution/possession_v11.py
```

To validate the generated CSV against official rules:

```bash
python evaluation/validate_submission.py evaluation/results/inka_analytics_v11.csv
```

To generate or launch the 2D interactive pitch visualizer:

```bash
# Genera o actualiza el visualizador 2D interactivo (HTML)
python generar_visualizador.py --minuto 0 --duracion 60 --sub evaluation/results/inka_analytics_v11.csv
```

## Architecture & Files

| File | Description |
|---|---|
| **`possession_v11.py`** | **Entry point de la versión final**. Incorpora reanudación dinámica por golpeo real, despolarización aérea y desambiguación defensiva de porteros. |
| `possession_v10.py` - `v5.py` | Versiones iterativas de calibración y análisis de hiperparámetros. |
| `possession_v4.py` | Versión base con ventanas de reanudación fijas. |
| `possession_v1.py` | Lector de tracking, evidencia de control y detector de balón muerto (dead-ball). |
| `possession_v2.py` | Decodificador Viterbi y ponderación de fases Smart Tagging. |
| `possession_v3.py` | Cálculo cinemático de velocidad real del balón. |
| `generar_visualizador.py` | Generador de la interfaz web 2D para visualización en tiempo real. |
