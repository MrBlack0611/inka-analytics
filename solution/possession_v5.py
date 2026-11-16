"""Quinta version: la v4 con radio de control adaptado al error de paralaje del tracking (3.0 m).

Motivacion:
La camara tactica cenital/oblicua introduce un error de proyeccion geometrica y perspectiva cuando
el jugador corre o controla el balon cerca de las bandas o en conduccion. La distancia aparente
entre las coordenadas 2D del balon y del jugador supera habitualmente los 2.0 m originales
(llegando a 2.5 - 3.5 m en posesiones reales verificadas visualmente, ej. frame 11363).

Cambios frente a la v4:
- CONTROL_RADIUS_M pasa de 2.0 m a 3.0 m. Esto absorbe el desfase optico sin ser tan amplio
  como para atribuir control a defensores que solo persiguen a distancia (a 4.0 m empieza a degradar).
- Mantiene el resto de la logica intacta: deteccion de balon muerto por actividad, salidas de campo,
  reanudaciones de ventanas largas (desfase 158 frames) y decodificacion de Viterbi.

    python solution/possession_v5.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v5.csv
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import possession_v1
# Ajuste de tolerancia fisica para compensar el paralaje de la camara tactica
possession_v1.CONTROL_RADIUS_M = 3.0

from possession_v1 import ROOT
from possession_v4 import (
    GAME, decode, frame_score_v4, load_match_v4, measured_offset, RESTART_OFFSET, write_csv)

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v5.csv")


def predict(game=GAME):
    match = load_match_v4(game)
    print("  radio de control usado: {} m (original v4 era 2.0 m)".format(possession_v1.CONTROL_RADIUS_M))
    print("  desfase medido {}  usado {}".format(measured_offset(match, game), RESTART_OFFSET))
    score, summary = frame_score_v4(match, game)
    print("  " + "  ".join("{} {}".format(key, value) for key, value in summary.items()))
    return decode(match, score)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="CSV de salida")
    args = parser.parse_args()

    result = predict()
    path = write_csv(args.out, result)
    counts = {label: int(np.sum(result == label)) for label in "HADX"}
    print("  H {H}  A {A}  D {D}  X {X}".format(**counts))
    print("escrito en {}".format(path))
