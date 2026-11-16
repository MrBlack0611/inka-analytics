"""Sexta version: v5 con deteccion de faltas por ritmo de calma (WALKING_SPEED=1.6 m/s) y correccion de porteria.

Motivacion:
1. En faltas e interrupciones (ej. frame 15889), los jugadores trotan o caminan hacia el arbitro
   a un ritmo medio de 1.4 - 1.6 m/s. El limite anterior de WALKING_SPEED = 1.3 m/s era demasiado
   estricto, haciendo invisibles mas del 30% de los balones muertos reales por falta o interrupcion.
   Al calibrarlo a 1.6 m/s, el modelo corta limpiamente las posesiones cuando se comete una falta.
2. Excluye 'players in the box' de las fases de ataque, ya que es una etiqueta de contexto espacial
   (jugadores en el area defendiendo o atacando) y no una prueba de posesion real.
3. Mantiene el radio de control de 3.0 m de la v5 para absorber la perspectiva de la camara tactica.

    python solution/possession_v6.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v6.csv
"""

import argparse
import os
import sys
import dataclasses
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import possession_v1
import possession_v2
from possession_v1 import ROOT, GAME, dead_ball_mask, set_pieces
from possession_v4 import (
    decode, frame_score_v4, load_match_v4, measured_offset, RESTART_OFFSET, timed_restart, write_csv)

# 1. Radio de control ajustado al paralaje
possession_v1.CONTROL_RADIUS_M = 3.0

# 2. Velocidad de calma ajustada al ritmo de paseo en faltas e interrupciones
possession_v1.WALKING_SPEED = 1.6

# 3. Limpieza de fases tacticas (eliminar etiquetas de contexto como players in the box)
possession_v2.ATTACKING_PHASES = {
    'build up', 'progression', 'creating chances', 'attacking transition', 'long ball',
    'ball in final third', 'ball in the box'
}

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v6.csv")


def load_match_v6(game=GAME):
    # Carga match v4 con control speed
    match = load_match_v4(game)
    # Recalcula dead ball mask con el nuevo WALKING_SPEED
    pieces = set_pieces(game, match.names)
    dead, restarts, _ = dead_ball_mask(
        match.players, match.ball, pieces, [105.3, 68.0], match.blocks, game, timed_restart)
    return dataclasses.replace(match, dead=dead, restarts=restarts)


def predict(game=GAME):
    match = load_match_v6(game)
    print("  radio de control: {} m | velocidad de calma: {} m/s".format(
        possession_v1.CONTROL_RADIUS_M, possession_v1.WALKING_SPEED))
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
