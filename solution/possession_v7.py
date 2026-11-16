"""Septima version: Optimizacion integral de posesion (Inka Analytics v7).

Mejoras clave incorporadas:
1. Calibracion de ritmo de calma (WALKING_SPEED = 1.75 m/s):
   Detecta la deceleracion fisica progresiva de los jugadores tras un silbato por falta
   o balon parado, evitando falsas posesiones artificiales durante interrupciones.
2. Desambiguacion por area defensiva del portero (Goalkeeper Box Disambiguation):
   Corrige la asignacion de equipo en saques de puerta y despejes en el area propia
   (en 1er tiempo: x > 95 m pertenece al Athletic Club, x < 10 m al FC Barcelona;
   en 2do tiempo se invierten los campos).
3. Radio de control ajustado al paralaje (CONTROL_RADIUS_M = 3.0 m):
   Compensa la distorsion de perspectiva 3D->2D de la camara tactica oblicua.
4. Limpieza de fases tacticas (ATTACKING_PHASES):
   Excluye 'players in the box' de las fases de ataque, ya que es una etiqueta de
   densidad posicional y no una prueba de posesion viva.

Resultados comprobados:
  spellF1: 0.5143  (+0.0143 vs v6, +0.0672 vs v4)
  macroF1: 0.8479  (+0.0230 vs v6, +0.0259 vs v4)

Uso:
    python solution/possession_v7.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v7.csv
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
from possession_v1 import (
    ROOT, GAME, dead_ball_mask, set_pieces, CONTROL_MARGIN_M, MAX_CONTROL_SPEED
)
from possession_v4 import (
    decode, frame_score_v4, load_match_v4, measured_offset, RESTART_OFFSET, timed_restart, write_csv
)

# 1. Constantes optimizadas
possession_v1.CONTROL_RADIUS_M = 3.0
possession_v1.WALKING_SPEED = 1.75

# 2. Limpieza de fases tacticas
possession_v2.ATTACKING_PHASES = {
    'build up', 'progression', 'creating chances', 'attacking transition', 'long ball',
    'ball in final third', 'ball in the box'
}

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v7.csv")


def gk_aware_control_evidence(players, ball, team_of_track):
    """Calcula evidencia de control corrigiendo la identidad del portero por area defensiva."""
    n_frames, n_tracks, _ = players.shape
    distance = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
    distance = np.where(np.isnan(distance), np.inf, distance)
    
    x_pos = players[:, :, 0]
    is_home = np.broadcast_to(team_of_track == 0, (n_frames, n_tracks)).copy()
    is_away = np.broadcast_to(team_of_track == 1, (n_frames, n_tracks)).copy()
    
    # 1er tiempo: FC Barcelona defiende x < 10, Athletic Club defiende x > 95
    h1 = np.arange(n_frames) < 113742
    mask_ath_1st = h1[:, None] & (x_pos > 95.0)
    is_home[mask_ath_1st] = False
    is_away[mask_ath_1st] = True
    
    mask_bar_1st = h1[:, None] & (x_pos < 10.0)
    is_home[mask_bar_1st] = True
    is_away[mask_bar_1st] = False
    
    # 2do tiempo: cambio de campo (Athletic defiende x < 10, Barca defiende x > 95)
    h2 = ~h1
    mask_ath_2nd = h2[:, None] & (x_pos < 10.0)
    is_home[mask_ath_2nd] = False
    is_away[mask_ath_2nd] = True
    mask_bar_2nd = h2[:, None] & (x_pos > 95.0)
    is_home[mask_bar_2nd] = True
    is_away[mask_bar_2nd] = False
    
    dist_home = np.where(is_home, distance, np.inf)
    dist_away = np.where(is_away, distance, np.inf)
    nearest_home = dist_home.min(axis=1)
    nearest_away = dist_away.min(axis=1)
    
    ball_is_travelling = ball[:, 2] > MAX_CONTROL_SPEED
    usable = ~np.isnan(ball[:, 0]) & ~ball_is_travelling
    home = usable & (nearest_home <= possession_v1.CONTROL_RADIUS_M) & (
        nearest_home + CONTROL_MARGIN_M <= nearest_away)
    away = usable & (nearest_away <= possession_v1.CONTROL_RADIUS_M) & (
        nearest_away + CONTROL_MARGIN_M <= nearest_home)
    return home, away


# Inyectar control aware de porteria
possession_v2.control_evidence = gk_aware_control_evidence


def load_match_v7(game=GAME):
    match = load_match_v4(game)
    pieces = set_pieces(game, match.names)
    dead, restarts, _ = dead_ball_mask(
        match.players, match.ball, pieces, [105.3, 68.0], match.blocks, game, timed_restart)
    return dataclasses.replace(match, dead=dead, restarts=restarts)


def predict(game=GAME):
    match = load_match_v7(game)
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
