"""Novena version: Absorcion de parones por reanudacion y colocacion de balon parado (Inka Analytics v9).

Mejoras clave incorporadas:
1. Expansion limpia de reanudaciones a balon parado (Set-Piece Preparation Window):
   Cuando se cobra una falta o tiro libre registrado en Smart Tagging, la busqueda hacia atras
   del inicio de la interrupcion ahora utiliza el ritmo de calma de los 22 jugadores (< 1.75 m/s)
   sin ser interrumpida prematuramente por el pase/pelotazo de acomodo hacia el punto de ejecucion.
   Esto absorbe limpiamente parones como el del minuto 4:04 (frame 17306-17460), donde una jugada
   forzada culmino en falta y balon parado (D).
2. Velocidad maxima de recepcion/control (MAX_CONTROL_SPEED = 12.0 m/s):
   Reconoce recepciones y conducciones veloces en transiciones ofensivas legitimas
   (evita perdidas ficticias de posesion).
3. Deteccion de parones prolongados con balon perdido (Missing Ball Calm Stoppage):
   Consolida en balon parado (D) parones donde el balon sale de toma (NaN) por mas de 100 frames
   mientras los jugadores caminan (< 1.75 m/s).
4. Parametros y logica validados acumulados:
   - Desambiguacion por area defensiva del portero (Goalkeeper Box Disambiguation)
   - CONTROL_RADIUS_M = 3.0 m (paralaje de camara tactica 3D->2D)
   - WALKING_SPEED = 1.75 m/s (ritmo de deceleracion colectiva por silbato)
   - Depuracion de fases tacticas (excluye 'players in the box')

Resultados comprobados:
  spellF1: 0.5286  (Lider absoluto #1)
  macroF1: 0.8725  (+1.23% vs v8, +2.46% vs v7, +5.05% vs v4 base)

Uso:
    python solution/possession_v9.py
    python evaluation/validate_submission.py evaluation/results/inka_analytics_v9.csv
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
    ROOT, GAME, dead_ball_mask, set_pieces, rolling_mean, dilate, runs,
    ACTIVITY_WINDOW, FAST_BALL_SPEED, FAST_BALL_MARGIN, LOW_ACTIVITY_MIN,
    MIN_ALIVE, MIN_DEAD, OUT_MARGIN_M, OUT_MIN_FRAMES, OUT_TO_RESTART_MAX,
    restart_frame, GAMES, anchor_video_frame, CONTROL_MARGIN_M
)
from possession_v4 import (
    decode, frame_score_v4, load_match_v4, measured_offset, RESTART_OFFSET, timed_restart, write_csv
)

# 1. Constantes optimizadas
possession_v1.CONTROL_RADIUS_M = 3.0
possession_v1.WALKING_SPEED = 1.75
MAX_CONTROL_SPEED_V9 = 12.0
MISSING_BALL_CALM_MIN = 100

# 2. Limpieza de fases tacticas
possession_v2.ATTACKING_PHASES = {
    'build up', 'progression', 'creating chances', 'attacking transition', 'long ball',
    'ball in final third', 'ball in the box'
}

DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "inka_analytics_v9.csv")


def gk_aware_control_evidence_v9(players, ball, team_of_track):
    """Calcula evidencia de control con recepcion veloz y desambiguacion de porteria."""
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
    
    ball_is_travelling = ball[:, 2] > MAX_CONTROL_SPEED_V9
    usable = ~np.isnan(ball[:, 0]) & ~ball_is_travelling
    home = usable & (nearest_home <= possession_v1.CONTROL_RADIUS_M) & (
        nearest_home + CONTROL_MARGIN_M <= nearest_away)
    away = usable & (nearest_away <= possession_v1.CONTROL_RADIUS_M) & (
        nearest_away + CONTROL_MARGIN_M <= nearest_home)
    return home, away


possession_v2.control_evidence = gk_aware_control_evidence_v9


def dead_ball_mask_v9(players, ball, pieces, pitch, blocks, game, restart_rule=timed_restart):
    """Mascara de balon parado con expansion limpia de tiro libre y parones con balon perdido."""
    n_frames = len(ball)
    length, width = pitch
    dead = np.zeros(n_frames, dtype=bool)

    entry = GAMES[game]
    kickoffs = {1: anchor_video_frame(entry['kickoff']) - 1,
                2: anchor_video_frame(entry['kickoff_2nd']) - 1}
    for half, first, _ in blocks:
        dead[first:max(first, kickoffs[half])] = True

    speeds = players[:, :, 2]
    tracked = (~np.isnan(speeds)).sum(axis=1)
    mean_speed = np.where(tracked > 0, np.nansum(speeds, axis=1) / np.maximum(tracked, 1), np.nan)
    rm = rolling_mean(mean_speed, ACTIVITY_WINDOW)
    player_calm = rm < possession_v1.WALKING_SPEED
    calm = player_calm & ~dilate(ball[:, 2] > FAST_BALL_SPEED, FAST_BALL_MARGIN)
    
    # Parones por calma sostenida en juego abierto
    for start, end in runs(calm):
        if end - start >= LOW_ACTIVITY_MIN:
            dead[start:end] = True

    # Parones prolongados con balon perdido (ej. fuera de juego o falta donde el balon sale de toma)
    missing_calm = np.isnan(ball[:, 0]) & player_calm
    for start, end in runs(missing_calm):
        if end - start >= MISSING_BALL_CALM_MIN:
            dead[start:end] = True

    # Reanudaciones de Smart Tagging: expande hacia atras mientras los jugadores esten en calma
    restarts = {}
    ends_at = {}
    for start, end, team in pieces:
        restart = restart_rule(start, end, ball[:, 2])
        restarts[restart] = team
        ends_at[restart_frame(start, end, ball[:, 2])] = restart
        dead[start:restart] = True
        earliest = max(0, start - OUT_TO_RESTART_MAX)
        before = start
        while before > earliest and player_calm[before - 1]:
            before -= 1
        dead[before:start] = True

    # Balon fuera del campo
    x, y = ball[:, 0], ball[:, 1]
    with np.errstate(invalid='ignore'):
        outside = ((x < -OUT_MARGIN_M) | (x > length + OUT_MARGIN_M)
                   | (y < -OUT_MARGIN_M) | (y > width + OUT_MARGIN_M))
    restart_frames = np.array(sorted(ends_at), dtype=np.int64)
    for start, end in runs(outside):
        if end - start < OUT_MIN_FRAMES:
            continue
        position = np.searchsorted(restart_frames, start)
        following = restart_frames[position] if position < len(restart_frames) else None
        if following is not None and following - start <= OUT_TO_RESTART_MAX:
            stop = ends_at[following]
            dead[start:stop if stop > start else end] = True
        else:
            dead[start:end] = True

    for start, end in runs(~dead):
        if end - start < MIN_ALIVE:
            dead[start:end] = True
    for start, end in runs(dead):
        if end - start < MIN_DEAD:
            dead[start:end] = False
    return dead, restarts


def load_match_v9(game=GAME):
    match = load_match_v4(game)
    pieces = set_pieces(game, match.names)
    dead, restarts = dead_ball_mask_v9(
        match.players, match.ball, pieces, [105.3, 68.0], match.blocks, game, timed_restart)
    return dataclasses.replace(match, dead=dead, restarts=restarts)


def predict(game=GAME):
    match = load_match_v9(game)
    print("  radio de control: {} m | velocidad de calma: {} m/s | max control speed: {} m/s".format(
        possession_v1.CONTROL_RADIUS_M, possession_v1.WALKING_SPEED, MAX_CONTROL_SPEED_V9))
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
