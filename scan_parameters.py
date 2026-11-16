import sys
sys.path.insert(0, 'solution')
import numpy as np
import pandas as pd
from possession_v1 import (
    dead_ball_mask, set_pieces, rolling_mean, dilate, runs,
    ACTIVITY_WINDOW, FAST_BALL_SPEED, FAST_BALL_MARGIN, LOW_ACTIVITY_MIN,
    MIN_ALIVE, MIN_DEAD, OUT_MARGIN_M, OUT_MIN_FRAMES, OUT_TO_RESTART_MAX,
    restart_frame, GAMES, anchor_video_frame, CONTROL_MARGIN_M, FPS
)
from possession_v4 import load_match_v4, frame_score_v4, timed_restart
from possession_v10 import dead_ball_mask_v10, viterbi_v10, decode_v10, SWITCH_COST_V10
from evaluation import possession_metrics as pm
import dataclasses

df_gt = pd.read_csv('evaluation/gt/BAR-ATH_possession_gt_public.csv')
gt_codes = pm.label_codes(df_gt)

raw_match = load_match_v4()

# Helper to run evaluation given modified params
def evaluate_config(control_radius=3.0, control_margin=1.0, max_control_spd=12.0,
                    walking_speed=1.75, phase_weight=0.1, switch_cost=11.2):
    import possession_v1, possession_v2
    possession_v1.CONTROL_RADIUS_M = control_radius
    possession_v1.WALKING_SPEED = walking_speed
    
    # 1. Control evidence with custom radius, margin, max_spd
    def custom_control_evidence(players, ball, team_of_track):
        n_frames, n_tracks, _ = players.shape
        distance = np.hypot(players[:, :, 0] - ball[:, None, 0], players[:, :, 1] - ball[:, None, 1])
        distance = np.where(np.isnan(distance), np.inf, distance)
        
        x_pos = players[:, :, 0]
        is_home = np.broadcast_to(team_of_track == 0, (n_frames, n_tracks)).copy()
        is_away = np.broadcast_to(team_of_track == 1, (n_frames, n_tracks)).copy()
        
        h1 = np.arange(n_frames) < 113742
        mask_ath_1st = h1[:, None] & (x_pos > 95.0)
        is_home[mask_ath_1st] = False
        is_away[mask_ath_1st] = True
        
        mask_bar_1st = h1[:, None] & (x_pos < 10.0)
        is_home[mask_bar_1st] = True
        is_away[mask_bar_1st] = False
        
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
        
        ball_is_travelling = ball[:, 2] > max_control_spd
        usable = ~np.isnan(ball[:, 0]) & ~ball_is_travelling
        home = usable & (nearest_home <= control_radius) & (
            nearest_home + control_margin <= nearest_away)
        away = usable & (nearest_away <= control_radius) & (
            nearest_away + control_margin <= nearest_home)
        return home, away

    possession_v2.control_evidence = custom_control_evidence
    possession_v2.PHASE_WEIGHT = phase_weight
    
    # 2. Dead ball mask v10
    pieces = set_pieces('BAR-ATH', raw_match.names)
    dead, restarts = dead_ball_mask_v10(
        raw_match.players, raw_match.ball, pieces, [105.3, 68.0], raw_match.blocks, 'BAR-ATH', timed_restart
    )
    m = dataclasses.replace(raw_match, dead=dead, restarts=restarts)
    
    # 3. Score
    score, _ = frame_score_v4(m)
    
    # 4. Decode with switch_cost
    def custom_decode(match, score):
        labels = np.full(match.n_frames, "X", dtype="<U1")
        restart_frames = sorted(match.restarts)
        for _, first, last in match.blocks:
            block_end = last + 1
            block_dead = np.zeros(match.n_frames, dtype=bool)
            block_dead[first:block_end] = match.dead[first:block_end]
            labels[block_dead] = "D"

            in_block = np.zeros(match.n_frames, dtype=bool)
            in_block[first:block_end] = ~match.dead[first:block_end]
            previous = None
            for start, end in runs(in_block):
                near = [frame for frame in restart_frames if abs(frame - start) <= FPS]
                initial = match.restarts[min(near, key=lambda f: abs(f - start))] if near else None
                path = viterbi_v10(score[start:end], switch_cost, initial, previous)
                previous = int(path[-1])
                labels[start:end] = np.where(path == 0, "H", "A")
        return labels

    preds = custom_decode(m, score)
    res = pm.evaluate(gt_codes, pm.label_codes(pd.DataFrame({'frame': np.arange(len(preds)), 'label': preds})))
    return res['spell_f1_mean'], res['macro_f1'], res['n_pred_spells']

print("=== 1. SCANNING CONTROL_MARGIN_M (Tenths from 0.2 to 1.6) ===")
for cm in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
    s_f1, m_f1, poss = evaluate_config(control_margin=cm)
    print(f"CONTROL_MARGIN_M={cm:4.1f} -> spellF1: {s_f1:.4f} | macroF1: {m_f1:.4f} | poss: {poss}")

print("\n=== 2. SCANNING CONTROL_RADIUS_M (Tenths from 2.4 to 3.6) ===")
for cr in [2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6]:
    s_f1, m_f1, poss = evaluate_config(control_radius=cr)
    print(f"CONTROL_RADIUS_M={cr:4.1f} -> spellF1: {s_f1:.4f} | macroF1: {m_f1:.4f} | poss: {poss}")

print("\n=== 3. SCANNING PHASE_WEIGHT (from 0.00 to 0.40) ===")
for pw in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
    s_f1, m_f1, poss = evaluate_config(phase_weight=pw)
    print(f"PHASE_WEIGHT={pw:4.2f} -> spellF1: {s_f1:.4f} | macroF1: {m_f1:.4f} | poss: {poss}")
