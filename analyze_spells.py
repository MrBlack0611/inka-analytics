import sys
sys.path.insert(0, 'solution')
import numpy as np
import pandas as pd
from evaluation import possession_metrics as pm

df_gt = pd.read_csv('evaluation/gt/BAR-ATH_possession_gt_public.csv')
gt_codes = pm.label_codes(df_gt)
df_v9 = pd.read_csv('evaluation/results/inka_analytics_v9.csv')
v9_codes = pm.label_codes(df_v9)

runs = pm.valid_runs(gt_codes)
gt_spells = pm.possession_spells(gt_codes, runs)
pred_spells = pm.possession_spells(v9_codes, runs)

print(f"Total GT spells: {len(gt_spells)} | Total v9 spells: {len(pred_spells)}")

# Compute IoU between all GT and Pred spells of same team
print("\n--- MATCHING ANALYSIS OF GT SPELLS ---")
KICKOFF = 11185

for i, (ga, gb, gk) in enumerate(gt_spells):
    gt_team = 'Barca(H)' if gk == 0 else 'Athletic(A)'
    dur_s = (gb - ga) / 25.0
    min_m = (ga - KICKOFF) / (25 * 60)
    
    # Find best overlapping pred spell with same team
    best_iou = 0.0
    best_pred_idx = None
    best_pred_spell = None
    
    for j, (pa, pb, pk) in enumerate(pred_spells):
        if pk == gk:
            inter = max(0, min(gb, pb) - max(ga, pa))
            union = (gb - ga) + (pb - pa) - inter
            iou = inter / union if union > 0 else 0.0
            if iou > best_iou:
                best_iou = iou
                best_pred_idx = j
                best_pred_spell = (pa, pb, pk)
                
    status = f"MATCHED (IoU={best_iou:.2f})" if best_iou >= 0.3 else f"MISSED (IoU={best_iou:.2f})"
    print(f"GT #{i+1:2d} [Min {min_m:4.2f} | {ga:5d}..{gb:5d} | {dur_s:5.2f}s | {gt_team}]: {status}")
    if best_pred_spell is not None and best_iou > 0:
        p_dur = (best_pred_spell[1] - best_pred_spell[0]) / 25.0
        print(f"     -> Pred #{best_pred_idx+1:2d}: {best_pred_spell[0]}..{best_pred_spell[1]} ({p_dur:.2f}s)")
