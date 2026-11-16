import sys
sys.path.insert(0, 'solution')
import numpy as np
import pandas as pd
from scan_parameters import evaluate_config

print("Testing combination of PHASE_WEIGHT and SWITCH_COST:")
for pw in [0.00, 0.02, 0.04, 0.06, 0.08, 0.10]:
    for sc in [11.5, 11.2, 11.0, 10.8]:
        s_f1, m_f1, poss = evaluate_config(phase_weight=pw, switch_cost=sc)
        if s_f1 >= 0.58:
            print(f"PW={pw:.2f}, SC={sc:.1f} -> spellF1: {s_f1:.4f} | macroF1: {m_f1:.4f} | poss: {poss}")
