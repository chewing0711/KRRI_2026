"""
diagnose_loso_slip.py

Diagnoses why LOSO (Mode 4) stays at ~51% even with slip_ratio features.
Prints mean slip_ratio_fl per scenario for Normal vs Abnormal.
"""

import os
import pandas as pd
import numpy as np

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
FIXED_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(FIXED_DIR, "results")
INPUT_CSV = os.path.join(RESULTS_DIR, "unified_can_context_dataset.csv")

if os.path.exists(INPUT_CSV):
    df = pd.read_csv(INPUT_CSV)
    print("=" * 80)
    print(" 🔍 DIAGNOSTIC SUMMARY OF PHYSICAL SLIP RATIOS PER SCENARIO")
    print("=" * 80)
    
    for scen, group in df.groupby("scenario"):
        norm_grp = group[group["target"] == 0]
        abnorm_grp = group[group["target"] == 1]
        
        print(f"\n [Scenario: {scen}] (Samples: {len(group)})")
        print(f"  - Speed: {group['speed_kph'].iloc[0]} kph, Grade: {group['grade_pct'].iloc[0]} %")
        if len(norm_grp) > 0 and "slip_ratio_fl" in df.columns:
            print(f"  - Normal   Slip FL Mean: {norm_grp['slip_ratio_fl'].mean():.6f} (std: {norm_grp['slip_ratio_fl'].std():.6f})")
        if len(abnorm_grp) > 0 and "slip_ratio_fl" in df.columns:
            print(f"  - Abnormal Slip FL Mean: {abnorm_grp['slip_ratio_fl'].mean():.6f} (std: {abnorm_grp['slip_ratio_fl'].std():.6f})")
    print("=" * 80)
