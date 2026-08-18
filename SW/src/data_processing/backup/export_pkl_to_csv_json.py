"""
export_pkl_to_csv_json.py (gearbox_final_suite)

Converts multimodal_features_cache.pkl into human-readable CSV, JSON, and TXT files:
1. multimodal_cache_full.csv : Full DataFrame containing metadata + all feature columns
2. multimodal_cache_summary.json : Summary dictionary of keys, dimensions, and target distribution
3. feature_names.txt : Plain text list of all feature names

Rule 3 Compliance:
- Code comments & terminal output in Korean.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
EXPORT_DIR = os.path.join(RESULTS_DIR, "inspected_cache")

CACHE_FIXED = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_fault_diagnosis_fixed", "results", "multimodal_features_cache.pkl")
CACHE_SRC2 = os.path.join(os.path.dirname(SUITE_DIR), "gearbox_src_2", "results", "multimodal_features_cache.pkl")
CACHE_FILE = CACHE_FIXED if os.path.exists(CACHE_FIXED) else CACHE_SRC2

os.makedirs(EXPORT_DIR, exist_ok=True)


def export_pkl_contents():
    print("=" * 80)
    print(" 🛠 CONVERTING MULTIMODAL_FEATURES_CACHE.PKL TO CSV & JSON")
    print("=" * 80)

    if not os.path.exists(CACHE_FILE):
        print(f"[ERROR] PKL file missing: {CACHE_FILE}")
        return

    cache = joblib.load(CACHE_FILE)
    print(f" [PKL KEYS FOUND]: {list(cache.keys())}")

    df_mm = cache["df_mm"].copy()
    X_mm = cache["X_mm"]
    y = cache["y"]
    feature_names = cache["feature_names"]

    print(f"  - df_mm shape        : {df_mm.shape}")
    print(f"  - X_mm shape         : {X_mm.shape}")
    print(f"  - y shape            : {y.shape}")
    print(f"  - feature_names count: {len(feature_names)}")

    # 1. Export Full CSV
    df_features = pd.DataFrame(X_mm, columns=feature_names)
    df_full = pd.concat([df_mm.reset_index(drop=True), df_features], axis=1)
    
    # Remove duplicate columns if any
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]

    csv_path = os.path.join(EXPORT_DIR, "multimodal_cache_full.csv")
    df_full.to_csv(csv_path, index=False, encoding="utf-8-sig")

    # 2. Export Feature Names TXT
    txt_path = os.path.join(EXPORT_DIR, "feature_names.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for idx, fname in enumerate(feature_names):
            f.write(f"{idx:02d}: {fname}\n")

    # 3. Export Summary JSON
    summary_data = {
        "cache_filepath": CACHE_FILE,
        "num_samples": int(len(y)),
        "num_features": int(len(feature_names)),
        "target_distribution": {
            "normal_0": int(np.sum(y == 0)),
            "abnormal_1": int(np.sum(y == 1))
        },
        "scenarios_found": df_mm["scenario"].value_counts().to_dict(),
        "feature_names_sample": feature_names[:10]
    }

    json_path = os.path.join(EXPORT_DIR, "multimodal_cache_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)

    print(f"\n[SUCCESS] Conversion Complete! Exported Files:")
    print(f" - CSV  : {csv_path} (Size: {os.path.getsize(csv_path)} bytes)")
    print(f" - TXT  : {txt_path}")
    print(f" - JSON : {json_path}")
    print("=" * 80)


if __name__ == "__main__":
    export_pkl_contents()
