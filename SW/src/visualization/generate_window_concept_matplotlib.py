"""
generate_window_concept_matplotlib.py (gearbox_final_suite)

Generates a clean, information-first table layout diagram illustrating the 500-sample window slicing principle:
- Full Raw Time-Series Table (Rows 0~499, 500~999, 1000~1499)
- Window 0, Window 1, Window 2 brackets
- 1:1 mapping to Unified Dataset Rows (Row 0, Row 1, Row 2)

Rule 3 Compliance:
- Matplotlib labels, title, text in English.
- Korean code comments.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
SUITE_DIR = os.path.dirname(SRC_DIR)
RESULTS_DIR = os.path.join(SUITE_DIR, "results")
OUTPUT_PNG = os.path.join(RESULTS_DIR, "raw_500rows_to_1unified_window_diagram.png")


def generate_clean_table_window_diagram():
    fig, ax = plt.subplots(figsize=(15, 8.5), dpi=150)
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#ffffff')

    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Main Header
    ax.text(50, 95, "500-Sample Sliding Window Slicing Principle",
            color="#0f172a", fontsize=18, fontweight='bold', ha='center', va='center')
    ax.text(50, 91, "How continuous Raw CAN time-series rows map 1-to-1 into Unified Dataset Rows",
            color="#475569", fontsize=12, ha='center', va='center')

    # =========================================================================
    # LEFT PANEL: Full RAW CAN Time Series Table (Long Continuous Rows)
    # =========================================================================
    ax.text(25, 84, "RAW CAN CSV File (Full Time-Series)",
            color="#1e293b", fontsize=14, fontweight='bold', ha='center')

    # Table Header Box
    ax.add_patch(patches.Rectangle((5, 75), 40, 5, facecolor="#cbd5e1", edgecolor="#0f172a", linewidth=1.5))
    ax.text(8, 77.5, "Row #", color="#0f172a", fontsize=10, fontweight='bold', va='center')
    ax.text(17, 77.5, "Time (s)", color="#0f172a", fontsize=10, fontweight='bold', va='center')
    ax.text(27, 77.5, "WHL_SPD_FL", color="#0f172a", fontsize=10, fontweight='bold', va='center')
    ax.text(37, 77.5, "WHL_SPD_FR", color="#0f172a", fontsize=10, fontweight='bold', va='center')

    # Window 0 Block (Rows 0 ~ 499)
    ax.add_patch(patches.Rectangle((5, 52), 40, 21, facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=2, linestyle="--"))
    ax.text(8, 70, "0", color="#0369a1", fontsize=9, fontweight='bold')
    ax.text(17, 70, "0.000", color="#334155", fontsize=9)
    ax.text(27, 70, "18.34", color="#334155", fontsize=9)
    ax.text(37, 70, "18.47", color="#334155", fontsize=9)

    ax.text(8, 64, "1", color="#0369a1", fontsize=9, fontweight='bold')
    ax.text(17, 64, "0.003", color="#334155", fontsize=9)
    ax.text(27, 64, "18.34", color="#334155", fontsize=9)
    ax.text(37, 64, "18.47", color="#334155", fontsize=9)

    ax.text(25, 59, ". . .  (500 Samples: t = 0.00s ~ 1.47s)  . . .", color="#0284c7", fontsize=9, fontweight='bold', ha='center')

    ax.text(8, 54, "499", color="#0369a1", fontsize=9, fontweight='bold')
    ax.text(17, 54, "1.467", color="#334155", fontsize=9)
    ax.text(27, 54, "18.78", color="#334155", fontsize=9)
    ax.text(37, 54, "18.82", color="#334155", fontsize=9)

    # Window 1 Block (Rows 500 ~ 999)
    ax.add_patch(patches.Rectangle((5, 29), 40, 21, facecolor="#fef3c7", edgecolor="#d97706", linewidth=2, linestyle="--"))
    ax.text(8, 47, "500", color="#b45309", fontsize=9, fontweight='bold')
    ax.text(17, 47, "1.470", color="#334155", fontsize=9)
    ax.text(27, 47, "18.81", color="#334155", fontsize=9)
    ax.text(37, 47, "18.85", color="#334155", fontsize=9)

    ax.text(25, 40, ". . .  (500 Samples: t = 1.47s ~ 2.94s)  . . .", color="#d97706", fontsize=9, fontweight='bold', ha='center')

    ax.text(8, 32, "999", color="#b45309", fontsize=9, fontweight='bold')
    ax.text(17, 32, "2.937", color="#334155", fontsize=9)
    ax.text(27, 32, "19.12", color="#334155", fontsize=9)
    ax.text(37, 32, "19.15", color="#334155", fontsize=9)

    # Window 2 Block (Rows 1000 ~ 1499)
    ax.add_patch(patches.Rectangle((5, 10), 40, 17, facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=2, linestyle="--"))
    ax.text(8, 23, "1000", color="#15803d", fontsize=9, fontweight='bold')
    ax.text(17, 23, "2.940", color="#334155", fontsize=9)
    ax.text(27, 23, "19.18", color="#334155", fontsize=9)
    ax.text(37, 23, "19.22", color="#334155", fontsize=9)

    ax.text(25, 16, ". . .  (500 Samples: t = 2.94s ~ 4.41s)  . . .", color="#16a34a", fontsize=9, fontweight='bold', ha='center')

    # Label Brackets
    ax.text(46, 62.5, "Window 0\n(500 rows)", color="#0284c7", fontsize=10, fontweight='bold', va='center')
    ax.text(46, 39.5, "Window 1\n(500 rows)", color="#d97706", fontsize=10, fontweight='bold', va='center')
    ax.text(46, 18.5, "Window 2\n(500 rows)", color="#16a34a", fontsize=10, fontweight='bold', va='center')

    # =========================================================================
    # ARROWS & AGGREGATION ENGINE
    # =========================================================================
    ax.annotate('', xy=(62, 62.5), xytext=(53, 62.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#0284c7", lw=2.5))
    ax.annotate('', xy=(62, 39.5), xytext=(53, 39.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#d97706", lw=2.5))
    ax.annotate('', xy=(62, 18.5), xytext=(53, 18.5),
                arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6", color="#16a34a", lw=2.5))

    # =========================================================================
    # RIGHT PANEL: Unified Dataset Table (1 Row per Window)
    # =========================================================================
    ax.text(80, 84, "Unified CSV Dataset (Window-Level Summarized)",
            color="#1e293b", fontsize=14, fontweight='bold', ha='center')

    # Header Box Right
    ax.add_patch(patches.Rectangle((62, 75), 35, 5, facecolor="#cbd5e1", edgecolor="#0f172a", linewidth=1.5))
    ax.text(65, 77.5, "Unified Row", color="#0f172a", fontsize=9, fontweight='bold', va='center')
    ax.text(74, 77.5, "Source Window", color="#0f172a", fontsize=9, fontweight='bold', va='center')
    ax.text(84, 77.5, "speed_kph", color="#0f172a", fontsize=9, fontweight='bold', va='center')
    ax.text(92, 77.5, "slip_fl", color="#0f172a", fontsize=9, fontweight='bold', va='center')

    # Row 0
    ax.add_patch(patches.Rectangle((62, 54), 35, 17, facecolor="#e0f2fe", edgecolor="#0284c7", linewidth=2))
    ax.text(65, 62.5, "Row 0", color="#0369a1", fontsize=11, fontweight='bold', va='center')
    ax.text(74, 62.5, "Rows 0~499", color="#0f172a", fontsize=10, va='center')
    ax.text(84, 62.5, "80.0", color="#0f172a", fontsize=10, va='center')
    ax.text(92, 62.5, "0.1639", color="#0f172a", fontsize=10, va='center')

    # Row 1
    ax.add_patch(patches.Rectangle((62, 31), 35, 17, facecolor="#fef3c7", edgecolor="#d97706", linewidth=2))
    ax.text(65, 39.5, "Row 1", color="#b45309", fontsize=11, fontweight='bold', va='center')
    ax.text(74, 39.5, "Rows 500~999", color="#0f172a", fontsize=10, va='center')
    ax.text(84, 39.5, "80.0", color="#0f172a", fontsize=10, va='center')
    ax.text(92, 39.5, "0.1645", color="#0f172a", fontsize=10, va='center')

    # Row 2
    ax.add_patch(patches.Rectangle((62, 10), 35, 17, facecolor="#f0fdf4", edgecolor="#16a34a", linewidth=2))
    ax.text(65, 18.5, "Row 2", color="#15803d", fontsize=11, fontweight='bold', va='center')
    ax.text(74, 18.5, "Rows 1000~1499", color="#0f172a", fontsize=10, va='center')
    ax.text(84, 18.5, "80.0", color="#0f172a", fontsize=10, va='center')
    ax.text(92, 18.5, "0.1652", color="#0f172a", fontsize=10, va='center')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(OUTPUT_PNG, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close()

    print(f"\n[SUCCESS] Generated Table Slicing Diagram: {OUTPUT_PNG}")


if __name__ == "__main__":
    generate_clean_table_window_diagram()
