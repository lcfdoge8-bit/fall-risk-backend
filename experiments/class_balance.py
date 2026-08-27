# Class Balance & Weight Distribution — 解釋點解 balanced 重要
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
# 用支援中文嘅 font (macOS)
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

# ═══ Load data ═══
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
y = df["fall_risk_level"]

ORDER = ["LOW", "MEDIUM", "HIGH"]
counts = y.value_counts().reindex(ORDER).fillna(0)
total = len(y)

# ═══ 計 class_weight='balanced' 嘅權重 ═══
# sklearn formula: weight = n_samples / (n_classes * n_class_samples)
n = total
n_classes = 3
weights = {}
for cls in ORDER:
    n_c = counts[cls]
    weights[cls] = n / (n_classes * n_c) if n_c > 0 else 0

print("Class counts:", dict(counts))
print("Balanced weights:", {k: round(v, 3) for k, v in weights.items()})

# ═══ 圖 1: 原始 class 分佈 (Bar) ═══
colors = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#F44336"}
fig, ax = plt.subplots(1, 2, figsize=(14, 6))

bars = ax[0].bar(ORDER, counts.values, color=[colors[c] for c in ORDER])
for b, c in zip(bars, counts.values):
    ax[0].text(b.get_x() + b.get_width()/2, b.get_height() + 10,
               f"{c}\n({c/total:.1%})", ha="center", fontsize=12)
ax[0].set_title("原始 Class 分佈 (唔平衡)", fontsize=14)
ax[0].set_xlabel("風險級別"); ax[0].set_ylabel("人數")
ax[0].set_ylim(0, max(counts.values) * 1.15)

# ═══ 圖 2: balanced class weight (Bar) ═══
bars2 = ax[1].bar(ORDER, [weights[c] for c in ORDER], color=[colors[c] for c in ORDER])
for b, w in zip(bars2, [weights[c] for c in ORDER]):
    ax[1].text(b.get_x() + b.get_width()/2, b.get_height() + 0.02,
               f"{w:.2f}", ha="center", fontsize=12)
ax[1].set_title("class_weight='balanced' 權重", fontsize=14)
ax[1].set_xlabel("風險級別"); ax[1].set_ylabel("權重")
ax[1].set_ylim(0, max(weights.values()) * 1.15)

plt.tight_layout()
plt.savefig("class_balance_weight.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved class_balance_weight.png")
