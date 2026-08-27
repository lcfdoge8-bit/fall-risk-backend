# 2000 病人分佈 — LOW/MEDIUM/HIGH 3 色 dot 圖
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
y = df["fall_risk_level"]

ORDER = ["LOW", "MEDIUM", "HIGH"]
COLORS = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#F44336"}

# ═══ Figure: 2 panels ═══
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ── 左: Strip plot (每個病人一粒 dot) ──
ax = axes[0]
rng = np.random.default_rng(42)
for i, cls in enumerate(ORDER):
    pts = (y == cls).sum()
    xs = rng.normal(i, 0.12, pts)          # 水平散開 (jitter)
    ys = rng.uniform(0, 1, pts)            # 垂直隨機
    ax.scatter(xs, ys, s=12, alpha=0.6, color=COLORS[cls],
               label=f"{cls} ({pts}, {pts/len(df):.0%})")
ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(ORDER)
ax.set_xlim(-0.5, len(ORDER)-0.5)
ax.set_ylim(-0.1, 1.1)
ax.set_title("2000 病人風險分佈 (Strip Dot)", fontsize=14)
ax.set_xlabel("風險級別"); ax.set_ylabel("隨機位置")
ax.set_yticks([])
ax.legend(loc="upper right", fontsize=10)

# ── 右: Beeswarm (更密, 每個 class 一排) ──
ax2 = axes[1]
positions = []
for i, cls in enumerate(ORDER):
    pts = (y == cls).sum()
    # 垂直堆疊成條狀 (jitter 少啲, 似 beeswarm)
    xs = np.full(pts, i) + rng.uniform(-0.35, 0.35, pts)
    ys = np.linspace(0, 1, pts) + rng.uniform(-0.01, 0.01, pts)
    positions.append((xs, ys))
    ax2.scatter(xs, ys, s=14, alpha=0.6, color=COLORS[cls])
ax2.set_xticks(range(len(ORDER)))
ax2.set_xticklabels([f"{c}\n({(y==c).sum()})" for c in ORDER])
ax2.set_xlim(-0.5, len(ORDER)-0.5)
ax2.set_ylim(-0.1, 1.1)
ax2.set_title("Beeswarm — 每格人數", fontsize=14)
ax2.set_xlabel("風險級別 (人數)"); ax2.set_yticks([])

plt.tight_layout()
plt.savefig("risk_distribution_dots.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ Saved risk_distribution_dots.png")
print("LOW:", (y=="LOW").sum(), "| MEDIUM:", (y=="MEDIUM").sum(), "| HIGH:", (y=="HIGH").sum())
