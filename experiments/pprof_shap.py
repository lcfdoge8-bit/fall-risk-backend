"""
Professional SHAP Analysis — Elderly Fall Risk Model
=====================================================
Produces clean, professional charts for publication/presentation.
Coloured by feature value (red=high, blue=low) showing impact direction.
"""
import os
import pandas as pd, numpy as np, shap
from sklearn.linear_model import LogisticRegression
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Professional styling ──
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 18, "axes.labelsize": 15,
    "xtick.labelsize": 12, "ytick.labelsize": 12,
    "legend.fontsize": 11, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "#FAFAFA",
    "axes.grid": True, "grid.alpha": 0.3,
})

# ═══ Data ═══
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for c in ["high_risk_medication","orthostatic_hypotension"]:
    df[c] = df[c].map({True:1,False:0})

FEATURES = ["age","night_bed_exits","night_activity_duration_min",
            "past_falls","mobility_score","high_risk_medication",
            "cognitive_impairment","polypharmacy_count",
            "orthostatic_hypotension","tug_seconds"]
X = df[FEATURES]; y = df["fall_risk_level"]

model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
model.fit(X, y)

# ── SHAP explainer ──
masker = shap.maskers.Independent(X, max_samples=500)
explainer = shap.LinearExplainer(model, masker=masker)
shap_vals = explainer(X)  # Explanation object
sv_high = shap_vals[..., 0]  # class 0 = HIGH

# ═══ FIG 1: Professional beeswarm (10 features, 1 column, TALL) ═══
fig, ax = plt.subplots(figsize=(10, 14))
shap.plots.beeswarm(sv_high, max_display=10, show=False, color_bar=True)
plt.title("Feature Impact on HIGH Fall Risk (SHAP)", fontsize=22, pad=16)
plt.tight_layout()
plt.savefig("pprof_1_beeswarm.pdf", dpi=200, bbox_inches="tight")
plt.savefig("pprof_1_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 1. pprof_1_beeswarm.png (.pdf)")

# ═══ FIG 2: Professional bar chart ═══
fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.bar(sv_high, max_display=10, show=False)
plt.title("Mean Feature Importance | SHAP | (HIGH Risk)", fontsize=20, pad=12)
plt.tight_layout()
plt.savefig("pprof_2_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 2. pprof_2_bar.png")

# ═══ FIG 3: Violin-style distribution per feature (wider x-axis + jitter) ═══
SHAP_MEAN = np.abs(sv_high.values).mean(0)
top4_idx  = np.argsort(SHAP_MEAN)[-4:][::-1]
rng = np.random.default_rng(7)
fig, axes = plt.subplots(2, 2, figsize=(24, 13))   # 拉闊: 16 -> 24
axes = axes.flatten()
for ax, idx in zip(axes, top4_idx):
    vals = sv_high.values[:, idx]
    feat  = X.iloc[:, idx].values
    # jitter: 離散 feature (int) 加少少隨機，令重疊點散開
    n_uniq = len(np.unique(feat))
    jitter = 0.0 if n_uniq > 50 else (np.max(feat) - np.min(feat)) / n_uniq * 0.3
    feat_j = feat + rng.uniform(-jitter, jitter, len(feat))
    colors_pts = np.where(feat > np.median(feat), "#D32F2F", "#1976D2")
    ax.scatter(feat_j, vals, s=10, alpha=0.45, c=colors_pts, edgecolors="none")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    # 拉闊 x-axis: 加 8% margin 兩邊
    lo, hi = feat.min(), feat.max()
    pad = (hi - lo) * 0.08 if hi > lo else 0.5
    ax.set_xlim(lo - pad, hi + pad)
    # 如果離散 (int) 用整數 ticks
    if n_uniq <= 20:
        ax.set_xticks(np.unique(feat))
    ax.set_title(FEATURES[idx], fontsize=16)
    ax.set_xlabel("Feature value"); ax.set_ylabel("SHAP (HIGH risk)")
plt.suptitle("Dependency Plots — Top 4 Features vs SHAP (Wide View)", fontsize=22)
plt.tight_layout()
plt.savefig("pprof_3_dependence.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 3. pprof_3_dependence.png (wide x-axis + jitter)")

# ═══ FIG 4: Heatmap — all patients × top features ═══
fig, ax = plt.subplots(figsize=(14, 9))
shap.plots.heatmap(sv_high[:, SHAP_MEAN.argsort()[-10:][::-1]],
                   max_display=10, show=False, instance_order=sorted(range(500),
                   key=lambda i: sv_high.values[i].sum()))
plt.title("SHAP Heatmap — 500 Patients × 10 Features", fontsize=20, pad=12)
plt.tight_layout()
plt.savefig("pprof_4_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 4. pprof_4_heatmap.png")
print("\n🎉 4 professional SHAP charts ready!")
