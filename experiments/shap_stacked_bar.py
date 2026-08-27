# Phase 2 — Multi-Class Stacked SHAP Bar Chart
# =============================================
# Horizontal stacked bar: mean(|SHAP|) per feature, broken down by class
# (HIGH / LOW / MEDIUM), like the multi-class SHAP summary bar plot.
#
# Run:
#   cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && ../IA/.venv_shap/bin/python shap_stacked_bar.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Load P1 v2 data + engineer (same as phase2_model.py) ──
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})
df["sex"] = df["sex"].map({"F": 0, "M": 1})

df["mobility_tug_ratio"]      = df["tug_seconds"] / (df["mobility_score"] + 1)
df["night_falls_interaction"] = df["night_bed_exits"] * df["past_falls"]
df["med_poly_interaction"]    = df["high_risk_medication"] * df["polypharmacy_count"]
df["age_mobility_risk"]       = df["age"] * (10 - df["mobility_score"])
df["days_since_last_fall"]    = pd.to_numeric(df["days_since_last_fall"], errors="coerce").fillna(0)

BASE = ["age", "night_bed_exits", "night_activity_duration_min",
        "past_falls", "mobility_score", "high_risk_medication",
        "cognitive_impairment", "orthostatic_hypotension", "tug_seconds"]
ENGINEERED = ["mobility_tug_ratio", "night_falls_interaction",
              "med_poly_interaction", "age_mobility_risk"]
CLIN_BG = ["days_since_last_fall", "syncopal_fall", "fall_cluster_30d"]
FEATURES = ["sex"] + BASE + ENGINEERED + CLIN_BG

X = df[FEATURES]
y = df["fall_risk_level"]

le = LabelEncoder()
y_num = le.fit_transform(y)
class_names = list(le.classes_)   # e.g. ['HIGH', 'LOW', 'MEDIUM']

# ── Train LR on scaled data ──
scaler = StandardScaler()
X_s = scaler.fit_transform(X)

model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
model.fit(X_s, y_num)

# ── SHAP ──
explainer = shap.LinearExplainer(model, X_s)
sv = explainer.shap_values(X_s)   # multiclass: (n, features, classes) or list

if isinstance(sv, list):
    shap_per_class = sv                       # list of (n, features) per class
else:
    shap_per_class = [sv[:, :, c] for c in range(len(class_names))]

# mean |SHAP| per feature per class
mean_abs_by_class = np.array([np.abs(sc).mean(axis=0) for sc in shap_per_class])  # (classes, features)
total_importance = mean_abs_by_class.sum(axis=0)                                  # (features,)

# sort features by total importance (descending)
order = np.argsort(total_importance)[::-1]
features_sorted = [FEATURES[i] for i in order]
values_sorted = mean_abs_by_class[:, order]    # (classes, features_sorted)

print("Class order:", class_names)
print("\nmean |SHAP| by class (sorted by total):")
print(f"{'Feature':<28}" + "".join(f"{c:>12}" for c in class_names) + f"{'TOTAL':>10}")
for fi, feat in enumerate(features_sorted):
    row = values_sorted[:, fi]
    print(f"{feat:<28}" + "".join(f"{v:>12.3f}" for v in row) + f"{row.sum():>10.3f}")

# ── Plot: horizontal stacked bar ──
colors = ["#FF6B6B", "#4DA8DA", "#8BC34A"]   # HIGH red, LOW blue, MEDIUM green
fig, ax = plt.subplots(figsize=(13, 10))
y_pos = np.arange(len(features_sorted))
left = np.zeros(len(features_sorted))

for ci, cname in enumerate(class_names):
    ax.barh(y_pos, values_sorted[ci], left=left, color=colors[ci],
            label=cname, edgecolor="white", linewidth=0.3, height=0.7)
    left += values_sorted[ci]

ax.set_yticks(y_pos)
ax.set_yticklabels(features_sorted, fontsize=12)
ax.invert_yaxis()   # top = most important
ax.set_xlabel("mean(|SHAP value|)  (average impact on model output magnitude)", fontsize=13)
ax.set_title("Feature Importance — Multi-Class SHAP (mean |SHAP|, Phase 2 LR model)", fontsize=17)
ax.legend(title="Class", loc="lower right", fontsize=11)
ax.grid(axis="x", alpha=0.3)

# value labels at bar ends
for i, feat in enumerate(features_sorted):
    ax.text(left[i] + 0.02, i, f"{left[i]:.2f}", va="center", fontsize=9, color="#333")

plt.tight_layout()
plt.savefig("phase2_shap_stacked_bar.png", dpi=200, bbox_inches="tight")
plt.close()
print("\n✅ Saved phase2_shap_stacked_bar.png")
