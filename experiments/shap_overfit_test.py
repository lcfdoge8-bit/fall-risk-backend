# Phase 2 — SHAP Overfitting Test + Feature Balance
# =================================================
# 1) SHAP stability across 5 folds → prove no overfitting
#    (if feature ranking is same in every fold, model learned a real pattern)
# 2) Feature balance visualisation (distribution per risk level)
#
# Run with SHAP venv:
#   cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && ../IA/.venv_shap/bin/python shap_overfit_test.py
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score

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
HIGH_IDX = list(le.classes_).index("HIGH")

print("Loaded", df.shape, "| features:", len(FEATURES))

# ═══════════════════════════════════════════════
# PART 1 — SHAP OVERFITTING TEST (5-fold stability)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 1: SHAP STABILITY ACROSS 5 FOLDS (overfit test)")
print("=" * 60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_rankings = []   # top-5 feature ranking per fold
fold_acc = []
fold_rec = []

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
    ytr, yval = y.iloc[train_idx], y.iloc[val_idx]

    # StandardScaler — fixes LR convergence warning
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xval_s = scaler.transform(Xval)

    # train LR (Kero approach) on SCALED data
    model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    model.fit(Xtr_s, ytr)

    # fold metrics
    pred = model.predict(Xval_s)
    fold_acc.append(accuracy_score(yval, pred))
    fold_rec.append(recall_score(yval, pred, labels=["HIGH"], average=None, zero_division=0)[0])

    # SHAP on this fold's validation data (HIGH class) — use scaled data
    explainer = shap.LinearExplainer(model, Xtr_s)
    sv = explainer.shap_values(Xval_s)

    # sv shape: (n_samples, n_features) for binary; (n_samples, n_features, n_classes) for multiclass
    if isinstance(sv, list):
        sv_high = sv[HIGH_IDX]
    elif sv.ndim == 3:
        sv_high = sv[:, :, HIGH_IDX]
    else:
        sv_high = sv

    # mean |SHAP| per feature → ranking
    mean_abs = np.abs(sv_high).mean(axis=0)
    ranking = np.argsort(mean_abs)[::-1].tolist()
    top5 = [FEATURES[int(i)] for i in ranking[:5]]
    fold_rankings.append(top5)

    print(f"Fold {fold+1}: acc={fold_acc[-1]:.4f}  HIGH recall={fold_rec[-1]:.4f}")
    print(f"   Top-5: {top5}")

# ── Stability check: same top features across all folds? ──
print("\n" + "-" * 60)
print("STABILITY CHECK")
print("-" * 60)
all_top5 = set()
for r in fold_rankings:
    all_top5.update(r)
print(f"Unique features appearing in top-5 across folds: {len(all_top5)}")
print(f"  {sorted(all_top5)}")

# majority top-5 = features in top-5 of ≥4 folds
from collections import Counter
cnt = Counter([f for r in fold_rankings for f in r])
stable = [f for f, c in cnt.items() if c >= 4]
print(f"Stable features (top-5 in ≥4 folds): {stable}")

print(f"\nAccuracy across folds: {fold_acc}")
print(f"  mean={np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}")
print(f"HIGH recall across folds: {fold_rec}")
print(f"  mean={np.mean(fold_rec):.4f} ± {np.std(fold_rec):.4f}")

if len(stable) >= 4 and np.std(fold_rec) < 0.05:
    print("\n✅ NO OVERFITTING — stable features + low recall variance")
else:
    print("\n⚠️ Review — feature ranking varies a lot")

# ═══════════════════════════════════════════════
# PART 2 — FEATURE BALANCE (distribution per risk level)
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2: FEATURE BALANCE — distribution by risk level")
print("=" * 60)

ORDER = ["LOW", "MEDIUM", "HIGH"]
COLORS = {"LOW": "#4CAF50", "MEDIUM": "#FFC107", "HIGH": "#F44336"}

# target balance
counts = y.value_counts().reindex(ORDER).fillna(0)
total = len(y)
print("\nTarget balance:")
for c in ORDER:
    print(f"  {c:6s}: {counts[c]:4d}  ({counts[c]/total:.1%})")

# class_weight='balanced' weights
n, n_classes = total, 3
print("\nBalanced class weights:")
for c in ORDER:
    w = n / (n_classes * counts[c]) if counts[c] > 0 else 0
    print(f"  {c:6s}: {w:.3f}")

# selected numeric features balance (mean per risk level)
num_feats = ["age", "night_bed_exits", "past_falls", "tug_seconds", "mobility_score",
             "polypharmacy_count", "days_since_last_fall"]
print("\nFeature means per risk level:")
print(f"{'Feature':<28}{'LOW':>10}{'MEDIUM':>10}{'HIGH':>10}")
for f in num_feats:
    row = [df.loc[df["fall_risk_level"] == c, f].mean() for c in ORDER]
    print(f"{f:<28}{row[0]:>10.2f}{row[1]:>10.2f}{row[2]:>10.2f}")

# ── Plot: feature balance boxplots (top 4 by SHAP later) ──
fig, axes = plt.subplots(2, 2, figsize=(18, 11))
plot_feats = ["night_bed_exits", "age", "tug_seconds", "mobility_score"]
for ax, f in zip(axes.flatten(), plot_feats):
    for c in ORDER:
        vals = df.loc[df["fall_risk_level"] == c, f]
        ax.hist(vals, bins=15, alpha=0.5, color=COLORS[c], label=c)
    ax.set_title(f"{f} by risk level", fontsize=14)
    ax.set_xlabel(f); ax.set_ylabel("count")
    ax.legend()
plt.suptitle("Feature Balance — Distribution by Risk Level (Phase 2)", fontsize=18)
plt.tight_layout()
plt.savefig("phase2_feature_balance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✅ Saved phase2_feature_balance.png")

# ── Save SHAP stability result as text ──
with open("phase2_shap_overfit_report.txt", "w") as f:
    f.write("PHASE 2 — SHAP OVERFITTING TEST REPORT\n")
    f.write("=" * 45 + "\n")
    for i, r in enumerate(fold_rankings):
        f.write(f"Fold {i+1} top-5: {r}\n")
    f.write(f"\nStable features (≥4 folds): {stable}\n")
    f.write(f"Accuracy: {np.mean(fold_acc):.4f} ± {np.std(fold_acc):.4f}\n")
    f.write(f"HIGH recall: {np.mean(fold_rec):.4f} ± {np.std(fold_rec):.4f}\n")
    f.write(f"Verdict: {'NO OVERFITTING' if len(stable)>=4 and np.std(fold_rec)<0.05 else 'REVIEW'}\n")
print("✅ Saved phase2_shap_overfit_report.txt")
