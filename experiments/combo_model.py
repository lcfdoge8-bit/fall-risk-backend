# Combination model — Kero engineered features + Alex's polypharmacy & -1 handling
# ================================================================================
# Combines the best of both:
#   - Kero: 4 engineered features (mobility_tug_ratio, night_falls_interaction,
#           med_poly_interaction, age_mobility_risk)
#   - Alex: KEEPS polypharmacy_count (Kero dropped it)
#   - Alex: days_since_last_fall = -1 for 'never fell' (Kero used 0)
# Total features: sex + 10 base (INCLUDING polypharmacy) + 4 engineered + 3 clin_bg = 18
#
# Run:
#   cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && /opt/anaconda3/bin/python combo_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})
df["sex"] = df["sex"].map({"F": 0, "M": 1})

# Kero engineered features
df["mobility_tug_ratio"]      = df["tug_seconds"] / (df["mobility_score"] + 1)
df["night_falls_interaction"] = df["night_bed_exits"] * df["past_falls"]
df["med_poly_interaction"]    = df["high_risk_medication"] * df["polypharmacy_count"]
df["age_mobility_risk"]       = df["age"] * (10 - df["mobility_score"])

# Alex's -1 handling (never fell = -1)
df["days_since_last_fall"]    = pd.to_numeric(df["days_since_last_fall"], errors="coerce").fillna(-1)

# 18 features: sex + 10 base (WITH polypharmacy) + 4 engineered + 3 clin_bg
FEATURES = (["sex"] +
    ["age","night_bed_exits","night_activity_duration_min","past_falls",
     "mobility_score","high_risk_medication","cognitive_impairment",
     "polypharmacy_count","orthostatic_hypotension","tug_seconds"] +
    ["mobility_tug_ratio","night_falls_interaction","med_poly_interaction","age_mobility_risk"] +
    ["days_since_last_fall","syncopal_fall","fall_cluster_30d"])

X = df[FEATURES]
y = df["fall_risk_level"]

le = LabelEncoder()
y_num = le.fit_transform(y)
HIGH_IDX = list(le.classes_).index("HIGH")

print(f"Combo model: {len(FEATURES)} features")
print(FEATURES)

# ── PART 1: 5-fold CV (same protocol as fair_compare) ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accs, recs, precs = [], [], []
for tr, va in cv.split(X, y):
    m = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    m.fit(X.iloc[tr], y.iloc[tr])
    p = m.predict(X.iloc[va])
    yv = y.iloc[va]
    accs.append(accuracy_score(yv, p))
    recs.append(recall_score(yv, p, labels=["HIGH"], average=None, zero_division=0)[0])
    precs.append(precision_score(yv, p, labels=["HIGH"], average=None, zero_division=0)[0])

print("\n" + "=" * 62)
print("PART 1: 5-FOLD CV (combo, 18 features)")
print("=" * 62)
print(f"{'Acc':>10}{'HIGH rec':>12}{'HIGH prec':>12}")
print(f"{np.mean(accs):>10.4f}{np.mean(recs):>12.4f}{np.mean(precs):>12.4f}")

# ── PART 2: same 70/20/10 as fair_compare ──
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp)

X_fit = pd.concat([X_train, X_val])
y_fit = pd.concat([y_train, y_val])

m = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
m.fit(X_fit, y_fit)
p = m.predict(X_test)

acc = accuracy_score(y_test, p)
rec = recall_score(y_test, p, labels=["HIGH"], average=None, zero_division=0)[0]
prec = precision_score(y_test, p, labels=["HIGH"], average=None, zero_division=0)[0]
f1h = f1_score(y_test, p, labels=["HIGH"], average=None, zero_division=0)[0]

print("\n" + "=" * 62)
print("PART 2: SAME 70/20/10 TEST SET (combo, 18 features)")
print("=" * 62)
print(f"Accuracy:       {acc:.4f}")
print(f"HIGH recall:    {rec:.4f}")
print(f"HIGH precision: {prec:.4f}")
print(f"HIGH F1:        {f1h:.4f}")

# HIGH caught
n_high = (y_test.values == "HIGH").sum()
caught = ((p == "HIGH") & (y_test.values == "HIGH")).sum()
print(f"\nHIGH caught: {caught}/{n_high} | missed: {n_high - caught}")

# ── Comparison table vs fair_compare results ──
print("\n" + "=" * 62)
print("COMPARISON (same test set)")
print("=" * 62)
print(f"{'Model':<14}{'Acc':>10}{'HIGH rec':>12}{'HIGH prec':>12}")
print(f"{'Kero(17)':<14}{0.9100:>10.4f}{0.9811:>12.4f}{0.8814:>12.4f}")
print(f"{'Alex(14)':<14}{0.9150:>10.4f}{1.0000:>12.4f}{0.8689:>12.4f}")
print(f"{'COMBO(18)':<14}{acc:>10.4f}{rec:>12.4f}{prec:>12.4f}")
