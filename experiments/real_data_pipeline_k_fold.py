# cd ~/Desktop/IA/demo/IA && unset PYTHONPATH && /opt/anaconda3/bin/python real_data_pipeline_k_fold.py
# 5-Fold K-Fold Cross-Validation — improved Kero's model (v2)
# Improvements: +4 interaction features, -polypharmacy, +StandardScaler
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, classification_report

# Load data
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})

# ── Improved feature engineering (research-based) ──
df["mobility_tug_ratio"]   = df["tug_seconds"] / (df["mobility_score"] + 1)   # 行動效率
df["night_falls_interaction"] = df["night_bed_exits"] * df["past_falls"]      # 離床×跌倒史
df["med_poly_interaction"] = df["high_risk_medication"] * df["polypharmacy_count"]  # 藥物疊加
df["age_mobility_risk"]    = df["age"] * (10 - df["mobility_score"])          # 年齡×行動差

# ── Base 10 features MINUS polypharmacy (noise, per permutation) ──
REQUIRED = ["age", "night_bed_exits", "night_activity_duration_min",
            "past_falls", "mobility_score", "high_risk_medication",
            "cognitive_impairment", "orthostatic_hypotension", "tug_seconds"]
ENGINEERED = ["mobility_tug_ratio", "night_falls_interaction",
              "med_poly_interaction", "age_mobility_risk"]
FEATURES = REQUIRED + ENGINEERED   # 9 + 4 = 13 features

X = df[FEATURES]
y = df["fall_risk_level"]

print("Loaded", df.shape)
print(f"Features used: {len(FEATURES)} (9 base − polypharmacy + 4 engineered)")
print("  " + ", ".join(FEATURES))

le = LabelEncoder()
y_num = le.fit_transform(y)

# ── StandardScaler (LR convergence + fairness) ──
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=FEATURES)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "LR":  LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    "RF":  RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "XGB": XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss'),
}

print("\n" + "=" * 45)
print("5-FOLD CROSS-VALIDATION (HIGH recall) — v2")
print("=" * 45)

results = {}
for name, model in models.items():
    acc_list, high_list = [], []
    for train_idx, val_idx in cv.split(X_scaled, y):
        Xtr, Xval = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
        if name == "XGB":
            ytr = y_num[train_idx]
            model.fit(Xtr, ytr)
            pred = le.inverse_transform(model.predict(Xval))
        else:
            ytr = y.iloc[train_idx]
            model.fit(Xtr, ytr)
            pred = model.predict(Xval)
        yval = y.iloc[val_idx]
        acc_list.append(accuracy_score(yval, pred))
        high_list.append(recall_score(yval, pred, labels=["HIGH"], average=None, zero_division=0)[0])
    results[name] = {"acc": acc_list, "rec": high_list}
    print(f"{name}: HIGH recall={sum(high_list)/5:.4f} ± {pd.Series(high_list).std():.4f} | acc={sum(acc_list)/5:.4f}")

best = max(results, key=lambda k: sum(results[k]["rec"])/5)
print("\n" + "=" * 45)
print(f"🏆 Best by HIGH recall: {best} ({sum(results[best]['rec'])/5:.4f})")

# ── Final: train best on 80/20, show report ──
print("\n" + "=" * 45)
print(f"FINAL: {best} on 80/20 split")
print("=" * 45)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y)
final = models[best]
if best == "XGB":
    ytr = le.fit_transform(y_train)
    final.fit(X_train, ytr)
    pred = le.inverse_transform(final.predict(X_test))
else:
    final.fit(X_train, y_train)
    pred = final.predict(X_test)
print(classification_report(y_test, pred, target_names=le.classes_, zero_division=0))
high = recall_score(y_test, pred, average=None, zero_division=0)[0]
print(f"\nHIGH Recall (test): {high:.4f}")
