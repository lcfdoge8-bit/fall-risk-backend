# Phase 2 Model — IMPROVED Kero K-Fold model, on P1's v2 data (17 features)
# =========================================================================
# Base: P1 real_data_pipeline_k_fold.py (Kero's approach)
#   - 5-fold StratifiedKFold, HIGH recall scoring
#   - class_weight='balanced'
#   - 4 engineered features
#   - dropped polypharmacy_count (noise per Permutation Importance)
# Improvements in P2:
#   1. New P1 v2 data (17 cols, adds sex + clinical background fields)
#   2. Added sex (F=0/M=1) + 3 clinical background features
#   3. Kept the engineered features
#   4. Threshold tuning on validation (recall/acc balance)
#   5. Overfit check (val vs test gap)
# cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && /opt/anaconda3/bin/python phase2_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report

# ── Step 1: Load P1's v2 data ──
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})
df["sex"] = df["sex"].map({"F": 0, "M": 1})

print("Shape:", df.shape)
print("Columns:", list(df.columns))

# ── Step 2: Features — Kero v2 engineered + sex + clinical background ──
# base 9 (dropped polypharmacy) + 4 engineered + sex + 3 clinical bg = 17 features
# (fall_risk_score NOT used — target leakage!)
df["mobility_tug_ratio"]      = df["tug_seconds"] / (df["mobility_score"] + 1)
df["night_falls_interaction"] = df["night_bed_exits"] * df["past_falls"]
df["med_poly_interaction"]    = df["high_risk_medication"] * df["polypharmacy_count"]
df["age_mobility_risk"]       = df["age"] * (10 - df["mobility_score"])

# clinical background: days_since_last_fall 空字串 → 0 (past_falls=0 冇跌過)
df["days_since_last_fall"] = pd.to_numeric(df["days_since_last_fall"], errors="coerce").fillna(0)

BASE = ["age", "night_bed_exits", "night_activity_duration_min",
        "past_falls", "mobility_score", "high_risk_medication",
        "cognitive_impairment", "orthostatic_hypotension", "tug_seconds"]
ENGINEERED = ["mobility_tug_ratio", "night_falls_interaction",
              "med_poly_interaction", "age_mobility_risk"]
CLIN_BG = ["days_since_last_fall", "syncopal_fall", "fall_cluster_30d"]
FEATURES = ["sex"] + BASE + ENGINEERED + CLIN_BG   # 17 features

X = df[FEATURES]
y = df["fall_risk_level"]

print(f"\nFeatures ({len(FEATURES)}): {FEATURES}")
print("Target:", y.value_counts().to_dict())

# ── Step 3: Encode target ──
le = LabelEncoder()
y_num = le.fit_transform(y)
HIGH_IDX = list(le.classes_).index("HIGH")

# ── Step 4: 70/20/10 split (same as Kero) ──
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp)

# align numeric y by index (same splits)
y_num_series = pd.Series(y_num, index=X.index)
y_num_train = y_num_series.loc[X_train.index].values
y_num_val = y_num_series.loc[X_val.index].values
y_num_test = y_num_series.loc[X_test.index].values

print(f"\nTrain {len(X_train)} | Val {len(X_val)} | Test {len(X_test)}")

# ── Step 5: 5-fold CV (Kero's approach) ──
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
models = {
    "LR":  LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42),
    "RF":  RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
    "XGB": XGBClassifier(n_estimators=200, max_depth=6, random_state=42, eval_metric='mlogloss'),
}

print("\n" + "=" * 45)
print("5-FOLD CV (HIGH recall) — Kero approach on v2 data")
print("=" * 45)
results = {}
for name, model in models.items():
    acc_list, high_list = [], []
    for train_idx, val_idx in cv.split(X, y):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        if name == "XGB":
            model.fit(Xtr, y_num[train_idx])
            pred = le.inverse_transform(model.predict(Xval))
        else:
            model.fit(Xtr, y.iloc[train_idx])
            pred = model.predict(Xval)
        yval = y.iloc[val_idx]
        acc_list.append(accuracy_score(yval, pred))
        high_list.append(recall_score(yval, pred, labels=["HIGH"], average=None, zero_division=0)[0])
    results[name] = {"acc": np.mean(acc_list), "rec": np.mean(high_list)}
    print(f"{name}: HIGH recall={np.mean(high_list):.4f} ± {np.std(high_list):.4f} | acc={np.mean(acc_list):.4f}")

best = max(results, key=lambda k: results[k]["rec"])
print(f"\n🏆 Best by HIGH recall: {best} ({results[best]['rec']:.4f})")

# ── Step 6: Train best on train+val, threshold tune on validation ──
X_fit = pd.concat([X_train, X_val])
y_fit = pd.concat([y_train, y_val])
y_num_fit = np.concatenate([y_num_train, y_num_val])

final = models[best]
if best == "XGB":
    final.fit(X_fit, y_num_fit)
else:
    final.fit(X_fit, y_fit)

proba_val = final.predict_proba(X_val)
proba_test = final.predict_proba(X_test)

def predict_with_threshold(proba, thr):
    preds = []
    for row in proba:
        if row[HIGH_IDX] >= thr:
            preds.append("HIGH")
        else:
            others = [(c, row[i]) for i, c in enumerate(le.classes_) if c != "HIGH"]
            preds.append(max(others, key=lambda x: x[1])[0])
    return np.array(preds)

print("\n" + "=" * 45)
print(f"THRESHOLD TUNING ({best}) on VALIDATION")
print("=" * 45)
best_thr, best_score = 0.5, -1
for thr in [0.50, 0.45, 0.40, 0.35, 0.30, 0.25, 0.20]:
    p = predict_with_threshold(proba_val, thr)
    acc = accuracy_score(y_val, p)
    rec = recall_score(y_val, p, labels=["HIGH"], average=None, zero_division=0)[0]
    score = min(acc/0.90, 1.0)*0.5 + min(rec/0.95, 1.0)*0.5
    print(f"thr={thr:.2f}  acc={acc:.4f}  HIGH recall={rec:.4f}  score={score:.4f}")
    if score > best_score:
        best_thr, best_score = thr, score
print(f"→ Best threshold = {best_thr:.2f}")

# ── Step 7: Final test + overfit check ──
print("\n" + "=" * 45)
print(f"FINAL: {best} (thr={best_thr:.2f}) on TEST + OVERFIT CHECK")
print("=" * 45)
test_pred = predict_with_threshold(proba_test, best_thr)
val_pred = predict_with_threshold(proba_val, best_thr)
test_acc = accuracy_score(y_test, test_pred)
test_rec = recall_score(y_test, test_pred, labels=["HIGH"], average=None, zero_division=0)[0]
val_acc = accuracy_score(y_val, val_pred)
val_rec = recall_score(y_val, val_pred, labels=["HIGH"], average=None, zero_division=0)[0]
print(f"Validation: acc={val_acc:.4f}  HIGH recall={val_rec:.4f}")
print(f"Test:       acc={test_acc:.4f}  HIGH recall={test_rec:.4f}")
print(f"Gap:        acc diff={abs(val_acc-test_acc):.4f}  recall diff={abs(val_rec-test_rec):.4f}")
if abs(val_acc - test_acc) < 0.03 and abs(val_rec - test_rec) < 0.03:
    print("\n✅ NO OVERFITTING")
else:
    print("\n⚠️ WATCH OUT — possible overfitting")

print(classification_report(y_test, test_pred, target_names=le.classes_, zero_division=0))
print(f"\nTarget: acc≥90%? {'✅' if test_acc>=0.90 else '❌ '+f'{test_acc:.1%}'} | recall≥95%? {'✅' if test_rec>=0.95 else '❌ '+f'{test_rec:.1%}'}")
