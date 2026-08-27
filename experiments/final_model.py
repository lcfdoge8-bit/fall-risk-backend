# FINAL MODEL — LR + Alex 14 features + Kero validation method
# =============================================================
# Features:  Alex's 14 (proven best in fair comparison)
# Validation: Kero's rigorous approach (70/20/10 + 5-fold + threshold + overfit check)
# Model:      Logistic Regression (class_weight='balanced') — proven best across
#             both feature sets and all 3 model types (LR > XGB > RF)
#
# Run:
#   cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && /opt/anaconda3/bin/python final_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report
import joblib, json, os

# ── Load data ──
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})
df["sex"] = df["sex"].map({"F": 0, "M": 1})
df["days_since_last_fall"] = pd.to_numeric(df["days_since_last_fall"], errors="coerce").fillna(-1)

# ── Alex's 14 features (proven best) ──
FEATURES = ["sex", "age", "night_bed_exits", "night_activity_duration_min",
            "past_falls", "mobility_score", "high_risk_medication",
            "cognitive_impairment", "polypharmacy_count",
            "orthostatic_hypotension", "tug_seconds",
            "days_since_last_fall", "syncopal_fall", "fall_cluster_30d"]

X = df[FEATURES]
y = df["fall_risk_level"]

le = LabelEncoder()
y_num = le.fit_transform(y)
HIGH_IDX = list(le.classes_).index("HIGH")

print(f"FINAL MODEL — {len(FEATURES)} features (Alex) + Kero validation")
print(FEATURES)

# ═══════════════════════════════════════════════
# PART 1 — 5-FOLD CV (Kero's stability check)
# ═══════════════════════════════════════════════
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

print("\n" + "=" * 60)
print("PART 1: 5-FOLD CV (stability)")
print("=" * 60)
print(f"Accuracy:       {np.mean(accs):.4f} ± {np.std(accs):.4f}")
print(f"HIGH recall:    {np.mean(recs):.4f} ± {np.std(recs):.4f}")
print(f"HIGH precision: {np.mean(precs):.4f} ± {np.std(precs):.4f}")

# ═══════════════════════════════════════════════
# PART 2 — 70/20/10 + threshold tuning (Kero method)
# ═══════════════════════════════════════════════
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify=y_temp)

X_fit = pd.concat([X_train, X_val])
y_fit = pd.concat([y_train, y_val])

model = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
model.fit(X_fit, y_fit)

proba_val = model.predict_proba(X_val)
proba_test = model.predict_proba(X_test)

def predict_with_threshold(proba, thr):
    preds = []
    for row in proba:
        if row[HIGH_IDX] >= thr:
            preds.append("HIGH")
        else:
            others = [(c, row[i]) for i, c in enumerate(le.classes_) if c != "HIGH"]
            preds.append(max(others, key=lambda x: x[1])[0])
    return np.array(preds)

print("\n" + "=" * 60)
print("PART 2: THRESHOLD TUNING on VALIDATION (Kero method)")
print("=" * 60)
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

# ═══════════════════════════════════════════════
# PART 3 — FINAL TEST + OVERFIT CHECK
# ═══════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"PART 3: FINAL on TEST (thr={best_thr:.2f}) + OVERFIT CHECK")
print("=" * 60)
test_pred = predict_with_threshold(proba_test, best_thr)
val_pred = predict_with_threshold(proba_val, best_thr)

test_acc = accuracy_score(y_test, test_pred)
test_rec = recall_score(y_test, test_pred, labels=["HIGH"], average=None, zero_division=0)[0]
test_prec = precision_score(y_test, test_pred, labels=["HIGH"], average=None, zero_division=0)[0]
test_f1 = f1_score(y_test, test_pred, labels=["HIGH"], average=None, zero_division=0)[0]
val_acc = accuracy_score(y_val, val_pred)
val_rec = recall_score(y_val, val_pred, labels=["HIGH"], average=None, zero_division=0)[0]

print(f"Validation: acc={val_acc:.4f}  HIGH recall={val_rec:.4f}")
print(f"Test:       acc={test_acc:.4f}  HIGH recall={test_rec:.4f}")
print(f"Gap:        acc diff={abs(val_acc-test_acc):.4f}  recall diff={abs(val_rec-test_rec):.4f}")
if abs(val_acc - test_acc) < 0.03 and abs(val_rec - test_rec) < 0.03:
    print("✅ NO OVERFITTING")
else:
    print("⚠️ WATCH OUT — possible overfitting")

print("\n" + "-" * 60)
print("CLASSIFICATION REPORT (test)")
print("-" * 60)
print(classification_report(y_test, test_pred, target_names=le.classes_, zero_division=0))

n_high = (y_test.values == "HIGH").sum()
caught = ((test_pred == "HIGH") & (y_test.values == "HIGH")).sum()
print(f"\nHIGH caught: {caught}/{n_high} | missed: {n_high - caught}")
print(f"\n🎯 Target: acc≥90%? {'✅' if test_acc>=0.90 else '❌ '+f'{test_acc:.1%}'} | recall≥95%? {'✅' if test_rec>=0.95 else '❌ '+f'{test_rec:.1%}'}")

# ═══════════════════════════════════════════════
# SAVE FINAL MODEL (for API deployment)
# ═══════════════════════════════════════════════
os.makedirs("saved", exist_ok=True)
joblib.dump(model, "saved/final_model.pkl")
joblib.dump(le, "saved/label_encoder.pkl")
with open("saved/features.json", "w") as f:
    json.dump(FEATURES, f)
with open("saved/threshold.json", "w") as f:
    json.dump({"threshold": best_thr, "high_idx": HIGH_IDX, "classes": list(le.classes_)}, f)
print("\n✅ Saved: saved/final_model.pkl + label_encoder.pkl + features.json + threshold.json")
