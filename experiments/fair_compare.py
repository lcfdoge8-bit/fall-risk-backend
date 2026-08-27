# Fair Comparison — Kero (17 features) vs Alex (14 features)
# ===========================================================
# SAME split, SAME test set, SAME metrics, SAME evaluation protocol.
# Only difference = the feature set (the actual thing we're comparing).
#
# Protocol:
#   - 70/20/10 split (stratified, random_state=42) — same for both
#   - Both trained on the SAME train+val data
#   - Both evaluated on the SAME test set
#   - Metrics: accuracy, HIGH recall, HIGH precision, HIGH F1
#   - 5-fold CV on train data for stability (same folds for both)
#   - Threshold: default 0.5 for BOTH (no unfair threshold advantage)
#
# Run:
#   cd ~/Desktop/IA/demo/IA2 && unset PYTHONPATH && /opt/anaconda3/bin/python fair_compare.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, classification_report

# ── Load data (shared by both) ──
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})
df["sex"] = df["sex"].map({"F": 0, "M": 1})

y = df["fall_risk_level"]
le = LabelEncoder()
y_num = le.fit_transform(y)

# ── Feature set A: KERO (17 features) ──
df_k = df.copy()
df_k["mobility_tug_ratio"]      = df_k["tug_seconds"] / (df_k["mobility_score"] + 1)
df_k["night_falls_interaction"] = df_k["night_bed_exits"] * df_k["past_falls"]
df_k["med_poly_interaction"]    = df_k["high_risk_medication"] * df_k["polypharmacy_count"]
df_k["age_mobility_risk"]       = df_k["age"] * (10 - df_k["mobility_score"])
df_k["days_since_last_fall"]    = pd.to_numeric(df_k["days_since_last_fall"], errors="coerce").fillna(0)

KERO_FEATURES = (["sex"] +
    ["age","night_bed_exits","night_activity_duration_min","past_falls",
     "mobility_score","high_risk_medication","cognitive_impairment",
     "orthostatic_hypotension","tug_seconds"] +
    ["mobility_tug_ratio","night_falls_interaction","med_poly_interaction","age_mobility_risk"] +
    ["days_since_last_fall","syncopal_fall","fall_cluster_30d"])

# ── Feature set B: ALEX (14 features) ──
df_a = df.copy()
df_a["days_since_last_fall"] = pd.to_numeric(df_a["days_since_last_fall"], errors="coerce").fillna(-1)

ALEX_FEATURES = ["sex","age","night_bed_exits","night_activity_duration_min",
                 "past_falls","mobility_score","high_risk_medication",
                 "cognitive_impairment","polypharmacy_count",
                 "orthostatic_hypotension","tug_seconds",
                 "days_since_last_fall","syncopal_fall","fall_cluster_30d"]

print("Kero features:", len(KERO_FEATURES))
print("Alex features:", len(ALEX_FEATURES))

# ═══════════════════════════════════════════════
# PART 1 — SAME 5-FOLD CV (on full data, same folds)
# ═══════════════════════════════════════════════
print("\n" + "=" * 62)
print("PART 1: SAME 5-FOLD CV (same folds, default threshold 0.5)")
print("=" * 62)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

def run_cv(feat_list, df_src):
    accs, recs, precs = [], [], []
    for tr, va in cv.split(df_src[feat_list], y):
        m = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
        m.fit(df_src[feat_list].iloc[tr], y.iloc[tr])
        p = m.predict(df_src[feat_list].iloc[va])
        yv = y.iloc[va]
        accs.append(accuracy_score(yv, p))
        recs.append(recall_score(yv, p, labels=["HIGH"], average=None, zero_division=0)[0])
        precs.append(precision_score(yv, p, labels=["HIGH"], average=None, zero_division=0)[0])
    return np.mean(accs), np.mean(recs), np.mean(precs), np.std(accs), np.std(recs)

k_acc, k_rec, k_prec, k_acc_std, k_rec_std = run_cv(KERO_FEATURES, df_k)
a_acc, a_rec, a_prec, a_acc_std, a_rec_std = run_cv(ALEX_FEATURES, df_a)

print(f"{'Model':<10}{'Acc':>10}{'HIGH rec':>12}{'HIGH prec':>12}")
print(f"{'Kero(17)':<10}{k_acc:>10.4f}{k_rec:>12.4f}{k_prec:>12.4f}")
print(f"{'Alex(14)':<10}{a_acc:>10.4f}{a_rec:>12.4f}{a_prec:>12.4f}")

# ═══════════════════════════════════════════════
# PART 2 — SAME 70/20/10 SPLIT, SAME TEST SET
# ═══════════════════════════════════════════════
print("\n" + "=" * 62)
print("PART 2: SAME 70/20/10 split → SAME test set")
print("=" * 62)

# split once on the raw df (shared)
X_raw = df.drop(columns=["fall_risk_level"])
X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
    X_raw, y, test_size=0.30, random_state=42, stratify=y)
X_val_raw, X_test_raw, y_val, y_test = train_test_split(
    X_temp_raw, y_temp, test_size=0.333, random_state=42, stratify=y_temp)

test_idx = X_test_raw.index   # THE SAME test rows for both

def run_split(feat_list, df_src):
    # build train/val/test from the same indices
    X_train_s = df_src.loc[X_train_raw.index, feat_list]
    X_val_s   = df_src.loc[X_val_raw.index, feat_list]
    X_test_s  = df_src.loc[test_idx, feat_list]
    y_fit = pd.concat([y_train, y_val])
    X_fit = pd.concat([X_train_s, X_val_s])
    m = LogisticRegression(max_iter=5000, class_weight="balanced", random_state=42)
    m.fit(X_fit, y_fit)
    p = m.predict(X_test_s)
    return p

k_pred = run_split(KERO_FEATURES, df_k)
a_pred = run_split(ALEX_FEATURES, df_a)

def report(name, pred):
    acc = accuracy_score(y_test, pred)
    rec = recall_score(y_test, pred, labels=["HIGH"], average=None, zero_division=0)[0]
    prec = precision_score(y_test, pred, labels=["HIGH"], average=None, zero_division=0)[0]
    f1h = f1_score(y_test, pred, labels=["HIGH"], average=None, zero_division=0)[0]
    print(f"\n{name}:")
    print(f"  Accuracy:     {acc:.4f}")
    print(f"  HIGH recall:  {rec:.4f}")
    print(f"  HIGH precision: {prec:.4f}")
    print(f"  HIGH F1:      {f1h:.4f}")
    return acc, rec, prec, f1h

k_metrics = report("KERO (17 features)", k_pred)
a_metrics = report("ALEX (14 features)", a_pred)

# ═══════════════════════════════════════════════
# PART 3 — WHERE THEY DIFFER (confusion analysis)
# ═══════════════════════════════════════════════
print("\n" + "=" * 62)
print("PART 3: Where do they disagree?")
print("=" * 62)
disagree = (k_pred != a_pred).sum()
print(f"Predictions differ on {disagree}/{len(y_test)} test patients ({disagree/len(y_test):.1%})")

# who is right when they disagree
both_right = 0; kero_right = 0; alex_right = 0
for i in range(len(y_test)):
    if k_pred[i] != a_pred[i]:
        k_ok = k_pred[i] == y_test.iloc[i]
        a_ok = a_pred[i] == y_test.iloc[i]
        if k_ok and a_ok: both_right += 1
        elif k_ok: kero_right += 1
        elif a_ok: alex_right += 1
print(f"When disagreeing: Kero right={kero_right}, Alex right={alex_right}, both right={both_right}")

# HIGH-risk patients each catches
y_test_arr = y_test.values
k_caught = (k_pred == "HIGH") & (y_test_arr == "HIGH")
a_caught = (a_pred == "HIGH") & (y_test_arr == "HIGH")
print(f"\nHIGH patients caught: Kero={k_caught.sum()}/{ (y_test_arr=='HIGH').sum() }, "
      f"Alex={a_caught.sum()}/{(y_test_arr=='HIGH').sum()}")
missed_k = (y_test_arr == "HIGH") & (k_pred != "HIGH")
missed_a = (y_test_arr == "HIGH") & (a_pred != "HIGH")
print(f"HIGH patients missed: Kero={missed_k.sum()}, Alex={missed_a.sum()}")
