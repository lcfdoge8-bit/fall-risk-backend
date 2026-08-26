"""
Retrain the fall-risk Logistic Regression on the v2 feature set + 3 fall-detail features.
=======================================================================================
New feature order (14) — MUST match ml/predict.py REQUIRED:
  sex, age, night_bed_exits, night_activity_duration_min, past_falls, mobility_score,
  high_risk_medication, cognitive_impairment, polypharmacy_count, orthostatic_hypotension,
  tug_seconds,
  days_since_last_fall (-1 = never fell), syncopal_fall, fall_cluster_30d

Outputs (overwrite, keep .bak):
  ml/fall_risk_model.pkl   - the fitted LogisticRegression
  ml/train_data.npy        - the training X matrix (used by the LIME explainer)
  ml/top3_features.json    - static top-3 risk factors
"""
import os
import csv
import json
import numpy as np
import joblib
from collections import Counter
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(_DIR, "..", "data", "fall_risk_patients_2000_v2.csv")

# 14 features now (11 original + 3 fall-detail). ORDER MUST MATCH ml/predict.py REQUIRED.
FEATURES = [
    "sex", "age", "night_bed_exits", "night_activity_duration_min", "past_falls",
    "mobility_score", "high_risk_medication", "cognitive_impairment",
    "polypharmacy_count", "orthostatic_hypotension", "tug_seconds",
    "days_since_last_fall", "syncopal_fall", "fall_cluster_30d",
]
LABEL = "fall_risk_level"

def _bool(v):
    return 1 if str(v).strip().lower() in ("1", "true", "yes") else 0

def _rs(row, key):
    """days_since_last_fall: '' -> -1 means 'never fell' (no recent-fall info)."""
    s = str(row[key]).strip()
    if s == "":
        return -1.0
    return float(s)

def load_rows():
    with open(CSV, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))

def encode(row):
    vals = []
    for name in FEATURES:
        if name == "sex":
            vals.append(0.0 if row[name].strip().upper().startswith("F") else 1.0)
        elif name in ("high_risk_medication", "orthostatic_hypotension"):
            vals.append(float(_bool(row[name])))
        elif name == "days_since_last_fall":
            vals.append(_rs(row, name))
        else:
            vals.append(float(row[name]))
    return vals

def main():
    rows = load_rows()
    if not rows:
        raise SystemExit("No rows found in CSV")
    X = np.array([encode(row) for row in rows], dtype=np.float64)
    y = np.array([row[LABEL].strip() for row in rows])
    print(f"Loaded {len(rows)} rows, {X.shape[1]} features, {FEATURES}")
    print("Label distribution:", dict(Counter(y)))

    # class_weight='balanced' is REQUIRED: without it HIGH recall drops to ~87%
    # (validated by the team's imbalance experiments - see day22_smote.py)
    clf = LogisticRegression(max_iter=20000, class_weight="balanced", random_state=42)
    clf.fit(X, y)

    # train/test split just to report a sanity metric (model itself is trained on all)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf_sanity = LogisticRegression(max_iter=20000, class_weight="balanced", random_state=42).fit(X_tr, y_tr)
    pred = clf_sanity.predict(X_te)
    print(f"Sanity hold-out accuracy: {accuracy_score(y_te, pred):.3f}")
    print(classification_report(y_te, pred, digits=3))

    joblib.dump(clf, os.path.join(_DIR, "fall_risk_model.pkl"))
    np.save(os.path.join(_DIR, "train_data.npy"), X)

    # Rebuild the static top-3 (trained on all classes; use mean abs coefficient weight
    # across classes, weighted by class frequency for cross-class comparability).
    from collections import defaultdict
    weight_by_feat = defaultdict(float)
    freq = Counter(y)
    n = len(y)
    for cls, coef_row in zip(clf.classes_, clf.coef_):
        w = freq[cls] / n
        for feat, c in zip(FEATURES, coef_row):
            weight_by_feat[feat] += abs(c) * w
    ranked = sorted(weight_by_feat.items(), key=lambda kv: -kv[1])[:3]
    meanings = {
        "tug_seconds": "TUG test time (seconds)",
        "past_falls": "Number of past falls",
        "high_risk_medication": "Uses high-risk medication (0/1)",
        "mobility_score": "Mobility score (1-10)",
        "polypharmacy_count": "Number of medicines",
        "orthostatic_hypotension": "Dizzy when standing (0/1)",
    }
    top3 = [{"feature": f, "meaning": meanings.get(f, f), "weight": round(w, 4)}
            for f, w in ranked]
    with open(os.path.join(_DIR, "top3_features.json"), "w", encoding="utf-8") as jf:
        json.dump(top3, jf, indent=2)
    print("Top-3 features ->", top3)
    print(f"Saved model to ml/fall_risk_model.pkl, train_data to ml/train_data.npy")

if __name__ == "__main__":
    main()