"""
Fall Risk Prediction + LIME explanation + Minimal Counterfactual
=================================================================
Model: Logistic Regression (trained on v2 feature data, 2026-08-20)
- predict_fall_risk(features)      -> 'LOW' / 'MEDIUM' / 'HIGH'
- explain_patient(features)        -> LIME explanation (why + which way)
- get_top3_features()              -> Top-3 risk factors (by LIME)
- get_minimal_change(features)     -> smallest INTEGER change to lower HIGH risk
- recommend_intervention(features) -> easiest actionable intervention

Input features (11): sex ('F'/'M' or 0/1) + the original 10.
"""
import os
import json
import numpy as np
import joblib
import lime
import lime.lime_tabular

_DIR = os.path.dirname(os.path.abspath(__file__))
LABELS = ["LOW", "MEDIUM", "HIGH"]
REQUIRED = ["sex", "age", "night_bed_exits", "night_activity_duration_min",
            "past_falls", "mobility_score", "high_risk_medication",
            "cognitive_impairment", "polypharmacy_count",
            "orthostatic_hypotension", "tug_seconds"]

TOP_3_FEATURES = [
  {
    "feature": "tug_seconds",
    "meaning": "TUG test time (seconds)",
    "lime_weight": 0.3637
  },
  {
    "feature": "past_falls",
    "meaning": "Number of past falls",
    "lime_weight": 0.1366
  },
  {
    "feature": "high_risk_medication",
    "meaning": "Uses high-risk medication (0/1)",
    "lime_weight": 0.1185
  }
]

# "safe" direction: which way to move each feature to LOWER the risk.
# +1 = increase makes safer, -1 = decrease makes safer.
SAFE_DIRECTION = {
    "age": -1,
    "night_bed_exits": -1,
    "night_activity_duration_min": -1,
    "past_falls": -1,
    "mobility_score": +1,          # higher mobility = safer
    "high_risk_medication": -1,
    "cognitive_impairment": -1,
    "polypharmacy_count": -1,
    "orthostatic_hypotension": -1,
    "tug_seconds": -1,             # faster = safer
}

# features a clinician CANNOT change through intervention
UNCHANGEABLE = {"sex", "age", "past_falls", "cognitive_impairment"}

_model = joblib.load(os.path.join(_DIR, "fall_risk_model.pkl"))
_train = np.load(os.path.join(_DIR, "train_data.npy"))

_explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=_train,
    feature_names=REQUIRED,
    class_names=LABELS,
    discretize_continuous=True,
    random_state=42,
)


def _to_row(features: dict) -> np.ndarray:
    """Encode a feature dict into a model-ready row.
    sex: 'F'/'M' (or 0/1); booleans: True/False (or 0/1).
    """
    row = []
    for name in REQUIRED:
        v = features[name]
        if name == "sex" and isinstance(v, str):
            v = 0 if v.strip().upper().startswith("F") else 1
        elif isinstance(v, bool):
            v = int(v)
        row.append(float(v))
    return np.asarray(row)


def predict_fall_risk(features: dict) -> str:
    """features: dict with keys = REQUIRED -> 'LOW'/'MEDIUM'/'HIGH'"""
    X = _to_row(features).reshape(1, -1)
    return str(_model.predict(X)[0])


def explain_patient(features: dict, max_features: int = 5) -> list:
    """LIME explanation for ONE patient.
    Returns: list of {condition, weight, direction}
    weight > 0 -> pushes toward predicted class
    weight < 0 -> pulls away
    """
    row = _to_row(features)
    pred = str(_model.predict(row.reshape(1, -1))[0])
    label_idx = list(_model.classes_).index(pred)

    exp = _explainer.explain_instance(
        data_row=row, predict_fn=_model.predict_proba,
        num_features=len(REQUIRED), labels=[label_idx])

    out = []
    for feat_text, weight in exp.as_list(label=label_idx)[:max_features]:
        direction = f"push {pred}" if weight > 0 else "pull away"
        out.append({"condition": feat_text, "weight": round(weight, 4),
                     "direction": direction})
    return out


def get_top3_features() -> list:
    """Top-3 risk factors (global, by LIME) for the API."""
    return TOP_3_FEATURES


def get_minimal_change(features: dict) -> list:
    """Find the smallest feature change that lowers a HIGH-risk patient's risk.
    Returns a list of dicts, sorted by normalized_change (smallest first).
    Each dict: {feature, from, to, change, normalized_change, can_flip}
    normalized_change = change / feature_range, so different scales compare fairly.
    All change values are positive integers (clinically meaningful).
    """
    pred = predict_fall_risk(features)
    if pred != 'HIGH':
        return []

    row = _to_row(features)
    lo = _train.min(axis=0)
    hi = _train.max(axis=0)

    results = []
    for i, feat in enumerate(REQUIRED):
        if feat in UNCHANGEABLE:
            continue
        cur = int(round(row[i]))
        direction = SAFE_DIRECTION[feat]
        lo_i, hi_i = int(round(lo[i])), int(round(hi[i]))
        span = hi_i - lo_i if hi_i > lo_i else 1.0

        # enumerate INTEGER candidates in the safe direction (skip current value)
        if direction > 0:
            candidates = range(cur + 1, hi_i + 1)       # increase
        else:
            candidates = range(cur - 1, lo_i - 1, -1)   # decrease

        flip_val = None
        for val in candidates:
            trial = row.copy()
            trial[i] = val
            if str(_model.predict(trial.reshape(1, -1))[0]) != 'HIGH':
                flip_val = val
                break

        if flip_val is not None:
            results.append({
                "feature": feat,
                "from": cur,
                "to": int(flip_val),
                "change": int(abs(flip_val - cur)),
                "normalized_change": round(float(abs(flip_val - cur) / span), 4),
                "can_flip": True,
            })
        else:
            results.append({
                "feature": feat,
                "from": cur,
                "to": None,
                "change": None,
                "normalized_change": None,
                "can_flip": False,
            })

    results.sort(key=lambda r: (
        not r["can_flip"],
        r["normalized_change"] if r["normalized_change"] is not None else float("inf"),
    ))
    return results


def recommend_intervention(features: dict) -> dict:
    """Human-readable: the easiest actionable change to lower risk."""
    pred = predict_fall_risk(features)
    if pred != 'HIGH':
        return {"risk": pred, "note": "Not HIGH - no intervention needed."}

    changes = get_minimal_change(features)
    actionable = [c for c in changes if c["can_flip"]]
    best = actionable[0] if actionable else changes[0]

    return {
        "risk": pred,
        "easiest_intervention": {
            "feature": best["feature"],
            "from": best["from"],
            "to": best["to"],
            "normalized_change": best["normalized_change"],
        },
        "all_options": changes,
    }


if __name__ == "__main__":
    test = {"sex": "M", "age": 85, "night_bed_exits": 2,
            "night_activity_duration_min": 31.7, "past_falls": 2,
            "mobility_score": 3, "high_risk_medication": 1,
            "cognitive_impairment": 1, "polypharmacy_count": 1,
            "orthostatic_hypotension": 0, "tug_seconds": 24.4}
    print("=" * 62)
    print("Fall Risk Model Output (self-test)")
    print("=" * 62)
    print(f"\n[1] PREDICTION         : {predict_fall_risk(test)}")
    print(f"[2] TOP-3 RISK FACTORS : "
          f"{[t['feature'] for t in get_top3_features()]}")
    print(f"\n[3] WHY this patient is HIGH (LIME explanation):")
    for e in explain_patient(test):
        print(f"      {e}")
    print(f"\n[4] POSSIBLE CHANGES (ranked, all 7 changeable features):")
    for c in get_minimal_change(test):
        if c["can_flip"]:
            print(f"      [CAN FLIP] {c['feature']}: "
                  f"{c['from']} -> {c['to']}  (normalized {c['normalized_change']})")
        else:
            print(f"      [cannot ]  {c['feature']}  -- changing alone not enough")
    print(f"\n[5] ★ SUGGESTION (the recommended single change):")
    print(f"      {recommend_intervention(test)['easiest_intervention']}")
    print("\n" + "=" * 62)
    print("Note:  [1]-[3] explain the prediction.  [4] is the full menu.")
    print("       [5] is THE final answer to give to the clinician.")
    print("=" * 62)
