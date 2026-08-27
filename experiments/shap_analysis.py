# SHAP analysis — explain LR model with 10 features (multiple large plots)
import os
import pandas as pd
import numpy as np
import shap
from sklearn.linear_model import LogisticRegression

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ═══ Load data ═══
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))
for col in ["high_risk_medication", "orthostatic_hypotension"]:
    df[col] = df[col].map({True: 1, False: 0})

REQUIRED = ["age","night_bed_exits","night_activity_duration_min",
            "past_falls","mobility_score","high_risk_medication",
            "cognitive_impairment","polypharmacy_count",
            "orthostatic_hypotension","tug_seconds"]
X = df[REQUIRED]
y = df["fall_risk_level"]

# ═══ Train Kero's LR ═══
model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)
model.fit(X, y)

# ═══ SHAP explainer ═══
explainer = shap.LinearExplainer(model, X)
shap_values = explainer.shap_values(X)   # shape (2000, 10)

# ═══ 圖 1: Summary beeswarm (feature importance + 方向) ═══
plt.figure(figsize=(12, 8))
shap.summary_plot(shap_values, X, feature_names=REQUIRED, show=False, max_display=10)
plt.tight_layout()
plt.savefig("shap_1_summary_beeswarm.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 1. shap_1_summary_beeswarm.png")

# ═══ 圖 2: Bar plot (mean |SHAP| — 純 feature importance) ═══
plt.figure(figsize=(10, 7))
shap.summary_plot(shap_values, X, feature_names=REQUIRED,
                  plot_type="bar", show=False, max_display=10)
plt.tight_layout()
plt.savefig("shap_2_bar_importance.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 2. shap_2_bar_importance.png")

# ═══ 圖 3: 揀一個 HIGH-risk 病人做 waterfall (解釋個別案例) ═══
# LinearExplainer multi-class → shap_values 係 list (每 class 一個)
# 攞 HIGH class (le.classes_ = ['HIGH','LOW','MEDIUM'] → index 0)
sv_high = shap_values[0]
base_high = explainer.expected_value[0]

high_idx = np.where(y.values == "HIGH")[0][0]
plt.figure(figsize=(14, 7))
shap.plots.waterfall(
    shap.Explanation(sv_high[high_idx], base_values=base_high,
                     data=X.iloc[high_idx].values, feature_names=REQUIRED),
    max_display=10, show=False,
)
plt.tight_layout()
plt.savefig("shap_3_waterfall_HIGH.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 3. shap_3_waterfall_HIGH.png")

# ═══ 圖 4: Heatmap (所有病人嘅 SHAP pattern) ═══
plt.figure(figsize=(14, 9))
shap.plots.heatmap(
    shap.Explanation(sv_high, base_values=base_high,
                     data=X.values, feature_names=REQUIRED),
    max_display=10, show=False,
)
plt.tight_layout()
plt.savefig("shap_4_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()
print("✅ 4. shap_4_heatmap.png")

print("\n🎉 All 4 SHAP plots saved!")
