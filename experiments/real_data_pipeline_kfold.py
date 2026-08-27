# Real Data Pipeline — fall_risk_patients_2000.csv
# Load with pandas → feature engineering → train with model

import os
import pandas as pd

#Load real data
df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv"))

#The 10 required features from the blueprint
REQUIRED = ["age", "night_bed_exits", "night_activity_duration_min",
            "past_falls", "mobility_score", "high_risk_medication",
            "cognitive_impairment", "polypharmacy_count",
            "orthostatic_hypotension", "tug_seconds"]

print("Shape:", df.shape)
print("\nRequired features present:", [c for c in REQUIRED if c in df.columns])
print("Missing: ", [c for c in REQUIRED if c not in c in df.columns])
print("\nColumns:", list(df.columns))

# -- Step 2: Seperate features (X) and target (Y) --

# The 10 required features
X = df[REQUIRED]

# Target: fall_risk_level (classification)
y = df["fall_risk_level"]

print("X shape:", X.shape)
print("y value counts:")                # (2000, 10)
print(y.value_counts().to_string())     # check how many patient in each risk level

# -- Step 3: 70/20/10 Split --
from sklearn.model_selection import train_test_split

# First split 70% train and 30% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify = y
)

# Second split temp 30% to 20% and 10%
# test_size = 0.333 -> 1/3
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.333, random_state=42, stratify = y_temp
)

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# --Step 4: Train on train set, validate on val set --
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

model = LogisticRegression(max_iter=5000, class_weight='balanced', random_state=42)

# use train to learn
model.fit(X_train, y_train)

# use validation to test model
val_pred = model.predict(X_val)

print("=== Validation Result ===")
print(classification_report(y_val, val_pred, zero_division=0))

# --Step 5: Final test evaluation--
from sklearn.metrics import recall_score

# use test set to test the model
test_pred = model.predict(X_test)

print("=== Final Test Result ===")
print(classification_report(y_test, test_pred, zero_division=0))

# High class recall
recall_high = recall_score(y_test, test_pred, average=None, zero_division=0)[0]
print(f"\nHigh Recall: {recall_high:.4f}")

# -- Step 6: Compare models --
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# Random Forest
rf = RandomForestClassifier(
    n_estimators=200, max_depth=10,
    class_weight='balanced', random_state=42, n_jobs=-1
)

rf.fit(X_train,y_train)
rf_val = rf.predict(X_val)
print("=== Random Forest Result (validation) ===")
print(classification_report(y_val, rf_val, zero_division=0))

# -- Encode y for XGBoost (XGBoost needs numbers) --
from sklearn.preprocessing import LabelEncoder

# change number to text to generate a report
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)

# XGBoost
xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=6, random_state=42
)
xgb_model.fit(X_train, y_train_enc)
xgb_val_enc = xgb_model.predict(X_val)
xgb_val = le.inverse_transform(xgb_val_enc)

print("=== XGBosot Result (validation) ===")
print(classification_report(y_val,xgb_val, zero_division=0))




# -- Step 7: Compare & pick best by High recall--
def high_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, zero_division=0)[0]

result = {
    "Logistic Regression":      high_recall(y_val, val_pred),
    "Random_Forest":            high_recall(y_val, rf_val),
    "XGBoost":                  high_recall(y_val, xgb_val),
}

print("\n=== Validation HIGH Recall Comparison ===")
for name, r in result.items():
    print(f"    {name}: {r:.4f}")
best = max(result, key=result.get)
print(f"\n Best model by HIGH recall: {best} ({result[best]:.4f})")







