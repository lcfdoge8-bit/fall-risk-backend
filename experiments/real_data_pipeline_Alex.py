import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "fall_risk_patients_2000_v2.csv")
df = pd.read_csv(DATA, encoding="utf-8-sig")

# Preprocessing: drop id/name/leakage columns, convert boolean columns to 0/1
df = df.drop(columns=[c for c in ['patient_id', 'name', 'fall_risk_score'] if c in df.columns])
for col in ['high_risk_medication', 'orthostatic_hypotension']:
    df[col] = df[col].map({True: 1, False: 0})

X = df.drop(columns=['fall_risk_level'])
y = df['fall_risk_level']

# stratify=y keeps class proportions (important with imbalanced data)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

labels = ['LOW', 'MEDIUM', 'HIGH']
# XGBoost requires numeric labels
y_train_num = y_train.map({'LOW': 0, 'MEDIUM': 1, 'HIGH': 2})

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(C=100, max_iter=5000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6,
                             random_state=42, eval_metric='mlogloss'),
}

results = {}
for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_s, y_train)
        pred = model.predict(X_test_s)
    elif name == 'Random Forest':
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
    else:  # XGBoost uses numeric labels; map predictions back to text
        model.fit(X_train, y_train_num)
        pred = pd.Series(model.predict(X_test)).map({0:'LOW',1:'MEDIUM',2:'HIGH'}).values
    results[name] = {'pred': pred, 'acc': accuracy_score(y_test, pred),
                     'cm': confusion_matrix(y_test, pred, labels=labels)}

# Summary
for name, r in results.items():
    print(f"{name}: {r['acc']:.2%}")

# Classification report for each model
for name, r in results.items():
    print(f"\n{name} Accuracy: {r['acc']:.2%}")
    print(classification_report(y_test, r['pred'], target_names=labels, digits=2))

# HIGH recall comparison
hi = labels.index('HIGH')
for name, r in results.items():
    tp = r['cm'][hi, hi]; total = r['cm'][hi].sum()
    print(f"{name}: HIGH Recall = {tp}/{total} = {tp/total:.2%} (missed {total-tp})")

# Three confusion matrices side by side
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, (name, r) in zip(axes, results.items()):
    sns.heatmap(r['cm'], annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False, ax=ax)
    ax.set_title(f"{name}\nAcc = {r['acc']:.2%}")
plt.tight_layout()
plt.savefig("day8_three_models_cm.png", dpi=100, bbox_inches='tight')

def high_recall(y_true, y_pred):
    return recall_score(y_true, y_pred, average=None, zero_division=0)[0]

result = {
    "Logistic Regression": high_recall(y_test, results["Logistic Regression"]["pred"]),
    "Random_Forest":       high_recall(y_test, results["Random Forest"]["pred"]),
    "XGBoost":             high_recall(y_test, results["XGBoost"]["pred"]),
}

print("\n=== Validation HIGH Recall Comparison ===")
for name, r in result.items():
    print(f"    {name}: {r:.4f}")
best = max(result, key=result.get)
print(f"\n Best model by HIGH recall: {best} ({result[best]:.4f})")
