"""
K-Pop Scandal Impact Predictor — Model Training Script
Run this once to produce model_base.joblib and pipeline_config.json
Requirements: pip install pandas scikit-learn joblib
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import (classification_report, f1_score, accuracy_score,
                             roc_auc_score, confusion_matrix)
import joblib
import json
import os
import warnings
warnings.filterwarnings('ignore')

# === LOAD DATA ===
DATA_PATH = "kpop_scandals_FINALFINAL.csv"
df = pd.read_csv(DATA_PATH)

feature_cols = [
    'scandal_type', 'fandom_size_num', 'agency_tier', 'company_response',
    'response_delay_days', 'apology', 'international', 'is_solo',
    'prior_scandal', 'reaction_spike'
]

X = pd.get_dummies(df[feature_cols], columns=['scandal_type'], prefix='type')
y = (df['label_binary'] == 'high').astype(int)

feature_names = list(X.columns)

print(f"Dataset: {len(df)} rows, {y.sum()} high / {(1-y).sum()} not_high")
print(f"Features: {len(feature_names)}")
print(f"Majority baseline accuracy: {1 - y.mean():.3f}\n")

# === CROSS-VALIDATE ===
clf = RandomForestClassifier(
    n_estimators=200,
    class_weight='balanced',
    max_depth=8,
    min_samples_leaf=2,
    random_state=42
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
y_pred = cross_val_predict(clf, X, y, cv=cv, method='predict')
y_prob = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')[:, 1]

acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
auc = roc_auc_score(y, y_prob)
cm = confusion_matrix(y, y_pred)

print("=== 5-Fold Cross-Validation ===")
print(classification_report(y, y_pred, target_names=['not_high', 'high']))
print(f"AUC: {auc:.3f}")
print(f"Confusion Matrix: TN={cm[0,0]} FP={cm[0,1]} FN={cm[1,0]} TP={cm[1,1]}\n")

# Stability check
f1_scores = []
for seed in range(10):
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf_s = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   max_depth=8, min_samples_leaf=2, random_state=seed)
    y_pred_s = cross_val_predict(clf_s, X, y, cv=cv_s, method='predict')
    f1_scores.append(f1_score(y, y_pred_s))

print(f"Stability (10 seeds): F1 mean={np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
print(f"  Range: [{min(f1_scores):.3f}, {max(f1_scores):.3f}]\n")

# === TRAIN FINAL MODEL ON ALL DATA ===
clf.fit(X, y)

# Feature importances
importances = dict(zip(feature_names, clf.feature_importances_.tolist()))
print("=== Feature Importance ===")
for feat, imp in sorted(importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25s} {imp:.3f} {bar}")

# === EXPORT MODEL ===
joblib.dump(clf, 'model_base.joblib')
print(f"\nSaved model_base.joblib ({os.path.getsize('model_base.joblib') / 1024:.0f} KB)")

# === EXPORT PIPELINE CONFIG ===
pipeline_config = {
    'feature_columns': feature_names,
    'scandal_types': sorted(df['scandal_type'].unique().tolist()),
    'feature_importances': importances,
    'model_metrics': {
        'accuracy': round(acc, 3),
        'f1_high': round(f1, 3),
        'auc': round(auc, 3),
        'majority_baseline': round(1 - y.mean(), 3),
        'n_samples': len(df),
        'n_high': int(y.sum()),
        'n_not_high': int((1 - y).sum()),
        'cv_f1_mean': round(np.mean(f1_scores), 3),
        'cv_f1_std': round(np.std(f1_scores), 3)
    },
    'input_ranges': {
        'fandom_size_num': {'min': 2, 'max': 4, 'labels': {'2': 'Medium', '3': 'Large', '4': 'Mega'}},
        'agency_tier': {'min': 1, 'max': 3, 'labels': {'1': 'Small/Indie', '2': 'Mid-tier', '3': 'Big3/Big4 (SM, YG, JYP, HYBE)'}},
        'company_response': {'min': 0, 'max': 2, 'labels': {'0': 'Silence', '1': 'Partial/vague', '2': 'Strong defense'}},
        'response_delay_days': {'min': 0, 'max': 30},
        'apology': {'options': [0, 1]},
        'international': {'options': [0, 1]},
        'is_solo': {'options': [0, 1]},
        'prior_scandal': {'options': [0, 1]},
        'reaction_spike': {'min': -6.0, 'max': 14.0}
    }
}

with open('pipeline_config.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print(f"Saved pipeline_config.json ({os.path.getsize('pipeline_config.json') / 1024:.1f} KB)")

# === SANITY CHECK ===
print("\n=== Sanity Check ===")
test_cases = [
    ('criminal, small fan, high spike', {'scandal_type': 'criminal', 'fandom_size_num': 2,
     'agency_tier': 1, 'company_response': 0, 'response_delay_days': 7, 'apology': 0,
     'international': 1, 'is_solo': 0, 'prior_scandal': 1, 'reaction_spike': 8.0}),
    ('dating, mega fan, low spike', {'scandal_type': 'dating', 'fandom_size_num': 4,
     'agency_tier': 3, 'company_response': 2, 'response_delay_days': 0, 'apology': 0,
     'international': 0, 'is_solo': 0, 'prior_scandal': 0, 'reaction_spike': -1.0}),
]
for desc, tc in test_cases:
    row = pd.DataFrame([tc])
    row_encoded = pd.get_dummies(row, columns=['scandal_type'], prefix='type')
    for col in feature_names:
        if col not in row_encoded.columns:
            row_encoded[col] = 0
    row_encoded = row_encoded[feature_names]
    prob = clf.predict_proba(row_encoded)[0][1]
    print(f"  {desc}: P(high)={prob:.3f} → {'HIGH CRISIS' if prob > 0.5 else 'Manageable'}")

print("\nDone. Files ready for deployment.")
