"""
K-Pop Scandal Impact Predictor — Model Training Script
Produces TWO models:
  - model_base.joblib     → 9 features (no reaction_spike) for the main prediction
  - model_full.joblib     → 10 features (with reaction_spike) for sensitivity analysis
  - pipeline_config.json  → metadata for both models
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

y = (df['label_binary'] == 'high').astype(int)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print(f"Dataset: {len(df)} rows, {y.sum()} high / {(1-y).sum()} not_high")
print(f"Majority baseline accuracy: {1 - y.mean():.3f}\n")

# ─────────────────────────────────────────────────────────────
# MODEL 1: BASE (9 features — no reaction_spike)
# This is the main prediction model. Uses only features known at scandal time.
# ─────────────────────────────────────────────────────────────
base_feature_cols = [
    'scandal_type', 'fandom_size_num', 'agency_tier', 'company_response',
    'response_delay_days', 'apology', 'international', 'is_solo', 'prior_scandal'
]

X_base = pd.get_dummies(df[base_feature_cols], columns=['scandal_type'], prefix='type')
base_feature_names = list(X_base.columns)

clf_base = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', max_depth=8,
    min_samples_leaf=2, random_state=42
)

y_pred_base = cross_val_predict(clf_base, X_base, y, cv=cv)
y_prob_base = cross_val_predict(clf_base, X_base, y, cv=cv, method='predict_proba')[:, 1]

acc_base = accuracy_score(y, y_pred_base)
f1_base = f1_score(y, y_pred_base)
auc_base = roc_auc_score(y, y_prob_base)

print("=" * 60)
print("MODEL 1: BASE (9 features — no reaction_spike)")
print("=" * 60)
print(classification_report(y, y_pred_base, target_names=['not_high', 'high']))
print(f"AUC: {auc_base:.3f}\n")

f1_scores_base = []
for seed in range(10):
    cv_s = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    clf_s = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                                   max_depth=8, min_samples_leaf=2, random_state=seed)
    yp = cross_val_predict(clf_s, X_base, y, cv=cv_s)
    f1_scores_base.append(f1_score(y, yp))
print(f"Stability (10 seeds): F1 mean={np.mean(f1_scores_base):.3f} ± {np.std(f1_scores_base):.3f}")

clf_base.fit(X_base, y)
base_importances = dict(zip(base_feature_names, clf_base.feature_importances_.tolist()))

print("\nFeature Importance (base):")
for feat, imp in sorted(base_importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25s} {imp:.3f} {bar}")

joblib.dump(clf_base, 'model_base.joblib')
print(f"\nSaved model_base.joblib ({os.path.getsize('model_base.joblib') / 1024:.0f} KB)")

# ─────────────────────────────────────────────────────────────
# MODEL 2: FULL (10 features — with reaction_spike)
# Used for sensitivity analysis only.
# ─────────────────────────────────────────────────────────────
full_feature_cols = base_feature_cols + ['reaction_spike']

X_full = pd.get_dummies(df[full_feature_cols], columns=['scandal_type'], prefix='type')
full_feature_names = list(X_full.columns)

clf_full = RandomForestClassifier(
    n_estimators=200, class_weight='balanced', max_depth=8,
    min_samples_leaf=2, random_state=42
)

y_pred_full = cross_val_predict(clf_full, X_full, y, cv=cv)
y_prob_full = cross_val_predict(clf_full, X_full, y, cv=cv, method='predict_proba')[:, 1]

acc_full = accuracy_score(y, y_pred_full)
f1_full = f1_score(y, y_pred_full)
auc_full = roc_auc_score(y, y_prob_full)

print("\n" + "=" * 60)
print("MODEL 2: FULL (10 features — with reaction_spike)")
print("=" * 60)
print(classification_report(y, y_pred_full, target_names=['not_high', 'high']))
print(f"AUC: {auc_full:.3f}\n")

clf_full.fit(X_full, y)
full_importances = dict(zip(full_feature_names, clf_full.feature_importances_.tolist()))

print("Feature Importance (full):")
for feat, imp in sorted(full_importances.items(), key=lambda x: -x[1]):
    bar = "█" * int(imp * 50)
    print(f"  {feat:<25s} {imp:.3f} {bar}")

joblib.dump(clf_full, 'model_full.joblib')
print(f"\nSaved model_full.joblib ({os.path.getsize('model_full.joblib') / 1024:.0f} KB)")

# ─────────────────────────────────────────────────────────────
# PIPELINE CONFIG
# ─────────────────────────────────────────────────────────────
pipeline_config = {
    'base_feature_columns': base_feature_names,
    'full_feature_columns': full_feature_names,
    'scandal_types': sorted(df['scandal_type'].unique().tolist()),
    'base_feature_importances': base_importances,
    'full_feature_importances': full_importances,
    'base_model_metrics': {
        'accuracy': round(acc_base, 3),
        'f1_high': round(f1_base, 3),
        'auc': round(auc_base, 3),
        'majority_baseline': round(1 - y.mean(), 3),
        'n_samples': len(df),
        'n_high': int(y.sum()),
        'n_not_high': int((1 - y).sum()),
        'cv_f1_mean': round(np.mean(f1_scores_base), 3),
        'cv_f1_std': round(np.std(f1_scores_base), 3)
    },
    'full_model_metrics': {
        'accuracy': round(acc_full, 3),
        'f1_high': round(f1_full, 3),
        'auc': round(auc_full, 3),
    },
    'reaction_spike_stats': {
        'min': round(float(df['reaction_spike'].min()), 1),
        'max': round(float(df['reaction_spike'].max()), 1),
        'mean': round(float(df['reaction_spike'].mean()), 2),
        'std': round(float(df['reaction_spike'].std()), 2),
        'median': round(float(df['reaction_spike'].median()), 2),
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
    }
}

with open('pipeline_config.json', 'w') as f:
    json.dump(pipeline_config, f, indent=2)

print(f"\nSaved pipeline_config.json ({os.path.getsize('pipeline_config.json') / 1024:.1f} KB)")
print("\nDone. Files ready for deployment.")
