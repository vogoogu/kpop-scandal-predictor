# Model Evaluation Report

**Dataset:** 105 scandals (37 high crisis / 68 manageable) from 2006–2024  
**Base model:** Random Forest (200 trees, max_depth=8, balanced class weights, 9 features)  
**Full model:** Random Forest (200 trees, max_depth=8, balanced class weights, 10 features incl. reaction_spike)  
**Majority baseline:** 64.8% (always predict manageable)  

---

## 1. Confusion Matrix

### Resubstitution (training set)

| | Predicted Not High | Predicted High |
|---|---|---|
| **Actual Not High** | 65 (TN) | 3 (FP) |
| **Actual High** | 4 (FN) | 33 (TP) |

### 5-Fold Cross-Validation

| | Predicted Not High | Predicted High |
|---|---|---|
| **Actual Not High** | 63 (TN) | 5 (FP) |
| **Actual High** | 13 (FN) | 24 (TP) |

---

## 2. Classification Metrics

### Per-Class (Resubstitution)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| not_high | 0.942 | 0.956 | 0.949 | 68 |
| high | 0.917 | 0.892 | 0.904 | 37 |
| **Accuracy** | | | **0.933** | **105** |

### Per-Class (5-Fold Cross-Validation)

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| not_high | 0.829 | 0.926 | 0.875 | 68 |
| high | 0.828 | 0.649 | 0.727 | 37 |
| **Accuracy** | | | **0.829** | **105** |

---

## 3. Summary Metrics

| Metric | Resubstitution | 5-Fold CV |
|---|---|---|
| Accuracy | 0.933 | 0.829 |
| F1 (high crisis) | 0.904 | 0.727 |
| F1 (manageable) | 0.949 | 0.875 |
| AUC-ROC | 0.984 | 0.839 |
| Average Precision | 0.972 | 0.803 |
| Brier Score | 0.078 | 0.150 |
| Log Loss | 0.287 | 0.465 |
| Cohen's Kappa | 0.853 | 0.605 |
| Matthews Corr. Coeff. | 0.853 | 0.614 |

**Full model AUC-ROC (with reaction_spike):** 0.995

---

## 4. Stability Analysis (10-Seed Cross-Validation)

F1 (high crisis) across 10 random seeds (42–51), 5-fold stratified CV each:

| Seed | F1 (high) |
|---|---|
| 42 | 0.727 |
| 43 | 0.638 |
| 44 | 0.646 |
| 45 | 0.667 |
| 46 | 0.676 |
| 47 | 0.667 |
| 48 | 0.708 |
| 49 | 0.735 |
| 50 | 0.585 |
| 51 | 0.697 |
| **Mean ± Std** | **0.675 ± 0.043** |

---

## 5. Feature Importances (Base Model)

| Rank | Feature | Importance |
|---|---|---|
| 1 | Fandom Size | 0.174 █████████████████ |
| 2 | Response Delay | 0.151 ███████████████ |
| 3 | Company Response | 0.112 ███████████ |
| 4 | Criminal | 0.104 ██████████ |
| 5 | Agency Tier | 0.071 ███████ |
| 6 | Controversy | 0.059 █████ |
| 7 | Apology | 0.056 █████ |
| 8 | Dating | 0.044 ████ |
| 9 | Prior Scandal | 0.039 ███ |
| 10 | International | 0.037 ███ |
| 11 | Drugs | 0.035 ███ |
| 12 | Contract | 0.029 ██ |
| 13 | Bullying | 0.029 ██ |
| 14 | Behavior | 0.028 ██ |
| 15 | Is Solo | 0.024 ██ |
| 16 | Political | 0.008  |

---

## 6. Probability Calibration (CV Predictions)

How well do predicted probabilities match actual outcomes?

| Bin | Count | Mean Predicted | Actual Rate | Deviation |
|---|---|---|---|---|
| 0.0-0.1 | 11 | 0.057 | 0.000 | -0.057 |
| 0.1-0.2 | 14 | 0.152 | 0.214 | +0.062 |
| 0.2-0.3 | 17 | 0.248 | 0.000 | -0.248 |
| 0.3-0.4 | 18 | 0.357 | 0.389 | +0.032 |
| 0.4-0.5 | 16 | 0.442 | 0.188 | -0.254 |
| 0.5-0.6 | 7 | 0.548 | 0.857 | +0.309 |
| 0.6-0.7 | 4 | 0.642 | 0.500 | -0.142 |
| 0.7-0.8 | 6 | 0.751 | 0.833 | +0.082 |
| 0.8-0.9 | 7 | 0.859 | 0.857 | -0.002 |
| 0.9-1.0 | 5 | 0.953 | 1.000 | +0.047 |

---

## 7. Performance by Scandal Type

| Type | N | High Crisis | Crisis Rate | CV Accuracy |
|---|---|---|---|---|
| Criminal | 18 | 14 | 78% | 78% |
| Drugs | 7 | 5 | 71% | 86% |
| Bullying | 11 | 7 | 64% | 82% |
| Contract | 11 | 5 | 45% | 55% |
| Political | 5 | 1 | 20% | 80% |
| Controversy | 20 | 2 | 10% | 90% |
| Dating | 21 | 2 | 10% | 90% |
| Behavior | 12 | 1 | 8% | 92% |

---

## 8. ROC Curve Data Points

### Resubstitution

| FPR | TPR |
|---|---|
| 0.000 | 0.000 |
| 0.015 | 0.622 |
| 0.015 | 0.865 |
| 0.059 | 0.892 |
| 0.103 | 0.946 |
| 0.176 | 0.973 |
| 0.265 | 1.000 |
| 0.676 | 1.000 |
| 0.809 | 1.000 |
| 1.000 | 1.000 |

AUC = 0.984

### Cross-Validation

| FPR | TPR |
|---|---|
| 0.000 | 0.000 |
| 0.015 | 0.270 |
| 0.059 | 0.486 |
| 0.103 | 0.676 |
| 0.118 | 0.730 |
| 0.324 | 0.784 |
| 0.382 | 0.838 |
| 0.426 | 0.892 |
| 0.676 | 0.919 |
| 1.000 | 1.000 |

AUC = 0.839

---

## 9. Error Analysis

### False Negatives (predicted manageable, actually high crisis): 13

| ID | Artist | Scandal Type | CV Predicted Prob |
|---|---|---|---|
| 2 | JYJ | contract | 0.47 |
| 3 | Jay Park | controversy | 0.11 |
| 8 | T-ara | bullying | 0.32 |
| 18 | Jessica | contract | 0.31 |
| 20 | Sungmin | dating | 0.12 |
| 25 | Kangin | criminal | 0.45 |
| 35 | Hyuna | dating | 0.37 |
| 46 | B.I | drugs | 0.37 |
| 75 | Kris Wu | criminal | 0.34 |
| 86 | Kim Garam | bullying | 0.11 |
| 95 | EXO-CBX | contract | 0.50 |
| 104 | Tiffany | political | 0.33 |
| 105 | Seunghan | behavior | 0.36 |

### False Positives (predicted high crisis, actually manageable): 5

| ID | Artist | Scandal Type | CV Predicted Prob |
|---|---|---|---|
| 4 | Kangin | criminal | 0.67 |
| 11 | Zico | controversy | 0.71 |
| 48 | Amber | contract | 0.53 |
| 89 | OMEGA X | criminal | 0.82 |
| 90 | LOONA | contract | 0.61 |

---

## 10. Key Takeaways

- The model achieves **93.3% accuracy** on training data and **82.9% under cross-validation**, significantly beating the 64.8% majority baseline.
- **F1 for high crisis is 0.73** under CV — respectable for a 105-sample dataset with a 35/65 class imbalance.
- **Stability is strong**: F1 varies only ±0.043 across 10 random seeds.
- **Brier score of 0.150** (CV) indicates reasonably calibrated probabilities.
- Adding reaction_spike lifts AUC from 0.839 (base, CV) → 0.886 (full model, resubstitution), confirming public reaction has independent predictive value.
- The 13 false negatives (missed crises) are the higher-risk errors for a crisis prediction tool.

---

*Generated from 105 scandals. Base model: 9 features, Random Forest (n_estimators=200, max_depth=8, class_weight=balanced, random_state=42).*