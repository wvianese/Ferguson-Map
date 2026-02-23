# Statistical Model Retrain Report

Method: class-weighted L2 logistic regression, Newton-Raphson optimization, stratified 4-fold CV for regularization.

Rows: 8877, positives (abandoned): 138, negatives: 8739

Feature screening diagnostics (corr with is_abandoned):
- is_city_owned: +0.6861
- is_government_owned: +0.5927
- log_improvement_value: -0.3446
- log_square_feet: -0.2762
- log_property_value: -0.2372
- log_value_per_sqft: +0.1733
- is_owner_occupied: -0.1052
- log_land_value: -0.1014
- building_age: +0.0007

Selected model features:
- is_city_owned (corr=+0.6861)
- log_improvement_value (corr=-0.3446)

Forward feature selection path (CV-gain based):
- step 1: add is_city_owned, auc=0.9293, brier=0.0257, l2=1.0, n_features=1
- step 2: add log_improvement_value, auc=0.9723, brier=0.0314, l2=1.0, n_features=2

Model coefficients (standardized feature scale):
- intercept: -2.297947
- is_city_owned: +0.839431 (odds ratio per +1 SD = 2.315)
- log_improvement_value: -0.722102 (odds ratio per +1 SD = 0.486)

CV regularization search (mean AUC, mean Brier):
- l2=0.0: auc=0.9723, brier=0.0314
- l2=0.0001: auc=0.9723, brier=0.0314
- l2=0.001: auc=0.9723, brier=0.0314
- l2=0.01: auc=0.9723, brier=0.0314
- l2=0.05: auc=0.9723, brier=0.0314
- l2=0.1: auc=0.9723, brier=0.0314
- l2=0.5: auc=0.9723, brier=0.0314
- l2=1.0: auc=0.9723, brier=0.0314

Selected l2: 1.0
OOF blend search (AUC for ML-vs-CV blend alpha):
- alpha=0.00: auc=0.8230
- alpha=0.05: auc=0.8958
- alpha=0.10: auc=0.9112
- alpha=0.15: auc=0.9210
- alpha=0.20: auc=0.9263
- alpha=0.25: auc=0.9323
- alpha=0.30: auc=0.9433
- alpha=0.35: auc=0.9519
- alpha=0.40: auc=0.9637
- alpha=0.45: auc=0.9654
- alpha=0.50: auc=0.9669
- alpha=0.55: auc=0.9694
- alpha=0.60: auc=0.9720
- alpha=0.65: auc=0.9721
- alpha=0.70: auc=0.9722
- alpha=0.75: auc=0.9721
- alpha=0.80: auc=0.9721
- alpha=0.85: auc=0.9721
- alpha=0.90: auc=0.9721
- alpha=0.95: auc=0.9716
- alpha=1.00: auc=0.9637
Selected blend alpha (ML weight): 0.70

Full-data diagnostic metrics (reference only, not holdout):
- ML AUC: 0.9721
- ML Brier: 0.0313
- Ensemble AUC: 0.9751
- Ensemble Brier: 0.0316

Score distributions:
- ml_score min/max/mean: 1.607/99.637/11.568
- ensemble_score min/max/mean: 1.125/87.741/12.283
