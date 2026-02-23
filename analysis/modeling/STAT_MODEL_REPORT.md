# Statistical Model Retrain Report

Method: class-weighted L2 logistic regression, Newton-Raphson optimization, stratified 4-fold CV for regularization.

Rows: 8877, positives (abandoned): 138, negatives: 8739

Feature screening diagnostics (corr with is_abandoned):
- log_improvement_value: -0.3446
- log_square_feet: -0.2762
- log_property_value: -0.2372
- log_value_per_sqft: +0.1733
- is_owner_occupied: -0.1052
- log_land_value: -0.1014
- building_age: +0.0007

Selected model features:
- selected_set: baseline_core
- building_age
- log_property_value
- log_square_feet
- is_owner_occupied

Feature-set CV comparison (selected by highest AUC, tie-break lower Brier):
- baseline_core: auc=0.9250, brier=0.0958, l2=1.0, n_features=4
- add_value_per_sqft: auc=0.9250, brier=0.0958, l2=1.0, n_features=5
- economic_full: auc=0.9241, brier=0.0895, l2=0.001, n_features=7
- add_improvement: auc=0.9238, brier=0.0910, l2=1.0, n_features=5
- add_land: auc=0.9190, brier=0.0938, l2=1.0, n_features=5

Model coefficients (standardized feature scale):
- intercept: -1.578096
- log_property_value: -0.832473 (odds ratio per +1 SD = 0.435)
- is_owner_occupied: -0.654910 (odds ratio per +1 SD = 0.519)
- log_square_feet: -0.647065 (odds ratio per +1 SD = 0.524)
- building_age: +0.503916 (odds ratio per +1 SD = 1.655)

CV regularization search (mean AUC, mean Brier):
- l2=0.0: auc=0.9250, brier=0.0958
- l2=0.0001: auc=0.9250, brier=0.0958
- l2=0.001: auc=0.9250, brier=0.0958
- l2=0.01: auc=0.9250, brier=0.0958
- l2=0.05: auc=0.9250, brier=0.0958
- l2=0.1: auc=0.9250, brier=0.0958
- l2=0.5: auc=0.9250, brier=0.0958
- l2=1.0: auc=0.9250, brier=0.0958

Selected l2: 1.0
OOF blend search (AUC for ML-vs-CV blend alpha):
- alpha=0.00: auc=0.8230
- alpha=0.05: auc=0.8808
- alpha=0.10: auc=0.8904
- alpha=0.15: auc=0.8961
- alpha=0.20: auc=0.8994
- alpha=0.25: auc=0.9026
- alpha=0.30: auc=0.9089
- alpha=0.35: auc=0.9171
- alpha=0.40: auc=0.9262
- alpha=0.45: auc=0.9320
- alpha=0.50: auc=0.9345
- alpha=0.55: auc=0.9352
- alpha=0.60: auc=0.9358
- alpha=0.65: auc=0.9358
- alpha=0.70: auc=0.9351
- alpha=0.75: auc=0.9336
- alpha=0.80: auc=0.9321
- alpha=0.85: auc=0.9304
- alpha=0.90: auc=0.9280
- alpha=0.95: auc=0.9244
- alpha=1.00: auc=0.9207
Selected blend alpha (ML weight): 0.65

Full-data diagnostic metrics (reference only, not holdout):
- ML AUC: 0.9262
- ML Brier: 0.0954
- Ensemble AUC: 0.9387
- Ensemble Brier: 0.0696

Score distributions:
- ml_score min/max/mean: 0.347/99.987/22.109
- ensemble_score min/max/mean: 0.226/85.681/19.254
