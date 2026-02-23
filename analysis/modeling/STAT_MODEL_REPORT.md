# Statistical Model Retrain Report

Method: class-weighted logistic regression (pure Python), stratified 5-fold CV for L2 selection.

Selected features:
- building_age
- log1p(property_value)
- log1p(square_feet)
- is_owner_occupied
- is_city_owned
- is_government_owned

CV regularization search (mean AUC, mean Brier):
- l2=0.0: auc=0.9697, brier=0.0338
- l2=0.0001: auc=0.9698, brier=0.0338
- l2=0.0005: auc=0.9699, brier=0.0337
- l2=0.001: auc=0.9701, brier=0.0336
- l2=0.005: auc=0.9709, brier=0.0331
- l2=0.01: auc=0.9713, brier=0.0330

Selected l2: 0.01
Selected ensemble blend alpha (ML weight): 0.90

Full-data diagnostic metrics (for reference):
- ML AUC: 0.9731
- ML Brier: 0.0326
- Ensemble AUC: 0.9726
- Ensemble Brier: 0.0317

Score distributions:
- ml_score min/max/mean: 2.102/99.654/11.302
- ensemble_score min/max/mean: 1.988/95.614/11.567
