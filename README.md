# Ferguson Housing Abandonment Analysis Map

Interactive map and modeling workflow for identifying property abandonment risk in Ferguson, MO (ZIP 63135), combining parcel-level statistical modeling with Street View image analysis.

## 1) Project Goal

This project answers one core question:

- Which Ferguson properties are most likely abandoned or at high risk of abandonment, and where should interventions be prioritized?

It does this by combining:

- Structured parcel data (value, age, size, occupancy)
- AI image analysis from Google Street View (occupied vs abandoned vs vacant lot)
- A combined score surfaced in an interactive Leaflet map

## 2) Tech Stack (Languages + Packages)

### Languages

- Python (data prep, modeling, scoring refresh)
- HTML/CSS/JavaScript (interactive web map UI)
- JSON/CSV (data interchange + layer payloads)

### Frontend / Map

- [Leaflet 1.9.3](https://leafletjs.com/) (loaded via CDN in `ferguson_map.html`)

### Data Science / CV (used in notebook pipeline)

From `analysis/notebooks/streetview.ipynb`:

- `pandas`
- `geopandas`
- `requests`
- `Pillow`
- `torch`
- `clip` (OpenAI CLIP package)

### Current scoring refresh script

From `analysis/modeling/retrain_statistical_model.py`:

- Python standard library only (`csv`, `json`, `math`, `random`, `re`, etc.)
- Implements logistic regression and metrics manually, using the same statistical principles you would use with `sklearn`.

## 3) Repository Structure

- `ferguson_map.html`: Main interactive map app (UI + layer loading + search + popups)
- `index.html`: Landing page entry
- `ferguson_complete_data.csv`: Master scored dataset used by map and analysis
- `map_data/*.json`: Per-layer GeoJSON-like feature arrays (`ensemble`, `ml`, `cv`, `risk`, etc.)
- `analysis/notebooks/streetview.ipynb`: Original data + Street View + CLIP workflow
- `analysis/modeling/retrain_statistical_model.py`: Current defensible statistical retraining script
- `analysis/modeling/STAT_MODEL_REPORT.md`: Latest model diagnostics (features, coefficients, CV metrics)
- `INSIGHTS_REPORT.md`: Narrative findings and policy recommendations

## 4) Data Sources

As documented in app/help and notebook:

- St. Louis County parcel/open GIS data (parcel attributes)
- Ferguson Property Restoration Program list (confirmed abandoned labels)
- Google Street View Static API imagery

## 5) End-to-End Methodology

## 5.1 Labeling strategy

- Ground-truth target: `is_abandoned`
- Derived by joining official Ferguson restoration list to parcel records by parcel ID

## 5.2 Statistical model (current production refresh)

Implemented in `analysis/modeling/retrain_statistical_model.py`.

### Selected features

Current model intentionally uses pre-outcome, interpretable property factors:

- `building_age`
- `log_property_value`
- `log_square_feet`
- `is_owner_occupied`

Why these:

- They are conceptually causal/plausible for abandonment risk
- They align with correlation findings in this dataset
- They avoid leakage-like governance-status shortcuts

### Additional feature testing (what we tried and why we kept current features)

We tested additional non-leaky candidates to see if they improve statistical performance:

- `log_improvement_value`
- `log_land_value`
- `log_value_per_sqft`

Candidate feature sets were compared with stratified CV using the same class-weighted L2 logistic setup.  
Selection rule: highest CV AUC, tie-break by lower CV Brier.

Result in current data snapshot:

- `baseline_core` (current): AUC `0.9250`
- `add_value_per_sqft`: AUC `0.9250` (tie, no gain)
- `economic_full`: AUC `0.9241`
- `add_improvement`: AUC `0.9238`
- `add_land`: AUC `0.9190`

Decision:

- Keep `baseline_core` because no added non-leaky feature set improved AUC.
- This is documented in `analysis/modeling/STAT_MODEL_REPORT.md`.

Important note on code violations:

- `CODE_ENFOR` (code violations) exists in the older notebook pipeline but is **not present** in the current production CSV (`ferguson_complete_data.csv`), so it could not be included in this retrain.
- If you re-introduce that field into production data, we can rerun the same CV feature-set test and include it formally.

### Preprocessing

- Numeric coercion with median imputation for missing values
- Log transforms for skewed monetary/size variables (`log1p`)
- Standardization (z-score) before regression

### Model form

Class-weighted L2-regularized logistic regression:

- `P(abandoned=1 | x) = sigmoid(b0 + b'x)`
- Positive-class weighting handles strong class imbalance (138 positives vs 8,739 negatives)
- L2 regularization controls coefficient variance

### Optimization

- Newton-Raphson / IRLS-style updates
- Solves the Hessian system each iteration via Gaussian elimination

### Hyperparameter selection

- Stratified 4-fold cross-validation
- Grid over L2 values: `0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0`
- Selection rule favors robust regularization within small AUC tolerance of best score

### Metrics

- AUC (rank-discrimination ability)
- Brier score (probability calibration / squared error)

Latest diagnostics are in `analysis/modeling/STAT_MODEL_REPORT.md`.

## 5.3 AI image analysis (Street View + CLIP)

Implemented in `analysis/notebooks/streetview.ipynb`.

### Image acquisition

- Uses Google Street View Static API per address
- Saves images and tracks which parcels have imagery

### CLIP classification setup

- CLIP model: `ViT-B/32`
- 3 semantic prompts/classes:
  - Occupied structure
  - Abandoned/deteriorated structure
  - Vacant lot
- Similarity softmax gives class probabilities:
  - `prob_occupied`
  - `prob_abandoned`
  - `prob_vacant_lot`

### CV score logic

- If primary class = occupied: `cv_score = 0`
- If primary class = abandoned: `cv_score = prob_abandoned`
- If primary class = vacant_lot: `cv_score = 0.6 * prob_vacant_lot`

Those class probabilities are shown in the popup confidence subsection.

## 5.4 Combined score and risk score

### Combined score (`ensemble_score`)

Current production refresh chooses blend weight `alpha` from out-of-fold AUC search:

- `ensemble_prob = alpha * ml_prob + (1 - alpha) * cv_prob`
- `ensemble_score = 100 * ensemble_prob`

If no imagery exists, CV is missing and ensemble falls back to ML.

### Property risk score (`property_risk_score`)

Current map risk aggregation:

- `risk = 0.40 * ensemble + 0.20 * age_risk + 0.25 * value_risk + 0.15 * occupancy_risk`
- Clamped to `[0, 100]`
- Categorized to `Low / Medium / High / Critical`

## 6) Map Architecture and UX

From `ferguson_map.html`:

- Loads `ensemble` layer first for fastest perceived startup
- Streams remaining layers in background (`fetch('map_data/<layer>.json')`)
- Uses `cache: 'force-cache'` for layer fetches
- Search can operate even before all layers are loaded (fallback against ensemble index)
- Layer controls are custom (not Leaflet default), with `Layers` open by default
- Popups include:
  - Combined and Statistical scores
  - AI image category and confidence percentages
  - Property details (value/age/size/occupancy/owner)

## 7) How to Run Locally

From repo root:

```bash
cd /path/to/Ferguson-Vacant-Housing-Tool
python3 -m http.server 8000
```

Then open:

- `http://localhost:8000/` (landing page)
- `http://localhost:8000/ferguson_map.html` (map)

## 8) How to Recompute Scores After Data/Model Changes

Run:

```bash
python3 analysis/modeling/retrain_statistical_model.py
```

This updates:

- `ferguson_complete_data.csv`
- `map_data/ml.json`
- `map_data/ensemble.json`
- `map_data/risk.json`
- `analysis/modeling/STAT_MODEL_REPORT.md`

## 9) How to Recreate the Full AI Workflow (Notebook Path)

Open `analysis/notebooks/streetview.ipynb` and execute in order:

1. Install/import packages
2. Load parcels + confirmed list
3. Pull Street View imagery with Google API key
4. Run CLIP classification
5. Compute CV/ensemble outputs and export CSV
6. Regenerate map layers

Minimum keys/dependencies required:

- Google Street View Static API key
- PyTorch + CLIP environment

## 10) Interview Defense: Key Talking Points

If asked why your approach is rigorous:

- You used a clear supervised target from official program records
- You addressed class imbalance with class-weighted logistic regression
- You selected regularization by stratified cross-validation (not by eyeballing)
- You report both discrimination (AUC) and probability quality (Brier)
- You avoided leakage-prone predictors in final scoring
- You treat AI image model as complementary evidence and tune blend weights out-of-fold
- You keep interpretable features and publish coefficients + odds ratios

If asked why probabilities are not uniformly 0-100:

- Real calibrated probabilities reflect data prevalence and separability
- For rare events, most cases should cluster low; forcing artificial spread is statistically misleading

## 10.1) Validation Against Confirmed Abandoned List

To evaluate model classification performance against official labels, we used:

- Ground truth: `is_abandoned` (from Ferguson confirmed restoration list join)
- Positive prediction rule: score `>= 50` on a `0-100` scale
- Statistical model score: `ml_score`
- AI image model score: `cv_score`

### Exact figures (current dataset snapshot)

- Total parcels: `8,877`
- Confirmed abandoned positives: `138`

Statistical model (`ml_score >= 50`):

- Accuracy: `91.13%` (`8090 / 8877`)
- Confusion counts: `TP=111, TN=7979, FP=760, FN=27`
- Recall on confirmed abandoned: `80.43%`

AI image model (`cv_score >= 50`):

- Evaluated on `8,874` parcels with non-missing `cv_score` (3 missing imagery/model outputs)
- Accuracy: `86.89%` (`7711 / 8874`)
- Confusion counts: `TP=89, TN=7622, FP=1114, FN=49`
- Recall on confirmed abandoned: `64.49%`

Notes for rigorous interpretation:

- Dataset is highly imbalanced (138 positives vs 8,739 negatives), so accuracy alone can look high.
- For interviews, discuss accuracy with recall, specificity, and confusion matrix to avoid misleading conclusions.

### How these figures were obtained

We computed them directly from `ferguson_complete_data.csv` by thresholding model scores at `50` and comparing to `is_abandoned`:

```python
import csv

with open("ferguson_complete_data.csv", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

def eval_acc(score_col):
    vals = []
    for r in rows:
        if score_col == "cv_score" and r[score_col] in ("", "nan", "NaN", None):
            continue
        y = 1 if float(r["is_abandoned"]) >= 0.5 else 0
        s = float(r[score_col]) if r[score_col] not in ("", None) else 0.0
        yhat = 1 if s >= 50 else 0
        vals.append((y, yhat))
    acc = sum(1 for y, yhat in vals if y == yhat) / len(vals)
    return acc
```

## 11) Limitations and Next Improvements

- CLIP prompt-based classification is zero-shot; supervised local fine-tuning could improve calibration
- External validation on a held-out time period would strengthen generalization claims
- Spatial autocorrelation and neighborhood effects could be modeled explicitly
- Formal probability calibration (Platt/isotonic) can be added after holdout design

## 12) Quick Reproduction Checklist

- Get parcel + confirmed label data
- Build `is_abandoned` label
- Engineer age/value/size/occupancy features
- Fit class-weighted logistic with CV-selected L2
- Generate Street View imagery and CLIP class probabilities
- Blend ML + CV via out-of-fold alpha search
- Write scores to CSV + layer JSON
- Serve map as static site

---

## Current Project Status (as of February 23, 2026)

- Regression refresh committed and pushed to `main`
- Model/report files reflect leakage-safe feature set and CV-selected regularization
- UI includes confidence subsection in popups and functional address search
