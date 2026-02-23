#!/usr/bin/env python3
"""Retrain abandonment model with statistically defensible methods.

Design goals:
- No percentile forcing or artificial score spreading.
- Use regularized logistic regression probabilities.
- Select regularization with stratified cross-validation.
- Document feature screening based on target-correlation and collinearity.

Outputs:
- /tmp/ferguson_pushfix/ferguson_complete_data.csv
- /tmp/ferguson_pushfix/map_data/{ml,ensemble,risk}.json
- /tmp/ferguson_pushfix/analysis/modeling/STAT_MODEL_REPORT.md
"""

from __future__ import annotations

import csv
import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

CSV_PATH = Path('/tmp/ferguson_pushfix/ferguson_complete_data.csv')
MAP_DIR = Path('/tmp/ferguson_pushfix/map_data')
REPORT_PATH = Path('/tmp/ferguson_pushfix/analysis/modeling/STAT_MODEL_REPORT.md')
SEED = 42


def to_float(x: object) -> float:
    try:
        s = '' if x is None else str(x).strip()
        if s == '' or s.lower() in ('nan', 'none', 'null'):
            return math.nan
        return float(s)
    except Exception:
        return math.nan


def parse_binary(x: object) -> float:
    s = str(x or '').strip().lower()
    return 1.0 if s in ('1', '1.0', 'true', 'yes') else 0.0


def clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def color_score_0_100(score: float) -> str:
    # Match displayed one-decimal score so color buckets align with what users see.
    s = round(score, 1)
    if s < 25:
        return '#27ae60'
    if s < 50:
        return '#f1c40f'
    if s < 75:
        return '#e67e22'
    return '#e74c3c'


def risk_category(score: float) -> str:
    if score < 25:
        return 'Low Risk'
    if score < 50:
        return 'Medium Risk'
    if score < 75:
        return 'High Risk'
    return 'Critical Risk'


def sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def auc_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    pairs = sorted(zip(y_prob, y_true), key=lambda t: t[0])
    n_pos = sum(y_true)
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    rank = 1
    rank_sum_pos = 0.0
    i = 0
    n = len(pairs)
    while i < n:
        j = i + 1
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (rank + rank + (j - i) - 1) / 2.0
        pos_in_tie = sum(1 for _, y in pairs[i:j] if y == 1)
        rank_sum_pos += avg_rank * pos_in_tie
        rank += (j - i)
        i = j

    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_score(y_true: Sequence[int], y_prob: Sequence[float]) -> float:
    n = len(y_true)
    return sum((p - y) ** 2 for y, p in zip(y_true, y_prob)) / n


def corr(x: Sequence[float], y: Sequence[float]) -> float:
    pairs = [(a, b) for a, b in zip(x, y) if not (math.isnan(a) or math.isnan(b))]
    n = len(pairs)
    if n < 3:
        return 0.0
    mx = sum(a for a, _ in pairs) / n
    my = sum(b for _, b in pairs) / n
    sxx = sum((a - mx) ** 2 for a, _ in pairs)
    syy = sum((b - my) ** 2 for _, b in pairs)
    if sxx <= 1e-12 or syy <= 1e-12:
        return 0.0
    sxy = sum((a - mx) * (b - my) for a, b in pairs)
    return sxy / math.sqrt(sxx * syy)


def solve_linear_system(a: List[List[float]], b: List[float]) -> List[float]:
    """Solve Ax=b with Gaussian elimination + partial pivoting."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]

    for col in range(n):
        piv = col
        best = abs(m[col][col])
        for r in range(col + 1, n):
            v = abs(m[r][col])
            if v > best:
                best = v
                piv = r
        if best < 1e-12:
            raise ValueError('Singular matrix')
        if piv != col:
            m[col], m[piv] = m[piv], m[col]

        d = m[col][col]
        for c in range(col, n + 1):
            m[col][c] /= d

        for r in range(n):
            if r == col:
                continue
            factor = m[r][col]
            if abs(factor) < 1e-16:
                continue
            for c in range(col, n + 1):
                m[r][c] -= factor * m[col][c]

    return [m[i][n] for i in range(n)]


@dataclass
class Standardizer:
    means: List[float]
    stds: List[float]

    def transform(self, X: Sequence[Sequence[float]]) -> List[List[float]]:
        out: List[List[float]] = []
        for row in X:
            out.append([(v - m) / s for v, m, s in zip(row, self.means, self.stds)])
        return out


@dataclass
class LogisticModel:
    bias: float
    weights: List[float]
    standardizer: Standardizer

    def predict_proba(self, X_raw: Sequence[Sequence[float]]) -> List[float]:
        X = self.standardizer.transform(X_raw)
        probs: List[float] = []
        for row in X:
            z = self.bias + sum(w * x for w, x in zip(self.weights, row))
            probs.append(sigmoid(z))
        return probs


def fit_standardizer(X: Sequence[Sequence[float]]) -> Standardizer:
    n_features = len(X[0])
    means: List[float] = []
    stds: List[float] = []
    n = len(X)
    for j in range(n_features):
        col = [X[i][j] for i in range(n)]
        m = sum(col) / n
        var = sum((v - m) ** 2 for v in col) / n
        s = math.sqrt(var) if var > 1e-12 else 1.0
        means.append(m)
        stds.append(s)
    return Standardizer(means, stds)


def fit_logistic_regression(
    X_raw: Sequence[Sequence[float]],
    y: Sequence[int],
    l2: float,
    max_iter: int = 40,
    tol: float = 1e-6,
) -> LogisticModel:
    """Class-weighted L2 logistic via Newton-Raphson (IRLS style)."""
    n = len(y)
    m = len(X_raw[0])
    std = fit_standardizer(X_raw)
    Xs = std.transform(X_raw)

    # add intercept column at position 0
    X = [[1.0] + row for row in Xs]
    d = m + 1

    pos = sum(y)
    neg = n - pos
    w_pos = (neg / pos) if pos > 0 else 1.0

    beta = [0.0] * d

    for _ in range(max_iter):
        grad = [0.0] * d
        hess = [[0.0] * d for _ in range(d)]

        for i in range(n):
            xi = X[i]
            yi = y[i]
            eta = sum(beta[j] * xi[j] for j in range(d))
            p = sigmoid(eta)
            obs_w = w_pos if yi == 1 else 1.0

            # gradient contribution: x*(y-p)*w
            e = (yi - p) * obs_w
            for j in range(d):
                grad[j] += xi[j] * e

            # hessian contribution: x x^T p(1-p) * w
            s = p * (1.0 - p) * obs_w
            for j in range(d):
                xj = xi[j]
                for k in range(j, d):
                    h = xj * xi[k] * s
                    hess[j][k] += h
                    if k != j:
                        hess[k][j] += h

        # L2 penalty on non-intercept terms
        for j in range(1, d):
            grad[j] -= 2.0 * l2 * beta[j]
            hess[j][j] += 2.0 * l2

        try:
            # Newton step solves H * delta = grad, then beta <- beta + delta.
            delta = solve_linear_system(hess, grad)
        except ValueError:
            break

        step_norm = 0.0
        for j in range(d):
            beta[j] += delta[j]
            step_norm += delta[j] * delta[j]
        if step_norm < tol:
            break

    return LogisticModel(bias=beta[0], weights=beta[1:], standardizer=std)


def stratified_kfold_indices(y: Sequence[int], k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    pos_idx = [i for i, v in enumerate(y) if v == 1]
    neg_idx = [i for i, v in enumerate(y) if v == 0]
    rng = random.Random(seed)
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    pos_folds = [[] for _ in range(k)]
    neg_folds = [[] for _ in range(k)]
    for i, idx in enumerate(pos_idx):
        pos_folds[i % k].append(idx)
    for i, idx in enumerate(neg_idx):
        neg_folds[i % k].append(idx)

    folds: List[Tuple[List[int], List[int]]] = []
    all_idx = set(range(len(y)))
    for i in range(k):
        test = sorted(pos_folds[i] + neg_folds[i])
        train = sorted(all_idx.difference(test))
        folds.append((train, test))
    return folds


def select_l2_via_cv(X: List[List[float]], y: List[int], l2_grid: Sequence[float], k: int = 4) -> Tuple[float, Dict[float, Tuple[float, float]]]:
    folds = stratified_kfold_indices(y, k=k, seed=SEED)
    summary: Dict[float, Tuple[float, float]] = {}

    for l2 in l2_grid:
        aucs: List[float] = []
        briers: List[float] = []
        for train_idx, test_idx in folds:
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            model = fit_logistic_regression(X_train, y_train, l2=l2)
            p = model.predict_proba(X_test)
            aucs.append(auc_score(y_test, p))
            briers.append(brier_score(y_test, p))

        summary[l2] = (sum(aucs) / len(aucs), sum(briers) / len(briers))

    # Robustness rule: allow tiny AUC drop, then prefer stronger regularization.
    # This avoids fragile near-unregularized coefficients when CV differences are negligible.
    best_auc = max(v[0] for v in summary.values())
    eligible = [(l2, auc, br) for l2, (auc, br) in summary.items() if auc >= best_auc - 0.002]
    best = sorted(eligible, key=lambda t: (-t[0], t[2]))[0][0]
    return best, summary


def choose_blend_alpha_oof(X: List[List[float]], y: List[int], cv_prob: List[float], l2: float, k: int = 4) -> Tuple[float, Dict[float, float]]:
    folds = stratified_kfold_indices(y, k=k, seed=SEED + 7)
    oof_ml = [0.0] * len(y)

    for train_idx, test_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        model = fit_logistic_regression(X_train, y_train, l2=l2)
        p = model.predict_proba(X_test)
        for idx, prob in zip(test_idx, p):
            oof_ml[idx] = prob

    alpha_scores: Dict[float, float] = {}
    # Grid-search blend from pure CV (0.0) to pure ML (1.0) using OOF predictions.
    for step in range(0, 21):
        a = step / 20.0
        comb = []
        for mlp, cvp in zip(oof_ml, cv_prob):
            if math.isnan(cvp):
                comb.append(mlp)
            else:
                comb.append(a * mlp + (1.0 - a) * cvp)
        alpha_scores[a] = auc_score(y, comb)

    best_alpha = sorted(alpha_scores.items(), key=lambda kv: -kv[1])[0][0]
    return best_alpha, alpha_scores


def extract_address_from_popup(popup_html: str) -> str:
    m = re.search(r"<div style='font-weight:600;font-size:15px;margin-bottom:6px'>(.*?)</div>", popup_html)
    return m.group(1).strip().upper() if m else ''


def update_map_layers(addr_index: Dict[str, Dict[str, float]]) -> None:
    combined_re = re.compile(r"(<tr><td><b>Combined Score:</b></td><td style='text-align:right'><b style='color:)(#[0-9a-fA-F]{6})('>)([0-9]+(?:\.[0-9]+)?)(</b>/100</td></tr>)")
    ml_re = re.compile(r"(<tr><td>Statistical Model:</td><td style='text-align:right'>)([0-9]+(?:\.[0-9]+)?)(/100</td></tr>)")
    tooltip_color_re = re.compile(r"(background:)(#[0-9a-fA-F]{6})")

    for layer_name in ['ensemble', 'ml', 'risk']:
        fp = MAP_DIR / f'{layer_name}.json'
        features = json.loads(fp.read_text(encoding='utf-8'))
        changed = 0

        for feat in features:
            props = feat.get('properties', {})
            popup = props.get('p', '')
            addr = extract_address_from_popup(popup)
            rec = addr_index.get(addr)
            if not rec:
                continue

            layer_color = rec['ensemble_color'] if layer_name == 'ensemble' else rec['ml_color'] if layer_name == 'ml' else rec['risk_color']
            props['c'] = layer_color

            popup = combined_re.sub(
                lambda mm: f"{mm.group(1)}{rec['ensemble_color']}{mm.group(3)}{rec['ensemble']:.1f}{mm.group(5)}",
                popup,
                count=1,
            )
            popup = ml_re.sub(
                lambda mm: f"{mm.group(1)}{rec['ml']:.1f}{mm.group(3)}",
                popup,
                count=1,
            )
            props['p'] = popup

            if 't' in props and isinstance(props['t'], str):
                props['t'] = tooltip_color_re.sub(lambda mm: f"{mm.group(1)}{layer_color}", props['t'], count=1)

            changed += 1

        fp.write_text(json.dumps(features, separators=(',', ':')), encoding='utf-8')
        print(f'{layer_name}: updated {changed} features')


def build_feature_matrix(rows: List[Dict[str, str]]) -> Tuple[Dict[str, List[float]], List[int], List[float], Dict[str, float], Dict[str, float]]:
    cont_cols = ['building_age', 'property_value', 'improvement_value', 'land_value', 'square_feet', 'value_per_sqft']
    med: Dict[str, float] = {}

    for c in cont_cols:
        vals = [to_float(r.get(c)) for r in rows]
        vals = [v for v in vals if not math.isnan(v)]
        vals.sort()
        # Median imputation is robust to heavy right-skew in tax/value fields.
        med[c] = vals[len(vals) // 2] if vals else 0.0

    y: List[int] = [1 if parse_binary(r.get('is_abandoned')) == 1.0 else 0 for r in rows]

    # Restrict to pre-outcome property characteristics to avoid leakage.
    # These are interpretable and align with inspection-time parcel attributes.
    candidates: Dict[str, List[float]] = {
        'building_age': [],
        'log_property_value': [],
        'log_improvement_value': [],
        'log_land_value': [],
        'log_square_feet': [],
        'log_value_per_sqft': [],
        'is_owner_occupied': [],
        'is_city_owned': [],
        'is_government_owned': [],
    }

    for r in rows:
        age = to_float(r.get('building_age'))
        pv = to_float(r.get('property_value'))
        iv = to_float(r.get('improvement_value'))
        lv = to_float(r.get('land_value'))
        sf = to_float(r.get('square_feet'))
        vpsf = to_float(r.get('value_per_sqft'))

        if math.isnan(age):
            age = med['building_age']
        if math.isnan(pv):
            pv = med['property_value']
        if math.isnan(iv):
            iv = med['improvement_value']
        if math.isnan(lv):
            lv = med['land_value']
        if math.isnan(sf):
            sf = med['square_feet']
        if math.isnan(vpsf):
            vpsf = med['value_per_sqft']

        candidates['building_age'].append(age)
        candidates['log_property_value'].append(math.log1p(max(pv, 0.0)))
        candidates['log_improvement_value'].append(math.log1p(max(iv, 0.0)))
        candidates['log_land_value'].append(math.log1p(max(lv, 0.0)))
        candidates['log_square_feet'].append(math.log1p(max(sf, 0.0)))
        candidates['log_value_per_sqft'].append(math.log1p(max(vpsf, 0.0)))
        candidates['is_owner_occupied'].append(parse_binary(r.get('is_owner_occupied')))
        candidates['is_city_owned'].append(parse_binary(r.get('is_city_owned')))
        candidates['is_government_owned'].append(parse_binary(r.get('is_government_owned')))

    y_float = [float(v) for v in y]
    target_corr = {name: corr(vals, y_float) for name, vals in candidates.items()}

    selected = ['building_age', 'log_property_value', 'log_square_feet', 'is_owner_occupied']
    inter_corr_kept: Dict[str, float] = {}
    for i, a in enumerate(selected):
        for b in selected[i + 1:]:
            inter_corr_kept[f'{a}~{b}'] = corr(candidates[a], candidates[b])

    cv_prob: List[float] = []
    for r in rows:
        cv = to_float(r.get('cv_score'))
        cv_prob.append(math.nan if math.isnan(cv) else cv / 100.0)

    return candidates, y, cv_prob, target_corr, inter_corr_kept


def forward_select_features(
    candidates: Dict[str, List[float]],
    y: List[int],
    l2_grid: Sequence[float],
    target_corr: Dict[str, float],
) -> Tuple[List[str], float, Dict[float, Tuple[float, float]], List[Dict[str, float]]]:
    """Greedy CV feature selection with collinearity guard and minimum gain threshold."""
    available = sorted(candidates.keys(), key=lambda n: -abs(target_corr.get(n, 0.0)))
    selected: List[str] = []
    history: List[Dict[str, float]] = []

    best_auc = -1.0
    best_brier = 1e9
    best_l2 = 1.0
    best_summary: Dict[float, Tuple[float, float]] = {}

    min_auc_gain = 0.0010
    max_abs_pair_corr = 0.95

    for step in range(len(available)):
        trial_rows: List[Tuple[str, float, float, float]] = []
        for feat in available:
            if feat in selected:
                continue

            # Skip additions that are nearly duplicates of already-selected predictors.
            too_collinear = False
            for sf in selected:
                if abs(corr(candidates[feat], candidates[sf])) > max_abs_pair_corr:
                    too_collinear = True
                    break
            if too_collinear:
                continue

            feats = selected + [feat]
            X = [[candidates[f][i] for f in feats] for i in range(len(y))]
            l2, summary = select_l2_via_cv(X, y, l2_grid=l2_grid, k=4)
            auc, br = summary[l2]
            trial_rows.append((feat, auc, br, l2))

        if not trial_rows:
            break

        feat, auc, br, l2 = sorted(trial_rows, key=lambda t: (-t[1], t[2]))[0]

        if best_auc < 0:
            selected.append(feat)
            best_auc, best_brier, best_l2 = auc, br, l2
            X = [[candidates[f][i] for f in selected] for i in range(len(y))]
            _, best_summary = select_l2_via_cv(X, y, l2_grid=l2_grid, k=4)
            history.append({'step': float(step + 1), 'added': 0.0, 'auc': auc, 'brier': br, 'l2': l2, 'n_features': float(len(selected))})
            history[-1]['feature_name'] = feat  # type: ignore[index]
            continue

        # Add a feature only if it provides meaningful AUC lift, or equal AUC with lower Brier.
        improves_auc = auc >= (best_auc + min_auc_gain)
        improves_brier = (abs(auc - best_auc) <= 1e-9 and br < best_brier - 0.0005)
        if not (improves_auc or improves_brier):
            break

        selected.append(feat)
        best_auc, best_brier, best_l2 = auc, br, l2
        X = [[candidates[f][i] for f in selected] for i in range(len(y))]
        _, best_summary = select_l2_via_cv(X, y, l2_grid=l2_grid, k=4)
        history.append({'step': float(step + 1), 'added': 0.0, 'auc': auc, 'brier': br, 'l2': l2, 'n_features': float(len(selected))})
        history[-1]['feature_name'] = feat  # type: ignore[index]

    # Safety fallback to a known stable baseline if selection ends empty.
    if not selected:
        selected = ['building_age', 'log_property_value', 'log_square_feet', 'is_owner_occupied']
        X = [[candidates[f][i] for f in selected] for i in range(len(y))]
        best_l2, best_summary = select_l2_via_cv(X, y, l2_grid=l2_grid, k=4)

    return selected, best_l2, best_summary, history


def main() -> None:
    rows: List[Dict[str, str]] = []
    with CSV_PATH.open(newline='', encoding='utf-8') as f:
        r = csv.DictReader(f)
        fieldnames = r.fieldnames or []
        for row in r:
            rows.append(row)

    candidates, y, cv_prob, target_corr, inter_corr = build_feature_matrix(rows)

    # Candidate regularization strengths for CV model selection.
    l2_grid = [0.0, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1.0]
    selected_features, best_l2, cv_summary, selection_history = forward_select_features(candidates, y, l2_grid=l2_grid, target_corr=target_corr)
    X = [[candidates[f][i] for f in selected_features] for i in range(len(y))]
    best_alpha, alpha_scores = choose_blend_alpha_oof(X, y, cv_prob=cv_prob, l2=best_l2, k=4)

    model = fit_logistic_regression(X, y, l2=best_l2)
    ml_prob = model.predict_proba(X)

    # Standardization makes coefficient magnitudes comparable across features.
    coef_by_feature = {f: w for f, w in zip(selected_features, model.weights)}

    addr_index: Dict[str, Dict[str, float]] = {}
    for row, p_ml, p_cv in zip(rows, ml_prob, cv_prob):
        ml = clamp(p_ml * 100.0)
        ens = ml if math.isnan(p_cv) else clamp((best_alpha * p_ml + (1.0 - best_alpha) * p_cv) * 100.0)

        age_r = to_float(row.get('age_risk_score'))
        val_r = to_float(row.get('value_risk_score'))
        occ_r = to_float(row.get('occupancy_risk_score'))
        if math.isnan(age_r):
            age_r = 50.0
        if math.isnan(val_r):
            val_r = 50.0
        if math.isnan(occ_r):
            occ_r = 0.0 if parse_binary(row.get('is_owner_occupied')) == 1.0 else 100.0

        # Preserve the map's existing risk framework while refreshing model inputs.
        prisk = clamp(0.40 * ens + 0.20 * age_r + 0.25 * val_r + 0.15 * occ_r)

        row['ml_score'] = f'{ml:.6f}'
        row['ensemble_score'] = f'{ens:.6f}'
        row['property_risk_score'] = f'{prisk:.6f}'
        row['risk_category'] = risk_category(prisk)

        addr = (row.get('address') or '').strip().upper()
        if addr:
            addr_index[addr] = {
                'ml': ml,
                'ensemble': ens,
                'risk': prisk,
                'ml_color': color_score_0_100(ml),
                'ensemble_color': color_score_0_100(ens),
                'risk_color': color_score_0_100(prisk),
            }

    with CSV_PATH.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    update_map_layers(addr_index)

    ml_vals = [to_float(r['ml_score']) for r in rows]
    ens_vals = [to_float(r['ensemble_score']) for r in rows]
    y_ml = [v / 100.0 for v in ml_vals]
    y_ens = [v / 100.0 for v in ens_vals]

    auc_ml_full = auc_score(y, y_ml)
    auc_ens_full = auc_score(y, y_ens)
    brier_ml_full = brier_score(y, y_ml)
    brier_ens_full = brier_score(y, y_ens)

    lines = []
    lines.append('# Statistical Model Retrain Report')
    lines.append('')
    lines.append('Method: class-weighted L2 logistic regression, Newton-Raphson optimization, stratified 4-fold CV for regularization.')
    lines.append('')
    lines.append(f'Rows: {len(rows)}, positives (abandoned): {sum(y)}, negatives: {len(y)-sum(y)}')
    lines.append('')
    lines.append('Feature screening diagnostics (corr with is_abandoned):')
    for name, c in sorted(target_corr.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f'- {name}: {c:+.4f}')
    lines.append('')
    lines.append('Selected model features:')
    for f in selected_features:
        lines.append(f"- {f} (corr={target_corr.get(f, 0.0):+.4f})")
    lines.append('')
    lines.append('Forward feature selection path (CV-gain based):')
    for row in selection_history:
        fname = str(row.get('feature_name', ''))
        lines.append(f"- step {int(row['step'])}: add {fname}, auc={row['auc']:.4f}, brier={row['brier']:.4f}, l2={row['l2']}, n_features={int(row['n_features'])}")
    lines.append('')
    lines.append('Model coefficients (standardized feature scale):')
    lines.append(f'- intercept: {model.bias:+.6f}')
    for name, coef in sorted(coef_by_feature.items(), key=lambda kv: -abs(kv[1])):
        lines.append(f'- {name}: {coef:+.6f} (odds ratio per +1 SD = {math.exp(coef):.3f})')
    lines.append('')
    lines.append('CV regularization search (mean AUC, mean Brier):')
    for l2, (auc, br) in sorted(cv_summary.items(), key=lambda kv: kv[0]):
        lines.append(f'- l2={l2}: auc={auc:.4f}, brier={br:.4f}')
    lines.append('')
    lines.append(f'Selected l2: {best_l2}')
    lines.append('OOF blend search (AUC for ML-vs-CV blend alpha):')
    for a, auc in sorted(alpha_scores.items(), key=lambda kv: kv[0]):
        lines.append(f'- alpha={a:.2f}: auc={auc:.4f}')
    lines.append(f'Selected blend alpha (ML weight): {best_alpha:.2f}')
    lines.append('')
    lines.append('Full-data diagnostic metrics (reference only, not holdout):')
    lines.append(f'- ML AUC: {auc_ml_full:.4f}')
    lines.append(f'- ML Brier: {brier_ml_full:.4f}')
    lines.append(f'- Ensemble AUC: {auc_ens_full:.4f}')
    lines.append(f'- Ensemble Brier: {brier_ens_full:.4f}')
    lines.append('')
    lines.append('Score distributions:')
    lines.append(f'- ml_score min/max/mean: {min(ml_vals):.3f}/{max(ml_vals):.3f}/{sum(ml_vals)/len(ml_vals):.3f}')
    lines.append(f'- ensemble_score min/max/mean: {min(ens_vals):.3f}/{max(ens_vals):.3f}/{sum(ens_vals)/len(ens_vals):.3f}')

    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('done')
    print(f'features={selected_features}')
    print(f'best_l2={best_l2}, best_alpha={best_alpha:.2f}')
    print(f'ml range: {min(ml_vals):.3f}..{max(ml_vals):.3f}')
    print(f'ensemble range: {min(ens_vals):.3f}..{max(ens_vals):.3f}')


if __name__ == '__main__':
    main()
