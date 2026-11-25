"""README experiments: variability across seeds, a stricter recall target, no scaling, and data leakage.

Usage: uv run --with scikit-learn --with matplotlib --with pandas python experiments.py
"""
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

data = load_breast_cancer(as_frame=True)
X, y = data.data, (data.target == 0).astype(int)


def make_model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))


def run(seed, recall_target=0.97):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    m = make_model()
    cv_prob = cross_val_predict(m, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
    p, r, t = precision_recall_curve(y_tr, cv_prob)
    valid = [tt for tt, rr in zip(t, r[:-1]) if rr >= recall_target]
    threshold = max(valid) if valid else 0.5
    m.fit(X_tr, y_tr)
    pred = (m.predict_proba(X_te)[:, 1] >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
    return threshold, recall_score(y_te, pred), precision_score(y_te, pred), fn, fp, int(y_te.sum())


print("## Experiment 1: change the seed (same rule: threshold with recall >= 0.97 in validation)")
print("| Seed | Chosen threshold | Test recall | Test precision | FN | FP | Malignant in test |")
print("|---|---|---|---|---|---|---|")
recalls = []
for s in (0, 7, 42, 123, 2024):
    u, r, p, fn, fp, n = run(s)
    recalls.append(r)
    print(f"| {s} | {u:.3f} | {r:.3f} | {p:.3f} | {fn} | {fp} | {n} |")
print(f"\nTest recall: min {min(recalls):.3f}, max {max(recalls):.3f}, mean {np.mean(recalls):.3f}")

print("\n## Experiment 2: recall target >= 0.99 in validation (seed 42)")
u, r, p, fn, fp, n = run(42, 0.99)
print(f"threshold {u:.3f} → recall {r:.3f}, precision {p:.3f}, FN {fn}, FP {fp}  (with 0.97 see the table above)")

print("\n## Experiment 3: no feature scaling (seed 42, 5 folds, training only)")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    unscaled = cross_validate(LogisticRegression(max_iter=1000, random_state=0), X_tr, y_tr, cv=cv, scoring=["recall", "f1"])
    caught = sorted({type(x.message).__name__ for x in w})
scaled = cross_validate(make_model(), X_tr, y_tr, cv=cv, scoring=["recall", "f1"])
print(f"unscaled: recall {unscaled['test_recall'].mean():.3f}, F1 {unscaled['test_f1'].mean():.3f}, warnings: {caught or 'none'}")
print(f"scaled:   recall {scaled['test_recall'].mean():.3f}, F1 {scaled['test_f1'].mean():.3f}")

print("\n## Experiment 4: data leakage (scale EVERYTHING before splitting / validating)")
X_leaky = StandardScaler().fit_transform(X)  # the scaler also sees the validation and test data
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(X_leaky, y, test_size=0.2, stratify=y, random_state=42)
leaky = cross_validate(LogisticRegression(max_iter=1000, random_state=0), X_tr2, y_tr2, cv=cv, scoring=["recall", "f1"])
print(f"with leakage:          recall {leaky['test_recall'].mean():.3f}, F1 {leaky['test_f1'].mean():.3f}")
print(f"without (pipeline):    recall {scaled['test_recall'].mean():.3f}, F1 {scaled['test_f1'].mean():.3f}")
