"""Tumor classification (benign / malignant) with scikit-learn, step by step and with a careful evaluation.

Dataset: Breast Cancer Wisconsin (Diagnostic), bundled with scikit-learn (nothing to download).
Positive class = MALIGNANT, because the serious mistake is NOT detecting it: that is why recall matters.

Usage:  python train.py        (writes results.json and figures/*.png)
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# The baseline never predicts "malignant", so its precision is undefined (reported as 0): expected.
warnings.filterwarnings("ignore", message="Precision is ill-defined")
SEED = 42
Path("figures").mkdir(exist_ok=True)
out = {}

# ---------- 1. Data ----------
data = load_breast_cancer(as_frame=True)
X = data.data
y = (data.target == 0).astype(int)  # in the original data 0 = malignant; here 1 = malignant (positive class)
out["rows"], out["columns"] = X.shape
out["missing_values"] = int(X.isna().sum().sum())
out["malignant"], out["benign"] = int(y.sum()), int((1 - y).sum())
print(f"{X.shape[0]} rows, {X.shape[1]} columns, missing values: {out['missing_values']}")
print(f"malignant: {out['malignant']} ({y.mean():.1%})  benign: {out['benign']}")

# ---------- 2. Split BEFORE looking at anything else: the test set stays untouched until the end ----------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"train: {len(X_train)}  test: {len(X_test)}")

# ---------- 3. Models. Scaling lives INSIDE the pipeline to avoid leakage during cross-validation ----------
models = {
    "Baseline (always «benign»)": DummyClassifier(strategy="most_frequent"),
    "Logistic regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEED)),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
}

# ---------- 4. Compare with cross-validation (training data only) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
scoring = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
out["cross_validation"] = {}
print("\nCross-validation (5 folds, training only) — mean")
for name, m in models.items():
    r = cross_validate(m, X_train, y_train, cv=cv, scoring=scoring)
    out["cross_validation"][name] = {k: [round(float(r[f"test_{k}"].mean()), 4), round(float(r[f"test_{k}"].std()), 4)] for k in scoring}
    f = out["cross_validation"][name]
    print(f"  {name:28s} accuracy {f['accuracy'][0]:.3f}  precision {f['precision'][0]:.3f}  recall {f['recall'][0]:.3f}  F1 {f['f1'][0]:.3f}  AUC {f['roc_auc'][0]:.3f}")

# The model is CHOSEN with cross-validation (recall, then F1), not by looking at the test set.
chosen = max((n for n in models if not n.startswith("Baseline")), key=lambda n: tuple(out["cross_validation"][n][k][0] for k in ("recall", "f1")))
out["chosen_model"] = chosen
print(f"\nModel chosen by cross-validation: {chosen}")
model = models[chosen]

# ---------- 5. The threshold is also chosen WITHOUT the test set: using cross-validated predictions ----------
cv_prob = cross_val_predict(model, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thresholds = precision_recall_curve(y_train, cv_prob)
# Rule: the highest threshold that reaches recall >= 0.97 in validation (missing a malignant case is the serious error).
valid = [(u, p, r) for u, p, r in zip(thresholds, prec[:-1], rec[:-1]) if r >= 0.97]
threshold = max(valid)[0] if valid else 0.5
out["threshold"] = round(float(threshold), 4)
print(f"Threshold chosen in validation (recall >= 0.97): {threshold:.3f}")

# ---------- 6. Final evaluation on the test set: ONCE ----------
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]
results = {}
for label, u in (("threshold 0.5", 0.5), ("chosen threshold", threshold)):
    pred = (prob >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    results[label] = dict(threshold=round(float(u), 3), accuracy=round(accuracy_score(y_test, pred), 4), precision=round(precision_score(y_test, pred), 4),
                          recall=round(recall_score(y_test, pred), 4), f1=round(f1_score(y_test, pred), 4), TP=int(tp), FP=int(fp), TN=int(tn), FN=int(fn))
out["test"] = results
out["test"]["roc_auc"] = round(roc_auc_score(y_test, prob), 4)
# The "always benign" reference on the test set, to show why accuracy is misleading
base = models["Baseline (always «benign»)"].fit(X_train, y_train).predict(X_test)
out["test"]["baseline_always_benign"] = dict(accuracy=round(accuracy_score(y_test, base), 4), recall=round(recall_score(y_test, base), 4))
print("\nTEST (data the model never saw):")
for k, v in results.items():
    print(f"  {k:17s} {v}")
print(f"  AUC = {out['test']['roc_auc']}   |   baseline «always benign»: accuracy {out['test']['baseline_always_benign']['accuracy']}, recall {out['test']['baseline_always_benign']['recall']}")

# ---------- 7. Overfitting: compare training vs test ----------
acc_train = accuracy_score(y_train, model.predict(X_train))
acc_test = accuracy_score(y_test, (prob >= 0.5).astype(int))
out["overfitting"] = dict(train_accuracy=round(acc_train, 4), test_accuracy=round(acc_test, 4), gap=round(acc_train - acc_test, 4))
print(f"\nOverfitting: training accuracy {acc_train:.3f} vs test {acc_test:.3f} (gap {acc_train - acc_test:+.3f})")

# ---------- 8. Which variables matter ----------
clf = model[-1] if hasattr(model, "steps") else model
importance = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
top = sorted(zip(X.columns, importance), key=lambda t: -t[1])[:5]
out["top5_variables"] = [(n, round(float(v), 3)) for n, v in top]
print("Most influential variables:", ", ".join(f"{n} ({v:.2f})" for n, v in top))

# ---------- 9. Figures ----------
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_test, (prob >= threshold).astype(int), display_labels=["benign", "malignant"], ax=ax, colorbar=False)
ax.set_title(f"Confusion matrix (test)\n{chosen}, threshold {threshold:.2f}")
fig.tight_layout(); fig.savefig("figures/confusion_matrix.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
PrecisionRecallDisplay.from_predictions(y_test, prob, ax=ax)
ax.axvline(results["chosen threshold"]["recall"], color="gray", ls=":")
ax.set_title("Precision vs recall (test)")
fig.tight_layout(); fig.savefig("figures/precision_recall.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist([prob[y_test == 0], prob[y_test == 1]], bins=20, stacked=True, label=["benign", "malignant"], color=["#4c9", "#d55"])
ax.axvline(threshold, color="black", ls="--", label=f"threshold {threshold:.2f}")
ax.set_xlabel("predicted probability of «malignant»"); ax.set_ylabel("cases"); ax.legend(); ax.set_title("Where the test cases fall")
fig.tight_layout(); fig.savefig("figures/probability_distribution.png", dpi=140); plt.close(fig)

Path("results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nSaved: results.json and figures/")
