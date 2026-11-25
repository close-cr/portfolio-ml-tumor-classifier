🇬🇧 English · 🇪🇸 [Español](README.es.md)

# Tumor classification with scikit-learn (and why accuracy is misleading)

A supervised-learning project on the public **Breast Cancer Wisconsin (Diagnostic)** dataset, bundled with scikit-learn (569 cases, 30 numeric measurements each, no missing values). The goal is not to "reach 99 %": it is to show **how to evaluate a model properly**: split the data first, avoid leakage, choose the model and the threshold without looking at the test set, and report the metric that matters for the problem.

**Cost: zero. Runs in seconds on a laptop, no GPU.**

> This is a learning exercise. It is not a medical device and must not be used to diagnose anything.

## Run it

```bash
uv run --with scikit-learn --with matplotlib --with pandas python train.py
# or:  pip install -r requirements.txt && python train.py
```

It writes `results.json` and three figures in `figures/`. It is deterministic (seed 42).

## The problem and the metric that matters

The model predicts whether a tumor is **malignant** (positive class) or benign. The serious mistake is a **false negative**: saying "benign" when it was malignant. So the main metric is **recall** (of the truly malignant cases, how many are detected), while watching precision so we do not over-flag.

The data has 212 malignant (37.3 %) and 357 benign cases: not heavily imbalanced, but enough to show the accuracy problem.

## Method

1. **Split before exploring**: 80 % train (455) / 20 % test (114), stratified. The test set is not touched until the end.
2. **Pipeline with scaling inside**: the scaler is fitted only on each fold's training data, so there is no **data leakage** in cross-validation.
3. **Three models** compared with 5-fold cross-validation on the training data only: a baseline that always says "benign", logistic regression, and random forest.
4. **The model is chosen with cross-validation** (recall, then F1), not by looking at the test set.
5. **The decision threshold is also chosen without the test set**: with cross-validated predictions, the highest threshold that reaches recall ≥ 0.97.
6. **Final evaluation exactly once** on the test set, plus a train-vs-test comparison to check for overfitting.

## Results (real output of `train.py`)

**Cross-validation (training data, 5-fold mean):**

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Baseline "always benign" | 0.626 | 0.000 | **0.000** | 0.000 | 0.500 |
| Logistic regression | 0.974 | 0.977 | 0.953 | 0.964 | 0.996 |
| Random forest | 0.960 | 0.958 | 0.935 | 0.946 | 0.988 |

Chosen: **logistic regression** (it also wins on every metric, and it is simpler and easier to interpret).

**Test set (114 unseen cases), logistic regression:**

| Threshold | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
|---|---|---|---|---|---|---|---|---|
| 0.50 (default) | 0.965 | 0.975 | 0.929 | 0.951 | 39 | 1 | 71 | 3 |
| **0.315 (chosen in validation)** | **0.974** | 0.976 | **0.952** | 0.964 | 40 | 1 | 71 | **2** |

Test AUC: 0.996. Overfitting: accuracy 0.987 on train vs 0.965 on test (a 2-point gap, acceptable).

Most influential variables (logistic-regression coefficients on standardized features): `worst texture`, `radius error`, `worst symmetry`, `mean concave points`, `worst concavity`.

Figures: `figures/confusion_matrix.png`, `figures/precision_recall.png`, `figures/probability_distribution.png`.

## What the project shows

- **Accuracy is misleading**: the model that always says "benign" has 63 % accuracy and **recall 0**: it never detects a malignant case.
- **Lowering the threshold changes the type of error**: from 0.5 to 0.315 one false negative is avoided at no cost in this sample (1 false positive in both cases).
- **Choosing without the test set**: model and threshold came from cross-validation; the test set only reports.

## Experiments

Run with `uv run --with scikit-learn --with matplotlib --with pandas python experiments.py`. Real output.

### 1. Change the seed (same rule: threshold with recall ≥ 0.97 in validation)

| Seed | Chosen threshold | Test recall | Test precision | FN | FP | Malignant in test |
|---|---|---|---|---|---|---|
| 0 | 0.418 | 0.952 | 0.909 | 2 | 4 | 42 |
| 7 | 0.376 | 0.952 | 0.976 | 2 | 1 | 42 |
| 42 | 0.315 | 0.952 | 0.976 | 2 | 1 | 42 |
| 123 | 0.288 | 0.976 | 0.837 | 1 | 8 | 42 |
| 2024 | 0.444 | 0.929 | 0.975 | 3 | 1 | 42 |

Test recall: min 0.929, max 0.976, mean 0.952. **The chosen threshold varies a lot** (0.288 to 0.444) and so does precision (0.837 to 0.976). With 42 malignant cases in the test set, a single case moves recall 2.4 points: one seed's result is not a fixed number. That is why it is better to say "around 95 %" than to quote an exact figure.

### 2. Recall target ≥ 0.99 in validation (seed 42)

The threshold drops to 0.094: recall 0.976 (1 false negative) but precision 0.891 (5 false positives). Wanting more recall is paid for with more false alarms, and the 0.99 from validation is not reached on the test set either.

### 3. No feature scaling (seed 42, training only)

| | Recall | F1 | Warnings |
|---|---|---|---|
| Unscaled | 0.924 | 0.937 | `ConvergenceWarning` |
| Scaled | 0.953 | 0.964 | none |

Without scaling, logistic regression converges worse and performs worse.

### 4. Data leakage: scaling everything before splitting

| | Recall | F1 |
|---|---|---|
| With leakage | 0.953 | 0.964 |
| Without (pipeline) | 0.953 | 0.964 |

**On this dataset there is no visible difference**: leakage through scaling is subtle because the mean and standard deviation of the whole set are very close to those of the training set. That does **not** mean the leakage rule is useless: with missing-value imputation, feature selection, or encodings learned from the data, leakage can inflate results. Here the rule is followed as prevention, not because this experiment demonstrated it.

## Limitations (important)

- **The test set is small**: only 42 malignant cases. One more or one fewer false negative moves recall by about 2.4 points, so the gap between 0.929 and 0.952 **is not statistically solid**. No confidence intervals were computed.
- **The threshold did not reach on the test set the 0.97 recall targeted in validation** (it was 0.952): that is normal sample-to-sample variation, and a reason not to over-interpret a single split.
- One clean, public dataset; it was not tested at another hospital or with another instrument.
- No hyperparameter tuning and no feature selection were done.

## Next steps

1. Bootstrap confidence intervals on the test set.
2. Repeat with nested cross-validation to tune hyperparameters without bias.
3. Calibrate the probabilities and inspect the calibration curve.
4. Try another tabular dataset (for example one with very imbalanced classes) to see the effect of the threshold.
