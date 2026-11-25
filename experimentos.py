"""Experimentos del README: variabilidad por semilla, otro objetivo de recall, sin escalado y fuga de datos.

Uso: uv run --with scikit-learn --with matplotlib --with pandas python experimentos.py
"""
import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_curve, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

datos = load_breast_cancer(as_frame=True)
X, y = datos.data, (datos.target == 0).astype(int)


def modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=0))


def correr(semilla, recall_objetivo=0.97):
    Xe, Xp, ye, yp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=semilla)
    cv = StratifiedKFold(5, shuffle=True, random_state=semilla)
    m = modelo()
    prob_cv = cross_val_predict(m, Xe, ye, cv=cv, method="predict_proba")[:, 1]
    p, r, u = precision_recall_curve(ye, prob_cv)
    validos = [uu for uu, rr in zip(u, r[:-1]) if rr >= recall_objetivo]
    umbral = max(validos) if validos else 0.5
    m.fit(Xe, ye)
    pred = (m.predict_proba(Xp)[:, 1] >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(yp, pred).ravel()
    return umbral, recall_score(yp, pred), precision_score(yp, pred), fn, fp, int(yp.sum())


print("## Experimento 1: cambiar la semilla (misma regla: umbral con recall ≥ 0,97 en validación)")
print("| Semilla | Umbral elegido | Recall en prueba | Precisión en prueba | FN | FP | Malignos en prueba |")
print("|---|---|---|---|---|---|---|")
rec = []
for s in (0, 7, 42, 123, 2024):
    u, r, p, fn, fp, n = correr(s)
    rec.append(r)
    print(f"| {s} | {u:.3f} | {r:.3f} | {p:.3f} | {fn} | {fp} | {n} |")
print(f"\nRecall en prueba: mínimo {min(rec):.3f}, máximo {max(rec):.3f}, media {np.mean(rec):.3f}")

print("\n## Experimento 2: objetivo de recall ≥ 0,99 en validación (semilla 42)")
u, r, p, fn, fp, n = correr(42, 0.99)
print(f"umbral {u:.3f} → recall {r:.3f}, precisión {p:.3f}, FN {fn}, FP {fp}  (con 0,97 eran: ver tabla anterior)")

print("\n## Experimento 3: sin escalado (semilla 42, 5 pliegues, solo entrenamiento)")
Xe, Xp, ye, yp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    sin = cross_validate(LogisticRegression(max_iter=1000, random_state=0), Xe, ye, cv=cv, scoring=["recall", "f1"])
    avisos = sorted({type(x.message).__name__ for x in w})
con = cross_validate(modelo(), Xe, ye, cv=cv, scoring=["recall", "f1"])
print(f"sin escalado: recall {sin['test_recall'].mean():.3f}, F1 {sin['test_f1'].mean():.3f}, avisos: {avisos or 'ninguno'}")
print(f"con escalado: recall {con['test_recall'].mean():.3f}, F1 {con['test_f1'].mean():.3f}")

print("\n## Experimento 4: fuga de datos (escalar TODO antes de separar / de validar)")
Xmal = StandardScaler().fit_transform(X)  # el escalador ve también los datos de validación y de prueba
Xe2, Xp2, ye2, yp2 = train_test_split(Xmal, y, test_size=0.2, stratify=y, random_state=42)
fuga = cross_validate(LogisticRegression(max_iter=1000, random_state=0), Xe2, ye2, cv=cv, scoring=["recall", "f1"])
print(f"con fuga:      recall {fuga['test_recall'].mean():.3f}, F1 {fuga['test_f1'].mean():.3f}")
print(f"sin fuga (pipeline): recall {con['test_recall'].mean():.3f}, F1 {con['test_f1'].mean():.3f}")
