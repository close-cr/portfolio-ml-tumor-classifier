"""Clasificación de tumores (benigno / maligno) con scikit-learn, paso a paso y con cuidado en la evaluación.

Dataset: Breast Cancer Wisconsin (Diagnostic), incluido en scikit-learn (no hay que descargar nada).
Clase positiva = MALIGNO, porque lo grave es NO detectarlo: por eso importa el recall.

Uso:  python entrenar.py        (genera resultados.json y figuras/*.png)
"""
import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay, accuracy_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

# El modelo base nunca predice «maligno», así que su precisión no está definida (se reporta 0): es esperado.
warnings.filterwarnings("ignore", message="Precision is ill-defined")
SEMILLA = 42
Path("figuras").mkdir(exist_ok=True)
out = {}

# ---------- 1. Datos ----------
datos = load_breast_cancer(as_frame=True)
X = datos.data
y = (datos.target == 0).astype(int)  # en el original 0 = maligno; aquí 1 = maligno (clase positiva)
out["filas"], out["columnas"] = X.shape
out["valores_faltantes"] = int(X.isna().sum().sum())
out["malignos"], out["benignos"] = int(y.sum()), int((1 - y).sum())
print(f"{X.shape[0]} filas, {X.shape[1]} columnas, faltantes: {out['valores_faltantes']}")
print(f"malignos: {out['malignos']} ({y.mean():.1%})  benignos: {out['benignos']}")

# ---------- 2. Separar ANTES de mirar nada más: la prueba queda intacta hasta el final ----------
X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEMILLA)
print(f"entrenamiento: {len(X_ent)}  prueba: {len(X_pru)}")

# ---------- 3. Modelos. El escalado va DENTRO del pipeline para evitar fuga de datos en la validación cruzada ----------
modelos = {
    "Base (siempre «benigno»)": DummyClassifier(strategy="most_frequent"),
    "Regresión logística": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=SEMILLA)),
    "Random forest": RandomForestClassifier(n_estimators=300, random_state=SEMILLA),
}

# ---------- 4. Comparación con validación cruzada (solo con los datos de entrenamiento) ----------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA)
metricas = {"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1", "roc_auc": "roc_auc"}
out["validacion_cruzada"] = {}
print("\nValidación cruzada (5 pliegues, solo entrenamiento) — media ± desviación")
for nombre, m in modelos.items():
    r = cross_validate(m, X_ent, y_ent, cv=cv, scoring=metricas)
    out["validacion_cruzada"][nombre] = {k: [round(float(r[f"test_{k}"].mean()), 4), round(float(r[f"test_{k}"].std()), 4)] for k in metricas}
    f = out["validacion_cruzada"][nombre]
    print(f"  {nombre:28s} exactitud {f['accuracy'][0]:.3f}  precisión {f['precision'][0]:.3f}  recall {f['recall'][0]:.3f}  F1 {f['f1'][0]:.3f}  AUC {f['roc_auc'][0]:.3f}")

# El modelo se ELIGE con la validación cruzada (recall, luego F1), no mirando la prueba.
elegido = max((n for n in modelos if not n.startswith("Base")), key=lambda n: tuple(out["validacion_cruzada"][n][k][0] for k in ("recall", "f1")))
out["modelo_elegido"] = elegido
print(f"\nModelo elegido por validación cruzada: {elegido}")
modelo = modelos[elegido]

# ---------- 5. El umbral también se elige SIN mirar la prueba: con predicciones de validación cruzada ----------
prob_cv = cross_val_predict(modelo, X_ent, y_ent, cv=cv, method="predict_proba")[:, 1]
prec, rec, umbrales = precision_recall_curve(y_ent, prob_cv)
# Regla: el umbral más alto que logra recall >= 0,97 en validación (lo grave es no detectar un maligno).
validos = [(u, p, r) for u, p, r in zip(umbrales, prec[:-1], rec[:-1]) if r >= 0.97]
umbral = max(validos)[0] if validos else 0.5
out["umbral"] = round(float(umbral), 4)
print(f"Umbral elegido en validación (recall ≥ 0,97): {umbral:.3f}")

# ---------- 6. Evaluación final en la prueba: UNA sola vez ----------
modelo.fit(X_ent, y_ent)
prob = modelo.predict_proba(X_pru)[:, 1]
resultados = {}
for etiqueta, u in (("umbral 0,5", 0.5), ("umbral elegido", umbral)):
    pred = (prob >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_pru, pred).ravel()
    resultados[etiqueta] = dict(umbral=round(float(u), 3), accuracy=round(accuracy_score(y_pru, pred), 4), precision=round(precision_score(y_pru, pred), 4),
                                recall=round(recall_score(y_pru, pred), 4), f1=round(f1_score(y_pru, pred), 4), VP=int(tp), FP=int(fp), VN=int(tn), FN=int(fn))
out["prueba"] = resultados
out["prueba"]["roc_auc"] = round(roc_auc_score(y_pru, prob), 4)
# La referencia «siempre benigno» en prueba, para mostrar por qué la exactitud engaña
base = modelos["Base (siempre «benigno»)"].fit(X_ent, y_ent).predict(X_pru)
out["prueba"]["base_siempre_benigno"] = dict(accuracy=round(accuracy_score(y_pru, base), 4), recall=round(recall_score(y_pru, base), 4))
print("\nPRUEBA (datos que el modelo nunca vio):")
for k, v in resultados.items():
    print(f"  {k:15s} {v}")
print(f"  AUC = {out['prueba']['roc_auc']}   |   base «siempre benigno»: exactitud {out['prueba']['base_siempre_benigno']['accuracy']}, recall {out['prueba']['base_siempre_benigno']['recall']}")

# ---------- 7. Sobreajuste: comparar entrenamiento vs prueba ----------
acc_ent = accuracy_score(y_ent, modelo.predict(X_ent))
acc_pru = accuracy_score(y_pru, (prob >= 0.5).astype(int))
out["sobreajuste"] = dict(exactitud_entrenamiento=round(acc_ent, 4), exactitud_prueba=round(acc_pru, 4), brecha=round(acc_ent - acc_pru, 4))
print(f"\nSobreajuste: exactitud en entrenamiento {acc_ent:.3f} vs prueba {acc_pru:.3f} (brecha {acc_ent - acc_pru:+.3f})")

# ---------- 8. Qué variables pesan ----------
clf = modelo[-1] if hasattr(modelo, "steps") else modelo
imp = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
top = sorted(zip(X.columns, imp), key=lambda t: -t[1])[:5]
out["variables_top5"] = [(n, round(float(v), 3)) for n, v in top]
print("Variables más influyentes:", ", ".join(f"{n} ({v:.2f})" for n, v in top))

# ---------- 9. Figuras ----------
fig, ax = plt.subplots(figsize=(4.5, 4))
ConfusionMatrixDisplay.from_predictions(y_pru, (prob >= umbral).astype(int), display_labels=["benigno", "maligno"], ax=ax, colorbar=False)
ax.set_title(f"Matriz de confusión (prueba)\n{elegido}, umbral {umbral:.2f}")
fig.tight_layout(); fig.savefig("figuras/matriz_confusion.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(5, 4))
PrecisionRecallDisplay.from_predictions(y_pru, prob, ax=ax)
ax.axvline(resultados["umbral elegido"]["recall"], color="gray", ls=":")
ax.set_title("Precisión vs recall (prueba)")
fig.tight_layout(); fig.savefig("figuras/precision_recall.png", dpi=140); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
ax.hist([prob[y_pru == 0], prob[y_pru == 1]], bins=20, stacked=True, label=["benigno", "maligno"], color=["#4c9", "#d55"])
ax.axvline(umbral, color="black", ls="--", label=f"umbral {umbral:.2f}")
ax.set_xlabel("probabilidad predicha de «maligno»"); ax.set_ylabel("casos"); ax.legend(); ax.set_title("Dónde caen los casos de prueba")
fig.tight_layout(); fig.savefig("figuras/distribucion_probabilidades.png", dpi=140); plt.close(fig)

Path("resultados.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
print("\nGuardado: resultados.json y figuras/")
