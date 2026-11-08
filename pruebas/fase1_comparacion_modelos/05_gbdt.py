"""
Método adicional 5: GBDT (Gradient Boosting de árboles de decisión), comparado con CART.

Compara, con la MISMA partición y las mismas variables que los métodos 1 a 4:
  a) GBDT con hiperparámetros por defecto de scikit-learn.
  b) GBDT con hiperparámetros buscados (GroupKFold 5 sobre entrenamiento, exactitud).
  c) CART con hiperparámetros buscados de la misma forma (comparación pareja con b).
El árbol CART original (profundidad 4, hoja mínima 50, sin búsqueda) está en 02_arbol_decision.py.

Uso: python 05_gbdt.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit

# ---------------------------------------------------------------- datos y partición
CARPETA = Path(__file__).resolve().parent
datos = pd.read_csv(CARPETA.parents[1] / "data" / "cybermind_sintetico.csv")   # único archivo de datos

CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)   # clase positiva = REFUERZO
grupos = datos["id_estudiante"]

# Misma partición en los 4 métodos: 75 % / 25 %, agrupada por estudiante
ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups=grupos))
y_pru = y.iloc[pru].to_numpy()


def evaluar(pred):
    """Métricas en el conjunto de prueba + IC 95 % de la exactitud (bootstrap por estudiante)."""
    pred = np.asarray(pred)
    tn, fp, fn, tp = confusion_matrix(y_pru, pred).ravel()
    est = grupos.iloc[pru].to_numpy()
    df = pd.DataFrame({"est": est, "ok": (pred == y_pru).astype(int)}).groupby("est")["ok"].agg(["sum", "count"])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(df), size=(1000, len(df)))
    acc_boot = df["sum"].to_numpy()[idx].sum(axis=1) / df["count"].to_numpy()[idx].sum(axis=1)
    return {
        "n_entrenamiento": int(len(ent)), "n_prueba": int(len(pru)),
        "exactitud": round(float(accuracy_score(y_pru, pred)), 4),
        "ic95_exactitud": [round(float(np.percentile(acc_boot, 2.5)), 4), round(float(np.percentile(acc_boot, 97.5)), 4)],
        "precision_REFUERZO": round(float(precision_score(y_pru, pred)), 4),
        "recall_REFUERZO": round(float(recall_score(y_pru, pred)), 4),
        "F1_REFUERZO": round(float(f1_score(y_pru, pred)), 4),
        "matriz_confusion": {"VN_avance_ok": int(tn), "FP_refuerzo_innecesario": int(fp),
                             "FN_avance_sin_dominio": int(fn), "VP_refuerzo_ok": int(tp)},
    }


def guardar(nombre, res):
    (CARPETA / "resultados").mkdir(exist_ok=True)
    (CARPETA / "resultados" / f"{nombre}.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2, ensure_ascii=False))



from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# ---------------------------------------------------------------- método
def prep():
    # one-hot en las categóricas, numéricas sin escalar (los árboles no lo necesitan)
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
                             remainder="passthrough", verbose_feature_names_out=False)

cv = GroupKFold(n_splits=5)
salida = {}

# a) GBDT por defecto (n_estimators=100, learning_rate=0.1, max_depth=3, subsample=1.0)
gb_def = Pipeline([("prep", prep()), ("gb", GradientBoostingClassifier(random_state=0))])
gb_def.fit(X.iloc[ent], y.iloc[ent])
salida["GBDT por defecto"] = {"hiperparametros": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "subsample": 1.0, "random_state": 0},
                              **evaluar(gb_def.predict(X.iloc[pru]))}

# b) GBDT con búsqueda
rej_gb = {"gb__n_estimators": [50, 100, 200], "gb__learning_rate": [0.05, 0.1], "gb__max_depth": [2, 3],
          "gb__subsample": [0.8, 1.0], "gb__min_samples_leaf": [20, 50]}
gs_gb = GridSearchCV(Pipeline([("prep", prep()), ("gb", GradientBoostingClassifier(random_state=0))]),
                     rej_gb, scoring="accuracy", cv=cv, n_jobs=-1)
gs_gb.fit(X.iloc[ent], y.iloc[ent], groups=grupos.iloc[ent])
salida["GBDT con búsqueda"] = {"hiperparametros": {k.replace("gb__", ""): v for k, v in gs_gb.best_params_.items()},
                               "combinaciones": len(gs_gb.cv_results_["params"]), "exactitud_cv": round(float(gs_gb.best_score_), 4),
                               **evaluar(gs_gb.predict(X.iloc[pru]))}

# c) CART con la misma búsqueda
rej_ct = {"ct__max_depth": [2, 3, 4, 5, 6, 8], "ct__min_samples_leaf": [20, 50, 100]}
gs_ct = GridSearchCV(Pipeline([("prep", prep()), ("ct", DecisionTreeClassifier(random_state=0))]),
                     rej_ct, scoring="accuracy", cv=cv, n_jobs=-1)
gs_ct.fit(X.iloc[ent], y.iloc[ent], groups=grupos.iloc[ent])
salida["CART con búsqueda"] = {"hiperparametros": {k.replace("ct__", ""): v for k, v in gs_ct.best_params_.items()},
                               "combinaciones": len(gs_ct.cv_results_["params"]), "exactitud_cv": round(float(gs_ct.best_score_), 4),
                               **evaluar(gs_ct.predict(X.iloc[pru]))}

# importancia de variables del GBDT elegido
nombres = list(gs_gb.best_estimator_[0].get_feature_names_out())
imp = sorted(zip(nombres, gs_gb.best_estimator_[-1].feature_importances_), key=lambda t: -t[1])
salida["importancias_gbdt"] = {n: round(float(v), 4) for n, v in imp if v > 0.005}
guardar("05_gbdt", {"metodo": "GBDT vs CART", **salida})
