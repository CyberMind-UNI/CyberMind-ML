"""
Método 4: k vecinos más cercanos (kNN).

A diferencia de los métodos 2 y 3, aquí SÍ se buscaron los hiperparámetros: validación cruzada
agrupada de 5 particiones (GroupKFold) usando solo el conjunto de entrenamiento y exactitud como
métrica. El conjunto de prueba se usa una única vez al final.

Uso: python 04_knn.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.neighbors import KNeighborsClassifier

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


# ---------------------------------------------------------------- método
REJILLA = {"knn__n_neighbors": [1, 3, 5, 9, 15, 25, 35, 51, 75, 101],
           "knn__weights": ["uniform", "distance"],
           "knn__p": [1, 2]}          # 1 = distancia Manhattan, 2 = euclídea

pipe = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                                ("num", StandardScaler(), NUMERICAS)])),   # escalar es obligatorio en kNN
    ("knn", KNeighborsClassifier()),
])
gs = GridSearchCV(pipe, REJILLA, scoring="accuracy", cv=GroupKFold(n_splits=5), n_jobs=-1)
gs.fit(X.iloc[ent], y.iloc[ent], groups=grupos.iloc[ent])
pred = gs.predict(X.iloc[pru])

r = pd.DataFrame(gs.cv_results_)
por_k = r.groupby("param_knn__n_neighbors")["mean_test_score"].max().round(4)
res = {"metodo": "kNN", "hiperparametros": {k.replace("knn__", ""): v for k, v in gs.best_params_.items()},
       "rejilla_probada": {k.replace("knn__", ""): v for k, v in REJILLA.items()},
       "combinaciones_probadas": int(len(r)), "validacion": "GroupKFold(5) sobre entrenamiento, métrica = exactitud",
       "exactitud_cv_entrenamiento": round(float(gs.best_score_), 4),
       "exactitud_cv_por_k": {int(k): float(v) for k, v in por_k.items()},
       "variables": CATEGORICAS + NUMERICAS, **evaluar(pred)}
guardar("04_knn", res)
