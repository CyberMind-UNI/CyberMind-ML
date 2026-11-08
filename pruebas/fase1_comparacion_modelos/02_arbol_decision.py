"""
Método 2: árbol de decisión (CART).

Hiperparámetros fijados a mano (NO se buscaron con validación cruzada):
max_depth=4, min_samples_leaf=50. El resto son los valores por defecto de scikit-learn.

Uso: python 02_arbol_decision.py
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
from sklearn.tree import DecisionTreeClassifier, export_text

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
PARAMS = dict(max_depth=4, min_samples_leaf=50, random_state=0)   # criterion='gini', splitter='best', ccp_alpha=0 por defecto

modelo = Pipeline([
    # one-hot solo en las categóricas; las numéricas pasan sin escalar (el árbol no lo necesita)
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)],
                               remainder="passthrough", verbose_feature_names_out=False)),
    ("arbol", DecisionTreeClassifier(**PARAMS)),
])
modelo.fit(X.iloc[ent], y.iloc[ent])
pred = modelo.predict(X.iloc[pru])

nombres = list(modelo[0].get_feature_names_out())
imp = sorted(zip(nombres, modelo[-1].feature_importances_), key=lambda t: -t[1])
res = {"metodo": "Árbol de decisión (CART)", "hiperparametros": modelo[-1].get_params(),
       "variables": CATEGORICAS + NUMERICAS, **evaluar(pred),
       "importancias": {n: round(float(v), 4) for n, v in imp if v > 0},
       "reglas": export_text(modelo[-1], feature_names=nombres)}
guardar("02_arbol_decision", res)
