"""
Fase 3: buscar un factor que distinga a los 4 modelos, ya que su exactitud está prácticamente empatada.

Se repite la partición 75/25 por estudiante 30 veces (semillas 100-129) y en cada una se entrenan
los 4 modelos con los mismos hiperparámetros de la fase 2. Se miden:
  - exactitud (umbral 0,5)
  - log loss y Brier: calidad de la probabilidad que entrega el modelo (menor = mejor)
  - ECE: error de calibración (si dice 80 %, ¿acierta el 80 % de las veces?)
  - FN: estudiantes que pasan a AVANCE sin dominar el tema (el error más costoso)
Para cada métrica se cuenta en cuántas de las 30 particiones gana cada modelo.

Datos: SOLO data/cybermind_sintetico.csv. No se usa cybermind_sintetico_oculto.csv.
Uso:   python pruebas/fase3_seleccion_modelo/seleccion_modelo.py
"""
import json
import pickle
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

CARPETA = Path(__file__).resolve().parent
SALIDA = CARPETA / "resultados"
SALIDA.mkdir(exist_ok=True)
datos = pd.read_csv(CARPETA.parents[1] / "data" / "cybermind_sintetico.csv")

CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)
grupos = datos["id_estudiante"]


def prep():
    return ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                              ("num", StandardScaler(), NUMERICAS)])


# Mismos hiperparámetros que la fase 2 (kNN y GBDT: los elegidos por su búsqueda en rejilla)
MODELOS = {
    "Árbol de decisión": lambda: Pipeline([("p", prep()), ("m", DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=0))]),
    "Regresión logística": lambda: Pipeline([("p", prep()), ("m", LogisticRegression(C=1.0, max_iter=1000))]),
    "kNN": lambda: Pipeline([("p", prep()), ("m", KNeighborsClassifier(n_neighbors=51, p=1))]),
    "GBDT": lambda: Pipeline([("p", prep()), ("m", GradientBoostingClassifier(learning_rate=0.05, max_depth=3, min_samples_leaf=20,
                                                                           n_estimators=200, subsample=0.8, random_state=0))]),
}


def ece(y_real, p, n_bins=10):
    """Error de calibración esperado: diferencia media entre probabilidad predicha y frecuencia real."""
    bordes = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, bordes[1:-1]), 0, n_bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y_real[idx == b].mean())
                     for b in range(n_bins) if (idx == b).any()))


METRICAS = {"exactitud": max, "auc": max, "log_loss": min, "brier": min, "ece": min, "FN_avance_sin_dominio": min}
R = {n: {k: [] for k in METRICAS} for n in MODELOS}
for semilla in range(100, 130):
    ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=semilla).split(X, y, grupos))
    y_pru = y.iloc[pru].values
    for nombre, crear in MODELOS.items():
        p = crear().fit(X.iloc[ent], y.iloc[ent]).predict_proba(X.iloc[pru])[:, 1]
        R[nombre]["exactitud"].append(accuracy_score(y_pru, p >= 0.5))
        R[nombre]["auc"].append(roc_auc_score(y_pru, p))
        R[nombre]["log_loss"].append(log_loss(y_pru, p))
        R[nombre]["brier"].append(brier_score_loss(y_pru, p))
        R[nombre]["ece"].append(ece(y_pru, p))
        R[nombre]["FN_avance_sin_dominio"].append(int(((y_pru == 1) & (p < 0.5)).sum()))

resumen = {}
for k, mejor in METRICAS.items():
    victorias = {n: 0 for n in MODELOS}
    for i in range(30):
        valores = {n: R[n][k][i] for n in MODELOS}
        victorias[mejor(valores, key=valores.get)] += 1
    resumen[k] = {n: {"media": round(float(np.mean(R[n][k])), 4), "desv": round(float(np.std(R[n][k])), 4),
                      "gana_en_30": victorias[n]} for n in MODELOS}

# Costo de servirlo: tamaño del modelo guardado y tiempo de respuesta para un estudiante
ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, grupos))
servicio = {}
for nombre, crear in MODELOS.items():
    m = crear().fit(X.iloc[ent], y.iloc[ent])
    una = X.iloc[pru].iloc[[0]]
    t = time.perf_counter()
    for _ in range(200):
        m.predict_proba(una)
    servicio[nombre] = {"tamano_kb": round(len(pickle.dumps(m)) / 1024, 1),
                        "latencia_ms": round((time.perf_counter() - t) / 200 * 1000, 2)}

print(f"{'métrica':<24}" + "".join(f"{n:>22}" for n in MODELOS))
for k in METRICAS:
    print(f"{k:<24}" + "".join(f"{resumen[k][n]['media']:>13.4f} ({resumen[k][n]['gana_en_30']:>2}/30)" for n in MODELOS))
print("\nCosto de servicio:", servicio)

(SALIDA / "seleccion_modelo.json").write_text(json.dumps(
    {"datos": "cybermind_sintetico.csv", "particiones": "30 × GroupShuffleSplit 75/25 por estudiante, semillas 100-129",
     "resumen": resumen, "servicio": servicio,
     "por_particion": {n: {k: [float(v) for v in vals] for k, vals in R[n].items()} for n in MODELOS}},
    indent=2, ensure_ascii=False), encoding="utf-8")

# Gráfica: exactitud (empate) vs log loss (GBDT gana siempre) en las 30 particiones
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.5))
colores = {"Árbol de decisión": "#8a94a6", "Regresión logística": "#2a78d6", "kNN": "#7fb2ff", "GBDT": "#eb6834"}
for ax, k, titulo in [(ejes[0], "exactitud", "Exactitud (mayor = mejor)"), (ejes[1], "log_loss", "Log loss (menor = mejor)")]:
    for i, n in enumerate(MODELOS):
        v = R[n][k]
        ax.scatter(np.full(30, i) + np.random.default_rng(i).uniform(-0.15, 0.15, 30), v, s=14, color=colores[n], alpha=0.7)
        ax.hlines(np.mean(v), i - 0.3, i + 0.3, color="#1b2738", linewidth=2)
    ax.set_xticks(range(len(MODELOS)), list(MODELOS), fontsize=9)
    ax.set_title(f"{titulo}\n30 particiones por estudiante", fontsize=10)
    ax.grid(axis="y", color="#e5e4df")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
fig.suptitle("La exactitud empata; la calidad de la probabilidad separa a GBDT", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(SALIDA / "seleccion_modelo.png", dpi=150, bbox_inches="tight")
print("Guardado en", SALIDA)
