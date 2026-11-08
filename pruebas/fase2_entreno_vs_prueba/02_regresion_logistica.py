"""
Modelo 2 de 4: Regresión logística — evaluación con datos de ENTRENAMIENTO (75 %) y luego de PRUEBA (25 %).

Regresión logística (salida = probabilidad de REFUERZO; se decide REFUERZO si p >= 0,5).
Hiperparámetros fijados a mano (iguales al script original modelos/03_regresion_logistica.py):
C=1.0, regularización L2, solver lbfgs, max_iter=1000. Numéricas estandarizadas, categóricas en one-hot.

Pasos:
  0. Cargar cybermind_sintetico.csv y partir 75 / 25 por estudiante (misma partición en los 4 modelos).
  1. Entrenar el modelo con el 75 %.
  2. Evaluar con los datos de ENTRENAMIENTO y mostrar resultados.
  3. Evaluar con los datos de PRUEBA y mostrar resultados.
  4. Comparar entreno vs prueba, graficar y guardar en resultados/02_regresion_logistica.json y resultados/02_regresion_logistica.png.

Datos: SOLO cybermind_sintetico.csv. No se usa cybermind_sintetico_oculto.csv.
Uso:   python 02_regresion_logistica.py      (requiere pandas, numpy, scikit-learn, matplotlib)
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ---------------------------------------------------------------- 0. datos y partición
# Único archivo de datos: cybermind_sintetico.csv (NO se usa cybermind_sintetico_oculto.csv)
CARPETA = Path(__file__).resolve().parent
RUTA_DATOS = CARPETA.parents[1] / "data" / "cybermind_sintetico.csv"
SALIDA = CARPETA / "resultados"
SALIDA.mkdir(exist_ok=True)
datos = pd.read_csv(RUTA_DATOS)

CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)   # clase positiva (1) = REFUERZO, 0 = AVANCE
grupos = datos["id_estudiante"]

# Partición 75 % entrenamiento / 25 % prueba, AGRUPADA por estudiante (un estudiante nunca
# aparece en los dos conjuntos) y con semilla fija: es la misma en los 4 modelos.
ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups=grupos))
X_ent, y_ent, g_ent = X.iloc[ent], y.iloc[ent], grupos.iloc[ent]
X_pru, y_pru = X.iloc[pru], y.iloc[pru]

COLOR_ENT, COLOR_PRU = "#2a78d6", "#eb6834"   # azul = entrenamiento, naranja = prueba
NOMBRES_METRICAS = {"exactitud": "Exactitud\n(accuracy)", "precision_REFUERZO": "Precisión\n(precision)",
                    "recall_REFUERZO": "Sensibilidad\n(recall)", "F1_REFUERZO": "F1"}


def metricas(y_real, y_pred):
    """Métricas de clasificación; precisión, recall y F1 se calculan para la clase REFUERZO."""
    tn, fp, fn, tp = confusion_matrix(y_real, y_pred, labels=[0, 1]).ravel()
    return {
        "n_filas": int(len(y_real)),
        "exactitud": round(float(accuracy_score(y_real, y_pred)), 4),
        "precision_REFUERZO": round(float(precision_score(y_real, y_pred, zero_division=0)), 4),
        "recall_REFUERZO": round(float(recall_score(y_real, y_pred, zero_division=0)), 4),
        "F1_REFUERZO": round(float(f1_score(y_real, y_pred, zero_division=0)), 4),
        "matriz_confusion": {"VN_avance_ok": int(tn), "FP_refuerzo_innecesario": int(fp),
                             "FN_avance_sin_dominio": int(fn), "VP_refuerzo_ok": int(tp)},
    }


def mostrar(titulo, m):
    """Imprime las métricas de un conjunto en forma de tabla legible."""
    mc = m["matriz_confusion"]
    print(f"\n=== {titulo} — {m['n_filas']} filas ===")
    print(f"  Exactitud (accuracy) ......... {m['exactitud']:.4f}")
    print(f"  Precisión REFUERZO ........... {m['precision_REFUERZO']:.4f}")
    print(f"  Sensibilidad (recall) REFUERZO {m['recall_REFUERZO']:.4f}")
    print(f"  F1 REFUERZO .................. {m['F1_REFUERZO']:.4f}")
    print("  Matriz de confusión (filas = real, columnas = predicho):")
    print(f"                 pred AVANCE  pred REFUERZO")
    print(f"    real AVANCE   {mc['VN_avance_ok']:>9}  {mc['FP_refuerzo_innecesario']:>13}")
    print(f"    real REFUERZO {mc['FN_avance_sin_dominio']:>9}  {mc['VP_refuerzo_ok']:>13}")


def comparar(m_ent, m_pru):
    """Diferencia entrenamiento − prueba en cada métrica (positiva = el modelo rinde mejor en lo que ya vio)."""
    brecha = {k: round(m_ent[k] - m_pru[k], 4) for k in NOMBRES_METRICAS}
    print("\n=== Comparación entrenamiento vs prueba ===")
    print(f"  {'métrica':<20}{'entreno':>9}{'prueba':>9}{'diferencia':>12}")
    for k in NOMBRES_METRICAS:
        print(f"  {k:<20}{m_ent[k]:>9.4f}{m_pru[k]:>9.4f}{brecha[k]:>+12.4f}")
    return brecha


def _matriz(ax, m, titulo):
    mc = m["matriz_confusion"]
    cm = np.array([[mc["VN_avance_ok"], mc["FP_refuerzo_innecesario"]],
                   [mc["FN_avance_sin_dominio"], mc["VP_refuerzo_ok"]]])
    pct = cm / cm.sum() * 100
    ax.imshow(cm, cmap="Blues", vmin=0)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n({pct[i, j]:.1f} %)", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() * 0.55 else "#0b0b0b")
    ax.set_xticks([0, 1], ["AVANCE", "REFUERZO"])
    ax.set_yticks([0, 1], ["AVANCE", "REFUERZO"])
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(titulo, fontsize=10)


def graficar(nombre_archivo, titulo_modelo, m_ent, m_pru, mostrar_figura=False):
    """Barras entreno vs prueba por métrica + matriz de confusión de cada conjunto."""
    fig = plt.figure(figsize=(14, 4.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.7, 1, 1], wspace=0.45)
    ax = fig.add_subplot(gs[0])
    claves = list(NOMBRES_METRICAS)
    x = np.arange(len(claves))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, [m_ent[k] for k in claves], w, color=COLOR_ENT, label=f"Entrenamiento (75 %, n={m_ent['n_filas']})")
    b2 = ax.bar(x + w / 2 + 0.01, [m_pru[k] for k in claves], w, color=COLOR_PRU, label=f"Prueba (25 %, n={m_pru['n_filas']})")
    ax.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
    ax.set_xticks(x, [NOMBRES_METRICAS[k] for k in claves], fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Valor (eje desde 0,5)")
    ax.grid(axis="y", color="#e5e4df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, frameon=False, fontsize=9)
    ax.set_title("Métricas: entrenamiento vs prueba", fontsize=10)
    _matriz(fig.add_subplot(gs[1]), m_ent, "Matriz de confusión — entrenamiento")
    _matriz(fig.add_subplot(gs[2]), m_pru, "Matriz de confusión — prueba")
    fig.suptitle(titulo_modelo, fontsize=12, fontweight="bold")
    ruta = SALIDA / f"{nombre_archivo}.png"
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    print(f"\nGráfica guardada en {ruta}")
    if mostrar_figura:
        plt.show()
    else:
        plt.close(fig)


def guardar(nombre_archivo, resultado):
    ruta = SALIDA / f"{nombre_archivo}.json"
    ruta.write_text(json.dumps(resultado, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Resultados guardados en {ruta}")


# ---------------------------------------------------------------- 1. entrenar (75 %)
modelo = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                                ("num", StandardScaler(), NUMERICAS)])),
    ("logit", LogisticRegression(C=1.0, max_iter=1000)),
])
modelo.fit(X_ent, y_ent)   # se entrena SOLO con el 75 % (el escalador también se ajusta solo con el 75 %)
hiperparametros = {"C": 1.0, "regularizacion": "L2", "solver": "lbfgs", "max_iter": 1000, "umbral_probabilidad": 0.5}

# ---------------------------------------------------------------- 2. evaluar con ENTRENAMIENTO
pred_ent = modelo.predict(X_ent)
m_ent = metricas(y_ent, pred_ent)
mostrar("Regresión logística — ENTRENAMIENTO (75 %)", m_ent)

# ---------------------------------------------------------------- 3. evaluar con PRUEBA
pred_pru = modelo.predict(X_pru)
m_pru = metricas(y_pru, pred_pru)
mostrar("Regresión logística — PRUEBA (25 %)", m_pru)

# ---------------------------------------------------------------- 4. comparar, graficar y guardar
brecha = comparar(m_ent, m_pru)
graficar("02_regresion_logistica", "Regresión logística", m_ent, m_pru, mostrar_figura=False)
nombres = [n.replace("num__", "").replace("cat__", "") for n in modelo[0].get_feature_names_out()]
coeficientes = sorted(zip(nombres, modelo[-1].coef_[0]), key=lambda t: -abs(t[1]))
extra = {"intercepto": round(float(modelo[-1].intercept_[0]), 4),
         "coeficientes_estandarizados": {n: round(float(c), 4) for n, c in coeficientes}}
print("\nCoeficientes (positivo = empuja hacia REFUERZO):", extra["coeficientes_estandarizados"])
guardar("02_regresion_logistica", {"modelo": "Regresión logística", "datos": "cybermind_sintetico.csv",
                 "particion": "GroupShuffleSplit 75/25 por estudiante, random_state=0",
                 "hiperparametros": hiperparametros, "entrenamiento": m_ent, "prueba": m_pru,
                 "diferencia_entreno_menos_prueba": brecha, **extra})
