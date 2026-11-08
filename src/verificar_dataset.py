"""
Verificación rápida del dataset sintético: comprueba que la etiqueta se puede
aprender a partir de las variables observables y que no es trivial.

Compara, con partición por estudiante (las 5 filas de un estudiante quedan
juntas en entrenamiento o en prueba):
  - Regla fija: AVANCE si puntaje_tema >= 70.
  - Árbol de decisión (CART) de profundidad 4.
  - Regresión logística (referencia lineal).

No es el entrenamiento del modelo final; solo valida el dataset.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

# Único archivo de datos: data/cybermind_sintetico.csv (no se usa el archivo oculto)
RUTA_DATOS = Path(__file__).resolve().parents[1] / "data" / "cybermind_sintetico.csv"
datos = pd.read_csv(RUTA_DATOS)

CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)

split = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
ent, pru = next(split.split(X, y, groups=datos["id_estudiante"]))


def informe(nombre, pred):
    print(f"{nombre:<28} exactitud={accuracy_score(y.iloc[pru], pred):.3f}  "
          f"F1={f1_score(y.iloc[pru], pred):.3f}  "
          f"recall_REFUERZO={recall_score(y.iloc[pru], pred):.3f}")


informe("Regla puntaje_tema >= 70", (X.iloc[pru]["puntaje_tema"] < 70).astype(int))

arbol = make_pipeline(
    ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS)], remainder="passthrough",
                      verbose_feature_names_out=False),
    DecisionTreeClassifier(max_depth=4, min_samples_leaf=50, random_state=0),
)
arbol.fit(X.iloc[ent], y.iloc[ent])
informe("Árbol de decisión (prof. 4)", arbol.predict(X.iloc[pru]))

logistica = make_pipeline(
    ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                       ("num", StandardScaler(), NUMERICAS)]),
    LogisticRegression(max_iter=1000),
)
logistica.fit(X.iloc[ent], y.iloc[ent])
informe("Regresión logística", logistica.predict(X.iloc[pru]))

print(f"\nExactitud de predecir siempre REFUERZO: {y.iloc[pru].mean():.3f}")

print("\nReglas aprendidas por el árbol (1 = REFUERZO):")
nombres = arbol[0].get_feature_names_out()
print(export_text(arbol[-1], feature_names=list(nombres)))
