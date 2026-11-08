"""
kNN como modelo adicional, con la MISMA partición de src/verificar_dataset.py
(GroupShuffleSplit test_size=0.25, random_state=0, agrupado por estudiante).

A diferencia de src/verificar_dataset.py, aquí los hiperparámetros de kNN se eligen
con validación cruzada agrupada de 5 particiones usando SOLO el conjunto de
entrenamiento; el de prueba se toca una única vez al final.
"""
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

datos = pd.read_csv(Path(__file__).resolve().parents[2] / "data" / "cybermind_sintetico.csv")
CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)
grupos = datos["id_estudiante"]

ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups=grupos))

pipe = Pipeline([
    ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                                ("num", StandardScaler(), NUMERICAS)])),
    ("knn", KNeighborsClassifier()),
])
rejilla = {"knn__n_neighbors": [1, 3, 5, 9, 15, 25, 35, 51, 75, 101],
           "knn__weights": ["uniform", "distance"],
           "knn__p": [1, 2]}
gs = GridSearchCV(pipe, rejilla, scoring="accuracy", cv=GroupKFold(n_splits=5), n_jobs=-1)
gs.fit(X.iloc[ent], y.iloc[ent], groups=grupos.iloc[ent])

print("Mejores hiperparámetros:", gs.best_params_)
print(f"Exactitud CV (entrenamiento): {gs.best_score_:.3f}")
pred = gs.predict(X.iloc[pru])
print(f"kNN en prueba: exactitud={accuracy_score(y.iloc[pru], pred):.3f}  "
      f"F1={f1_score(y.iloc[pru], pred):.3f}  recall_REFUERZO={recall_score(y.iloc[pru], pred):.3f}")
print("\nExactitud CV por k (mejor combinación de weights/p para cada k):")
r = pd.DataFrame(gs.cv_results_)
print(r.groupby("param_knn__n_neighbors")["mean_test_score"].max().round(3).to_string())
