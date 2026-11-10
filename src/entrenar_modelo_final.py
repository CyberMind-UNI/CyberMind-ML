"""
Entrena el modelo FINAL de CyberMind (GBDT) y lo exporta en UN solo archivo para la app del celular:

    modelo/cybermind_gbdt.onnx

Por qué ONNX y no .pkl o .h5:
  - .pkl solo lo puede abrir Python, y en el celular no hay Python.
  - .h5 es el formato de Keras/TensorFlow (redes neuronales); un GBDT de scikit-learn no se guarda así.
  - .onnx es un formato abierto que ONNX Runtime ejecuta directamente en Android, iOS, Flutter y React Native,
    sin internet y sin servidor.

El archivo incluye el preprocesamiento (one-hot de las categóricas): la app le pasa las 8 variables tal cual
y recibe la decisión y la probabilidad. Dentro del mismo archivo (metadata_props) van la versión, el umbral,
los valores válidos de cada variable categórica y las métricas de referencia.

Pasos:
  1. Cargar data/cybermind_sintetico.csv (NO se usa cybermind_sintetico_oculto.csv).
  2. Medir el modelo con la partición 75/25 por estudiante de siempre (solo como referencia).
  3. Entrenar el modelo final con el 100 % de los datos.
  4. Convertirlo a ONNX y comprobar que da las mismas probabilidades que scikit-learn en las 7.500 filas.

Hiperparámetros: los elegidos en la fase 2 y confirmados en la fase 3 (pruebas/fase3_seleccion_modelo).
Nota: las variables numéricas entran sin estandarizar. Un árbol no cambia sus decisiones por la escala,
y así el modelo del celular reproduce exactamente al de Python (con el escalado, el redondeo a float32
movía algunas probabilidades hasta 0,01).

Uso:  pip install skl2onnx onnxruntime
      python src/entrenar_modelo_final.py
"""
import json
from datetime import date
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import sklearn
import skl2onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, StringTensorType
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, log_loss, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "modelo"
SALIDA.mkdir(exist_ok=True)
RUTA_MODELO = SALIDA / "cybermind_gbdt.onnx"
VERSION = "1.0.0"
UMBRAL = 0.5   # REFUERZO si la probabilidad de REFUERZO es >= 0,5

# ---------------------------------------------------------------- 1. datos
datos = pd.read_csv(RAIZ / "data" / "cybermind_sintetico.csv")   # único archivo de datos permitido
CATEGORICAS = ["tema", "experiencia_previa"]
NUMERICAS = ["aciertos_basico", "aciertos_intermedio", "aciertos_avanzado", "puntaje_tema",
             "puntaje_global", "tiempo_promedio_seg"]
X = datos[CATEGORICAS + NUMERICAS]
y = (datos["decision"] == "REFUERZO").astype(int)   # 1 = REFUERZO, 0 = AVANCE
grupos = datos["id_estudiante"]


def crear_modelo():
    return Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAS),
                                    ("num", "passthrough", NUMERICAS)])),
        ("gbdt", GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=3,
                                            min_samples_leaf=20, subsample=0.8, random_state=0)),
    ])


# ---------------------------------------------------------------- 2. referencia con 75/25
ent, pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, grupos))
p = crear_modelo().fit(X.iloc[ent], y.iloc[ent]).predict_proba(X.iloc[pru])[:, 1]
y_pru = y.iloc[pru]
referencia = {
    "particion": "GroupShuffleSplit 75/25 por estudiante, random_state=0",
    "exactitud": round(float(accuracy_score(y_pru, p >= UMBRAL)), 4),
    "auc": round(float(roc_auc_score(y_pru, p)), 4),
    "log_loss": round(float(log_loss(y_pru, p)), 4),
    "recall_REFUERZO": round(float(recall_score(y_pru, p >= UMBRAL)), 4),
}
print("Referencia en prueba (25 %):", referencia)

# ---------------------------------------------------------------- 3. modelo final con el 100 %
modelo = crear_modelo().fit(X, y)

# ---------------------------------------------------------------- 4. convertir a ONNX y verificar
entradas = [(c, StringTensorType([None, 1])) for c in CATEGORICAS] + \
           [(c, FloatTensorType([None, 1])) for c in NUMERICAS]
onx = convert_sklearn(modelo, "cybermind_gbdt", initial_types=entradas,
                      options={id(modelo): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})


def a_onnx(df):
    """Convierte un DataFrame al diccionario de entradas que espera el modelo ONNX."""
    feed = {c: df[[c]].astype(str).to_numpy() for c in CATEGORICAS}
    feed.update({c: df[[c]].astype(np.float32).to_numpy() for c in NUMERICAS})
    return feed


sesion = ort.InferenceSession(onx.SerializeToString(), providers=["CPUExecutionProvider"])
salidas = [o.name for o in sesion.get_outputs()]
etiqueta_onnx, prob_onnx = sesion.run(None, a_onnx(X))
dif = float(np.max(np.abs(prob_onnx[:, 1] - modelo.predict_proba(X)[:, 1])))
iguales = int(np.sum(etiqueta_onnx.ravel() == modelo.predict(X)))
print(f"ONNX vs Python: diferencia máxima de probabilidad = {dif:.1e}; "
      f"decisiones iguales en {iguales} de {len(X)} filas")
assert dif < 1e-4 and iguales == len(X), "El modelo ONNX no reproduce al de Python"

# ---------------------------------------------------------------- 5. información para la app, dentro del archivo
categorias = {c: [str(v) for v in cats]
              for c, cats in zip(CATEGORICAS, modelo[0].named_transformers_["cat"].categories_)}
info = {
    "version": VERSION,
    "fecha_entrenamiento": date.today().isoformat(),
    "datos": "cybermind_sintetico.csv, 7.500 filas (100 % para el modelo final)",
    "entradas": json.dumps({**{c: {"tipo": "string [1,1]", "valores": categorias[c]} for c in CATEGORICAS},
                            **{c: {"tipo": "float32 [1,1]"} for c in NUMERICAS}}, ensure_ascii=False),
    "salidas": json.dumps({salidas[0]: "int64 [1]: 1 = REFUERZO, 0 = AVANCE",
                           salidas[1]: "float32 [1,2]: [prob. AVANCE, prob. REFUERZO]"}, ensure_ascii=False),
    "umbral_REFUERZO": str(UMBRAL),
    "ruta_por_defecto": "si el modelo no carga: REFUERZO si puntaje_tema < 70",
    "metricas_referencia": json.dumps(referencia, ensure_ascii=False),
    "versiones": json.dumps({"scikit-learn": sklearn.__version__, "skl2onnx": skl2onnx.__version__}),
}
for clave, valor in info.items():
    prop = onx.metadata_props.add()
    prop.key, prop.value = clave, valor
onx.doc_string = "CyberMind: decide REFUERZO o AVANCE tras el diagnóstico de un tema (GBDT)."
RUTA_MODELO.write_bytes(onx.SerializeToString())
print(f"Guardado {RUTA_MODELO} ({RUTA_MODELO.stat().st_size / 1024:.0f} KB)")
print("Valores válidos:", categorias)
