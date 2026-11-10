"""
Prueba el archivo modelo/cybermind_gbdt.onnx tal como lo usaría la app: sin scikit-learn, solo ONNX Runtime.
Es el mismo flujo que se programa en el celular (onnxruntime-android, onnxruntime-objc, onnxruntime-react-native
o el paquete de Flutter): cargar el archivo, armar las 8 entradas y leer la probabilidad de REFUERZO.

Uso:  pip install onnxruntime
      python src/probar_modelo_onnx.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort

RUTA = Path(__file__).resolve().parents[1] / "modelo" / "cybermind_gbdt.onnx"
sesion = ort.InferenceSession(str(RUTA), providers=["CPUExecutionProvider"])

# Información guardada dentro del propio archivo
meta = sesion.get_modelmeta().custom_metadata_map
print("Versión del modelo:", meta["version"], "| umbral:", meta["umbral_REFUERZO"])

# Un estudiante de ejemplo (resultado de su diagnóstico en un tema)
estudiante = {
    "tema": "criptografia",               # valores válidos en meta["entradas"]
    "experiencia_previa": "cursos",
    "aciertos_basico": 3,
    "aciertos_intermedio": 1,
    "aciertos_avanzado": 0,
    "puntaje_tema": 44.4,
    "puntaje_global": 61.0,
    "tiempo_promedio_seg": 38.0,
}

entradas = {k: np.array([[v]], dtype=object if isinstance(v, str) else np.float32) for k, v in estudiante.items()}
etiqueta, probabilidades = sesion.run(None, entradas)
p_refuerzo = float(probabilidades[0, 1])
decision = "REFUERZO" if p_refuerzo >= float(meta["umbral_REFUERZO"]) else "AVANCE"
print(f"Decisión: {decision} (probabilidad de REFUERZO = {p_refuerzo:.2f})")
