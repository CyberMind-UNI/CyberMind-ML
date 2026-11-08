"""
Generador del dataset sintético de CyberMind (decisión REFUERZO / AVANCE).

Cada fila es un par (estudiante, tema): lo que el servicio de IA recibiría tras
la evaluación diagnóstica inicial, más la decisión que debería tomar.

El dataset se construye en tres capas:
  1. Variables latentes (no observables): experiencia, habilidad general y
     dominio real por tema (theta).
  2. Variables observables: respuestas al diagnóstico simuladas con un modelo
     de Teoría de Respuesta al Ítem de 3 parámetros (3PL) y tiempos de
     respuesta. Son las únicas que el modelo puede usar.
  3. Etiqueta: AVANCE si la probabilidad de aprobar la unidad siguiente,
     calculada con el dominio real, es >= 0,70; si no, REFUERZO. Después se
     invierte un 5 % de etiquetas para simular desacuerdo entre expertos.

Uso:
    python generar_dataset.py                  # 1500 estudiantes, semilla 42
    python generar_dataset.py --n 3000 --semilla 7 --salida otra_carpeta
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TEMAS = {
    # tema: dificultad relativa del tema (escala logit; >0 = más difícil)
    "fundamentos_redes": -0.30,
    "criptografia": 0.40,
    "seguridad_web": 0.30,
    "ingenieria_social": -0.50,
    "gestion_vulnerabilidades": 0.20,
}

EXPERIENCIA = {
    # nivel: (proporción en la población, media de habilidad general)
    "ninguna": (0.50, -0.50),
    "cursos": (0.35, 0.30),
    "laboral": (0.15, 0.90),
}

# Ítems del diagnóstico por tema: 3 por nivel, con su dificultad base (logit)
NIVELES_ITEM = {"basico": -1.0, "intermedio": 0.0, "avanzado": 1.0}
ITEMS_POR_NIVEL = 3
DISCRIMINACION = 1.2      # parámetro a del modelo 3PL
ADIVINANZA = 0.25         # parámetro c: preguntas de 4 opciones
D = 1.7                   # constante de escala de la IRT

UMBRAL_DOMINIO = 0.70     # probabilidad mínima de aprobar para recomendar AVANCE
DIFICULTAD_UNIDAD = 0.0   # la unidad siguiente es de nivel intermedio
RUIDO_ETIQUETA = 0.05     # fracción de etiquetas invertidas
PROP_ADIVINADORES = 0.04  # estudiantes que responden al azar y rápido


def sigmoide(x):
    return 1.0 / (1.0 + np.exp(-x))


def generar(n_estudiantes: int, semilla: int):
    rng = np.random.default_rng(semilla)
    temas = list(TEMAS)
    n_temas = len(temas)

    # --- Capa 1: latentes por estudiante ---------------------------------
    niveles = list(EXPERIENCIA)
    probs = [EXPERIENCIA[k][0] for k in niveles]
    experiencia = rng.choice(niveles, size=n_estudiantes, p=probs)
    media_g = np.array([EXPERIENCIA[e][1] for e in experiencia])
    habilidad_general = rng.normal(media_g, 0.8)
    adivinador = rng.random(n_estudiantes) < PROP_ADIVINADORES

    # Dominio real por tema: mezcla de habilidad general y variación propia
    theta = 0.75 * habilidad_general[:, None] + rng.normal(0, 0.65, (n_estudiantes, n_temas))
    dif_tema = np.array([TEMAS[t] for t in temas])

    # --- Capa 2: observables del diagnóstico ------------------------------
    aciertos = {}
    for nivel, b in NIVELES_ITEM.items():
        p = ADIVINANZA + (1 - ADIVINANZA) * sigmoide(
            D * DISCRIMINACION * (theta - b - dif_tema[None, :])
        )
        p = np.where(adivinador[:, None], ADIVINANZA, p)
        aciertos[nivel] = rng.binomial(ITEMS_POR_NIVEL, p)

    total_items = ITEMS_POR_NIVEL * len(NIVELES_ITEM)
    puntaje_tema = sum(aciertos.values()) / total_items * 100
    puntaje_global = puntaje_tema.mean(axis=1, keepdims=True).repeat(n_temas, axis=1)

    # Tiempo medio por pregunta (s): log-normal, algo menor con más dominio;
    # los adivinadores responden muy rápido
    tiempo = 45 * np.exp(-0.15 * theta + rng.normal(0, 0.35, theta.shape))
    tiempo = np.where(adivinador[:, None], rng.uniform(4, 10, theta.shape), tiempo)
    tiempo = np.clip(tiempo, 3, 180)


    # --- Capa 3: etiqueta --------------------------------------------------
    p_aprobar = sigmoide(D * DISCRIMINACION * (theta - DIFICULTAD_UNIDAD - dif_tema[None, :]))
    decision_limpia = np.where(p_aprobar >= UMBRAL_DOMINIO, "AVANCE", "REFUERZO")
    invertir = rng.random(theta.shape) < RUIDO_ETIQUETA
    decision = np.where(
        invertir, np.where(decision_limpia == "AVANCE", "REFUERZO", "AVANCE"), decision_limpia
    )

    # --- Formato largo: una fila por (estudiante, tema) -------------------
    ids = np.array([f"E{i:05d}" for i in range(n_estudiantes)])
    filas = {
        "id_estudiante": ids.repeat(n_temas),
        "tema": np.tile(temas, n_estudiantes),
        "experiencia_previa": experiencia.repeat(n_temas),
        "aciertos_basico": aciertos["basico"].ravel(),
        "aciertos_intermedio": aciertos["intermedio"].ravel(),
        "aciertos_avanzado": aciertos["avanzado"].ravel(),
        "puntaje_tema": np.round(puntaje_tema.ravel(), 1),
        "puntaje_global": np.round(puntaje_global.ravel(), 1),
        "tiempo_promedio_seg": np.round(tiempo.ravel(), 1),
        "decision": decision.ravel(),
    }
    datos = pd.DataFrame(filas)

    oculto = pd.DataFrame({
        "id_estudiante": filas["id_estudiante"],
        "tema": filas["tema"],
        "habilidad_general": np.round(habilidad_general.repeat(n_temas), 4),
        "theta_dominio_real": np.round(theta.ravel(), 4),
        "p_aprobar_unidad": np.round(p_aprobar.ravel(), 4),
        "adivinador_rapido": adivinador.repeat(n_temas),
        "decision_sin_ruido": decision_limpia.ravel(),
        "etiqueta_invertida": invertir.ravel(),
    })
    return datos, oculto


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1500, help="número de estudiantes")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--salida", type=Path, default=Path(__file__).resolve().parents[1] / "data")
    args = ap.parse_args()

    datos, oculto = generar(args.n, args.semilla)
    args.salida.mkdir(parents=True, exist_ok=True)
    datos.to_csv(args.salida / "cybermind_sintetico.csv", index=False)
    oculto.to_csv(args.salida / "cybermind_sintetico_oculto.csv", index=False)

    print(f"Filas: {len(datos)} ({args.n} estudiantes x {len(TEMAS)} temas), semilla {args.semilla}")
    print("Distribución de la etiqueta:")
    print(datos["decision"].value_counts(normalize=True).round(3).to_string())
    print("% REFUERZO por tema:")
    print((datos.groupby("tema")["decision"].apply(lambda s: (s == "REFUERZO").mean()) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
