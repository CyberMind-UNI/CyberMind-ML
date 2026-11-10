"""
Regenera en MÁXIMA CALIDAD las 3 gráficas que usa la presentación final, con el mismo diseño de las originales.

Guarda cada gráfica dos veces en docs/graficas_presentacion/:
  - .svg  vectorial: se ve nítida a cualquier tamaño (es la que usan las diapositivas)
  - .png  a 300 ppp, para Word o PowerPoint

Gráficas (los datos salen de los resultados ya guardados; no se reentrena nada):
  1. importancia_variables_gbdt   ← docs/analisis_variables.json (importancia por permutación del GBDT)
  2. entreno_vs_prueba            ← pruebas/fase2_entreno_vs_prueba/resultados/0*.json
  3. seleccion_modelo             ← pruebas/fase3_seleccion_modelo/resultados/seleccion_modelo.json

Datos: solo resultados derivados de data/cybermind_sintetico.csv. No se usa cybermind_sintetico_oculto.csv.
Uso:   python src/graficas_presentacion.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

RAIZ = Path(__file__).resolve().parents[1]
SALIDA = RAIZ / "docs" / "graficas_presentacion"
SALIDA.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Inter", "DejaVu Sans"],
    "svg.fonttype": "path",        # el texto va como trazos: se ve igual en cualquier equipo
    "figure.dpi": 100,
})
GRIS_REJILLA = "#e5e4df"
COLOR_ENT, COLOR_PRU = "#2a78d6", "#eb6834"


def limpiar(ax, eje="y"):
    ax.grid(axis=eje, color=GRIS_REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def guardar(fig, nombre):
    for ext, extra in (("svg", {}), ("png", {"dpi": 300})):
        fig.savefig(SALIDA / f"{nombre}.{ext}", bbox_inches="tight", facecolor="white", **extra)
    plt.close(fig)
    print("Guardada", SALIDA / f"{nombre}.svg", "y .png (300 ppp)")


# ---------------------------------------------------------------- 1. importancia de variables (GBDT)
analisis = json.loads((RAIZ / "docs" / "analisis_variables.json").read_text(encoding="utf-8"))
imp = pd.DataFrame(analisis["perm_importancia_gbdt"], index=["media", "desv"]).T * 100   # a puntos %
imp = imp.sort_values("media")
fig, ax = plt.subplots(figsize=(6.2, 4.8))
ax.barh(imp.index, imp["media"], xerr=imp["desv"], color=COLOR_PRU, height=0.55,
        error_kw={"elinewidth": 0.9, "capsize": 0})
ax.set_xlabel("Pérdida de exactitud al revolverla (puntos %)")
ax.set_title("Importancia por permutación (GBDT)\ndatos de prueba", fontsize=10)
limpiar(ax, "x")
guardar(fig, "importancia_variables_gbdt")

# ---------------------------------------------------------------- 2. entrenamiento vs prueba (fase 2)
FASE2 = RAIZ / "pruebas" / "fase2_entreno_vs_prueba" / "resultados"
archivos = ["01_arbol_decision", "02_regresion_logistica", "03_knn", "04_gbdt"]
cortos = ["Árbol", "Regresión\nlogística", "kNN", "GBDT"]
res = [json.loads((FASE2 / f"{a}.json").read_text(encoding="utf-8")) for a in archivos]
ent = np.array([r["entrenamiento"]["exactitud"] for r in res])
pru = np.array([r["prueba"]["exactitud"] for r in res])

# Referencia: exactitud de responder siempre REFUERZO en la misma partición de prueba
datos = pd.read_csv(RAIZ / "data" / "cybermind_sintetico.csv")
y = (datos["decision"] == "REFUERZO").astype(int)
_, idx_pru = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(datos, y, groups=datos["id_estudiante"]))
base = y.iloc[idx_pru].mean()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2, 1], "wspace": 0.3})
x = np.arange(len(archivos))
w = 0.38
b1 = ax1.bar(x - w / 2 - 0.01, ent, w, color=COLOR_ENT, label="Entrenamiento (75 %)")
b2 = ax1.bar(x + w / 2 + 0.01, pru, w, color=COLOR_PRU, label="Prueba (25 %)")
ax1.bar_label(b1, fmt="%.3f", fontsize=8, padding=2)
ax1.bar_label(b2, fmt="%.3f", fontsize=8, padding=2)
ax1.axhline(base, color="#52514e", linestyle="--", linewidth=1)
ax1.text(len(archivos) - 0.5, base + 0.005, f"siempre REFUERZO = {base:.3f}", ha="right", fontsize=8, color="#52514e")
ax1.set_xticks(x, cortos, fontsize=9)
ax1.set_ylim(0.6, 1.0)
ax1.set_ylabel("Exactitud (eje desde 0,6)")
ax1.set_title("Exactitud por modelo: entrenamiento vs prueba", fontsize=10)
ax1.legend(loc="upper left", frameon=False, fontsize=9)
dif = (ent - pru) * 100
b3 = ax2.barh(x, dif, color="#52514e", height=0.5)
ax2.bar_label(b3, fmt="%+.1f pts", fontsize=8, padding=3)
ax2.axvline(0, color="#0b0b0b", linewidth=0.8)
ax2.set_yticks(x, cortos, fontsize=9)
ax2.invert_yaxis()
ax2.set_xlim(min(-2.5, dif.min() - 1.5), max(3, dif.max() + 2))
ax2.set_xlabel("Puntos porcentuales")
ax2.set_title("Diferencia de exactitud (entreno − prueba)", fontsize=10)
limpiar(ax1, "y")
limpiar(ax2, "x")
guardar(fig, "entreno_vs_prueba")

# ---------------------------------------------------------------- 3. selección del modelo (fase 3)
sel = json.loads((RAIZ / "pruebas" / "fase3_seleccion_modelo" / "resultados" / "seleccion_modelo.json")
                 .read_text(encoding="utf-8"))["por_particion"]
modelos = list(sel)
colores = {"Árbol de decisión": "#8a94a6", "Regresión logística": "#2a78d6", "kNN": "#7fb2ff", "GBDT": "#eb6834"}
fig, ejes = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, k, titulo in [(ejes[0], "exactitud", "Exactitud (mayor = mejor)"), (ejes[1], "log_loss", "Log loss (menor = mejor)")]:
    for i, n in enumerate(modelos):
        v = sel[n][k]
        ax.scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v,
                   s=14, color=colores[n], alpha=0.7)
        ax.hlines(np.mean(v), i - 0.3, i + 0.3, color="#1b2738", linewidth=2)
    ax.set_xticks(range(len(modelos)), modelos, fontsize=9)
    ax.set_title(f"{titulo}\n30 particiones por estudiante", fontsize=10)
    limpiar(ax, "y")
fig.tight_layout()
guardar(fig, "seleccion_modelo")
