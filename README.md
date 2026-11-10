# CyberMind · Decisión de refuerzo o avance con aprendizaje de máquina

Proyecto universitario de Inteligencia Artificial. **CyberMind** es una plataforma de aprendizaje de ciberseguridad; este repositorio contiene **solo el componente de IA**: el servicio que, después de una evaluación diagnóstica, decide si un estudiante debe hacer **REFUERZO** de un tema o puede pasar al **AVANCE**.

El repositorio muestra el avance de las pruebas: cómo se generó el dataset, qué modelos se probaron, con qué resultados y cómo reproducir todo.

> **Estado (noviembre 2026):** dataset sintético validado, 5 métodos comparados (regla fija + 4 modelos de ML) y evaluación entrenamiento vs. prueba terminada. **Todavía no se ha elegido el modelo final**: el equipo espera la recomendación del profesor.

---

## Contenido

1. [Problema](#1-problema)
2. [Dataset sintético](#2-dataset-sintético)
3. [Modelos probados](#3-modelos-probados)
4. [Resultados](#4-resultados)
5. [Estructura del repositorio](#5-estructura-del-repositorio)
6. [Cómo reproducir](#6-cómo-reproducir)
7. [Documentación](#7-documentación)
8. [Próximos pasos](#8-próximos-pasos)
9. [Equipo](#9-equipo)

---

## 1. Problema

Tras la retroalimentación del Avance II, el alcance se redujo a **una sola decisión mínima y verificable**:

| | |
|---|---|
| **Entrada** | Resultado del diagnóstico de un estudiante en un tema: 9 preguntas (3 básico, 3 intermedio, 3 avanzado), puntajes, tiempo promedio, tema y experiencia previa. |
| **Salida** | `REFUERZO` o `AVANCE` para ese par (estudiante, tema). |
| **Tipo de tarea** | Aprendizaje supervisado, clasificación binaria. Clase positiva: `REFUERZO`. |
| **Temas** | Fundamentos de redes, criptografía, seguridad web, ingeniería social, gestión de vulnerabilidades. |

La IA está **desacoplada de la app**: se plantea como un servicio REST independiente (`POST /v1/decision`, `GET /salud`). Si el servicio no responde, la app usa una ruta por defecto (refuerzo si el puntaje es menor a 70 %).

## 2. Dataset sintético

No hay datos reales de estudiantes, así que se generó un **dataset sintético reproducible** (semilla 42):

- **7.500 filas** = 1.500 estudiantes × 5 temas.
- Respuestas simuladas con **teoría de respuesta al ítem (IRT 3PL)**, con dificultad distinta por tema y por nivel de pregunta, e incluye un 4 % de "adivinadores rápidos".
- La etiqueta sale del **dominio real** simulado (criterio de mastery learning: AVANCE si la probabilidad de aprobar la unidad siguiente es ≥ 0,70) y se invierte un 5 % al azar para simular desacuerdo entre evaluadores.
- Distribución: **65,4 % REFUERZO / 34,6 % AVANCE**.

Variables de entrada (8): `tema`, `experiencia_previa`, `aciertos_basico`, `aciertos_intermedio`, `aciertos_avanzado`, `puntaje_tema`, `puntaje_global`, `tiempo_promedio_seg`. El campo `id_estudiante` no es entrada: solo se usa para que la partición entrenamiento/prueba sea **por estudiante** y no haya fuga de información.

Detalle completo de cada variable: [`data/diccionario_de_datos.md`](data/diccionario_de_datos.md).

> **Regla del proyecto:** el único archivo de datos que usan las pruebas y los notebooks es `data/cybermind_sintetico.csv`. El generador también puede producir `cybermind_sintetico_oculto.csv` (variables latentes del simulador), pero ese archivo **no está en el repositorio** (`.gitignore`) y **no se usa** en ninguna prueba.

## 3. Modelos probados

Todos usan la **misma partición**: 75 % entrenamiento / 25 % prueba, agrupada por estudiante (`GroupShuffleSplit`, `random_state=0`): 5.625 filas para entrenar y 1.875 para probar.

| # | Método | Tipo | Configuración |
|---|---|---|---|
| 0 | Regla fija | Línea base (no se entrena) | AVANCE si `puntaje_tema` ≥ 70 |
| 1 | Árbol de decisión (CART) | ML, interpretable | profundidad 4, ≥ 50 muestras por hoja |
| 2 | Regresión logística | ML, lineal | regularización L2, C = 1, variables estandarizadas |
| 3 | k vecinos más cercanos (kNN) | ML, basado en instancias | k = 51, distancia Manhattan; elegido con validación cruzada agrupada (5 particiones) |
| 4 | Gradient Boosting (GBDT) | ML, ensamble de árboles | 100 árboles, profundidad 3 (también se probó con búsqueda de hiperparámetros) |

## 4. Resultados

### Fase 1 · Comparación de métodos (conjunto de prueba)

| Método | Exactitud | IC 95 % | F1 (REFUERZO) | Recall (REFUERZO) |
|---|---|---|---|---|
| Siempre REFUERZO (referencia) | 0,676 | | | |
| Regla fija | 0,813 | [0,795; 0,832] | 0,862 | 0,866 |
| Árbol de decisión | 0,818 | [0,798; 0,836] | 0,869 | 0,898 |
| Regresión logística | 0,825 | [0,806; 0,842] | 0,874 | 0,899 |
| kNN | 0,830 | [0,813; 0,848] | 0,877 | 0,894 |
| GBDT | 0,831 | [0,813; 0,849] | 0,878 | 0,902 |

El IC 95 % se calculó con bootstrap por estudiante. **Los intervalos se solapan**: con estos datos los modelos quedan prácticamente empatados.

![Exactitud por método](pruebas/fase1_comparacion_modelos/resultados/comparacion_exactitud.png)

*La gráfica muestra los 4 primeros métodos (cuaderno de la fase 1); GBDT se añadió después con `05_gbdt.py`.*

### Fase 2 · Entrenamiento (75 %) vs. prueba (25 %)

Se pidió comprobar si los modelos sobreajustan comparando su desempeño en entrenamiento y en prueba.

| Modelo | Exactitud entreno | Exactitud prueba | Diferencia | F1 entreno | F1 prueba |
|---|---|---|---|---|---|
| Árbol de decisión (CART) | 0,814 | 0,818 | −0,003 | 0,861 | 0,869 |
| Regresión logística | 0,810 | 0,825 | −0,015 | 0,857 | 0,874 |
| kNN | 0,823 | 0,830 | −0,008 | 0,866 | 0,877 |
| GBDT | 0,836 | 0,830 | +0,006 | 0,877 | 0,878 |

**Lectura:** ningún modelo muestra sobreajuste (diferencias menores a 1,5 puntos). En prueba los cuatro quedan entre **0,818 y 0,830** de exactitud.

![Entrenamiento vs. prueba](pruebas/fase2_entreno_vs_prueba/resultados/comparacion_4_modelos.png)

Cada modelo tiene además su propia figura (matriz de confusión y métricas) en [`pruebas/fase2_entreno_vs_prueba/resultados/`](pruebas/fase2_entreno_vs_prueba/resultados/).

### Relevancia de variables

`puntaje_tema` es la variable que más aporta (solo con ella se obtiene 0,813 en prueba frente a 0,825 con las 8). Quitar cualquier otra variable cambia la exactitud en menos de 0,5 puntos porque se solapan entre sí (`puntaje_tema` es la suma de los `aciertos_*`). El análisis completo está en [`docs/`](docs/).

![Relevancia de variables](docs/relevancia_variables.png)

> Estos resultados validan el procedimiento sobre **datos sintéticos**; no miden todavía el efecto pedagógico con estudiantes reales.

## 5. Estructura del repositorio

```
CyberMind-ML/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── cybermind_sintetico.csv          # dataset (7.500 filas, 10 columnas)
│   └── diccionario_de_datos.md          # descripción de variables y etiqueta
├── src/
│   ├── generar_dataset.py               # genera el dataset (semilla 42)
│   └── verificar_dataset.py             # comprueba que la etiqueta es aprendible
├── pruebas/
│   ├── fase1_comparacion_modelos/       # regla fija + 4 modelos, métricas en prueba
│   │   ├── 01_regla_fija.py
│   │   ├── 02_arbol_decision.py
│   │   ├── 03_regresion_logistica.py
│   │   ├── 04_knn.py
│   │   ├── 05_gbdt.py
│   │   ├── comparar_knn.py
│   │   └── resultados/                  # JSON por método + gráficas
│   └── fase2_entreno_vs_prueba/         # 4 modelos: entrenamiento vs. prueba
│       ├── 01_arbol_decision.py
│       ├── 02_regresion_logistica.py
│       ├── 03_knn.py
│       ├── 04_gbdt.py
│       └── resultados/                  # JSON + PNG por modelo y comparación
├── notebooks/
│   ├── 01_generacion_dataset.ipynb      # genera y explora el dataset
│   ├── 02_pruebas_modelos.ipynb         # fase 1 en un solo cuaderno
│   └── 03_entreno_vs_prueba.ipynb       # fase 2 en un solo cuaderno
└── docs/                                # documentos entregados al profesor
```

## 6. Cómo reproducir

Requisitos: Python 3.10 o superior.

```bash
git clone https://github.com/CyberMind-UNI/CyberMind-ML.git
cd CyberMind-ML
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Ejecutar las pruebas (desde la raíz del repositorio):

```bash
# Validar el dataset
python src/verificar_dataset.py

# Fase 1: comparación de métodos
python pruebas/fase1_comparacion_modelos/01_regla_fija.py
python pruebas/fase1_comparacion_modelos/02_arbol_decision.py
python pruebas/fase1_comparacion_modelos/03_regresion_logistica.py
python pruebas/fase1_comparacion_modelos/04_knn.py
python pruebas/fase1_comparacion_modelos/05_gbdt.py

# Fase 2: entrenamiento vs. prueba
python pruebas/fase2_entreno_vs_prueba/01_arbol_decision.py
python pruebas/fase2_entreno_vs_prueba/02_regresion_logistica.py
python pruebas/fase2_entreno_vs_prueba/03_knn.py
python pruebas/fase2_entreno_vs_prueba/04_gbdt.py
```

Cada script guarda sus resultados en la carpeta `resultados/` que tiene al lado. Con la semilla fija, los números salen iguales a los de este README.

Notebooks: `jupyter notebook notebooks/`, o abrirlos en Google Colab y subir `data/cybermind_sintetico.csv` al panel de archivos.

Regenerar el dataset (opcional, sale idéntico con la semilla 42):

```bash
python src/generar_dataset.py
```

## 7. Documentación

| Documento | Contenido |
|---|---|
| [`docs/Avance_II_CyberMind_corregido.docx`](docs/Avance_II_CyberMind_corregido.docx) | Planeación corregida tras la retroalimentación del Avance II |
| [`docs/Resumen_proyecto_CyberMind.docx`](docs/Resumen_proyecto_CyberMind.docx) | Resumen general del proyecto |
| [`docs/Documentacion_pruebas_modelos.docx`](docs/Documentacion_pruebas_modelos.docx) | Documentación de la fase 1 de pruebas |
| [`docs/Documentacion_CyberMind_problema_variables_modelos.docx`](docs/Documentacion_CyberMind_problema_variables_modelos.docx) | Problema, justificación de variables, modelos y comparación |
| [`docs/analisis_variables.json`](docs/analisis_variables.json) | Datos del análisis de relevancia de variables |

## 8. Próximos pasos

- [ ] Elegir el modelo final con la recomendación del profesor.
- [ ] Cerrar la justificación de variables (posible retiro de `aciertos_basico`).
- [ ] Construir el servicio REST de decisión (`POST /v1/decision`).
- [ ] Validar con datos reales cuando estén disponibles.

## 9. Equipo

- David Escobar
- Nicolás Sotelo
- Juan Tellez

Proyecto académico de Inteligencia Artificial, 2026.
