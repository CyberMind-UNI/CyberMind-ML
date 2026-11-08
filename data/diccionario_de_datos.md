# Diccionario de datos: dataset sintético CyberMind

**Archivos**

| Archivo | Contenido |
|---|---|
| `src/generar_dataset.py` | Generador reproducible. `python src/generar_dataset.py` (por defecto 1500 estudiantes, semilla 42). Requiere numpy y pandas. |
| `cybermind_sintetico.csv` | Dataset para entrenar y probar el modelo: 7500 filas (1500 estudiantes × 5 temas), 10 columnas. |
| `cybermind_sintetico_oculto.csv` | **No se incluye en este repositorio** (está en `.gitignore`). Lo crea `generar_dataset.py` con las variables latentes del simulador; ninguna prueba ni notebook lo usa. |
| `src/verificar_dataset.py` | Comprueba que la etiqueta es aprendible y no trivial. Requiere además scikit-learn. |

**Unidad de análisis:** un par (estudiante, tema) después de la evaluación diagnóstica. Es exactamente la solicitud que la app enviaría al servicio de decisión (`POST /v1/decision`).

## Variables de entrada (`cybermind_sintetico.csv`)

| Variable | Tipo | Rango | Cómo se generó | Por qué se incluye |
|---|---|---|---|---|
| `id_estudiante` | texto | E00000–E01499 | Secuencial. | **No es variable de entrada.** Sirve para dividir entrenamiento/prueba por estudiante y evitar fuga de información. |
| `tema` | categórica | fundamentos_redes, criptografia, seguridad_web, ingenieria_social, gestion_vulnerabilidades | Los 5 temas para cada estudiante. Cada tema tiene una dificultad relativa (−0,5 a +0,4 en escala logit). | Algunos temas son más difíciles; el mismo puntaje no significa lo mismo en todos. |
| `experiencia_previa` | categórica | ninguna (50 %), cursos (35 %), laboral (15 %) | Muestreo con esas proporciones (supuesto: público mayoritariamente principiante). | Se pregunta en el registro; da información sobre el dominio más allá de 9 preguntas. |
| `aciertos_basico` | entero | 0–3 | Respuestas a 3 preguntas de nivel básico simuladas con IRT 3PL (ver abajo). | Separa a quien domina lo básico de quien no. |
| `aciertos_intermedio` | entero | 0–3 | Igual, 3 preguntas de nivel intermedio. | Es el nivel de la unidad siguiente; la variable más ligada a la decisión. |
| `aciertos_avanzado` | entero | 0–3 | Igual, 3 preguntas de nivel avanzado. | Distingue a estudiantes que pueden avanzar con holgura. |
| `puntaje_tema` | numérica | 0–100 | Aciertos totales del tema / 9 × 100. | Resumen directo; es la variable de la regla base. |
| `puntaje_global` | numérica | 0–100 | Promedio de `puntaje_tema` del estudiante en los 5 temas. | Estima la habilidad general y reduce el ruido de solo 9 preguntas por tema. |
| `tiempo_promedio_seg` | numérica | 3–180 | Log-normal con mediana ~45 s, algo menor con más dominio. El 4 % de "adivinadores rápidos" tiene 4–10 s. | Permite detectar respuestas al azar (puntaje no confiable). |

## Variable objetivo

| Variable | Valores | Cómo se generó |
|---|---|---|
| `decision` | REFUERZO (65,4 %), AVANCE (34,6 %) | 1) Se calcula la probabilidad de aprobar la unidad siguiente (nivel intermedio) con el **dominio real** θ, no con el puntaje observado: `p = sigmoide(1,7 · 1,2 · (θ − dificultad_tema))`. 2) AVANCE si p ≥ 0,70, si no REFUERZO (criterio de mastery learning). 3) Se invierte el 5 % de las etiquetas al azar para simular desacuerdo entre evaluadores. |

El desbalance (más REFUERZO) es coherente con una población mayoritariamente principiante. Por tema va de 48,9 % (ingeniería social) a 77,1 % (criptografía).

## Modelo generador (capas)

1. **Latentes por estudiante:** experiencia → habilidad general `g ~ N(μ_exp; 0,8)` con μ = −0,5 / 0,3 / 0,9. Dominio por tema `θ = 0,75·g + N(0; 0,65)`.
2. **Respuestas al diagnóstico (IRT 3PL):** P(acierto) = 0,25 + 0,75 · sigmoide(1,7 · 1,2 · (θ − b_nivel − dificultad_tema)), con b = −1 / 0 / +1 para básico / intermedio / avanzado. El 0,25 es la probabilidad de acertar al azar con 4 opciones. Los adivinadores rápidos responden con P = 0,25.
3. **Etiqueta:** como se describe arriba.

Todos los números son supuestos declarados como constantes al inicio de `generar_dataset.py`; cambiarlos y volver a correr produce otro dataset coherente.

## Variables ocultas (`cybermind_sintetico_oculto.csv`)

`habilidad_general`, `theta_dominio_real`, `p_aprobar_unidad`, `adivinador_rapido`, `decision_sin_ruido`, `etiqueta_invertida`. Nunca se usan como entrada.

## Verificación (`verificar_dataset.py`, prueba con 25 % de estudiantes)

| Modelo | Exactitud | F1 | Recall REFUERZO |
|---|---|---|---|
| Regla: AVANCE si puntaje_tema ≥ 70 | 0,813 | 0,862 | 0,866 |
| Árbol de decisión (prof. 4, ≥50 por hoja) | 0,818 | 0,869 | 0,897 |
| Regresión logística | 0,825 | 0,874 | 0,899 |
| Techo teórico (5 % de ruido en etiqueta) | 0,947 | | |

Lectura: la etiqueta es aprendible, no es trivial, y un modelo simple ya captura casi todo lo que estas variables permiten. La distancia al techo se debe sobre todo a que 9 preguntas por tema miden el dominio con ruido. Estas cifras validan el dataset, no el modelo final ni el efecto pedagógico.
