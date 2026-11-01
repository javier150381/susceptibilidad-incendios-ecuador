# Susceptibilidad de incendios forestales en Ecuador

Código de investigación para comparar Random Forest y una MLP con atención,
analizar SHAP, revisar la estacionalidad y auditar los datos del proyecto.

## Ejecutar el proceso completo en Colab

Cada cuaderno contiene el código visible en celdas editables, comentarios,
inspección de los CSV, parámetros, entrenamiento o análisis, métricas y gráficos.
No requiere cargar ZIP ni clonar scripts. Abre un cuaderno, ejecuta las celdas
en orden y carga únicamente los CSV que indique. No se incluyen los datasets.

**Empieza por la comparación con pseudoausencias cercanas (cuaderno 06).**

| Cuaderno | Ejecución |
|---|---|
| [Prueba inicial con ausencias sintéticas](notebooks/paso_a_paso/01_demo_sintetica.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/01_demo_sintetica.ipynb) |
| [Random Forest con presencias y pseudoausencias GEE](notebooks/paso_a_paso/02_rf_gee.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/02_rf_gee.ipynb) |
| [Importancia Gini de las variables](notebooks/paso_a_paso/03_importancia_rf.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/03_importancia_rf.ipynb) |
| [Comparación inicial: Random Forest y red de atención](notebooks/paso_a_paso/04_comparacion_inicial.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/04_comparacion_inicial.ipynb) |
| [Comparación aleatoria y grupos ambientales](notebooks/paso_a_paso/05_bloques_ambientales.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/05_bloques_ambientales.ipynb) |
| [Comparación con pseudoausencias cercanas](notebooks/paso_a_paso/06_hard_negatives.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/06_hard_negatives.ipynb) |
| [Explicabilidad SHAP de Random Forest y red de atención](notebooks/paso_a_paso/07_shap.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/07_shap.ipynb) |
| [Curvas ROC y figuras comparativas del artículo](notebooks/paso_a_paso/08_figuras_articulo.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/08_figuras_articulo.ipynb) |
| [Estacionalidad histórica de los incendios](notebooks/paso_a_paso/09_estacionalidad.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/09_estacionalidad.ipynb) |
| [Estacionalidad en una figura de dos paneles](notebooks/paso_a_paso/10_estacionalidad_panel.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/10_estacionalidad_panel.ipynb) |
| [Auditoría de registros y muestras de entrenamiento](notebooks/paso_a_paso/11_auditoria.ipynb) | [Abrir en Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/paso_a_paso/11_auditoria.ipynb) |

Los cuadernos 01 (ausencias sintéticas) y 05 (grupos ambientales) documentan
pruebas preliminares. El 05 no es validación espacial ni validación cruzada
por provincias. SHAP y las figuras del artículo vuelven a entrenar modelos;
sus configuraciones no son idénticas entre versiones.

## Qué se observa en cada cuaderno

1. Objetivo, archivos de entrada y límites del protocolo.
2. Bibliotecas, parámetros y carpeta de la ejecución.
3. Muestra de datos, dimensiones, tipos, nulos y balance de clases cuando corresponde.
4. Código de preparación, arquitectura, entrenamiento y análisis por etapas.
5. Progreso de las épocas de la red y figuras PNG al guardarse.
6. Tablas, descarga de salidas y registro de parámetros, versiones y hashes.
7. Plantilla para describir una hipótesis, una modificación y sus resultados.

Los comentarios originales se conservan. Se añade visualización y registro;
los valores predeterminados del protocolo histórico se mantienen. En el cuaderno
SHAP se corrige la asociación entre filas explicadas y colores del gráfico.
Los pesos de atención, Gini y SHAP son indicadores distintos y no demuestran causalidad.

## Colaborar

Lee [CONTRIBUTING.md](CONTRIBUTING.md) y la [guía de cuadernos](docs/CUADERNOS.md).
Abre un Issue para discutir propuestas y un pull request para revisar el código.
Antes de guardar un cuaderno en GitHub, borra sus salidas y comprueba que no
contengan datos privados. Las salidas de una sesión no se publican automáticamente.

## Estado

Se comprobó la sintaxis de los 11 cuadernos y la ejecución secuencial del código
de RF con GEE, importancia RF y auditoría usando datos locales. En esa comprobación
se sustituyeron solo las funciones de presentación de Jupyter; falta revisar su
renderizado y el flujo completo en Colab. Las redes, SHAP y estacionalidad aún no
se han ejecutado como parte de esta migración.

La extracción satelital GEE y los mapas no están reproducidos aquí. Los cuadernos
trabajan con CSV ya extraídos. La licencia del código y la distribución de los
datos siguen pendientes de definición. Los resultados históricos no se incluyen
en esta actualización pública.

El primer cuaderno `notebooks/01_entrenamiento_colab.ipynb` y `src/train.py`
se conservan como versión inicial de ejecución mediante script.
