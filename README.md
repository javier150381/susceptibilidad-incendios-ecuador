# Susceptibilidad de incendios forestales en Ecuador

## Un solo cuaderno para el estudio

**[Abrir el estudio completo en Google Colab](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/00_ESTUDIO_COMPLETO_INCENDIOS.ipynb)**

Ejecuta las celdas en orden o selecciona **Entorno de ejecución → Ejecutar todas**.
Carga una sola vez `dataset_modelo_incendios_sierra_2019_2025.csv`. El cuaderno
contiene el código completo y conecta auditoría, comparación de modelos,
validación espacial y temporal, ablaciones, 13 figuras, SHAP y una guía para
redactar el paper. No necesita ZIP ni resultados de otros cuadernos.

Con Drive activado guarda los datos, avances, configuraciones y resultados en
`MyDrive/Susceptibilidad_Incendios_Estudio`. Las etapas terminadas se reutilizan
si coinciden datos, código, parámetros y entorno. El modo predeterminado es
**COMPLETO**; puede requerir varias horas en CPU. `RAPIDO` comprueba el recorrido
y etiqueta los resultados como piloto. Una etapa interrumpida durante su ajuste
se repite; las etapas completas compatibles se conservan.

La [guía del cuaderno único](docs/CUADERNO_UNICO.md) explica entradas, etapas,
resultados guardados y alcance de la reproducibilidad.

## Protocolo y límites

Se comparan regresión logística, Random Forest, boosting, MLP, MLP de capacidad
similar y red con atención sobre las mismas particiones externas. Se registran
semillas, versiones, hashes, métricas, predicciones y controles de independencia.
Los modelos usan ocho covariables físicas; orientación circular genera nueve
canales. Las coordenadas, provincia y fecha se usan para trazabilidad y validación.

La reproducción comienza en el CSV trazable. La extracción GEE exacta todavía
debe reconciliarse con la bitácora; véase la [revisión del script](docs/REVISION_GEE.md).
Las pseudoausencias ya están en el CSV. Campañas adicionales y mapas continuos
requieren insumos propios y no se fabrican a partir de resultados anteriores.
El test temporal es retrospectivo; los scores caso-control no son probabilidades
poblacionales de incendio. El protocolo no garantiza aceptación editorial Q1.

Consulta el [protocolo de validación](docs/PROTOCOLO_VALIDACION.md) y la
[guía de figuras](docs/FIGURAS_ARTICULO.md) para los detalles científicos.

## Colaborar y ejecutar localmente

El código reutilizable está en `src/`; instala `requirements-figures.txt`.
Ejecuta los controles con `python -m unittest discover -s tests -v`.
Propón cambios en una rama y pull request siguiendo [CONTRIBUTING.md](CONTRIBUTING.md).
El cuaderno muestra las funciones completas en celdas editables y comentadas.
Antes de publicar cambios, borra sus salidas para no incluir datos de la sesión.

## Versiones anteriores

Los cuadernos de `notebooks/paso_a_paso/` y `notebooks/investigacion/` se conservan
como antecedentes y herramientas de análisis separado. **No son requisitos del
cuaderno principal**. Su documentación está en [CUADERNOS.md](docs/CUADERNOS.md).
Las comparaciones históricas tienen configuraciones y datos diferentes; sus
resultados no deben mezclarse automáticamente con la ejecución nueva.

No se distribuyen los datos ni resultados privados. La fuente institucional,
licencia y permisos de redistribución de los datos siguen pendientes de documentar.
