# Protocolo de validación de la base trazable 2019–2025

Este protocolo fortalece el diseño experimental y su reproducibilidad. No es una
prerregistración pública ni una garantía de aceptación editorial. Los cuadernos
históricos se conservan como antecedentes; los nuevos exigen coordenadas, fecha,
provincia e identificadores de referencia. No se deben mezclar versiones del dataset.

## Pregunta y alcance

¿Cuánto cambia la discriminación de presencias frente a pseudoausencias al evaluar
territorios y periodos distintos, y qué aporta la atención una vez controlados
la representación de las variables y el presupuesto aproximado de parámetros?

La unidad de observación es un punto-fecha del muestreo caso-control. La etiqueta
cero significa pseudoausencia, no ausencia demostrada. El área definida por
provincias no equivale automáticamente al ecosistema de páramo. Los scores no
son probabilidades poblacionales de incendio. Calibración, AP y Brier deben
interpretarse bajo la prevalencia del conjunto muestreado.

## Hipótesis y contrastes

1. **Transferencia territorial:** comparar partición aleatoria por pares, bloques
   proyectados y exclusión de una provincia completa. La diferencia entre diseños
   es descriptiva si cambian sus conjuntos de prueba; no es una comparación pareada.
2. **Aportación de la atención:** comparar una red con atención con una red sin
   atención que comparte tokenización y diseño de clasificador, y con otra red
   sin atención de presupuesto aproximadamente equivalente. Las predicciones se
   contrastan sobre exactamente las mismas observaciones externas.
3. **Dependencia del muestreo:** repetir la evaluación con campañas que cambien
   exclusivamente distancia de exclusión o semilla de pseudoausencias, manteniendo
   las presencias y la trazabilidad temporal/provincial.

La atención es un modelo de referencia nuevo para la ablación. No se presenta
como recuperación del checkpoint original ni como una arquitectura novedosa sin
revisión de antecedentes. La complejidad por sí sola no constituye una contribución.

## Procedencia y controles antes de modelar

El esquema requerido es `id_registro`, `date`, `provincia`, `latitud`, `longitud`,
`clase` y ocho variables ambientales. Se comprueban nulos, finitud, duplicados
espacio-temporales, rangos e identidad de las referencias emparejadas. Si se
aportan tablas fuente, se contrastan IDs, valores, fechas y controles de Sentinel-2.

La metodología documenta Sentinel-2 durante los 90 días anteriores al evento,
ERA5-Land durante los 30 días previos, SCL 4/5 y ausencia de imputación. Esto
debe verificarse con el código exacto de extracción o metadatos de adquisiciones;
los valores de un CSV y un contador de imágenes no demuestran las ventanas.
`filterDate` en Earth Engine usa inicio inclusivo y fin exclusivo.
[Referencia oficial](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterdate).

El script aportado durante la revisión usa otra variante (Imbabura, 2018–2025,
clima mensual de doce meses y anillo de 1–8 km). La selección preevento de ese
script es coherente, pero no implementa todos los parámetros de la bitácora nueva.
Consulta la [revisión del script GEE](REVISION_GEE.md) antes de atribuirlo al CSV final.

También deben conservarse escalas y proyecciones de extracción. Muestrear una
variable climática sobre puntos separados 30 m no le da resolución efectiva de
30 m: el catálogo ERA5-Land informa píxeles de aproximadamente 11 km.
[Catálogo de ERA5-Land](https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_LAND_DAILY_AGGR).

## Particiones externas

- **Desarrollo:** años hasta 2023. Una división aleatoria por referencia emparejada
  sirve como diagnóstico, no como prueba de independencia geográfica.
- **Bloques:** cinco folds con grupos definidos por celdas nominales de 20 km en
  EPSG:6933. La proyección es de área equivalente; sus celdas no son cuadrados
  geodésicos perfectos. Se elimina del entrenamiento cualquier referencia del test
  y cualquier punto dentro de 5 km esféricos de sus observaciones.
- **Provincias:** cada una de las provincias actúa una vez como test. Se aplica
  la misma exclusión de referencias y vecinos. Las métricas por provincia son
  esenciales: un promedio puede esconder diferencias territoriales relevantes.
- **Tiempo:** test de 2024 y entrenamiento anterior, con un embargo de 90 días
  entre el final del entrenamiento y el inicio del test. Se permiten lugares
  recurrentes porque la pregunta es transferencia temporal, no espacial.
- **2025:** bloqueado por defecto en esta implementación. Su independencia
  respecto a experimentos históricos debe demostrarse antes de llamarlo test
  confirmatorio. Haber reservado un año aquí no revierte su posible uso previo.

Las pseudoausencias existentes se seleccionaron excluyendo incendios conocidos
de todo 2019–2025. Por eso el contraste temporal es retrospectivo y condicionado
a ese diseño; no prueba capacidad prospectiva operacional. Para evaluarla hace
falta reconstruir muestreo y covariables usando solo información disponible al
corte, además de una validación externa realmente independiente.

La separación por grupos responde a la dependencia de las muestras; sus decisiones
deben justificarse y explorarse con sensibilidad.
[Roberts et al. 2017](https://doi.org/10.1111/ecog.02881),
[documentación de validación por grupos](https://scikit-learn.org/stable/modules/cross_validation.html).

## Modelos y ajuste

Los modelos son regresión logística, Random Forest, boosting por histogramas,
MLP con embeddings, MLP sin atención con capacidad aproximada equivalente y
atención sobre tokens de características. Se representan ocho variables físicas;
orientación se transforma en seno y coseno, dando nueve canales por defecto.
La variante `raw_aspect` conserva ocho canales como sensibilidad explícita.

Los hiperparámetros del protocolo base están fijados en el código, no elegidos por
el test. Las redes usan una validación interna espacial o temporal formada solo
con entrenamiento externo para seleccionar el número de épocas; luego se
reinicializan y reajustan con todo ese entrenamiento. Los escaladores se ajustan
siempre dentro de los conjuntos de entrenamiento correspondientes.

Esta versión **no implementa búsqueda de hiperparámetros con CV anidada**. Si se
incorpora, debe usar exclusivamente folds internos y dar presupuestos comparables
a los modelos. Elegir la mejor configuración con las métricas externas sesgaría
la evaluación. [Referencia sobre CV anidada](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

El presupuesto de la MLP de capacidad aproximada se obtiene de las dimensiones,
sin consultar resultados. Se registran número exacto de parámetros, ancho de
capa y épocas seleccionadas. Las semillas base son 42, 137 y 271; se usa CPU y
operaciones deterministas de PyTorch. La reproducibilidad entre versiones y
plataformas requiere verificar el entorno registrado.

## Métricas e incertidumbre

ROC-AUC es el contraste principal preespecificado; AP, balanced accuracy, precision,
recall, F1, Brier y log-loss lo complementan. El umbral es 0,5 dentro del diseño
caso-control. Se guardan también matrices de confusión y prevalencia del test.

Se calculan diferencias pareadas de AUC entre atención y las MLP sobre el mismo
test. El bootstrap remuestrea bloques espaciales completos, con 500 réplicas en
el protocolo base. Sus intervalos son aproximados y dependen del tamaño de bloque;
no incluyen toda la incertidumbre del inventario, el muestreo o el ajuste.
Los folds y semillas repetidos no se tratan como réplicas independientes para
obtener errores estándar. Comparaciones múltiples se declaran exploratorias.

## Sensibilidades planificadas

- Bloques nominales de 10, 20 y 40 km; desplazamiento de origen de media celda.
- Buffers de separación de 5 y 12 km, con reporte de cuántas muestras se retiran.
- Conjunto completo, sin índices espectrales y sin clima; orientación cruda o circular.
- Campañas de pseudoausencias con exclusión de 1, 2 y 5 km y semillas de muestreo
  17, 43, 101, 211 y 307. Son escenarios de sensibilidad, no distancias ecológicas universales.

El cuaderno 14 exige igualdad de presencias, fechas y covariables entre campañas.
No reconstruye una extracción desconocida. Antes de generar nuevas campañas
deben recuperarse y versionarse el código GEE vigente, escalas, proyecciones,
reglas de reemplazo y ámbito temporal del inventario de exclusión. La sensibilidad
al muestreo está documentada en modelos de presencia/pseudoausencia; su magnitud
en este estudio de incendios es una pregunta empírica.
[Barbet-Massin et al. 2012](https://doi.org/10.1111/j.2041-210X.2011.00172.x).

## Interpretación y cierre

Los pesos de atención no se identifican con SHAP ni con causas físicas. La
interpretación debe repetirse entre particiones y semillas, examinando colinealidad
y dominio ambiental. Los mapas deben declarar fecha, soporte espacial, cobertura
y disponibilidad de las covariables, y diferenciar interpolación de extrapolación.

Antes del manuscrito final deben identificarse la institución y versión oficial
del inventario, licencias y acceso, código exacto de extracción, uso histórico de
los conjuntos de prueba y disponibilidad de una evaluación externa independiente.
Si la atención no aporta una mejora robusta, ese resultado debe conservarse.
