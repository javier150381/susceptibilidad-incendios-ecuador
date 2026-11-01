# Guía del cuaderno único

Abre `notebooks/00_ESTUDIO_COMPLETO_INCENDIOS.ipynb` desde el enlace principal del
README. Este cuaderno sustituye el recorrido manual entre cuadernos para el
análisis de la base trazable. Todo el código de cada etapa está incluido.

## Entrada y ejecución

1. Conserva `MODO='COMPLETO'` para ejecutar el protocolo configurado. Elige
   `RAPIDO` únicamente para verificar funcionamiento; sus figuras indican PILOT.
2. Ejecuta todas las celdas y autoriza Drive para conservar los avances.
3. Carga `dataset_modelo_incendios_sierra_2019_2025.csv` cuando aparezca la única
   solicitud de archivos. La base antigua `dataset_entrenamiento_sierra_realista.csv`
   no contiene el esquema de trazabilidad necesario y no es intercambiable.
4. Si dispones de ellas, selecciona en la misma entrada las tablas fuente
   `incendios_sierra_2019_2025_modelo.csv`,
   `pseudoausencias_sierra_2019_2025_final.csv` e `incendios_coordenadas_gee.csv`.
   Son opcionales para ejecutar modelos y figuras, y necesarias para contrastar
   emparejamientos y distancias contra el inventario de origen.
5. Revisa los resultados, las figuras y la guía del manuscrito al final del
   mismo cuaderno. No hay que importar resultados de ninguna otra sesión.

Los datos quedan en `MyDrive/Susceptibilidad_Incendios_Estudio/datos` y se detectan
al volver a ejecutar. Para sustituirlos usa `FORZAR_CARGA_DATOS=True`. La celda de
entrada muestra el archivo, conteos y SHA-256; confirma la versión antes de entrenar.

## Qué realiza

- Auditoría de columnas, fechas, clases, duplicados y pares presencia-pseudoausencia.
- Plan de particiones y controles de separación antes de ajustar los modelos.
- Seis modelos, tres semillas y cuatro diseños: aleatorio por pares como
  diagnóstico, bloques espaciales, exclusión de provincias y test temporal 2024.
- Validación interna para elegir épocas; escalado ajustado sobre entrenamiento.
- Ablaciones de covariables con los tres modelos clásicos y una semilla.
- Figuras de muestreo, estacionalidad, distribuciones, ROC/PR, matrices,
  calibración del muestreo, provincias, aprendizaje y comparaciones pareadas.
- SHAP de RF y atención con los mismos casos y background; permutación y routing.
  Antes de explicar se reproduce el ajuste y se contrastan sus predicciones.
- Guía de redacción generada a partir de los resultados disponibles.

Con diez provincias, la configuración completa produce 306 evaluaciones
principales y 153 de ablaciones. El piloto usa una partición por diseño y una
semilla: 24 y 36, respectivamente. El tiempo depende de CPU y entorno; el estudio
completo puede requerir varias horas. Esta implementación utiliza CPU.

## Guardado y repetición

Cada estudio tiene su propia carpeta bajo `estudios`, identificada por datos,
código y configuración. Guarda auditoría, plan, logs, predicciones, métricas,
figuras PNG/SVG/PDF, leyendas, guía, versiones y dependencias fijadas de la ejecución.
El código ejecutado se identifica mediante huellas de funciones, incluyendo
ediciones en las celdas. Conserva también tu copia del cuaderno si lo modificas.

Una etapa completa se reutiliza solo si coinciden datos, configuración, código,
Python y bibliotecas relevantes. Una etapa interrumpida se repite desde su inicio.
Las explicaciones tienen un caché independiente comprobado contra predicciones,
datos, parámetros y versión del explainer. `RECALCULAR_ETAPAS=True` fuerza el cálculo.
La figura se puede redibujar sin repetir SHAP compatible. Un entorno diferente
puede cambiar los valores numéricos; las versiones guardadas permiten identificarlo.

Drive es el almacenamiento persistente recomendado. Sin Drive, los archivos en
`/content` desaparecen al terminar la sesión. La descarga de una copia al final es
opcional; no constituye una entrada para otro cuaderno.

## Alcance para el paper

La reproducibilidad implementada parte del CSV con covariables ya extraídas.
Las pseudoausencias ya existen. No se regenera su muestreo ni se afirma haber
reconstruido la extracción satelital: hay discrepancias pendientes entre bitácora
y script GEE sobre periodos, clima, máscara y distancias de exclusión.

La estacionalidad usa el inventario aportado o, si falta, solo las presencias de
modelado, con esa identificación. F01 muestra puntos y particiones, no una
superficie de susceptibilidad. Un mapa continuo necesita predictores espaciales
compatibles y comprobaciones de dominio. Campañas independientes requieren sus
propios datos; el cuaderno no las sustituye por simulaciones.

El test temporal es retrospectivo por el diseño global de exclusión de
pseudoausencias. Los scores no son probabilidades poblacionales. El año 2025
permanece bloqueado por defecto, pero su independencia histórica debe comprobarse.
Las figuras y la guía explicitan estos límites antes de preparar el envío.

## Colaboraciones

Describe hipótesis, modificación, versión de datos y validación en un Issue o PR.
Conserva las particiones y baselines comparables. Reinicia el entorno y ejecuta
el cuaderno antes de proponer cambios; borra salidas antes del commit. No publiques
datos o resultados privados sin autorización. La publicación del código no envía
los archivos de tu Drive al repositorio.
