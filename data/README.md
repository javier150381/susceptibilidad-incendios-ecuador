# Datos de entrada

Carga `dataset_entrenamiento_sierra_realista.csv` desde tu equipo o Drive.
El archivo original está en la subcarpeta `resultados` del proyecto local.
No se incluye en este paquete: documentar procedencia, permisos y licencia antes de distribuirlo.

Columnas requeridas: clase, elevacion, pendiente, orientacion, ndvi, evi, nbr, temp_c, precip_mm.
`clase`: 1 presencia; 0 pseudoausencia. Cada fila es una muestra de entrenamiento.
El script elimina filas con valores faltantes y rechaza variables no numéricas o infinitas.
No se requieren credenciales de Google Earth Engine para esta fase: las covariables ya están extraídas.
