# Estado y límites de esta adaptación

- Se adapta `train_realistic_models.py`: rutas portables, validación básica de entrada,
  métricas guardadas y registro de ejecución. Se mantienen arquitectura y configuración
  principal del experimento original. No se han vuelto a validar las conclusiones del artículo.
- La división 80/20 estratificada mide discriminación sobre muestras aleatorias;
  no demuestra generalización geográfica ni elimina autocorrelación espacial.
- El script local `train_spatial_validation.py` construye grupos mediante cuantiles
  de elevación + temperatura * 100. Eso no define provincias ni bloques geográficos.
  Por ese motivo no se presenta aquí como validación espacial.
- El CSV de entrenamiento inspeccionado tiene clase y ocho covariables, sin coordenadas
  ni provincia. Para evaluar transferencia territorial se debe recuperar la trazabilidad
  a ubicaciones reales y definir grupos geográficos independientes.
- Los pesos de atención no son equivalentes a importancia Gini, SHAP ni efectos causales.
  La figura heredada muestra ambos indicadores: su interpretación requiere esta distinción.
- En el módulo heredado, la media de los scores antes del softmax reduce el mecanismo
  de interacción entre características. Su revisión debe ser otro experimento explícito.
- El origen y buffer de las pseudoausencias provienen de la descripción del script local;
  el cuaderno no reconstruye ni verifica su muestreo geográfico.
- Esta es una base de migración. La ejecución completa en Colab queda pendiente.
