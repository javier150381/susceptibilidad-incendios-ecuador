# Figuras para el artículo

El cuaderno 15 recupera la presentación gráfica del estudio y la amplía con los
diagnósticos del protocolo nuevo. Las imágenes históricas pertenecen a otra
versión del dataset; no deben insertarse junto a métricas de la base trazable
sin comprobar que describen el mismo experimento.

## Usar la galería

1. Abre `notebooks/investigacion/15_figuras_articulo.ipynb` en Colab.
2. Carga el ZIP de resultados del cuaderno 13 y el CSV utilizado en esa ejecución.
3. Opcionalmente carga el inventario histórico para la estacionalidad completa.
4. Para mostrar todas las provincias o ablaciones, carga sus corridas reales como
   entradas adicionales; el piloto por sí solo no contiene esos experimentos completos.
5. Ejecuta la galería y descarga sus archivos PNG a 600 dpi, SVG y PDF. Los
   manifiestos y leyendas registran versión, diseño, semilla y alcance de cada figura.
6. Activa `CALCULATE_EXPLANATIONS=True` si necesitas SHAP y routing de atención.
   Se reentrena solo el ajuste seleccionado y se comprueba coincidencia con los
   scores exportados antes de generar explicaciones. Si no coincide, se detiene.

Las curvas de desempeño se generan desde las predicciones guardadas; no necesitan
reentrenamiento. SHAP sí necesita acceso al modelo. La reconstrucción requiere
un entorno compatible; si no se reproduce el ajuste, deben conservarse checkpoints
y escaladores originales, en lugar de explicar una red distinta.

Después de calcular SHAP una vez, `render_explanation_cache(OUT)` permite
redibujar esas figuras desde el caché numérico, sin volver a ajustar ni explicar.

## Selección de figuras

La cantidad final depende de las hipótesis y de la revista. Esta propuesta separa
las figuras que sostienen los argumentos principales de los diagnósticos extensos.

| Código | Figura | Uso propuesto |
|---|---|---|
| F01 | Localización de las muestras y partición geográfica real | Principal; complementar con límites cartográficos oficiales |
| F02 | Frecuencia mensual y mapa de calor año–mes | Principal si la estacionalidad forma parte del objetivo |
| F03 | Distribución de ocho covariables por clase | Suplementario o caracterización de datos |
| F04 | ROC y precisión–recall de modelos sobre el mismo test | Principal |
| F05 | Matrices con conteos y porcentajes por clase | Suplementario, o principal si importan tipos de error |
| F06 | Fiabilidad del score y Brier dentro del muestreo | Suplementario; no calibración poblacional |
| F07 | Desempeño por provincia excluida | Principal |
| F08 | Curvas internas de entrenamiento y validación | Suplementario |
| F09 | Aportación de atención frente a dos controles MLP | Principal |
| F10 | Ablaciones de espectro, clima y codificación de orientación | Principal o suplemento según hipótesis |
| F11 | Distribución SHAP con orden y ejes comunes | Principal |
| F12 | Mean absolute SHAP y reducción de AUC por permutación | Principal o suplemento |
| F13 | Matriz de routing real entre tokens de entrada | Suplementario; no importancia causal |

El diagrama metodológico del manuscrito debe actualizarse cuando se cierre la
versión de GEE: periodo, ventanas, máscara y muestreo deben coincidir con el CSV.
La galería no inventa un mapa de susceptibilidad a partir de puntos de entrenamiento.
Ese mapa requiere predictores continuos, dominio válido y un ajuste documentado.

## Criterios visuales y científicos

- Paleta consistente por modelo, legible también con deficiencias de visión cromática.
- Texto, unidades, nombres de modelos y leyendas en inglés para mantener coherencia.
- Escalas comunes en comparaciones de SHAP, matrices y desempeño territorial.
- Las curvas ROC/PR y matrices se calculan de las mismas observaciones externas.
- Los folds territoriales distintos no se promedian por su número como si fueran
  una misma región; las provincias sí tienen identidad estable entre semillas.
- La variación entre semillas o folds no se convierte en un intervalo de confianza
  que asuma independencia. El gráfico de atención utiliza los intervalos registrados
  del bootstrap por bloques y conserva el cero como referencia.
- SHAP usa los mismos casos y background de entrenamiento en ambos modelos.
  La transformación circular de orientación ocurre dentro de la función predictora.
  Las coaliciones interventionales no eliminan las dependencias entre covariables;
  las atribuciones no se presentan como causas.
- La importancia por permutación se muestra en reducción de AUC y SHAP en unidades
  de score. No se fuerzan medidas diferentes a sumar 100% para aparentar equivalencia.
- Las figuras de un piloto se identifican como PILOT y deben reemplazarse por
  las corridas completas para el manuscrito definitivo.
- El ejemplo de precipitaciones conserva unidades de milímetros sin atribuirle
  una ventana de 30 días o 12 meses hasta reconciliar el código y la bitácora.

## Colaborar

Las funciones de cada figura están visibles y comentadas. Un pull request debe
indicar si cambia solo presentación o también muestras, estimador, partición o
definición de métricas. Los cambios científicos requieren una nueva ejecución.
Borra las salidas de los cuadernos antes del commit. Los datos, scores y figuras
calculadas no se publican automáticamente al modificar el código de la galería.
