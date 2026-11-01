# Revisión del script GEE aportado

La revisión es estática: no se ejecutó el script en una cuenta de Earth Engine.
Confirma lo que hace el código aportado, pero no identifica automáticamente qué
archivo final se exportó con él ni qué transformaciones posteriores se realizaron.

## Ventanas previas al evento

Para Sentinel-2 se usa `filterDate(fechaEvento - 90 días, fechaEvento)`.
El fin es exclusivo, por lo que esta selección no incluye escenas con fecha
igual o posterior al evento. Para clima se seleccionan doce meses completos
anteriores al mes del evento; tampoco se incluye su mes. La lógica temporal de
estas dos consultas es coherente con usar información previa.
[Referencia oficial de filterDate](https://developers.google.com/earth-engine/apidocs/ee-imagecollection-filterdate).

Esta verificación no permite llamar prospectivo al conjunto: la selección de
pseudoausencias utiliza incendios conocidos de todo el periodo histórico.

## Diferencias entre variantes

| Decisión | Script aportado | Bitácora de la base trazable 2019–2025 |
|---|---|---|
| Área ejecutada | Imbabura; el ROI se limita a esa provincia | Diez provincias de la Sierra |
| Inicio de modelado | 2018 | 2019 tras auditar disponibilidad espectral |
| Pseudoausencias | Anillo de 1–8 km; exclusión histórica dentro del ROI | Muestreo provincial con exclusión de 2 km |
| Fecha de referencia | Se hereda del incendio más cercano | Una pseudoausencia por presencia, misma fecha y provincia |
| Clima | ERA5-Land mensual; 12 meses completos anteriores | ERA5-Land diario; 30 días anteriores |
| Sentinel-2 | 90 días; filtro de nubosidad de escena menor de 30 % | 90 días; bitácora indica retirada del filtro global de escena |
| Máscara SCL | Excluye varias clases; permite agua 6 y no clasificado 7 | Conserva únicamente clases 4 y 5 |
| Puntos sin datos | Se eliminan independientemente por clase | Reemplazos espaciales de pseudoausencias incompletas |
| Campos temporales | Exporta inicio y fin de las ventanas | La tabla limpia de modelado no conserva esos campos |

Las diferencias de clima y máscara requieren contrastar la versión de extracción;
no se resuelven renombrando columnas. No debe atribuirse a un dataset el protocolo
de otro. Los datos existentes pueden corresponder a una revisión posterior.

## Aspectos que deben documentarse o corregirse en una nueva versión

- Generar tantas pseudoausencias como presencias no crea una correspondencia
  uno a uno: varias pseudoausencias pueden heredar la misma presencia más cercana.
- Al filtrar faltantes por separado no queda garantizado el balance ni que sobreviva
  la presencia de referencia de cada pseudoausencia. Debe auditarse en la exportación.
- El inventario de exclusión se filtra previamente al ROI provincial. Una nueva
  implementación debe especificar si también excluye incendios cercanos al otro
  lado del límite provincial.
- La tolerancia geométrica de 100 m del anillo y del join no sustituye una auditoría
  de distancias en el archivo exportado.
- Extraer conjuntamente a `scale: 30` no da resolución de 30 m a ERA5-Land.
  Deben registrarse proyección y resolución efectiva de cada predictor.
- La cabecera anuncia distancia a zonas urbanizadas, pero el código aportado no
  calcula ni exporta esa variable. Debe ajustarse la descripción a lo implementado.

## Cómo cerrar la trazabilidad

Conserva el script como una variante histórica identificable. Para asociarlo a un
CSV concreto hacen falta la exportación con `event_date`, `s2_start`,
`s2_end_exclusive`, `climate_start` y `climate_end_exclusive`, el asset de entrada,
configuración de provincia, tarea y versión del código. Después debe documentarse
la cadena de limpieza y cualquier reextracción hasta el dataset final.

Si se adopta la bitácora 2019–2025 como protocolo vigente, habrá que recuperar su
implementación de clima diario, máscara SCL 4/5 y muestreo de 2 km antes de generar
campañas comparables. Los experimentos nuevos sobre CSV se interpretan como
retrospectivos y condicionados a la versión disponible mientras se cierra ese punto.
