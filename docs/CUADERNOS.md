# Editar y colaborar en los cuadernos

## Flujo de trabajo

1. Elige un cuaderno desde el README y abre su enlace a Colab.
2. Guarda una copia en tu Drive si deseas conservar cambios de la sesión.
3. Ejecuta las celdas en orden. La configuración inicial permite modificar semilla,
   árboles y profundidad de RF, épocas, tasa de aprendizaje y tamaño de tanda.
   Solo se usan los parámetros pertinentes al protocolo de cada cuaderno.
4. Carga los CSV solicitados. La inspección muestra sus columnas y calidad básica;
   el código posterior conserva los filtros y transformaciones originales.
5. Registra la ejecución original antes de probar una variante. Mantén la misma
   partición y dataset para comparar un hiperparámetro; si cambia el protocolo,
   identifica esa comparación como un experimento nuevo.
6. Completa la sección de propuesta al final del cuaderno.
7. Descarga el cuaderno modificado, elimina las salidas, súbelo a una rama de tu
   fork o del repositorio si tienes permisos y abre un pull request.

## Qué adjuntar a un pull request

- Pregunta e hipótesis; celda o bloque modificado y justificación.
- Parámetros, semilla, hash del dataset, protocolo y versiones.
- Métricas originales y de la variante, cuando sean pertinentes, con permiso para compartirlas.
- Cómo ejecutar el cambio y limitaciones observadas.

La celda de parámetros y el JSON de ejecución facilitan comparar contribuciones.
El conjunto de prueba no debe convertirse en objetivo de ajustes repetidos.
Si cambias parámetros, reinicia y vuelve a ejecutar desde el principio para evitar
variables residuales de una ejecución anterior.

## Diferencias respecto a los scripts históricos

El código de cada `main()` se expone en celdas secuenciales. Las rutas usan un
directorio de datos y otro de resultados por ejecución. Los errores de entrada
detienen el cuaderno. Se muestran PNG después de guardarlos y progreso de las
épocas. Los parámetros usan una configuración central con sus valores originales.
Se corrige el emparejamiento de datos y valores SHAP del análisis independiente.
No se sustituyen los grupos ambientales del experimento antiguo por provincias reales.

No hay salidas privadas incrustadas, historial de figuras ni modelos preentrenados
en estos cuadernos. Los resultados aparecen al ejecutar con tus propios CSV.
