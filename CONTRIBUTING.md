# Cómo contribuir

Abre un Issue indicando problema, propuesta y forma de reproducirlo.
Trabaja en una rama y envía un pull request con descripción y evidencia de ejecución.
Para cambios en modelos registra semilla, versiones, hash del dataset, partición y métricas.
No incluyas credenciales, datos sin permiso ni salidas grandes de los cuadernos.
Describe cualquier cambio del protocolo experimental para distinguir sus resultados de los anteriores.

Pendientes científicos: procedencia y licencia de los datos; extracción GEE exacta;
campañas independientes de pseudoausencias y predictores para mapas continuos.
Los controles de validación y capacidad de atención ya forman parte del estudio.

## Contribuciones en Colab

El punto de entrada es `notebooks/00_ESTUDIO_COMPLETO_INCENDIOS.ipynb`.
Contiene todo el código y conecta las etapas; consulta `docs/CUADERNO_UNICO.md`.
Describe en el PR la hipótesis, el cambio, el hash de los datos y su validación.
El código equivalente reutilizable está en `src/`; conserva ambas versiones coherentes.
Después de editar `src/`, ejecuta `python scripts/build_master_notebook.py` desde
la raíz del repositorio para regenerar el cuaderno con las funciones actualizadas.
Reinicia el entorno y ejecuta todas las etapas antes de enviar cambios.
Borra salidas antes del commit para evitar publicar datos de la sesión.
