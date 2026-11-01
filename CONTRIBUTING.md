# Cómo contribuir

Abre un Issue indicando problema, propuesta y forma de reproducirlo.
Trabaja en una rama y envía un pull request con descripción y evidencia de ejecución.
Para cambios en modelos registra semilla, versiones, hash del dataset, partición y métricas.
No incluyas credenciales, datos sin permiso ni salidas grandes de los cuadernos.
Describe cualquier cambio del protocolo experimental para distinguir sus resultados de los anteriores.

Primeras tareas: procedencia y licencia de los datos; extracción GEE reproducible;
validación con coordenadas o límites provinciales reales; dependencias verificadas;
revisión del módulo de atención y comparación con una MLP sin atención.

## Contribuciones en Colab

Los cuadernos de `notebooks/paso_a_paso/` contienen el código completo.
Consulta `docs/CUADERNOS.md` y completa la plantilla de propuesta al final del cuaderno.
Reinicia el entorno y ejecuta todas las etapas antes de enviar cambios.
Borra salidas antes del commit para evitar publicar datos de la sesión.
