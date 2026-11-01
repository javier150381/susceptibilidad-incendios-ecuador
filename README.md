# Susceptibilidad de incendios forestales en la Sierra ecuatoriana

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/javier150381/susceptibilidad-incendios-ecuador/blob/main/notebooks/01_entrenamiento_colab.ipynb)

Primera adaptación del experimento local para Google Colab y colaboración en GitHub.
Compara Random Forest y una red MLP con el módulo de atención del código original,
usando ocho covariables y pseudoausencias cercanas. Esta versión cubre entrenamiento
y evaluación aleatoria; no reproduce aún la extracción satelital ni genera mapas.

## Ejecutar en Colab

1. Pulsa **Abrir en Colab** arriba.
2. Ejecuta las celdas en orden. El cuaderno clonará este repositorio e instalará las dependencias.
3. Carga `dataset_entrenamiento_sierra_realista.csv` desde el proyecto original.
4. Ejecuta el entrenamiento y descarga los resultados antes de cerrar la sesión.

También puedes cargar el cuaderno manualmente en Colab. La URL del repositorio
ya está configurada; déjala vacía solo si prefieres cargar el paquete ZIP.

## Ejecutar localmente

```bash
pip install -r requirements.txt
python src/train.py
```

Coloca el CSV en `data/`. También puedes configurar `FIRE_DATA`, `FIRE_OUTPUT`
y `FIRE_EPOCHS` como variables de entorno. La ejecución predeterminada usa CPU y 50 épocas.
Genera métricas CSV, comparación de factores CSV/PNG y un registro JSON con semilla,
versiones de dependencias y hash del dataset. Las dependencias no están fijadas todavía;
el JSON permite identificar las versiones utilizadas en cada ejecución.

## Colaborar

Consulta `CONTRIBUTING.md` y `docs/ESTADO_METODOLOGICO.md`.
Publica preguntas y propuestas en Issues y cambios en ramas con pull requests.
Pendientes: titularidad y licencia del código, permisos de los datos
y versiones verificadas en Colab. No se ha asignado una licencia.
Los artículos PDF, manuscritos, credenciales, datos y modelos originales no forman parte del paquete.
