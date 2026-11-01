from pathlib import Path
import ast
import hashlib
import json

root=Path(__file__).resolve().parents[1];cells=[]
core_hash=hashlib.sha256((root/'src/research_validation.py').read_bytes()).hexdigest()
def add(kind,text,role=None):
    c=dict(cell_type=kind,id=f'master-{len(cells):03d}',metadata={},source=text.splitlines(keepends=True))
    if role:c['metadata']['workflow_role']=role
    if kind=='code':ast.parse(text);c.update(execution_count=None,outputs=[])
    cells.append(c)

add('markdown','''# Estudio completo de susceptibilidad a incendios en un solo Colab

**Este es el cuaderno principal.** Contiene datos de entrada, auditoría, modelos,
validación geográfica y temporal, ablaciones, figuras, explicabilidad y orientación
para escribir el paper. No necesita resultados ni ZIP de otros cuadernos.

### Cómo usarlo

1. Ejecuta las celdas en orden, o usa **Entorno de ejecución → Ejecutar todas**.
2. Autoriza Drive si quieres conservar los datos y avances entre sesiones.
3. Carga una sola vez `dataset_modelo_incendios_sierra_2019_2025.csv`.
4. Espera la ejecución y revisa las figuras y la guía del manuscrito al final.

Los datos obligatorios ya contienen las pseudoausencias y sus covariables:
**no se vuelven a generar**. El CSV correcto está en tu equipo, dentro de
`INCENDIOS_BASE_DATOS/BASE DE DATOS`. No lo sustituyas por la base antigua
`dataset_entrenamiento_sierra_realista.csv`.

El modo **COMPLETO** ejecuta el protocolo configurado; puede tardar varias horas
en CPU. El modo RAPIDO sirve para comprobar el flujo y marca sus resultados como
piloto. El cuaderno guarda y reutiliza etapas compatibles con los mismos datos,
código, bibliotecas y parámetros; no promete recuperar una etapa interrumpida a mitad de ajuste.

### Recorrido del estudio

Configuración → entrada única de datos → auditoría → particiones → modelos →
ablaciones → figuras y SHAP → informe orientativo para el paper → guardado.

La reproducción comienza en el CSV trazable. No se afirma haber reconstruido
la extracción GEE: la diferencia entre la bitácora y el script aportado sigue
pendiente de reconciliar. Tampoco se inventan campañas nuevas de pseudoausencias
ni superficies cartográficas que no estén respaldadas por predictores espaciales.
''')

add('markdown','''## A. Configuración única

Los valores siguientes dejan preparado el análisis completo. No necesitas
modificar rutas de otros cuadernos. Para una prueba técnica puedes cambiar
`MODO` a `RAPIDO`. Las etapas completadas se guardan por versión.
''')
add('code',f'''# Controles principales del único flujo de trabajo.
MODO='COMPLETO'                 # COMPLETO para el estudio; RAPIDO solo para comprobar.
GUARDAR_EN_DRIVE=True
FORZAR_CARGA_DATOS=False
RECALCULAR_ETAPAS=False
EJECUTAR_ABLACIONES=True
EJECUTAR_SHAP=True
ARCHIVO_PRINCIPAL='dataset_modelo_incendios_sierra_2019_2025.csv'

# Información que debes confirmar antes de cerrar el manuscrito.
FUENTE_INSTITUCIONAL=''         # Institución y nombre oficial de la base.
VERSION_INVENTARIO=''
ESTADO_EXTRACCION='Pendiente: reconciliar ventanas, máscara y muestreo entre bitácora y script GEE'

# Identidad de la implementación: se registra junto a los resultados.
NOTEBOOK_TEMPLATE_SHA={core_hash!r}
NOTEBOOK_IMPLEMENTATION_SHA=NOTEBOOK_TEMPLATE_SHA
''','configuration')

add('markdown','''## B. Preparar entorno y almacenamiento

Se instalan solo dependencias que falten. Los datos y salidas quedan en tu
carpeta de Drive `Susceptibilidad_Incendios_Estudio`; no se publican en GitHub.
Si desactivas Drive, el almacenamiento será temporal y tendrás que descargarlo
antes de que termine la sesión.
''')
add('code', '''import importlib.util
import subprocess
import sys
from pathlib import Path
from IPython.display import display,Image,Markdown
DEPENDENCIAS={'numpy':'numpy','pandas':'pandas','scipy':'scipy','sklearn':'scikit-learn',
              'matplotlib':'matplotlib','pyproj':'pyproj','torch':'torch','shap':'shap'}
faltantes=[p for m,p in DEPENDENCIAS.items() if importlib.util.find_spec(m) is None]
if faltantes:subprocess.run([sys.executable,'-m','pip','install',*faltantes],check=True)
if GUARDAR_EN_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    ALMACEN=Path('/content/drive/MyDrive/Susceptibilidad_Incendios_Estudio')
else:
    ALMACEN=Path('/content/Susceptibilidad_Incendios_Estudio')
DATOS=ALMACEN/'datos'
DATOS.mkdir(parents=True,exist_ok=True)
print('Carpeta del estudio:',ALMACEN)
''','environment')

add('markdown','''## C. Código completo, visible y comentado

Las siguientes celdas definen los procedimientos que usa este mismo cuaderno.
Contienen el código real de auditoría, particiones, modelos, métricas y figuras.
No se clonan repositorios ni se ejecutan scripts descargados en segundo plano.
Las funciones se llaman en las etapas D–G. Puedes usar el índice lateral de
Colab para desplazarte entre las secciones.
''')
modules=[
    ('research_audit.py','Auditoría de estructura, emparejamiento y tablas fuente'),
    ('research_validation.py','Particiones externas, modelos y ablación de atención'),
    ('paper_figures.py','Gráficos de datos, desempeño, geografía y aprendizaje'),
    ('paper_explainability.py','Recuperación comprobada de modelos y explicabilidad'),
    ('master_workflow.py','Conexión de todas las etapas y guía de manuscrito'),
]
for filename,title in modules:
    text=(root/'src'/filename).read_text(encoding='utf-8').split('# %% Interfaz de terminal')[0]
    add('markdown','### '+title)
    for i,section in enumerate(text.split('# %% ')):
        if i==0:continue
        heading,body=section.split('\n',1)
        add('markdown','#### '+heading)
        add('code',body,'definitions')

add('markdown','''## D. Entrada única de datos

Carga el CSV de modelado. Si los tienes disponibles, puedes seleccionar también
en esta misma entrada las presencias fuente, pseudoausencias fuente e inventario
histórico. Son **datos originales**, no resultados de otros cuadernos.

- Obligatorio: el CSV con `id_registro`, fecha, provincia, latitud/longitud,
  ocho covariables y clase.
- Opcionales: `incendios_sierra_2019_2025_modelo.csv`,
  `pseudoausencias_sierra_2019_2025_final.csv`, `incendios_coordenadas_gee.csv`.

Sin el inventario histórico, la estacionalidad se calculará solo sobre las
presencias de modelado y se etiquetará así. Sin las tres tablas fuente no se
certificará el buffer ni el origen de las observaciones satelitales desde el CSV final.
''')
add('code', '''# Esta es la única solicitud de archivos de todo el cuaderno.
if FORZAR_CARGA_DATOS or not (DATOS/ARCHIVO_PRINCIPAL).is_file():
    from google.colab import files
    cargados=files.upload()
    for nombre,contenido in cargados.items():
        if Path(nombre).name!=nombre or not nombre.lower().endswith('.csv'):
            raise ValueError('Selecciona únicamente CSV de datos; no se necesitan ZIP.')
        (DATOS/nombre).write_bytes(contenido)
    del cargados

columnas_modelo=set(FEATURES+['id_registro','date','provincia','latitud','longitud','clase'])
candidatos=[];PRESENCIAS=None;PSEUDOAUSENCIAS=None;INVENTARIO=None
for ruta in sorted(DATOS.glob('*.csv')):
    columnas=set(pd.read_csv(ruta,nrows=0).columns)
    if columnas_modelo.issubset(columnas):candidatos.append(ruta)
    elif {'id_pseudo','id_presencia_referencia','latitud','longitud'}.issubset(columnas):PSEUDOAUSENCIAS=ruta
    elif {'id','n_observaciones_S2_validas','latitud','longitud'}.issubset(columnas):PRESENCIAS=ruta
    elif {'id','date','provincia','latitude','longitude'}.issubset(columnas):INVENTARIO=ruta

principal=DATOS/ARCHIVO_PRINCIPAL
if principal in candidatos:DATA=principal
elif len(candidatos)==1:DATA=candidatos[0]
else:raise ValueError('No se identificó un único dataset trazable. Revisa ARCHIVO_PRINCIPAL y las columnas; no renombres la versión antigua.')
datos=load_data(DATA)
print('Dataset:',DATA.name,'| filas:',len(datos),'| SHA-256:',hashlib.sha256(DATA.read_bytes()).hexdigest())
display(datos.head())
display(datos.groupby(['year','clase']).size().unstack(fill_value=0))
print('Tablas fuente disponibles:',{'presencias':PRESENCIAS is not None,'pseudoausencias':PSEUDOAUSENCIAS is not None,'inventario':INVENTARIO is not None})
''','inputs')

add('markdown','''## E. Ejecutar todo el estudio

Esta celda conecta las etapas: los modelos alimentan directamente las figuras;
las tablas y figuras alimentan la guía del paper. No hay que exportar, volver a
subir ni buscar resultados de otro cuaderno.

El protocolo completo compara seis modelos con tres semillas, cinco folds
geográficos, exclusión de cada provincia y transferencia temporal retrospectiva.
Las ablaciones configuradas comparan los tres modelos clásicos con una semilla;
no se atribuyen esos resultados a redes que no se hayan evaluado en ellas.

Los hiperparámetros están fijados. Las épocas se eligen con validación interna
del entrenamiento; no se optimiza contra el test externo. 2025 permanece bloqueado.
Si la sesión se interrumpe, vuelve a ejecutar el mismo cuaderno con los mismos
parámetros: reutiliza las etapas que hayan terminado y sean compatibles.
''')
add('code', '''AJUSTES=copy.deepcopy(MASTER_DEFAULTS)
AJUSTES.update(mode=MODO,force=RECALCULAR_ETAPAS,run_ablations=EJECUTAR_ABLACIONES,
    run_explanations=EJECUTAR_SHAP,source_institution=FUENTE_INSTITUCIONAL,
    source_version=VERSION_INVENTARIO,extraction_status=ESTADO_EXTRACCION)
# Parámetros científicos visibles. Cambiarlos crea una configuración distinta.
AJUSTES.update(block_km=20,buffer_km=5,grid_offset_km=0,n_splits=5,
    epochs=40,patience=8,seeds=[42,137,271],rf_trees=300,
    ablation_models=['logistic','rf','histgb'],ablation_seeds=[42],
    explanation_cases=300,background_cases=50,coalitions=256,
    permutation_cases=1000,permutation_repeats=5)
if MODO=='RAPIDO':
    AJUSTES.update(explanation_cases=120,background_cases=30)

display(AJUSTES)
ESTUDIO=complete_study(DATA,ALMACEN,globals(),AJUSTES,
    presence=PRESENCIAS,pseudo=PSEUDOAUSENCIAS,inventory=INVENTARIO,
    core_hash=NOTEBOOK_IMPLEMENTATION_SHA)
''','execute')

add('markdown','''## F. Revisar resultados y figuras en este cuaderno

Estas salidas proceden de la ejecución anterior. Comprueba primero auditoría,
separación geográfica y diferencias por provincia. No concluyas superioridad de
la atención mirando solo un fold favorable. Las figuras conservan la etiqueta
PILOT si ejecutaste el modo rápido.
''')
add('code', '''display(ESTUDIO['audit'])
display(ESTUDIO['scores'].groupby(['scheme','model'])[['roc_auc','average_precision','brier']].agg(['mean','std']))
for figura in ESTUDIO['figures']:
    display(Markdown('### '+figura['figure']+'\\n\\n'+figura['caption']))
    display(Image(filename=str(ESTUDIO['folder']/'figuras'/(figura['figure']+'.png')),width=950))
''','display')

add('markdown','''## G. Orientación para redactar el paper

La guía siguiente usa los conteos, configuración, métricas y figuras realmente
generados. Distingue lo ejecutado de lo pendiente. Es una base de redacción,
no un texto que invente resultados o garantice aceptación editorial.

Incluye qué informar en introducción, métodos, resultados, discusión, figuras
principales y material suplementario. Revisa especialmente la procedencia del
inventario, la versión de extracción y la independencia del conjunto de prueba.
''')
add('code', "display(Markdown(ESTUDIO['report'].read_text(encoding='utf-8')))\nprint('Todo el estudio se encuentra en:',ESTUDIO['folder'])\n",'paper')

add('markdown','''## H. Guardado final y colaboración

Con Drive activado ya quedan guardados datos, auditoría, logs, particiones,
métricas, predicciones, figuras y la guía del paper. No necesitas trasladarlos
a otro cuaderno. La descarga de una copia completa es opcional y solo ocurre al final.

Para colaborar en GitHub, propone cambios mediante una rama y pull request.
No guardes salidas del cuaderno con datos privados en un commit. El cuaderno
público se distribuye con todas las celdas de código visibles y sin resultados incrustados.
''')
add('code', '''DESCARGAR_COPIA_FINAL=False
if DESCARGAR_COPIA_FINAL:
    import shutil
    from google.colab import files
    copia=shutil.make_archive(str(ESTUDIO['folder'])+'_copia','zip',ESTUDIO['folder'])
    files.download(copia)
print('Trabajo guardado:',ESTUDIO['folder'])
''','export')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python'),colab=dict(name='00_ESTUDIO_COMPLETO_INCENDIOS.ipynb')),cells=cells)
path=root/'notebooks/00_ESTUDIO_COMPLETO_INCENDIOS.ipynb'
path.write_text(json.dumps(nb,ensure_ascii=False,indent=2),encoding='utf-8')
print('Cuaderno único:',path,'|',len(cells),'celdas | carga de archivos única')
