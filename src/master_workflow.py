"""Orquestación del estudio en un solo cuaderno, con etapas reutilizables."""
# %% Configuración del flujo completo
from pathlib import Path
import contextlib
import copy
import hashlib
import json
import sys
import importlib.metadata
import inspect
import types
import pandas as pd

MASTER_DEFAULTS=dict(mode='COMPLETO',force=False,run_ablations=True,run_explanations=True,
    epochs=40,patience=8,seeds=[42,137,271],rf_trees=300,
    block_km=20,buffer_km=5,grid_offset_km=0,n_splits=5,
    ablation_models=['logistic','rf','histgb'],ablation_seeds=[42],
    explanation_cases=300,background_cases=50,coalitions=256,
    permutation_cases=1000,permutation_repeats=5,
    source_institution='',source_version='',extraction_status='Pendiente: reconciliar bitácora y script GEE')

class WorkflowLog:
    def __init__(self,*streams):self.streams=streams
    def write(self,text):
        for stream in self.streams:stream.write(text);stream.flush()
        return len(text)
    def flush(self):
        for stream in self.streams:stream.flush()


def callable_fingerprint(function):
    """Detecta ediciones de una función sin depender del nombre temporal de la celda."""
    try:
        content=inspect.getsource(function).replace('\r\n','\n')
    except (OSError,TypeError):
        def normalized(code):
            constants=[]
            for value in code.co_consts:
                constants.append(normalized(value) if isinstance(value,types.CodeType) else repr(value))
            return dict(bytecode=code.co_code.hex(),constants=constants,names=code.co_names,
                variables=code.co_varnames,argcount=code.co_argcount,kwonly=code.co_kwonlyargcount,
                freevars=code.co_freevars,cellvars=code.co_cellvars)
        content=json.dumps(normalized(function.__code__),sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()


def runtime_fingerprint(api):
    names=['load_data','design_matrix','spatial_groups','nearest_m','purge','make_splits',
        'inner_validation','classic_model','neural_model','fit_neural','metrics','paired_block_bootstrap','run_validation']
    values=dict(functions={name:callable_fingerprint(api[name]) for name in names},
        features=api['FEATURES'],defaults=api['DEFAULT'])
    return hashlib.sha256(json.dumps(values,sort_keys=True).encode()).hexdigest(),values


def stage_notice(stage,message):
    print('\n'+'='*72+'\n'+stage+'\n'+message+'\n'+'='*72,flush=True)


def master_config(settings,api):
    cfg=copy.deepcopy(api['DEFAULT'])
    for name in ['epochs','patience','seeds','rf_trees','block_km','buffer_km','grid_offset_km','n_splits']:
        cfg[name]=copy.deepcopy(settings[name])
    if settings['mode']=='RAPIDO':
        cfg.update(pilot=True,seeds=[42],epochs=min(cfg['epochs'],8),patience=min(cfg['patience'],3),rf_trees=min(cfg['rf_trees'],100),bootstrap_reps=100)
    elif settings['mode']=='COMPLETO':cfg['pilot']=False
    else:raise ValueError('Elige COMPLETO o RAPIDO.')
    return cfg


def reuse_or_run(dataset,folder,config,api,code_hash,force=False):
    """Reutiliza únicamente una ejecución íntegra de los mismos datos/código/configuración."""
    folder=Path(folder);meta_file=folder/'execution.json'
    desired=copy.deepcopy(api['DEFAULT']);desired.update(config)
    digest=hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    if not force and meta_file.exists():
        previous=json.loads(meta_file.read_text(encoding='utf-8'))
        files=['metrics.csv','predictions.csv','split_audit.csv']
        versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','pyproj']}
        if previous.get('status')=='completed' and previous.get('config')==desired and previous.get('data_sha256')==digest and previous.get('code_sha256')==code_hash and previous.get('versions')==versions and previous.get('python')==sys.version and all((folder/name).is_file() for name in files):
            print('Etapa ya completada y compatible; se reutiliza:',folder.name,flush=True)
            return pd.read_csv(folder/'metrics.csv')
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'console.log').open('w',encoding='utf-8') as log:
        with contextlib.redirect_stdout(WorkflowLog(sys.stdout,log)):
            return api['run_validation'](dataset,folder,config)


def table_markdown(frame):
    cols=[str(c) for c in frame.columns]
    lines=['| '+' | '.join(cols)+' |','|'+'|'.join(['---']*len(cols))+'|']
    for row in frame.itertuples(index=False,name=None):
        values=[]
        for value in row:
            if isinstance(value,float):values.append('—' if pd.isna(value) else f'{value:.4f}')
            else:values.append(str(value).replace('|','/'))
        lines.append('| '+' | '.join(values)+' |')
    return '\n'.join(lines)

# %% Informe orientado al manuscrito, basado solo en resultados disponibles
def write_paper_guide(folder,df,scores,audit,settings,config,figure_manifest):
    folder=Path(folder)
    summary=scores.groupby(['scheme','model']).agg(AUC=('roc_auc','mean'),AP=('average_precision','mean'),Brier=('brier','mean'),evaluaciones=('roc_auc','size')).reset_index()
    summary.to_csv(folder/'resumen_modelos.csv',index=False)
    province=scores[scores.scheme=='leave_province_out'].groupby(['fold','model'])[['roc_auc','average_precision']].mean().reset_index()
    province.to_csv(folder/'resumen_provincias.csv',index=False)
    status='PILOTO TÉCNICO; no resultados definitivos' if config['pilot'] else 'Ejecución del protocolo completo configurado; requiere revisión científica'
    count=df.clase.value_counts().to_dict()
    figures='\n'.join('- **'+m['figure']+'**: '+m['caption'] for m in figure_manifest)
    source_checks=audit.get('source_checks')
    source_status='Contrastadas las tablas fuente aportadas; revisar el JSON de auditoría.' if isinstance(source_checks,dict) else 'No se aportaron todas las tablas fuente: no se verificaron contra ellas el buffer ni los controles de Sentinel-2.'
    guide=f'''# Guía de redacción del paper

Estado de esta ejecución: **{status}**.

## Qué datos y versión se analizaron

Observaciones: {len(df):,}. Presencias: {count.get(1,0):,}. Pseudoausencias: {count.get(0,0):,}.
Periodo del CSV: {df.date.min().date()} a {df.date.max().date()}.
Las coordenadas recurrentes en años distintos no deben tratarse como eventos independientes sin control geográfico.

Fuente institucional declarada: {settings['source_institution'] or 'PENDIENTE DE DOCUMENTAR'}.
Versión del inventario: {settings['source_version'] or 'PENDIENTE DE DOCUMENTAR'}.
Estado de extracción: {settings['extraction_status']}.
{source_status}

## Introducción

Explica el problema de transferencia entre territorios y periodos y formula una
pregunta verificable sobre el valor añadido de la atención. La arquitectura por sí
sola no demuestra novedad. La revisión del estado del arte y la revista objetivo
siguen siendo decisiones del manuscrito, no resultados que este código invente.

## Métodos que corresponden a esta ejecución

Se usaron ocho covariables físicas; orientación circular genera nueve canales
en la configuración completa. ID, fecha, provincia y coordenadas se conservan
para trazabilidad y particiones, no como predictores del modelo.

Desarrollo hasta {config['development_end']}; bloques nominales de {config['block_km']} km
en EPSG:6933; buffer entre entrenamiento y test de {config['buffer_km']} km esféricos;
test temporal {config['temporal_years']} con embargo de 90 días. Semillas: {config['seeds']}.
2025 no se evalúa por defecto. Su independencia histórica todavía debe comprobarse.

Los escaladores se ajustan solo sobre entrenamiento. Las épocas neuronales se
seleccionan con validación interna y luego se reajusta con el entrenamiento externo
completo. Se fijan hiperparámetros; esta implementación no hace búsqueda con CV anidada.
Los controles incluyen MLP sin atención y MLP con presupuesto de parámetros similar.

La exclusión global de pseudoausencias basada en incendios conocidos de todo el
periodo hace que el contraste temporal sea retrospectivo. No debe describirse
como un pronóstico operacional. Tampoco debe afirmarse calibración poblacional:
se trata de un muestreo caso-control.

## Resultados disponibles

Los promedios siguientes son descriptivos. Folds y semillas no son réplicas
independientes para construir errores estándar o pruebas de significación.

{table_markdown(summary)}

Examina `resumen_provincias.csv` antes de generalizar un promedio nacional.
Compara los tipos de error, PR y Brier además de ROC-AUC.
Las diferencias pareadas de atención y sus intervalos están en
`principal/attention_deltas.csv`: usa exactamente el mismo test y no selecciones
solo los folds favorables. Un intervalo que incluya cero no sostiene una ventaja
clara en ese contraste.

## Ablaciones y explicabilidad

Ablaciones ejecutadas: {settings['run_ablations']}. Modelos de esas ablaciones:
{settings['ablation_models']}; semillas: {settings['ablation_seeds']}.
No atribuyas resultados de los modelos clásicos a redes que no se hayan evaluado
en esas ablaciones. Las campañas nuevas de pseudoausencias no se simulan en este
flujo: requieren el código GEE exacto y extracciones independientes comparables.

Explicabilidad solicitada: {settings['run_explanations']}. SHAP y permutación
describen el modelo y dependen del background, correlaciones y muestra. Los
pesos de atención son routing entre tokens, no efectos causales ni SHAP.
Comprueba `figuras/explanation_checks.json` si se generó esta etapa.

## Figuras y su función en el argumento

{figures}

F01 muestra puntos y particiones, no una superficie de susceptibilidad. Para un
mapa continuo deben prepararse predictores espaciales compatibles, dominio de
aplicación, soporte temporal y extrapolación. Los mapas históricos no se importan
como si pertenecieran automáticamente a este ajuste.

## Discusión y conclusiones

Relaciona los resultados territoriales con cobertura ambiental y limitaciones
de muestreo antes de proponer mecanismos ecológicos. Conserva resultados
negativos: la atención puede no mejorar al control de capacidad.
No reutilices cifras, importancias o interpretaciones de la versión antigua
del dataset en las figuras nuevas sin contrastar su procedencia.

## Pendientes antes del envío

- Fuente y versión oficiales del inventario, permisos y acceso a datos.
- Reconciliar clima de 30 días frente a 12 meses y exclusión de 2 km frente a 1–8 km
  entre la bitácora y el script aportado. Este cuaderno no modifica los CSV para ocultarlo.
- Verificar la adquisición preevento con el código de extracción exacto.
- Revisar sensibilidad de bloques/buffers y campañas reales de pseudoausencias.
- Comprobar un conjunto externo realmente independiente del trabajo histórico.
- Revisar estabilidad de SHAP y el dominio de los mapas antes de afirmar causalidad o utilidad operacional.

La ejecución reproducible organiza la evidencia; no garantiza aceptación Q1.
'''
    target=folder/'GUIA_PARA_REDACTAR_EL_PAPER.md';target.write_text(guide,encoding='utf-8')
    return target

# %% Ejecución única: auditoría → validación → ablaciones → figuras → informe
def complete_study(dataset,storage,api,settings=None,presence=None,pseudo=None,inventory=None,core_hash=None):
    cfg_settings=copy.deepcopy(MASTER_DEFAULTS);cfg_settings.update(settings or {})
    config=master_config(cfg_settings,api)
    df=api['load_data'](dataset)
    data_hash=hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    template_hash=api.get('NOTEBOOK_TEMPLATE_SHA',core_hash)
    core_hash,fingerprint=runtime_fingerprint(api)
    # La huella efectiva corresponde al código ejecutado, incluso si se editó una celda.
    api['NOTEBOOK_IMPLEMENTATION_SHA']=core_hash
    api['run_validation'].__globals__['NOTEBOOK_IMPLEMENTATION_SHA']=core_hash
    explanation_hash=callable_fingerprint(api['explain_paper_figures'])
    api['EXPLANATION_CODE_SHA']=explanation_hash
    api['explain_paper_figures'].__globals__['EXPLANATION_CODE_SHA']=explanation_hash
    identity=hashlib.sha256(json.dumps(dict(config=config,ablations=cfg_settings['ablation_models'],ablation_seeds=cfg_settings['ablation_seeds']),sort_keys=True).encode()).hexdigest()
    folder=Path(storage)/'estudios'/f'{data_hash[:10]}_{core_hash[:8]}_{identity[:8]}'
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'runtime_code_fingerprint.json').write_text(json.dumps(dict(sha256=core_hash,template_sha256=template_hash,**fingerprint),indent=2),encoding='utf-8')
    environment={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','pyproj','matplotlib','shap']}
    (folder/'environment.json').write_text(json.dumps(dict(python=sys.version,packages=environment),indent=2),encoding='utf-8')
    (folder/'requirements_ejecucion.txt').write_text('\n'.join(f'{name}=={version}' for name,version in environment.items())+'\n',encoding='utf-8')
    (folder/'configuration.json').write_text(json.dumps(dict(settings=cfg_settings,config=config,data_sha256=data_hash,core_sha256=core_hash),indent=2,ensure_ascii=False),encoding='utf-8')
    stage_notice('1/6 · Auditoría','Se comprueban datos y tablas fuente disponibles.')
    audit=api['run_audit'](dataset,folder/'auditoria',presence,pseudo,inventory)
    stage_notice('2/6 · Plan de validación','Se comprueban todas las particiones antes de entrenar.')
    planned=[]
    for seed in config['seeds']:
        for scheme,fold,train,test in api['make_splits'](df,dict(config,seed=seed)):
            planned.append(dict(scheme=scheme,fold=fold,seed=seed,train=len(train),test=len(test)))
    plan=pd.DataFrame(planned);plan.to_csv(folder/'plan_particiones.csv',index=False)
    print('Evaluaciones principales:',len(plan)*len(config['models']),flush=True)
    print(plan.to_string(index=False),flush=True)
    if not config['pilot']:print('Modo COMPLETO: puede tardar varias horas en CPU. Las etapas terminadas se conservan y reutilizan.',flush=True)
    stage_notice('3/6 · Comparación principal','Todos los modelos usan las mismas particiones externas.')
    scores=reuse_or_run(dataset,folder/'principal',config,api,core_hash,cfg_settings['force'])
    ablations={}
    if cfg_settings['run_ablations']:
        stage_notice('4/6 · Ablaciones','Se retira espectro o clima y se contrasta orientación cruda.')
        for feature_set in ['no_spectral','no_climate','raw_aspect']:
            variant=copy.deepcopy(config);variant.update(feature_set=feature_set,models=cfg_settings['ablation_models'],seeds=cfg_settings['ablation_seeds'])
            if config['pilot']:variant['seeds']=[42]
            destination=folder/('ablacion_'+feature_set)
            reuse_or_run(dataset,destination,variant,api,core_hash,cfg_settings['force'])
            ablations[feature_set]=destination
    stage_notice('5/6 · Galería de figuras','Se usan directamente las predicciones de las etapas anteriores.')
    figures=folder/'figuras'
    manifest=api['build_paper_figures'](folder/'principal',figures,dataset=dataset,inventory=inventory,
        reference_results=folder/'principal',ablations=ablations,seed=config['seeds'][0])
    if cfg_settings['run_explanations']:
        expected=dict(n_explained=min(cfg_settings['explanation_cases'],len(df[df.year==2024])),background_n=cfg_settings['background_cases'],coalitions=cfg_settings['coalitions'],
            permutation_cases=min(cfg_settings['permutation_cases'],len(df[df.year==2024])),permutation_repeats=cfg_settings['permutation_repeats'],
            predictions_sha256=hashlib.sha256((folder/'principal/predictions.csv').read_bytes()).hexdigest(),explainer_version=importlib.metadata.version('shap'),explanation_code_sha256=explanation_hash)
        cache=figures/'explanation_render_metadata.json';reuse=False
        if cache.exists() and (figures/'explanation_plot_data.npz').exists() and not cfg_settings['force']:
            previous=json.loads(cache.read_text(encoding='utf-8'))['provenance']
            reuse=all(previous.get(k)==v for k,v in expected.items()) and previous.get('data_sha256')==data_hash and previous.get('seed')==config['seeds'][0]
        if reuse:
            print('SHAP ya calculado: se redibuja desde el caché comprobado.',flush=True)
            manifest=api['render_explanation_cache'](figures)
        else:
            manifest=api['explain_paper_figures'](folder/'principal',figures,dataset,api=api,
                n_shap=cfg_settings['explanation_cases'],n_background=cfg_settings['background_cases'],coalitions=cfg_settings['coalitions'],
                permutation_n=cfg_settings['permutation_cases'],permutation_repeats=cfg_settings['permutation_repeats'],seed=config['seeds'][0])
    stage_notice('6/6 · Orientación para el manuscrito','Se genera una guía basada en los resultados realmente disponibles.')
    report=write_paper_guide(folder,df,scores,audit,cfg_settings,config,manifest)
    (folder/'workflow_completed.json').write_text(json.dumps(dict(status='completed',data_sha256=data_hash,core_sha256=core_hash,figures=len(manifest)),indent=2),encoding='utf-8')
    print('Estudio guardado en:',folder,'\nGuía para el paper:',report,flush=True)
    return dict(folder=folder,scores=scores,audit=audit,figures=manifest,report=report,config=config)
