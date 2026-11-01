"""Validación reproducible de la base trazable 2019–2025.

Las predicciones son discriminación caso-control retrospectiva, no probabilidades
poblacionales de incendio. Los resultados se escriben localmente, nunca se suben.
"""
# %% Bibliotecas y configuración
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.metadata
import json
import random
import time
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupShuffleSplit, LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.neighbors import BallTree
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ['elevacion', 'pendiente', 'orientacion', 'temperatura_C', 'precipitacion_mm', 'NDVI', 'EVI', 'NBR']
DEFAULT = dict(seed=42, block_km=20, grid_offset_km=0, buffer_km=5, n_splits=5,
    development_end=2023, temporal_years=[2024], allow_final_2025=False,
    models=['logistic', 'rf', 'histgb', 'mlp', 'mlp_matched', 'attention'],
    feature_set='complete', epochs=40, patience=8, batch_size=256,
    learning_rate=0.001, token_dim=16, head_width=64, rf_trees=300, rf_depth=15,
    seeds=[42, 137, 271], pilot=False, bootstrap_reps=500,
    schemes=['random_pairs','spatial_blocks','leave_province_out','temporal_retrospective'])

# %% Esquema y trazabilidad
def load_data(path):
    """Exige el esquema nuevo. No mezcla la base antigua de 25.901 filas."""
    df = pd.read_csv(path)
    required = set(FEATURES + ['id_registro', 'date', 'provincia', 'latitud', 'longitud', 'clase'])
    if required - set(df.columns):
        raise ValueError('Columnas faltantes: ' + str(sorted(required-set(df.columns))))
    if df[list(required)].isna().any().any():
        raise ValueError('Hay valores faltantes; audita el origen antes de modelar.')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    df['year'] = df.date.dt.year
    if not df.year.between(2019, 2025).all() or set(df.clase.unique()) != {0, 1}:
        raise ValueError('Periodo o clases fuera del protocolo 2019–2025.')
    if not np.isfinite(df[FEATURES + ['latitud','longitud']].to_numpy(dtype=float)).all():
        raise ValueError('Hay valores no finitos.')
    if not df.latitud.between(-90,90).all() or not df.longitud.between(-180,180).all():
        raise ValueError('Coordenadas inválidas.')
    if df.id_registro.duplicated().any() or df.duplicated(['clase','date','latitud','longitud']).any():
        raise ValueError('Hay IDs o eventos espacio-temporales duplicados.')
    # El sufijo de reemplazo de una pseudoausencia no cambia su presencia de referencia.
    df['pair_id'] = df.id_registro.astype(str).str.extract(r'^(?:P_|PA_)(\d+)', expand=False)
    if df.pair_id.isna().any():
        raise ValueError('No se puede recuperar la referencia; conserva su columna explícita.')
    grouped = df.groupby('pair_id')
    if not grouped.size().eq(2).all() or not grouped.clase.nunique().eq(2).all():
        raise ValueError('Cada referencia debe tener una presencia y una pseudoausencia.')
    if grouped.date.nunique().max() != 1 or grouped.provincia.nunique().max() != 1:
        raise ValueError('El emparejamiento temporal/provincial no coincide.')
    return df.reset_index(drop=True)


def design_matrix(df, feature_set='complete'):
    """Ocho variables físicas; orientación circular se convierte a seno y coseno."""
    names = FEATURES.copy()
    if feature_set == 'no_spectral':
        names = [n for n in names if n not in ['NDVI', 'EVI', 'NBR']]
    elif feature_set == 'no_climate':
        names = [n for n in names if n not in ['temperatura_C', 'precipitacion_mm']]
    elif feature_set not in ['complete', 'raw_aspect']:
        raise ValueError('Ablación desconocida: ' + feature_set)
    X = df[names].copy()
    if feature_set != 'raw_aspect':
        angle = np.deg2rad(X.pop('orientacion'))
        X['orientacion_sin'] = np.sin(angle)
        X['orientacion_cos'] = np.cos(angle)
    return X


def spatial_groups(df, block_km, offset_km=0):
    # Proyección métrica de área equivalente; identificadores de celda, nunca predictores.
    transformer = Transformer.from_crs('EPSG:4326', 'EPSG:6933', always_xy=True)
    x, y = transformer.transform(df.longitud.to_numpy(), df.latitud.to_numpy())
    gx, gy = np.floor((np.asarray(x)-offset_km*1000)/(block_km*1000)).astype(int), np.floor((np.asarray(y)-offset_km*1000)/(block_km*1000)).astype(int)
    return pd.Series([f'{a}:{b}' for a,b in zip(gx,gy)], index=df.index)


def nearest_m(train, test):
    if not len(train) or not len(test):
        return np.array([])
    tree = BallTree(np.deg2rad(test[['latitud','longitud']].to_numpy()), metric='haversine')
    return tree.query(np.deg2rad(train[['latitud','longitud']].to_numpy()), k=1)[0][:,0] * 6371008.8


def purge(df, train, test, buffer_km=0):
    """Retira referencias emparejadas y, cuando corresponde, vecinos del test."""
    train, test = np.asarray(train), np.asarray(test)
    train = train[~df.loc[train,'pair_id'].isin(df.loc[test,'pair_id']).to_numpy()]
    if buffer_km:
        train = train[nearest_m(df.loc[train],df.loc[test]) >= buffer_km*1000]
    if len(train)<100 or df.loc[train,'clase'].nunique()!=2 or df.loc[test,'clase'].nunique()!=2:
        raise ValueError('Partición insuficiente después de la exclusión. Revisa diseño y cobertura.')
    if set(train)&set(test) or set(df.loc[train,'pair_id'])&set(df.loc[test,'pair_id']):
        raise AssertionError('Fuga de filas o referencias entre conjuntos.')
    return train, test

# %% Particiones externas independientes del ajuste de modelos
def make_splits(df, config):
    """2025 permanece reservado; el piloto usa un fold por diseño."""
    cfg = config
    if 2025 in cfg['temporal_years'] and not cfg['allow_final_2025']:
        raise ValueError('2025 está reservado. Congela el protocolo antes de habilitarlo.')
    if cfg['development_end'] >= 2025 and not cfg['allow_final_2025']:
        raise ValueError('El desarrollo no puede incluir 2025 mientras permanezca reservado.')
    dev = df.index[df.year <= cfg['development_end']].to_numpy()
    groups = spatial_groups(df,cfg['block_km'],cfg.get('grid_offset_km',0))
    rows = []
    splitter = GroupShuffleSplit(n_splits=1,test_size=.2,random_state=cfg['seed'])
    tr,te = next(splitter.split(dev,df.loc[dev,'clase'],df.loc[dev,'pair_id']))
    rows.append(('random_pairs', 'diagnostic', *purge(df,dev[tr],dev[te])))
    splitter = StratifiedGroupKFold(n_splits=cfg['n_splits'],shuffle=True,random_state=cfg['seed'])
    for i,(tr,te) in enumerate(splitter.split(dev,df.loc[dev,'clase'],groups.loc[dev])):
        train,test = purge(df,dev[tr],dev[te],cfg['buffer_km'])
        rows.append(('spatial_blocks',f'fold_{i+1}',train,test))
        if cfg['pilot']:
            break
    for tr,te in LeaveOneGroupOut().split(dev,df.loc[dev,'clase'],df.loc[dev,'provincia']):
        train,test = purge(df,dev[tr],dev[te],cfg['buffer_km'])
        province = str(df.loc[test,'provincia'].iloc[0])
        rows.append(('leave_province_out',province,train,test))
        if cfg['pilot']:
            break
    for year in cfg['temporal_years']:
        start = pd.Timestamp(year=year,month=1,day=1)
        # Embargo de 90 días: evita superponer ventanas espectrales a ambos lados del corte.
        tr = df.index[df.date < start-pd.Timedelta(days=90)].to_numpy()
        te = df.index[df.year == year].to_numpy()
        train,test = purge(df,tr,te)
        rows.append(('temporal_retrospective',str(year),train,test))
    return rows


def inner_validation(df, train, scheme, config):
    """Validación interna para early stopping; nunca consulta el test externo."""
    if scheme == 'temporal_retrospective':
        last_year = int(df.loc[train,'year'].max())
        boundary = pd.Timestamp(year=last_year,month=1,day=1)
        tr = train[df.loc[train,'date'].to_numpy() < (boundary-pd.Timedelta(days=90)).to_datetime64()]
        va = train[df.loc[train,'year'].to_numpy() == last_year]
        return purge(df,tr,va)
    grouping = spatial_groups(df,config['block_km'],config.get('grid_offset_km',0)) if scheme != 'random_pairs' else df.pair_id
    split = GroupShuffleSplit(n_splits=1,test_size=.2,random_state=config['seed'])
    tr,va = next(split.split(train,df.loc[train,'clase'],grouping.loc[train]))
    return purge(df,train[tr],train[va],config['buffer_km'] if scheme!='random_pairs' else 0)

# %% Modelos de referencia con hiperparámetros fijados antes del test
def classic_model(name, cfg):
    if name == 'logistic':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=1000,random_state=cfg['seed']))
    if name == 'rf':
        return RandomForestClassifier(n_estimators=cfg['rf_trees'],max_depth=cfg['rf_depth'],min_samples_leaf=2,n_jobs=-1,random_state=cfg['seed'])
    if name == 'histgb':
        # Evita la validación aleatoria interna automática del estimador.
        return HistGradientBoostingClassifier(max_iter=150,max_leaf_nodes=15,l2_regularization=1,early_stopping=False,random_state=cfg['seed'])
    raise ValueError(name)

# %% Ablación de atención con representación y clasificador compartidos
def neural_model(features, attention, cfg):
    import torch
    from torch import nn

    class FeatureTokens(nn.Module):
        def __init__(self):
            super().__init__()
            dim = cfg['token_dim']
            width = cfg.get('head_width',64)
            self.scale = nn.Parameter(torch.randn(features,dim)*.02)
            self.bias = nn.Parameter(torch.randn(features,dim)*.02)
            self.attention = nn.MultiheadAttention(dim,4,dropout=.1,batch_first=True) if attention else None
            self.norm = nn.LayerNorm(dim)
            self.head = nn.Sequential(nn.Flatten(),nn.Linear(features*dim,width),nn.ReLU(),nn.Dropout(.1),nn.Linear(width,32),nn.ReLU(),nn.Linear(32,1))

        def forward(self,x):
            # [tanda, variables] → [tanda, variables, dimensión del token].
            tokens = x.unsqueeze(-1)*self.scale.unsqueeze(0)+self.bias.unsqueeze(0)
            if self.attention is not None:
                mixed,_ = self.attention(tokens,tokens,tokens,need_weights=False)
                tokens = tokens+mixed
            return self.head(self.norm(tokens)).squeeze(-1)

    return FeatureTokens()


def fit_neural(X, y, df, train, test, scheme, attention, cfg, on_fitted=None):
    """Selecciona épocas internamente y vuelve a ajustar en todo el train externo."""
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    torch.set_num_threads(min(4,torch.get_num_threads()))
    torch.use_deterministic_algorithms(True)
    device = torch.device('cpu')  # Comparabilidad local/Colab; habilitar GPU exige otra verificación.
    tr,va = inner_validation(df,train,scheme,cfg)
    scaler = StandardScaler().fit(X[tr])
    xtr = torch.tensor(scaler.transform(X[tr]),dtype=torch.float32)
    xva = torch.tensor(scaler.transform(X[va]),dtype=torch.float32,device=device)
    ytr = torch.tensor(y[tr],dtype=torch.float32)
    yva = torch.tensor(y[va],dtype=torch.float32,device=device)
    def reset_model():
        random.seed(cfg['seed']); np.random.seed(cfg['seed']); torch.manual_seed(cfg['seed'])
        return neural_model(X.shape[1],attention,cfg).to(device)
    model = reset_model()
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(xtr,ytr),batch_size=cfg['batch_size'],shuffle=True)
    history=[]; best=float('inf'); best_epoch=1; stale=0
    for epoch in range(1,cfg['epochs']+1):
        model.train(); total=0.
        for bx,by in loader:
            optimizer.zero_grad(); loss=criterion(model(bx.to(device)),by.to(device)); loss.backward(); optimizer.step()
            total+=float(loss.detach())*len(bx)
        model.eval()
        with torch.no_grad():
            val_loss=float(criterion(model(xva),yva))
        history.append(dict(epoch=epoch,training_loss=total/len(tr),validation_loss=val_loss))
        print(f"  {'attention' if attention else 'mlp'} época {epoch}: train={total/len(tr):.4f}, val={val_loss:.4f}",flush=True)
        if val_loss<best-1e-5:
            best=val_loss; best_epoch=epoch; stale=0
        else:
            stale+=1
        if stale>=cfg['patience']:
            break
    # Reajuste completo: el test externo sigue sin utilizarse.
    scaler=StandardScaler().fit(X[train]); model=reset_model()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    dataset=TensorDataset(torch.tensor(scaler.transform(X[train]),dtype=torch.float32),torch.tensor(y[train],dtype=torch.float32))
    loader=DataLoader(dataset,batch_size=cfg['batch_size'],shuffle=True)
    for epoch in range(best_epoch):
        model.train()
        for bx,by in loader:
            optimizer.zero_grad(); loss=criterion(model(bx.to(device)),by.to(device)); loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        probabilities=torch.sigmoid(model(torch.tensor(scaler.transform(X[test]),dtype=torch.float32,device=device))).cpu().numpy()
    if on_fitted is not None:
        # Permite explicar el modelo reajustado sin alterar predicciones ni métricas.
        on_fitted(model,scaler)
    return probabilities,dict(selected_epochs=best_epoch,parameters=sum(p.numel() for p in model.parameters()),history=history)

# %% Métricas y registro de cada partición
def metrics(y,p):
    predicted=(p>=.5).astype(int)
    tn,fp,fn,tp=confusion_matrix(y,predicted,labels=[0,1]).ravel()
    return dict(roc_auc=roc_auc_score(y,p),average_precision=average_precision_score(y,p),
        accuracy=accuracy_score(y,predicted),balanced_accuracy=balanced_accuracy_score(y,predicted),
        precision=precision_score(y,predicted,zero_division=0),recall=recall_score(y,predicted,zero_division=0),
        f1=f1_score(y,predicted,zero_division=0),brier=brier_score_loss(y,p),log_loss=log_loss(y,p,labels=[0,1]),
        tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))


def paired_block_bootstrap(frame,baseline='mlp',challenger='attention',reps=500,seed=42):
    """Intervalo aproximado de diferencia AUC; remuestrea bloques, no filas."""
    pivot=frame.pivot(index=['id_registro','clase','spatial_group'],columns='model',values='probability').dropna(subset=[baseline,challenger]).reset_index()
    groups=pivot.spatial_group.unique()
    if len(groups)<5:
        return dict(status='insufficient_blocks',blocks=len(groups))
    rng=np.random.default_rng(seed); estimates=[]
    locations={g:np.flatnonzero(pivot.spatial_group.to_numpy()==g) for g in groups}
    for _ in range(reps):
        sample=np.concatenate([locations[g] for g in rng.choice(groups,len(groups),replace=True)])
        subset=pivot.iloc[sample]
        if subset.clase.nunique()==2:
            estimates.append(roc_auc_score(subset.clase,subset[challenger])-roc_auc_score(subset.clase,subset[baseline]))
    if not estimates:
        return dict(status='insufficient_classes',blocks=len(groups))
    return dict(status='approximate_spatial_block_bootstrap',blocks=len(groups),replicates=len(estimates),
        delta_auc=float(roc_auc_score(pivot.clase,pivot[challenger])-roc_auc_score(pivot.clase,pivot[baseline])),
        lower=float(np.quantile(estimates,.025)),upper=float(np.quantile(estimates,.975)))


def run_validation(data_path,output_dir,config=None):
    cfg=copy.deepcopy(DEFAULT); cfg.update(config or {})
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    df=load_data(data_path); matrix=design_matrix(df,cfg['feature_set']); X=matrix.to_numpy(dtype=np.float32); y=df.clase.to_numpy()
    groups=spatial_groups(df,cfg['block_km'],cfg.get('grid_offset_km',0))
    metadata=dict(config=cfg,data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),features=list(matrix.columns),
        python=__import__('sys').version,versions={},status='running',interpretation='retrospective case-control discrimination; exclusion uses known fires across 2019-2025',
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest() if '__file__' in globals() else globals().get('NOTEBOOK_IMPLEMENTATION_SHA','not_recorded'))
    for package in ['numpy','pandas','scikit-learn','torch','pyproj']:
        try: metadata['versions'][package]=importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError: pass
    (output_dir/'execution.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    scores=[]; predictions=[]; split_rows=[]; deltas=[]
    try:
        for seed in cfg['seeds']:
            current=dict(cfg,seed=seed)
            for scheme,fold,train,test in make_splits(df,current):
                if scheme not in cfg['schemes']:
                    continue
                print(f'\n{scheme}/{fold}, seed={seed}, train={len(train)}, test={len(test)}',flush=True)
                distances=nearest_m(df.loc[train],df.loc[test])
                info=dict(scheme=scheme,fold=fold,seed=seed,train=len(train),test=len(test),
                    test_prevalence=float(y[test].mean()),minimum_train_test_m=float(distances.min()),
                    shared_spatial_cells=len(set(groups.loc[train])&set(groups.loc[test])),
                    shared_provinces=len(set(df.loc[train,'provincia'])&set(df.loc[test,'provincia'])))
                split_rows.append(info)
                assignment=df.loc[np.r_[train,test],['id_registro','pair_id']].copy()
                assignment['partition']=['train']*len(train)+['test']*len(test)
                assignment.to_csv(output_dir/f'split_{scheme}_{fold}_{seed}.csv',index=False)
                for name in cfg['models']:
                    started=time.monotonic()
                    if name in ['mlp','mlp_matched','attention']:
                        neural_cfg=current.copy()
                        if name=='mlp_matched':
                            # Equipara aproximadamente el presupuesto de parámetros sin usar el test.
                            dim=current['token_dim']; width=current.get('head_width',64)
                            neural_cfg['head_width']=width+round((4*dim*dim+4*dim)/(X.shape[1]*dim+33))
                        probability,training=fit_neural(X,y,df,train,test,scheme,name=='attention',neural_cfg)
                        training['head_width']=neural_cfg.get('head_width',64)
                        history=training.pop('history')
                        pd.DataFrame(history).to_csv(output_dir/f'learning_{scheme}_{fold}_{seed}_{name}.csv',index=False)
                    else:
                        model=classic_model(name,current); model.fit(X[train],y[train]); probability=model.predict_proba(X[test])[:,1]; training={}
                    score=dict(scheme=scheme,fold=fold,seed=seed,model=name,feature_set=cfg['feature_set'],**metrics(y[test],probability),**training,seconds=time.monotonic()-started)
                    scores.append(score); print(f"  {name}: AUC={score['roc_auc']:.4f}, AP={score['average_precision']:.4f}, Brier={score['brier']:.4f}",flush=True)
                    pred=df.loc[test,['id_registro','clase','provincia','year']].copy()
                    pred['spatial_group']=groups.loc[test].to_numpy(); pred['probability']=probability
                    pred['model']=name; pred['scheme']=scheme; pred['fold']=fold; pred['seed']=seed
                    predictions.append(pred)
                    pd.DataFrame(scores).to_csv(output_dir/'metrics.csv',index=False)
                if 'mlp' in cfg['models'] and 'attention' in cfg['models']:
                    frame=pd.concat(predictions[-len(cfg['models']):],ignore_index=True)
                    delta=paired_block_bootstrap(frame,reps=cfg['bootstrap_reps'],seed=seed)
                    deltas.append(dict(scheme=scheme,fold=fold,seed=seed,baseline='mlp',**delta))
                    if 'mlp_matched' in cfg['models']:
                        delta=paired_block_bootstrap(frame,baseline='mlp_matched',reps=cfg['bootstrap_reps'],seed=seed)
                        deltas.append(dict(scheme=scheme,fold=fold,seed=seed,baseline='mlp_matched',**delta))
                pd.DataFrame(split_rows).to_csv(output_dir/'split_audit.csv',index=False)
                pd.concat(predictions,ignore_index=True).to_csv(output_dir/'predictions.csv',index=False)
                pd.DataFrame(deltas).to_csv(output_dir/'attention_deltas.csv',index=False)
        metadata['status']='completed'
    except BaseException as exc:
        metadata['status']='failed'; metadata['error']=repr(exc); raise
    finally:
        (output_dir/'execution.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return pd.DataFrame(scores)

# %% Interfaz de terminal; el cuaderno muestra las funciones anteriores completas
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data',type=Path); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pilot',action='store_true'); parser.add_argument('--classical-only',action='store_true')
    args=parser.parse_args(); config=copy.deepcopy(DEFAULT)
    if args.pilot:
        config.update(pilot=True,seeds=[42],epochs=8,patience=3,rf_trees=100,bootstrap_reps=100)
    if args.classical_only:
        config['models']=['logistic','rf','histgb']
    run_validation(args.data,args.output,config)
