"""Auditoría de trazabilidad sin entrenar modelos ni alterar datos originales."""
# %% Bibliotecas y reglas del protocolo documentado
from pathlib import Path
import hashlib
import json
import unicodedata
import numpy as np
import pandas as pd
from pyproj import Geod
from sklearn.neighbors import BallTree

NUMERIC_FEATURES=['elevacion','pendiente','orientacion','temperatura_C','precipitacion_mm','NDVI','EVI','NBR']
SIERRA={'azuay','bolivar','canar','carchi','chimborazo','cotopaxi','imbabura','loja','pichincha','tungurahua'}

def province_name(value):
    text=''.join(c for c in unicodedata.normalize('NFKD',str(value).lower()) if not unicodedata.combining(c)).strip(" '")
    return {'bolvar':'bolivar','caar':'canar'}.get(text,text)

# %% Controles estructurales y emparejamiento
def audit_dataset(dataset):
    df=pd.read_csv(dataset)
    required=NUMERIC_FEATURES+['id_registro','date','provincia','latitud','longitud','clase']
    if set(required)-set(df):
        raise ValueError('Esquema distinto al de la base trazable.')
    dates=pd.to_datetime(df.date,errors='coerce')
    result=dict(file=Path(dataset).name,sha256=hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),rows=len(df),
        nulls=df[required].isna().sum().astype(int).to_dict(),invalid_dates=int(dates.isna().sum()),
        nonfinite=int((~np.isfinite(df[NUMERIC_FEATURES+['latitud','longitud']].to_numpy(dtype=float))).sum()),
        duplicate_ids=int(df.id_registro.duplicated().sum()),
        duplicate_spacetime=int(df.duplicated(['clase','date','latitud','longitud']).sum()),
        recurrent_locations=int(df[['latitud','longitud']].duplicated().sum()),
        date_min=str(dates.min().date()),date_max=str(dates.max().date()),
        classes={str(k):int(v) for k,v in df.clase.value_counts().items()})
    df['pair_id']=df.id_registro.astype(str).str.extract(r'^(?:P_|PA_)(\d+)',expand=False)
    groups=df.groupby('pair_id')
    result['pair_checks']=dict(unparseable=int(df.pair_id.isna().sum()),
        wrong_size=int((groups.size()!=2).sum()),wrong_classes=int((groups.clase.nunique()!=2).sum()),
        different_date=int((groups.date.nunique()!=1).sum()),different_province=int((groups.provincia.nunique()!=1).sum()))
    result['coordinate_range_valid']=bool(df.latitud.between(-90,90).all() and df.longitud.between(-180,180).all())
    result['spectral_range_violations']=dict(NDVI=int((~df.NDVI.between(-1,1)).sum()),NBR=int((~df.NBR.between(-1,1)).sum()))
    result['negative_precipitation']=int((df.precipitacion_mm<0).sum())
    return df,result

# %% Contraste con tablas fuente y ámbito explícito de exclusión
def audit_sources(df,presence_path,pseudo_path,inventory_path):
    presence=pd.read_csv(presence_path); pseudo=pd.read_csv(pseudo_path); raw=pd.read_csv(inventory_path)
    positive=df[df.clase==1].copy(); positive['id']=positive.id_registro.str.removeprefix('P_').astype(int)
    negative=df[df.clase==0].copy()
    psource=presence.drop_duplicates(['date','latitud','longitud'])
    cols=['latitud','longitud',*NUMERIC_FEATURES]
    pcheck=positive.merge(psource,on='id',suffixes=('_model','_source'),validate='one_to_one')
    acheck=negative.merge(pseudo,left_on='id_registro',right_on='id_pseudo',suffixes=('_model','_source'),validate='one_to_one')
    refs=pseudo.merge(positive[['id','date','provincia']],left_on='id_presencia_referencia',right_on='id',how='left',suffixes=('_a','_p'),validate='one_to_one')
    result=dict(source_presence_rows=len(presence),unique_presence_spacetime=len(psource),
        positive_matches=len(pcheck),negative_matches=len(acheck),
        positive_values_equal=all(np.allclose(pcheck[c+'_model'],pcheck[c+'_source']) for c in cols),
        negative_values_equal=all(np.allclose(acheck[c+'_model'],acheck[c+'_source']) for c in cols),
        missing_references=int(refs.id.isna().sum()),different_reference_date=int((refs.date_a!=refs.date_p).sum()),
        different_reference_province=int((refs.provincia_a!=refs.provincia_p).sum()),
        zero_valid_s2_presence=int((presence.n_observaciones_S2_validas<=0).sum()),
        zero_valid_s2_pseudo=int((pseudo.n_observaciones_S2_validas<=0).sum()))
    years=pd.to_datetime(raw.date,errors='coerce').dt.year
    # La bitácora define 2019–2025 y diez provincias, no todo el inventario 2010–2026.
    known=raw[years.between(2019,2025)&raw.provincia.map(province_name).isin(SIERRA)].dropna(subset=['latitude','longitude'])
    result['exclusion_scope']='Sierra administrative provinces, all known events 2019–2025, including spectrally unusable events'
    result['known_reference_rows']=len(known)
    if len(known):
        tree=BallTree(np.deg2rad(known[['latitude','longitude']].to_numpy()),metric='haversine')
        # Filtra con los cinco vecinos esféricos y refina en el elipsoide WGS84.
        # El elipsoide cambia el orden solo localmente; k=5 es un refinamiento operativo.
        _,indices=tree.query(np.deg2rad(negative[['latitud','longitud']].to_numpy()),k=min(5,len(known)))
        k=indices.shape[1]
        _,_,dist=Geod(ellps='WGS84').inv(np.repeat(negative.longitud.to_numpy(),k),np.repeat(negative.latitud.to_numpy(),k),
            known.longitude.to_numpy()[indices.ravel()],known.latitude.to_numpy()[indices.ravel()])
        distances=dist.reshape(-1,k).min(axis=1)
        result['minimum_exclusion_distance_wgs84_m']=float(distances.min())
        result['pseudo_under_2000_m']=int((distances<2000).sum())
        result['exclusion_distance_quantiles_m']={str(q):float(np.quantile(distances,q)) for q in [0,.01,.5,.99,1]}
    result['temporal_acquisition_verified']=False
    result['temporal_note']='CSV values and QC counters do not contain per-image acquisition timestamps; pre-event windows are documented but require extraction code/provenance to verify.'
    return result

# %% Ejecución y salidas locales
def run_audit(dataset,output,presence=None,pseudo=None,inventory=None):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    df,result=audit_dataset(dataset)
    if all(p is not None for p in [presence,pseudo,inventory]):
        result['source_checks']=audit_sources(df,presence,pseudo,inventory)
    else:
        result['source_checks']='not_run: upload source presence, pseudoabsence and full inventory CSVs'
    df['year']=pd.to_datetime(df.date).dt.year
    for group in ['provincia','year']:
        df.groupby([group,'clase']).size().unstack(fill_value=0).to_csv(output/f'balance_{group}.csv')
    df[NUMERIC_FEATURES].describe(percentiles=[.01,.5,.99]).to_csv(output/'feature_ranges.csv')
    (output/'audit.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    return result
