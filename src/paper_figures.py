"""Figuras reproducibles del artículo, calculadas desde salidas de validación.

No mezcla resultados de versiones de datos ni convierte un piloto en evidencia final.
Cada figura se exporta a PNG 600 dpi, SVG y PDF vectorial, con manifiesto y leyenda.
"""
# %% Estilo, nombres y exportación
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_curve,roc_auc_score,precision_recall_curve,average_precision_score,confusion_matrix,brier_score_loss
from sklearn.calibration import calibration_curve

ORDER=['logistic','rf','histgb','mlp','mlp_matched','attention']
LABELS={'logistic':'Logistic regression','rf':'Random Forest','histgb':'Histogram boosting','mlp':'MLP','mlp_matched':'MLP (capacity control)','attention':'Feature attention'}
COLORS={'logistic':'#666666','rf':'#0072B2','histgb':'#009E73','mlp':'#CC79A7','mlp_matched':'#56B4E9','attention':'#D55E00'}
SCHEMES={'random_pairs':'Random pairs','spatial_blocks':'Spatial blocks','leave_province_out':'Province held out','temporal_retrospective':'Temporal transfer'}
FEATURE_LABELS={'elevacion':'Elevation (m)','pendiente':'Slope (degrees)','orientacion':'Aspect (degrees)','temperatura_C':'Temperature (°C)','precipitacion_mm':'Precipitation (mm)','NDVI':'NDVI','EVI':'EVI','NBR':'NBR'}

def paper_style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
        'legend.fontsize':8,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})


def export_figure(fig,out,name,caption,manifest,provenance):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    if not getattr(fig,'_paper_layout_fixed',False):fig.tight_layout()
    files=[]
    for suffix in ['png','svg','pdf']:
        path=out/f'{name}.{suffix}'
        fig.savefig(path,dpi=600,bbox_inches='tight',pad_inches=.08)
        files.append(path.name)
    plt.close(fig)
    manifest.append(dict(figure=name,files=files,caption=caption,**provenance))
    (out/'figures_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    (out/'CAPTIONS.md').write_text('# Figure captions\n\n'+'\n\n'.join('**'+m['figure']+'** '+m['caption'] for m in manifest),encoding='utf-8')


def read_results(folder):
    folder=Path(folder)
    meta=json.loads((folder/'execution.json').read_text(encoding='utf-8'))
    if meta['status']!='completed':raise ValueError('La ejecución no está completada.')
    scores=pd.read_csv(folder/'metrics.csv',dtype={'fold':str})
    predictions=pd.read_csv(folder/'predictions.csv',dtype={'fold':str,'id_registro':str,'spatial_group':str})
    if predictions.duplicated(['scheme','fold','seed','model','id_registro']).any():raise ValueError('Predicciones duplicadas.')
    if not predictions.probability.between(0,1).all():raise ValueError('Scores fuera de rango.')
    return meta,scores,predictions


def choose_test(predictions,scheme,fold,seed):
    selected=predictions[(predictions.scheme==scheme)&(predictions.fold==str(fold))&(predictions.seed==seed)].copy()
    if selected.empty:raise ValueError('No existe la partición elegida.')
    ids=None
    for model in selected.model.unique():
        current=set(selected[selected.model==model].id_registro)
        if ids is not None and current!=ids:raise ValueError('Los modelos no comparten el mismo test.')
        ids=current
    if selected.groupby('id_registro').clase.nunique().max()!=1:raise ValueError('Etiquetas inconsistentes.')
    return selected

# %% Galería calculada desde predicciones guardadas: no reentrena
def build_paper_figures(results,out,dataset=None,inventory=None,reference_results=None,ablations=None,
                        scheme='temporal_retrospective',fold='2024',seed=42):
    paper_style();results=Path(results);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    meta,scores,pred=read_results(results);selected=choose_test(pred,scheme,fold,seed)
    manifest=[]
    status='PILOT' if meta['config'].get('pilot') else 'VALIDATION'
    stamp=f'{status} | {SCHEMES.get(scheme,scheme)}: {fold} | seed {seed}'
    source=dict(data_sha256=meta['data_sha256'],run_status=status,scheme=scheme,fold=str(fold),seed=seed)
    models=[m for m in ORDER if m in selected.model.unique()]
    def save(fig,name,caption,extra=None):
        export_figure(fig,out,name,caption,manifest,dict(source,**(extra or {})))
    df=None
    if dataset:
        if hashlib.sha256(Path(dataset).read_bytes()).hexdigest()!=meta['data_sha256']:
            raise ValueError('El dataset no corresponde a esta ejecución: hash distinto.')
        df=pd.read_csv(dataset,dtype={'id_registro':str});df['date']=pd.to_datetime(df.date)

    # F01: localización de las muestras y partición geográfica.
    if df is not None:
        geographic=scores[scores.scheme=='spatial_blocks']
        if not geographic.empty:
            f=str(geographic.iloc[0]['fold']);s=int(geographic.iloc[0]['seed'])
            split=pd.read_csv(results/f'split_spatial_blocks_{f}_{s}.csv',dtype={'id_registro':str})
            plotting=df.merge(split[['id_registro','partition']],on='id_registro',how='left')
            fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),sharex=True,sharey=True)
            for label,color in [(0,'#56B4E9'),(1,'#D55E00')]:
                part=plotting[plotting.clase==label]
                axes[0].scatter(part.longitud,part.latitud,s=1.4,color=color,alpha=.4,rasterized=True,label='Presence' if label else 'Pseudoabsence')
            for label,color in [('train','#0072B2'),('test','#D55E00')]:
                part=plotting[plotting.partition==label]
                axes[1].scatter(part.longitud,part.latitud,s=1.8,color=color,alpha=.45,rasterized=True,label=label.title())
            axes[0].set_title('(a) Available observations');axes[1].set_title(f'(b) Geographic partition: {f}')
            for ax in axes:
                ax.set_xlabel('Longitude (degrees)');ax.set_aspect(1/np.cos(np.deg2rad(plotting.latitud.mean())))
                ax.legend(markerscale=4,loc='best',frameon=False)
            axes[0].set_ylabel('Latitude (degrees)')
            save(fig,'F01_sampling_and_spatial_split','Sampling locations (all modeling years) and one geographic train/test partition from the development period. Points excluded by the spatial buffer are not part of the displayed train set. These are sampling points, not a susceptibility surface or administrative boundary map.',dict(scheme='spatial_blocks',fold=f,seed=s))

        # F02: el periodo histórico se mantiene explícitamente separado del dataset de modelado.
        if inventory:
            import unicodedata
            history=pd.read_csv(inventory)
            def clean(v):
                v=''.join(c for c in unicodedata.normalize('NFKD',str(v).lower()) if not unicodedata.combining(c)).strip(" '")
                return {'bolvar':'bolivar','caar':'canar'}.get(v,v)
            names={'azuay','bolivar','canar','carchi','chimborazo','cotopaxi','imbabura','loja','pichincha','tungurahua'}
            history['date']=pd.to_datetime(history.date,errors='coerce')
            history=history[history.provincia.map(clean).isin(names)&history.date.dt.year.between(2010,2025)].copy()
            description='Historical regional inventory, 2010–2025; recorded events before model-data cleaning'
        else:
            history=df[df.clase==1].copy();description='Modeling presences only, 2019–2025; not the complete historical inventory'
        history['year']=history.date.dt.year;history['month']=history.date.dt.month
        monthly=history.month.value_counts().reindex(range(1,13),fill_value=0).sort_index()
        heat=pd.crosstab(history.year,history.month).reindex(columns=range(1,13),fill_value=0).sort_index()
        fig,axes=plt.subplots(2,1,figsize=(7.2,6.0),gridspec_kw={'height_ratios':[1,1.5]})
        axes[0].bar(range(12),monthly,color=['#D55E00' if m in [8,9,10,11] else '#999999' for m in range(1,13)],width=.75)
        for i,value in enumerate(monthly):axes[0].text(i,value+monthly.max()*.015,f'{value:,}',ha='center',va='bottom',fontsize=6.5)
        axes[0].set_ylim(0,monthly.max()*1.15)
        axes[0].set(xticks=range(12),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],ylabel='Recorded fires',title='(a) Monthly distribution')
        axes[0].grid(axis='y',alpha=.15)
        image=axes[1].imshow(heat.to_numpy(),aspect='auto',cmap='YlOrRd',origin='lower')
        axes[1].set(xticks=range(12),xticklabels=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'],yticks=range(len(heat)),yticklabels=heat.index,ylabel='Year',title='(b) Year–month distribution')
        fig.colorbar(image,ax=axes[1],label='Recorded fires',fraction=.025,pad=.02)
        save(fig,'F02_seasonality',description+'. Counts are descriptive and depend on inventory coverage; zero counts do not demonstrate an absence of fire.',dict(scheme='descriptive',fold=None,seed=None,run_status='DESCRIPTIVE',historical_inventory_sha256=hashlib.sha256(Path(inventory).read_bytes()).hexdigest() if inventory else None))
        heat.to_csv(out/'seasonality_year_month.csv')

        # F03: las colas se recortan solo para dibujar; nunca se altera el modelado.
        fig,axes=plt.subplots(2,4,figsize=(7.2,4.8))
        for ax,(feature,label) in zip(axes.flat,FEATURE_LABELS.items()):
            lo,hi=df[feature].quantile([.005,.995]);bins=np.linspace(lo,hi,28)
            for cls,color in [(0,'#56B4E9'),(1,'#D55E00')]:
                values=df.loc[df.clase==cls,feature];shown=values[values.between(lo,hi)]
                ax.hist(shown,bins=bins,density=True,histtype='step',linewidth=1.25,color=color,label='Presence' if cls else 'Pseudoabsence')
            ax.set_title(label,fontsize=9);ax.tick_params(axis='x',labelsize=7)
        axes[0,0].legend(frameon=False,fontsize=7)
        axes[0,0].set_ylabel('Density');axes[1,0].set_ylabel('Density')
        save(fig,'F03_feature_distributions','Predictor distributions by class. Each panel displays the central 99% of the pooled values for legibility; all observations are retained in model fitting. Precipitation-window provenance must be reconciled with the extraction version.',dict(scheme='descriptive',fold=None,seed=None,run_status='DESCRIPTIVE'))

    # F04: ROC y PR usan exactamente los mismos IDs de test entre modelos.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5))
    for model in models:
        p=selected[selected.model==model];fpr,tpr,_=roc_curve(p.clase,p.probability);precision,recall,_=precision_recall_curve(p.clase,p.probability)
        axes[0].plot(fpr,tpr,color=COLORS[model],lw=1.3,label=f'{LABELS[model]}: {roc_auc_score(p.clase,p.probability):.3f}')
        axes[1].plot(recall,precision,color=COLORS[model],lw=1.3,label=f'{LABELS[model]}: {average_precision_score(p.clase,p.probability):.3f}')
    axes[0].plot([0,1],[0,1],ls='--',color='#999999',lw=.8)
    prevalence=selected[selected.model==models[0]].clase.mean();axes[1].axhline(prevalence,ls='--',color='#999999',lw=.8)
    axes[0].set(xlabel='False positive rate',ylabel='True positive rate',title='(a) ROC curves',xlim=(0,1),ylim=(0,1.01))
    axes[1].set(xlabel='Recall',ylabel='Precision',title='(b) Precision–recall curves',xlim=(0,1),ylim=(0,1.01))
    for ax in axes:ax.legend(frameon=False,fontsize=6.5,loc='lower left' if ax is axes[1] else 'lower right')
    fig.suptitle(stamp,fontsize=9);save(fig,'F04_roc_precision_recall',stamp+'. AUC and average precision are computed from the saved external-test predictions. PR baseline is the sampled test prevalence, not population prevalence.')

    # F05: frecuencias y proporciones por clase verdadera, con escala común.
    fig,axes=plt.subplots(2,3,figsize=(7.2,4.8));image=None
    for ax,model in zip(axes.flat,models):
        p=selected[selected.model==model];counts=confusion_matrix(p.clase,p.probability>=.5,labels=[0,1]);rates=counts/counts.sum(axis=1,keepdims=True)
        image=ax.imshow(rates,cmap='Blues',vmin=0,vmax=1)
        for (i,j),n in np.ndenumerate(counts):ax.text(j,i,f'{n:,}\n{rates[i,j]:.1%}',ha='center',va='center',fontsize=8,color='white' if rates[i,j]>.6 else 'black')
        ax.set(title=LABELS[model],xticks=[0,1],xticklabels=['PA','Fire'],yticks=[0,1],yticklabels=['PA','Fire'])
        ax.set_xlabel('Predicted label');ax.set_ylabel('Observed label');ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(models):]:ax.axis('off')
    fig.suptitle(stamp,fontsize=9);save(fig,'F05_confusion_matrices',stamp+'. Confusion matrices at score threshold 0.5. PA denotes pseudoabsence and Fire denotes recorded presence. Counts and row-normalized percentages are shown on a common 0–100% scale.')

    # F06: calibración exclusivamente dentro del muestreo caso-control.
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.5));briers=[]
    for model in models:
        p=selected[selected.model==model];observed,predicted=calibration_curve(p.clase,p.probability,n_bins=10,strategy='quantile')
        axes[0].plot(predicted,observed,'o-',color=COLORS[model],ms=3,lw=1,label=LABELS[model]);briers.append(brier_score_loss(p.clase,p.probability))
    axes[0].plot([0,1],[0,1],'--',color='#999999',lw=.8);axes[0].set(xlabel='Mean case-control score',ylabel='Observed presence fraction',title='(a) Sample reliability',xlim=(0,1),ylim=(0,1))
    axes[0].legend(frameon=False,fontsize=6.5)
    axes[1].barh([LABELS[m] for m in models],briers,color=[COLORS[m] for m in models]);axes[1].set(xlabel='Brier score (lower is better)',title='(b) Sample Brier score');axes[1].invert_yaxis()
    fig.suptitle(stamp,fontsize=9);save(fig,'F06_sample_calibration',stamp+'. Reliability curves use ten quantile bins. These diagnose calibration in the case-control sample; they do not establish real-world ignition probabilities.')

    # F07: provincias estables; nunca se promedian folds territoriales diferentes por su número.
    ref_meta,ref_scores,_=read_results(reference_results or results)
    if ref_meta['data_sha256']!=meta['data_sha256']:raise ValueError('La referencia usa otra versión del dataset.')
    province=ref_scores[ref_scores.scheme=='leave_province_out'].groupby(['fold','model']).roc_auc.mean().unstack()
    if not province.empty:
        columns=[m for m in ORDER if m in province]
        fig,ax=plt.subplots(figsize=(7.2,4.5));image=ax.imshow(province[columns],cmap='viridis',vmin=.5,vmax=1,aspect='auto')
        for i in range(len(province)):
            for j in range(len(columns)):
                value=province.iloc[i][columns[j]]
                if np.isfinite(value):ax.text(j,i,f'{value:.3f}',ha='center',va='center',color='white' if value<.8 else 'black',fontsize=8)
        ax.set(yticks=range(len(province)),yticklabels=province.index,xticks=range(len(columns)),xticklabels=[LABELS[m] for m in columns],title='Leave-one-province-out discrimination')
        fig.suptitle('PILOT' if ref_meta['config'].get('pilot') else 'VALIDATION',fontsize=9)
        if len(columns)>3:plt.setp(ax.get_xticklabels(),rotation=35,ha='right',fontsize=7)
        fig.colorbar(image,ax=ax,label='ROC-AUC',fraction=.03,pad=.03)
        save(fig,'F07_province_transfer','External-test AUC by held-out province, averaged over the available fitting seeds. All values are descriptive; repeated seeds do not create independent territories.',dict(scheme='leave_province_out',fold='all_available_provinces',seed=None,seeds=ref_meta['config'].get('seeds'),run_status='PILOT' if ref_meta['config'].get('pilot') else 'VALIDATION',reference_code_sha256=ref_meta.get('code_sha256')))

    # F08: curvas registradas en validación interna, no en el test externo.
    fig,axes=plt.subplots(2,2,figsize=(7.2,5.4));available=scores[['scheme','fold','seed']].drop_duplicates()
    panels=[]
    for s in SCHEMES:
        part=available[(available.scheme==s)&(available.seed==seed)]
        if not part.empty:panels.append((s,str(part.iloc[0]['fold'])))
    used=False
    for ax,(s,f) in zip(axes.flat,panels):
        for model in ['mlp','mlp_matched','attention']:
            file=results/f'learning_{s}_{f}_{seed}_{model}.csv'
            if file.exists():
                h=pd.read_csv(file);used=True
                ax.plot(h.epoch,h.validation_loss,color=COLORS[model],lw=1.3,label=LABELS[model]);ax.plot(h.epoch,h.training_loss,color=COLORS[model],ls=':',lw=.9)
        ax.set(title=f'{SCHEMES[s]}: {f}',xlabel='Epoch',ylabel='Binary cross-entropy');ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(panels):]:ax.axis('off')
    if used:
        axes[0,0].legend(frameon=False,fontsize=6.5)
        fig.suptitle(f'{status} | Internal optimization trajectories',fontsize=9)
        save(fig,'F08_learning_curves','Internal training (dotted) and validation (solid) losses. External-test data are not used for early stopping. One available partition per design is displayed; these are not the final full-training refit trajectories.',dict(scheme='multiple_designs',fold='one_representative_per_design'))
    else:plt.close(fig)

    # F09: diferencias pareadas y sus intervalos aproximados, sin ocultar el cero.
    delta_path=results/'attention_deltas.csv'
    if delta_path.exists() and delta_path.stat().st_size>1:
        deltas=pd.read_csv(delta_path,dtype={'fold':str})
        if not deltas.empty:
            deltas['baseline']=deltas.get('baseline',pd.Series('mlp',index=deltas.index)).fillna('mlp')
            deltas=deltas[(deltas.seed==seed)&(deltas.status=='approximate_spatial_block_bootstrap')].copy()
            comparisons=list(deltas[['scheme','fold']].drop_duplicates().itertuples(index=False,name=None))
            def comparison_label(s,f):
                if s=='random_pairs':return 'Random pairs'
                if s=='spatial_blocks':return str(f).replace('fold_','Block ')
                if s=='leave_province_out':return 'Province: '+str(f)
                return 'Year: '+str(f)
            labels=[comparison_label(s,f) for s,f in comparisons]
            baselines=[b for b in ['mlp','mlp_matched'] if b in deltas.baseline.unique()]
            fig,axes=plt.subplots(1,len(baselines),figsize=(7.2,max(3.5,len(comparisons)*.27)),sharex=True,sharey=True,squeeze=False)
            for ax,baseline in zip(axes.flat,baselines):
                for i,(s,f) in enumerate(comparisons):
                    rows=deltas[(deltas.scheme==s)&(deltas.fold==f)&(deltas.baseline==baseline)]
                    if rows.empty:continue
                    row=rows.iloc[0];color=COLORS[baseline]
                    ax.hlines(i,row.lower,row.upper,color=color,lw=1.3)
                    ax.vlines([row.lower,row.upper],i-.07,i+.07,color=color,lw=.8)
                    ax.plot(row.delta_auc,i,'o',color=color,ms=3.5)
                ax.axvline(0,color='#777777',ls='--',lw=.8)
                ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='AUC difference',title='vs MLP' if baseline=='mlp' else 'vs capacity control')
                ax.tick_params(labelsize=7);ax.grid(axis='x',alpha=.15)
            axes[0,0].invert_yaxis();fig.suptitle(f'{status}: attention ablation | seed {seed}',fontsize=9)
            save(fig,'F09_attention_ablation','Paired AUC differences on the same external-test cases. Bars are approximate spatial-block bootstrap intervals from the recorded run. They are conditional on the sample and block design and are not a confirmatory claim of superiority.',dict(scheme='multiple_designs',fold='recorded_comparisons'))

    # F10: ablaciones reales alineadas por diseño, fold, semilla y modelo.
    if ablations:
        full=ref_scores[ref_scores.seed==seed].copy();keys=['scheme','fold','seed','model']
        records=[]
        for name,folder in ablations.items():
            ameta,ascores,_=read_results(folder)
            if ameta['data_sha256']!=meta['data_sha256']:raise ValueError('Ablación de otra versión del dataset.')
            merged=ascores.merge(full,on=keys,suffixes=('_variant','_full'),validate='one_to_one')
            for (s,m),part in merged.groupby(['scheme','model']):records.append(dict(variant=name,scheme=s,model=m,delta=float((part.roc_auc_variant-part.roc_auc_full).mean())))
        table=pd.DataFrame(records)
        short_schemes={'random_pairs':'Random pairs','spatial_blocks':'Spatial blocks','leave_province_out':'Province','temporal_retrospective':'Temporal'}
        short_models={'logistic':'Logistic','rf':'RF','histgb':'Boosting'}
        table['row']=table.scheme.map(short_schemes)+' | '+table.model.map(short_models)
        heat=table.pivot(index='row',columns='variant',values='delta')
        limit=max(.005,float(np.nanmax(np.abs(heat.to_numpy()))))
        fig,ax=plt.subplots(figsize=(7.2,max(4,len(heat)*.28)));image=ax.imshow(heat,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        for (i,j),value in np.ndenumerate(heat.to_numpy()):
            if np.isfinite(value):ax.text(j,i,f'{value:+.3f}',ha='center',va='center',fontsize=7,color='white' if abs(value)>.65*limit else 'black')
        variant_labels={'no_climate':'Without climate','no_spectral':'Without spectral\nindices','raw_aspect':'Raw aspect'}
        ax.set(yticks=range(len(heat)),yticklabels=heat.index,xticks=range(len(heat.columns)),xticklabels=[variant_labels.get(c,c) for c in heat.columns],title='Covariate sensitivity: change relative to full features')
        fig.suptitle(('PILOT' if ref_meta['config'].get('pilot') else 'VALIDATION')+f' | seed {seed}',fontsize=9)
        fig.colorbar(image,ax=ax,label='Change in ROC-AUC',fraction=.03,pad=.03)
        save(fig,'F10_covariate_ablation','Mean paired differences from the full-feature model for matching external partitions and fitting seed. Negative values indicate lower AUC after the change. These are descriptive differences, not independent-fold confidence intervals.',dict(scheme='multiple_designs',fold='matching_external_partitions',run_status='PILOT' if ref_meta['config'].get('pilot') else 'VALIDATION',reference_code_sha256=ref_meta.get('code_sha256')))
        table.to_csv(out/'covariate_ablation_values.csv',index=False)
    return manifest
