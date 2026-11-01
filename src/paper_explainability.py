"""Explicabilidad de un ajuste externo reproducido y contrastado con sus scores."""
# %% Recuperación comprobada del modelo y muestras de explicación
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

def explain_paper_figures(results,out,dataset,api=None,n_shap=120,n_background=30,coalitions=256,
                          permutation_n=1000,permutation_repeats=5,scheme='temporal_retrospective',fold='2024',seed=42):
    import shap
    import torch
    if api is None:
        import research_validation
        api=vars(research_validation)
    # En el cuaderno estas funciones están visibles en las celdas anteriores.
    if 'export_figure' not in globals():
        from paper_figures import export_figure,FEATURE_LABELS,COLORS,LABELS,paper_style,read_results,choose_test
    else:
        export_figure=globals()['export_figure'];FEATURE_LABELS=globals()['FEATURE_LABELS'];COLORS=globals()['COLORS'];LABELS=globals()['LABELS'];paper_style=globals()['paper_style'];read_results=globals()['read_results'];choose_test=globals()['choose_test']
    paper_style();results=Path(results);out=Path(out)
    meta,scores,predictions=read_results(results);cfg=meta['config'].copy();cfg['seed']=seed
    if cfg['feature_set']!='complete':raise ValueError('SHAP comparativo requiere la configuración completa de ocho variables físicas.')
    import hashlib
    if hashlib.sha256(Path(dataset).read_bytes()).hexdigest()!=meta['data_sha256']:raise ValueError('Hash de dataset distinto.')
    df=api['load_data'](dataset);features=api['FEATURES'];matrix=api['design_matrix'](df,cfg['feature_set']);X=matrix.to_numpy(dtype=np.float32);y=df.clase.to_numpy()
    assignment=pd.read_csv(results/f'split_{scheme}_{fold}_{seed}.csv',dtype={'id_registro':str})
    lookup=pd.Series(df.index,index=df.id_registro.astype(str))
    train=lookup.loc[assignment.loc[assignment.partition=='train','id_registro']].to_numpy()
    test=lookup.loc[assignment.loc[assignment.partition=='test','id_registro']].to_numpy()
    recorded=choose_test(predictions,scheme,fold,seed)
    if not {'rf','attention'}.issubset(set(recorded.model)):
        raise ValueError('Esta explicación comparativa requiere RF y attention en la corrida seleccionada.')
    rf=api['classic_model']('rf',cfg);rf.fit(X[train],y[train])
    fitted={}
    def keep(model,scaler):fitted.update(model=model,scaler=scaler)
    probability,training=api['fit_neural'](X,y,df,train,test,scheme,True,cfg,on_fitted=keep)
    def transform(raw):
        return api['design_matrix'](pd.DataFrame(np.asarray(raw),columns=features),cfg['feature_set']).to_numpy(dtype=np.float32)
    def predict_rf(raw):return rf.predict_proba(transform(raw))[:,1]
    def predict_attention(raw):
        values=fitted['scaler'].transform(transform(raw));parts=[]
        with torch.no_grad():
            for start in range(0,len(values),2048):
                tensor=torch.tensor(values[start:start+2048],dtype=torch.float32)
                parts.append(torch.sigmoid(fitted['model'](tensor)).numpy())
        return np.concatenate(parts)
    raw=df[features].to_numpy()
    predictors={'rf':predict_rf,'attention':predict_attention}
    max_errors={}
    for name,predictor in predictors.items():
        recalculated=pd.DataFrame({'id_registro':df.loc[test,'id_registro'].to_numpy(),'recalculated':predictor(raw[test])})
        joined=recorded[recorded.model==name].merge(recalculated,on='id_registro',validate='one_to_one')
        if len(joined)!=len(test):raise ValueError('Faltan predicciones para comprobar el ajuste.')
        error=float(np.max(np.abs(joined.probability-joined.recalculated)));max_errors[name]=error
        if error>1e-5:raise ValueError(f'No se reprodujo el modelo {name}: error {error:.6g}. Usa el entorno original o sus checkpoints; no se generarán explicaciones de otro ajuste.')
    rng=np.random.default_rng(seed)
    test_sample=rng.choice(test,min(n_shap,len(test)),replace=False)
    background=rng.choice(train,min(n_background,len(train)),replace=False)
    permutation_sample=rng.choice(test,min(permutation_n,len(test)),replace=False)
    shap_values={};permutations={};additivity={}
    manifest_path=out/'figures_manifest.json';manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else []
    provenance=dict(data_sha256=meta['data_sha256'],run_status='PILOT' if cfg.get('pilot') else 'VALIDATION',scheme=scheme,fold=str(fold),seed=seed,
        reproduction_max_abs_error=max_errors,n_explained=len(test_sample),background_n=len(background),coalitions=coalitions)
    for name,predictor in predictors.items():
        print('Calculando SHAP:',name,flush=True)
        # Mismo explainer, background y casos; orientación se transforma conjuntamente a seno/coseno.
        np.random.seed(seed)
        explainer=shap.KernelExplainer(predictor,raw[background],link='identity')
        values=np.asarray(explainer.shap_values(raw[test_sample],nsamples=coalitions,l1_reg=0.0,silent=True))
        if values.ndim==3:values=values[:,:,0]
        if values.shape!=(len(test_sample),len(features)):raise ValueError('Forma SHAP inesperada.')
        shap_values[name]=values
        additivity[name]=float(np.max(np.abs(values.sum(axis=1)+float(np.asarray(explainer.expected_value).ravel()[0])-predictor(raw[test_sample]))))
        if additivity[name]>1e-3:raise ValueError('Falla la comprobación aditiva de SHAP.')
        pd.DataFrame(values,columns=features).assign(id_registro=df.loc[test_sample,'id_registro'].to_numpy()).to_csv(out/f'shap_values_{name}.csv',index=False)
        base=roc_auc_score(y[permutation_sample],predictor(raw[permutation_sample]));drops=np.empty((len(features),permutation_repeats))
        for j in range(len(features)):
            for repeat in range(permutation_repeats):
                shuffled=raw[permutation_sample].copy();order=np.random.default_rng(seed+j*100+repeat).permutation(len(shuffled));shuffled[:,j]=shuffled[order,j]
                drops[j,repeat]=base-roc_auc_score(y[permutation_sample],predictor(shuffled))
        permutations[name]=drops

    with torch.no_grad():
        scaled=fitted['scaler'].transform(X[test_sample]);tensor=torch.tensor(scaled,dtype=torch.float32);model=fitted['model']
        tokens=tensor.unsqueeze(-1)*model.scale.unsqueeze(0)+model.bias.unsqueeze(0)
        _,weights=model.attention(tokens,tokens,tokens,need_weights=True,average_attn_weights=False)
        routing=weights.mean(dim=(0,1)).numpy()
    np.testing.assert_allclose(routing.sum(axis=1),1,atol=1e-5)
    np.savez_compressed(out/'explanation_plot_data.npz',raw_samples=raw[test_sample],shap_rf=shap_values['rf'],shap_attention=shap_values['attention'],
        permutation_rf=permutations['rf'],permutation_attention=permutations['attention'],routing=routing,features=np.asarray(features),matrix_columns=np.asarray(matrix.columns))
    render_meta=dict(provenance=provenance,max_errors=max_errors,additivity=additivity,selected_epochs=training['selected_epochs'])
    (out/'explanation_render_metadata.json').write_text(json.dumps(render_meta,indent=2),encoding='utf-8')
    return render_explanation_cache(out)

# %% Redibujar explicabilidad sin ajustar modelos ni recalcular SHAP
def render_explanation_cache(out):
    """Modifica presentación a partir de los valores originales guardados."""
    if 'export_figure' not in globals():
        from paper_figures import export_figure,FEATURE_LABELS,COLORS,LABELS,paper_style
    else:
        export_figure=globals()['export_figure'];FEATURE_LABELS=globals()['FEATURE_LABELS'];COLORS=globals()['COLORS'];LABELS=globals()['LABELS'];paper_style=globals()['paper_style']
    out=Path(out);paper_style()
    meta=json.loads((out/'explanation_render_metadata.json').read_text(encoding='utf-8'))
    provenance=meta['provenance'];max_errors=meta['max_errors'];additivity=meta['additivity'];training=dict(selected_epochs=meta['selected_epochs'])
    cfg=dict(pilot=provenance['run_status']=='PILOT');scheme=provenance['scheme'];fold=provenance['fold']
    with np.load(out/'explanation_plot_data.npz',allow_pickle=False) as cache:
        raw=cache['raw_samples'].copy();features=cache['features'].tolist();test_sample=np.arange(len(raw))
        shap_values={name:cache['shap_'+name].copy() for name in ['rf','attention']}
        permutations={name:cache['permutation_'+name].copy() for name in ['rf','attention']}
        routing=cache['routing'].copy();matrix=pd.DataFrame(columns=cache['matrix_columns'].tolist())
    predictors={'rf':None,'attention':None}
    manifest=json.loads((out/'figures_manifest.json').read_text(encoding='utf-8'))
    manifest=[m for m in manifest if not m['figure'].startswith(('F11_','F12_','F13_'))]
    # F11: orden y escala comunes; colores enlazados a las mismas filas explicadas.
    importance=np.mean([np.abs(values).mean(axis=0) for values in shap_values.values()],axis=0)
    order=np.argsort(importance)
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.4),sharex=True,sharey=True)
    stamp=f"{'PILOT' if cfg.get('pilot') else 'VALIDATION'} | {scheme.replace('_',' ')}: {fold} | {len(test_sample)} explained cases"
    fig.suptitle(stamp,fontsize=8.5)
    limit=max(np.max(np.abs(v)) for v in shap_values.values())*1.06
    cmap=plt.get_cmap('coolwarm')
    for ax,(name,values) in zip(axes,shap_values.items()):
        for position,j in enumerate(order):
            v=values[:,j];x=raw[test_sample,j];lo,hi=np.quantile(x,[.01,.99]);normalized=np.clip((x-lo)/(hi-lo if hi>lo else 1),0,1)
            edges=np.linspace(float(v.min())-1e-8,float(v.max())+1e-8,25);bins=np.digitize(v,edges);offset=np.zeros(len(v))
            for b in np.unique(bins):
                idx=np.flatnonzero(bins==b);offset[idx]=np.linspace(-.25,.25,len(idx)) if len(idx)>1 else 0
            ax.scatter(v,position+offset,c=normalized,cmap=cmap,vmin=0,vmax=1,s=9,alpha=.8,edgecolors='none',rasterized=True)
        ax.axvline(0,color='#888888',lw=.7);ax.set(title=LABELS[name],xlabel='SHAP contribution to case-control score',xlim=(-limit,limit),yticks=range(len(features)),yticklabels=[FEATURE_LABELS[features[j]] for j in order]);ax.grid(axis='y',alpha=.1)
    # Barra de color separada para no reducir un panel respecto al otro.
    scalar=plt.cm.ScalarMappable(norm=plt.Normalize(0,1),cmap=cmap)
    colorbar=fig.colorbar(scalar,ax=axes.tolist(),fraction=.025,pad=.03);colorbar.set_ticks([0,1],labels=['Low','High']);colorbar.set_label('Relative feature value')
    fig.subplots_adjust(right=.87,wspace=.1)
    # El exportador usa tight_layout; la barra se coloca como eje explícito después.
    fig.delaxes(colorbar.ax)
    fig.tight_layout(rect=[0,0,.93,1]);cax=fig.add_axes([.94,.2,.014,.6]);fig.colorbar(scalar,cax=cax,ticks=[0,1]).ax.set_yticklabels(['Low','High'])
    fig._paper_layout_fixed=True
    export_figure(fig,out,'F11_shap_comparison','SHAP distributions for reproduced RF and feature-attention fits on the same external-test sample and training-only reference background. Both use KernelExplainer on the eight physical variables; the orientation sine/cosine transformation is inside the prediction function. A common order, x scale and color mapping are used. These are model associations, not causal effects.',manifest,provenance)

    # F12: dos indicadores con unidades distintas; no se normalizan artificialmente al 100%.
    fig,axes=plt.subplots(1,2,figsize=(7.2,4.3),sharey=True)
    fig.suptitle(stamp,fontsize=8.5)
    positions=np.arange(len(features));width=.32
    for k,name in enumerate(['rf','attention']):
        offset=(k-.5)*width
        axes[0].barh(positions+offset,np.abs(shap_values[name]).mean(axis=0)[order],height=width,color=COLORS[name],label=LABELS[name])
        means=permutations[name].mean(axis=1)[order];std=permutations[name].std(axis=1)[order]
        axes[1].barh(positions+offset,means,height=width,color=COLORS[name],xerr=std,error_kw={'elinewidth':.7,'capsize':2})
    axes[0].set(yticks=positions,yticklabels=[FEATURE_LABELS[features[j]] for j in order],xlabel='Mean |SHAP| (score units)',title='(a) Global attribution')
    axes[1].set(xlabel='ROC-AUC reduction after permutation',title='(b) Permutation importance');axes[1].axvline(0,color='gray',lw=.7)
    axes[0].legend(frameon=False,fontsize=7)
    export_figure(fig,out,'F12_global_attribution_permutation','Global mean absolute SHAP and external-test permutation importance. Permutation bars show mean and standard deviation across repeated permutations, not confidence intervals. Correlated predictors can share or mask importance; the two panels have different units.',manifest,provenance)

    # F13: matriz real de routing de atención, no importancia causal de variables.
    # Matriz real recuperada del caché numérico del ajuste comprobado.
    names=[FEATURE_LABELS.get(n,{'orientacion_sin':'Aspect: sine','orientacion_cos':'Aspect: cosine'}.get(n,n)) for n in matrix.columns]
    fig,ax=plt.subplots(figsize=(7.2,5.4));image=ax.imshow(routing,cmap='Blues',vmin=0,vmax=max(.2,float(routing.max())))
    fig.suptitle(stamp,fontsize=8.5)
    ax.set(xticks=range(len(names)),xticklabels=names,yticks=range(len(names)),yticklabels=names,xlabel='Key token',ylabel='Query token',title='Mean attention routing between input tokens')
    plt.setp(ax.get_xticklabels(),rotation=55,ha='right')
    for (i,j),v in np.ndenumerate(routing):ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=6.5,color='white' if v>image.norm.vmax*.6 else 'black')
    fig.colorbar(image,ax=ax,label='Mean attention weight',fraction=.035,pad=.03)
    export_figure(fig,out,'F13_attention_routing','Attention routing averaged over heads and explained test samples. Query rows sum to one. Orientation is represented by two derived tokens, so the matrix has nine channels for eight physical variables. These routing weights are not SHAP, permutation importance or causal influence.',manifest,provenance)
    summary=[]
    for name in predictors:
        for j,feature in enumerate(features):summary.append(dict(model=name,feature=feature,mean_abs_shap=float(np.abs(shap_values[name][:,j]).mean()),permutation_auc_drop=float(permutations[name][j].mean()),permutation_sd=float(permutations[name][j].std())))
    pd.DataFrame(summary).to_csv(out/'explainability_summary.csv',index=False)
    pd.DataFrame(routing,index=matrix.columns,columns=matrix.columns).to_csv(out/'attention_routing.csv')
    (out/'explanation_checks.json').write_text(json.dumps(dict(reproduction_max_abs_error=max_errors,shap_additivity_error=additivity,selected_epochs=training['selected_epochs'],**{k:v for k,v in provenance.items() if k not in ['reproduction_max_abs_error']}),indent=2),encoding='utf-8')
    return manifest
