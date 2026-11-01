"""Controles de reanudación: no mezclar datos, código o entornos distintos."""
from pathlib import Path
import copy
import hashlib
import importlib.metadata
import json
import sys
import tempfile
import unittest
from unittest.mock import patch
import types
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import master_workflow as mw
import research_validation as rv


class ResumeTests(unittest.TestCase):
    def test_notebook_input_uploads_once_and_reopens_saved_csv(self):
        from test_research_validation import toy_data
        notebook=json.loads((Path(__file__).resolve().parents[1]/'notebooks/00_ESTUDIO_COMPLETO_INCENDIOS.ipynb').read_text(encoding='utf-8'))
        cell=next(c for c in notebook['cells'] if c['metadata'].get('workflow_role')=='inputs')
        root=(Path.cwd()/'work/master-test-temp').resolve();root.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=root)
        try:
            calls=[];filename='dataset_modelo_incendios_sierra_2019_2025.csv'
            def upload():calls.append(True);return {filename:toy_data().to_csv(index=False).encode()}
            colab=types.ModuleType('google.colab');colab.files=types.SimpleNamespace(upload=upload)
            scope=dict(DATOS=Path(temp.name),ARCHIVO_PRINCIPAL=filename,FORZAR_CARGA_DATOS=False,
                Path=Path,pd=pd,FEATURES=rv.FEATURES,load_data=rv.load_data,hashlib=hashlib,display=lambda *a:None)
            with patch.dict(sys.modules,{'google.colab':colab}):
                exec(''.join(cell['source']),scope)
                exec(''.join(cell['source']),scope)
            self.assertEqual(len(calls),1)
            self.assertEqual(scope['DATA'],Path(temp.name)/filename)
            self.assertEqual(len(scope['datos']),2400)
        finally:
            self.assertTrue(Path(temp.name).resolve().is_relative_to(root));temp.cleanup()

    def test_live_code_and_defaults_change_fingerprint(self):
        api=vars(rv).copy()
        original,_=mw.runtime_fingerprint(api)
        self.assertEqual(original,mw.runtime_fingerprint(api)[0])
        def changed_metrics(*args):return {'changed':True}
        api['metrics']=changed_metrics
        self.assertNotEqual(original,mw.runtime_fingerprint(api)[0])
        api=vars(rv).copy();api['DEFAULT']=copy.deepcopy(api['DEFAULT'])
        api['DEFAULT']['buffer_km']=10
        self.assertNotEqual(original,mw.runtime_fingerprint(api)[0])

    def test_only_compatible_complete_stage_is_reused(self):
        root=(Path.cwd()/'work/master-test-temp').resolve();root.mkdir(parents=True,exist_ok=True)
        temp=tempfile.TemporaryDirectory(dir=root)
        try:
            folder=Path(temp.name);data=folder/'input.csv';data.write_text('example',encoding='utf-8')
            config={'seed':42};calls=[]
            def train(*args):calls.append(args);return pd.DataFrame({'trained':[True]})
            api={'DEFAULT':config,'run_validation':train}
            meta=dict(status='completed',config=config,code_sha256='code-a',
                data_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),python=sys.version,
                versions={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','scikit-learn','torch','pyproj']})
            (folder/'execution.json').write_text(json.dumps(meta),encoding='utf-8')
            for name in ['metrics.csv','predictions.csv','split_audit.csv']:
                pd.DataFrame({'saved':[1]}).to_csv(folder/name,index=False)
            reused=mw.reuse_or_run(data,folder,config,api,'code-a')
            self.assertEqual(len(calls),0);self.assertEqual(reused.saved.tolist(),[1])
            mw.reuse_or_run(data,folder,config,api,'code-b');self.assertEqual(len(calls),1)
            data.write_text('changed input',encoding='utf-8')
            mw.reuse_or_run(data,folder,config,api,'code-a');self.assertEqual(len(calls),2)
            data.write_text('example',encoding='utf-8');meta['status']='failed'
            (folder/'execution.json').write_text(json.dumps(meta),encoding='utf-8')
            mw.reuse_or_run(data,folder,config,api,'code-a');self.assertEqual(len(calls),3)
            meta['status']='completed';meta['python']='different python'
            (folder/'execution.json').write_text(json.dumps(meta),encoding='utf-8')
            mw.reuse_or_run(data,folder,config,api,'code-a');self.assertEqual(len(calls),4)
        finally:
            self.assertTrue(Path(temp.name).resolve().is_relative_to(root));temp.cleanup()

if __name__=='__main__':unittest.main()
