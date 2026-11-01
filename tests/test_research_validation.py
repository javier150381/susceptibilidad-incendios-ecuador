"""Controles de independencia y de uso del test, con datos sintéticos."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import research_validation as rv

def toy_data():
    rng=np.random.default_rng(26); rows=[]
    for i in range(1200):
        year=2019+i%7; province='A' if (i//7)%2 else 'B'
        date=f'{year}-{1+i%10:02d}-15'
        for label in [1,0]:
            # Pares de la misma provincia y fecha, en lugares distintos.
            x=int(rng.integers(0,8)); y=int(rng.integers(0,8))
            row=dict(id_registro=('P_' if label else 'PA_')+str(i),date=date,provincia=province,
                latitud=-4+y*.25+rng.uniform(0,.005),longitud=-80+x*.25+(2 if province=='B' else 0)+rng.uniform(0,.005),clase=label)
            row.update({n:float(rng.uniform(.1,.9)) for n in rv.FEATURES})
            rows.append(row)
    return pd.DataFrame(rows)

class IndependenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_root=(Path.cwd()/'work/research-test-temp').resolve()
        self.temp_root.mkdir(parents=True,exist_ok=True)
        self.tmp=tempfile.TemporaryDirectory(dir=self.temp_root); self.path=Path(self.tmp.name)/'toy.csv'
        toy_data().to_csv(self.path,index=False); self.df=rv.load_data(self.path)
        self.cfg=dict(rv.DEFAULT,pilot=True,seeds=[42],epochs=1,patience=1,n_splits=3,buffer_km=2)

    def tearDown(self):
        # Verifica el destino antes de la limpieza recursiva de TemporaryDirectory.
        self.assertTrue(Path(self.tmp.name).resolve().is_relative_to(self.temp_root))
        self.tmp.cleanup()

    def test_rejects_broken_pairs(self):
        frame=toy_data(); frame.loc[0,'date']='2018-01-01'; frame.to_csv(self.path,index=False)
        with self.assertRaises(ValueError): rv.load_data(self.path)

    def test_final_year_locked(self):
        with self.assertRaises(ValueError): rv.make_splits(self.df,dict(self.cfg,temporal_years=[2025]))
        with self.assertRaises(ValueError): rv.make_splits(self.df,dict(self.cfg,development_end=2025))

    def test_capacity_control(self):
        dimensions=9; dim=self.cfg['token_dim']
        width=64+round((4*dim*dim+4*dim)/(dimensions*dim+33))
        attention=rv.neural_model(dimensions,True,self.cfg)
        matched=rv.neural_model(dimensions,False,dict(self.cfg,head_width=width))
        a=sum(p.numel() for p in attention.parameters()); b=sum(p.numel() for p in matched.parameters())
        self.assertLess(abs(a-b)/a,.01)

    def test_spatial_and_province_separation(self):
        groups=rv.spatial_groups(self.df,self.cfg['block_km'])
        for scheme,fold,tr,te in rv.make_splits(self.df,self.cfg):
            self.assertFalse(set(self.df.loc[tr,'pair_id'])&set(self.df.loc[te,'pair_id']))
            self.assertFalse((self.df.loc[np.r_[tr,te],'year']==2025).any())
            if scheme in ['spatial_blocks','leave_province_out']:
                self.assertGreaterEqual(rv.nearest_m(self.df.loc[tr],self.df.loc[te]).min(),2000)
            if scheme=='spatial_blocks': self.assertFalse(set(groups.loc[tr])&set(groups.loc[te]))
            if scheme=='leave_province_out': self.assertFalse(set(self.df.loc[tr,'provincia'])&set(self.df.loc[te,'provincia']))
            inner,validation=rv.inner_validation(self.df,tr,scheme,self.cfg)
            self.assertTrue(set(inner)|set(validation)<=set(tr))
            self.assertFalse((set(inner)|set(validation))&set(te))

    def test_test_labels_do_not_change_neural_predictions(self):
        scheme,fold,tr,te=rv.make_splits(self.df,self.cfg)[0]
        X=rv.design_matrix(self.df).to_numpy(dtype=np.float32); y=self.df.clase.to_numpy()
        first,_=rv.fit_neural(X,y,self.df,tr,te,scheme,True,self.cfg)
        changed=y.copy(); changed[te]=1-changed[te]
        second,_=rv.fit_neural(X,changed,self.df,tr,te,scheme,True,self.cfg)
        np.testing.assert_array_equal(first,second)

if __name__=='__main__': unittest.main()
