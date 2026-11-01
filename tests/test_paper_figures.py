"""Evita comparar curvas obtenidas de casos o etiquetas distintos."""
from pathlib import Path
import sys
import unittest
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from paper_figures import choose_test

class ComparablePredictionsTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([dict(scheme='spatial_blocks',fold='fold_1',seed=42,model=m,id_registro=i,clase=c,probability=p)
            for m in ['rf','attention'] for i,c,p in [('P_1',1,.8),('PA_1',0,.2)]])

    def test_rejects_different_test_cases(self):
        frame=self.frame().iloc[:-1]
        with self.assertRaises(ValueError):choose_test(frame,'spatial_blocks','fold_1',42)

    def test_rejects_conflicting_labels(self):
        frame=self.frame();frame.loc[2,'clase']=0
        with self.assertRaises(ValueError):choose_test(frame,'spatial_blocks','fold_1',42)

    def test_identical_test_cases_accepted(self):
        selected=choose_test(self.frame(),'spatial_blocks','fold_1',42)
        self.assertEqual(len(selected),4)

if __name__=='__main__':unittest.main()
