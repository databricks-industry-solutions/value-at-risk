import unittest

import numpy as np

from utils.var_utils import *


class UtilTest(unittest.TestCase):

    def test_seed_df(self):
        df = create_seed_df(10)
        jf = df.to_json()
        self.assertEqual(jf, '{"trial_id":{"0":0,"1":1,"2":2,"3":3,"4":4,"5":5,"6":6,"7":7,"8":8,"9":9}}')

    def test_var(self):
        simulations = np.arange(0, 100, 1)
        var = get_var(simulations, 95)
        self.assertEqual(round(var), 5)

    def test_shortfall(self):
        simulations = np.arange(0, 100, 1)
        shortfall = get_shortfall(simulations, 89)
        self.assertEqual(round(shortfall), 5)

    def test_features(self):
        fs = [round(f) for f in non_linear_features([1, 4])]
        self.assertEqual(fs, [1, 1, 1, 1, 4, 16, 64, 2])
        
    def test_predict(self):
        fs = [round(f) for f in non_linear_features([1, 4])]
        ps = [p + 0.01 for p in np.zeros(9)]
        self.assertEqual(predict_non_linears(ps, fs), (np.sum(fs) + 1) / 100)


if __name__ == '__main__':
    unittest.main()
