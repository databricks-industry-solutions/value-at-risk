import unittest

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession


class UdfTest(unittest.TestCase):

    def setUp(self):
        self.spark = SparkSession \
            .builder \
            .appName("UNIT_TEST") \
            .master("local") \
            .getOrCreate()

    def tearDown(self):
        self.spark.stop()

    def test_return_df(self):

        from utils.var_udf import compute_return

        # input data
        pdf = pd.DataFrame([[2, 3]], columns=['first', 'close'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf)
        df = df.withColumn('return', compute_return('first', 'close'))
        df.show()

        # assertion
        rt = df.toPandas().iloc[0]
        self.assertEqual(round(rt['return'], 1), 0.4)

    def test_wsse(self):

        from utils.var_udf import wsse_udf

        # input data
        pdf = pd.DataFrame([[2, 3]], columns=['actual', 'prediction'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf)
        df = df.withColumn('wsse', wsse_udf('actual', 'prediction'))
        df.show()

        # assertion
        rt = df.toPandas().iloc[0]
        self.assertEqual(rt['wsse'], 1)

    def test_var(self):

        from utils.var_udf import get_var_udf, get_shortfall_udf
        from utils.var_utils import get_var, get_shortfall
        from pyspark.ml.linalg import Vectors
        from pyspark.sql.functions import col, lit
        import numpy as np
        import pandas as pd

        # input data
        simulation_array = [[1, Vectors.dense([float(i) for i in np.arange(0, 100, 1)])]]
        local_var = round(get_var(simulation_array[0][1], 95))
        local_shortfall = round(get_shortfall(simulation_array[0][1], 95))
        pdf = pd.DataFrame(simulation_array, columns=['dummy', 'simulation'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf)
        df = df.withColumn("var", get_var_udf(col('simulation'), lit(95)))
        df = df.withColumn("shortfall", get_shortfall_udf(col('simulation'), lit(95)))
        df.show()

        # assertion
        rt = df.toPandas().iloc[0]
        self.assertEqual(round(rt['var']), local_var)
        self.assertEqual(round(rt['shortfall']), local_shortfall)

    def test_weighted_returns(self):

        from utils.var_udf import weighted_returns
        from pyspark.ml.linalg import Vectors
        import numpy as np
        import pandas as pd

        # input data
        vector = [float(i) for i in np.arange(0, 100, 1)]
        simulation_array = [[2, Vectors.dense(vector)]]
        pdf = pd.DataFrame(simulation_array, columns=['weight', 'simulation'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf)
        df = df.withColumn('weighted', weighted_returns('simulation', 'weight'))
        df.show()

        # assertion
        actual = list(df.toPandas().iloc[0]['weighted'].toArray())
        expected = [v * 2 for v in vector]
        self.assertListEqual(actual, expected)

    def test_avg(self):

        import pandas as pd
        from utils.var_udf import compute_avg, compute_cov
        from pyspark.sql.functions import collect_list

        # input data
        data = [
            ['DBX', [1, 2, 3, 4, 5]],
            ['DBX', [2, 3, 4, 5, 1]],
            ['DBX', [3, 4, 5, 1, 2]],
            ['DBX', [4, 5, 1, 2, 3]],
            ['DBX', [5, 1, 2, 3, 4]],
        ]

        pdf = pd.DataFrame(data, columns=['ticker', 'features'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf).groupBy('ticker').agg(collect_list('features').alias('features'))
        df = df.withColumn('average', compute_avg('features'))
        df = df.withColumn('covariance', compute_cov('features'))
        df.show()

        # assertion
        rt = df.toPandas().iloc[0]
        average = list(rt['average'])
        self.assertListEqual([3.0, 3.0, 3.0, 3.0, 3.0], average)

        covariance = list(rt['covariance'])
        covariance_list = list(pd.DataFrame(covariance).sum(axis=1))
        self.assertListEqual([0, 0, 0, 0, 0], covariance_list)
        # 2.50  0.00 -1.25 -1.25  0.00

    def test_simulations(self):

        import pandas as pd
        from utils.var_udf import compute_avg, compute_cov, simulate_market
        from pyspark.sql.functions import collect_list

        # input data
        data = [
            ['DBX', [1, 2, 3, 4, 5]],
            ['DBX', [2, 3, 4, 5, 1]],
            ['DBX', [3, 4, 5, 1, 2]],
            ['DBX', [4, 5, 1, 2, 3]],
            ['DBX', [5, 1, 2, 3, 4]],
        ]

        pdf = pd.DataFrame(data, columns=['ticker', 'features'])

        # spark dataframe
        df = self.spark.createDataFrame(pdf).groupBy('ticker').agg(collect_list('features').alias('features'))
        df = df.withColumn('average', compute_avg('features'))
        df = df.withColumn('covariance', compute_cov('features'))
        df = df.join(self.spark.createDataFrame(pd.DataFrame([1, 1, 1, 1, 1], columns=['seed'])))
        df = df.withColumn('market', simulate_market('average', 'covariance', 'seed'))
        df.show()

        # assertion
        rt = pd.DataFrame(list(df.toPandas()['market']))
        for i in rt.columns:
            # trials should all be the same
            self.assertEqual(len(np.unique(rt[i]).tolist()), 1)


if __name__ == '__main__':
    unittest.main()
