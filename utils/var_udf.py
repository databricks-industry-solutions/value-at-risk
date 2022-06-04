from pyspark.sql.functions import udf
from pyspark.ml.linalg import Vectors, VectorUDT


@udf("double")
def compute_return(first, close):
  import numpy as np
  return float(np.log(close / first))


@udf("float")
def wsse_udf(p, a):
  return float((p - a)**2)


@udf('float')
def get_var_udf(simulations, var):
  from utils.var_utils import get_var
  return get_var(simulations, var)


@udf('int')
def count_breaches(xs, var):
  breaches = len([x for x in xs if x <= var])
  if breaches <= 3:
    return 0
  elif breaches < 10:
    return 1
  else:
    return 2


@udf('float')
def get_shortfall_udf(simulations, var):
  from utils.var_utils import get_shortfall
  return get_shortfall(simulations, var)


@udf(VectorUDT())
def weighted_returns(returns, weight):
  return Vectors.dense(returns.toArray() * weight)


@udf('array<double>')
def compute_avg(xs):
  import numpy as np
  mean = np.array(xs).mean(axis=0)
  return mean.tolist()

  
@udf('array<array<double>>')
def compute_cov(xs):
  import pandas as pd
  return pd.DataFrame(xs).cov().values.tolist()

# provided covariance matrix and average of market indicators, we sample from a multivariate distribution
# we allow a seed to be passed for reproducibility
# whilst many data scientists may add a seed as np.random.seed(seed), we have to appreciate the distributed nature 
# of our process and the undesired side effects settings seeds globally
# instead, use rng = np.random.default_rng(seed)

@udf('array<float>')
def simulate_market(vol_avg, vol_cov, seed):
  import numpy as np
  rng = np.random.default_rng(seed)
  return rng.multivariate_normal(vol_avg, vol_cov).tolist()