# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Monte Carlo
# MAGIC In this notebook, we use our model created in previous stage and run monte carlo simulations in parallel using **Apache Spark**. For each simulated market condition sampled from a multi variate distribution, we will predict our hypothetical instrument returns. By storing all of our data back into **Delta Lake**, we will create a data asset that can be queried on-demand across multiple down stream use cases

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

# MAGIC %run ./utils/var_utils

# COMMAND ----------

import datetime
from datetime import timedelta
import pandas as pd
import datetime

# We will generate monte carlo simulation for every week since we've built our model
today = datetime.datetime.strptime(config['yfinance_stop'], '%Y-%m-%d')
first = datetime.datetime.strptime(config['model_training_date'], '%Y-%m-%d')
run_dates = pd.date_range(first, today, freq='w')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Market volatility
# MAGIC As we've pre-computed all statistics at ingest time, we can easily retrieve the most recent statistical distribution of market indicators for each date we want to run monte carlo simulation against. We can access temporal information using asof join of our tempo `library`

# COMMAND ----------

from tempo import *
market_tsdf = TSDF(spark.read.table(config['volatility_table']), ts_col='date')
rdates_tsdf = TSDF(spark.createDataFrame(pd.DataFrame(run_dates, columns=['date'])), ts_col='date')

# COMMAND ----------

from pyspark.sql import functions as F

volatility_df = rdates_tsdf.asofJoin(market_tsdf).df.select(
  F.col('date'),
  F.col('right_vol_cov').alias('vol_cov'),
  F.col('right_vol_avg').alias('vol_avg')
)

display(volatility_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribute trials
# MAGIC By fixing a seed strategy, we ensure that each trial will be independant (no random number will be the same) as well as enforcing full reproducibility should we need to process the same experiment twice

# COMMAND ----------

# create a dataframe of seeds so that each trial will result in a different simulation
# each executor is responsible for num_instruments * ( total_runs / num_executors ) trials
import pandas as pd
import numpy as np

def create_seed_df():
  runs = config['num_runs']
  seed_df = pd.DataFrame(list(np.arange(0, runs)), columns = ['trial_id'])
  return spark.createDataFrame(seed_df)

# COMMAND ----------

# provided covariance matrix and average of market indicators, we sample from a multivariate distribution
# we allow a seed to be passed for reproducibility
# whilst many data scientists may add a seed as np.random.seed(seed), we have to appreciate the distributed nature 
# of our process and the undesired side effects settings seeds globally
# instead, use rng = np.random.default_rng(seed)
from pyspark.sql.functions import udf

@udf('array<float>')
def simulate_market(vol_avg, vol_cov, seed):
  import numpy as np
  rng = np.random.default_rng(seed)
  return rng.multivariate_normal(vol_avg, vol_cov).tolist()

# COMMAND ----------

market_conditions = (
  volatility_df
    .join(create_seed_df())
    .withColumn('features', simulate_market('vol_avg', 'vol_cov', 'trial_id'))
    .select('date', 'features', 'trial_id')
)

# COMMAND ----------

display(market_conditions)

# COMMAND ----------

# MAGIC %md
# MAGIC Since this was an expensive operation to cross join each trial ID with each simulated market condition, we can save that table as a delta table that we can process downstream. Furthermore, this table is generic as we only sampled points from known market volatility and did not take investment returns into account. New models and new trading strategies could be executed off the back of the exact same data without having to run this expensive process.

# COMMAND ----------

_ = (
  market_conditions
    .repartition(config['num_executors'], 'date')
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(config['monte_carlo_table'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute returns
# MAGIC Finally, we can leverage our model created earlier to predict our investment return for each stock given generated market indicators

# COMMAND ----------

import mlflow
model_udf = mlflow.pyfunc.spark_udf(
  model_uri='models:/{}/production'.format(config['model_name']), 
  result_type='float', 
  spark=spark
)

# COMMAND ----------

simulations = (
  spark.read.table(config['monte_carlo_table'])
    .join(spark.read.table(config['portfolio_table']).select('ticker'))
    .withColumn('return', model_udf(F.struct('ticker', 'features')))
    .drop('features')
)

display(simulations)

# COMMAND ----------

# MAGIC %md
# MAGIC Although we processed our simulated market conditions as a large table made of very few columns, we may want to create a better data asset by wraping all trials into well defined vectors. This asset will help us manipulate vectors through simple aggregated functions using the `Summarizer` class from `pyspark.ml.stat` (see next notebook)

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT

@udf(VectorUDT())
def to_vector(xs, ys):
  return Vectors.sparse(config['num_runs'], zip(xs, ys))

# COMMAND ----------

simulations_vectors = (
  simulations
    .groupBy('date', 'ticker')
    .agg(
      F.collect_list('trial_id').alias('xs'),
      F.collect_list('return').alias('ys')
    )
    .select(
      F.col('date'),
      F.col('ticker'),
      to_vector(F.col('xs'), F.col('ys')).alias('returns')
    )
)

# COMMAND ----------

_ = (
  simulations_vectors
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(config['trials_table'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we make it easy to extract specific slices of our data asset by optimizing our table for faster read.

# COMMAND ----------

_ = sql('OPTIMIZE {} ZORDER BY (`date`, `ticker`)'.format(config['trials_table']))
