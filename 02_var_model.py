# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Model building
# MAGIC In this notebook, we retrieve last 2 years worth of market indicator data to train a model that could predict our instrument returns. As our portfolio is made of 40 equities, we want to train 40 predictive models in parallel, collecting all weights into a single coefficient matrix for monte carlo simulations. We show how to have a more discipline approach to model development by leveraging **MLFlow** capabilities.

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

import datetime
from datetime import timedelta
import pandas as pd
import datetime

# We will generate monte carlo simulation for every week since we've built our model
# Alternatively, only select run_date to a given day to run once
today = datetime.date.today()
first = datetime.datetime.strptime(config['model_training_date'], '%Y-%m-%d')
run_dates = list(pd.date_range(first, today, freq='w'))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute returns
# MAGIC In the previous notebook, we already computed daily returns for each of our market indicators. Those will be used as features for our model, trying to predict investment returns for each of our instrument.

# COMMAND ----------

import datetime
run_date = datetime.datetime.strptime(config['model_training_date'], '%Y-%m-%d')

# COMMAND ----------

from pyspark.sql import functions as F
import pandas as pd

market_df = spark.read.table(config['volatility_table']).filter(F.col('date') < run_date).select('date', 'features')
market_pd = pd.DataFrame(market_df.toPandas()['features'].to_list(), columns=config['feature_names'])
display(market_pd)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's compute daily returns of our investments. Given the size of a typical portfolio, we can leverage Window functions on spark to do so.

# COMMAND ----------

import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

@udf("double")
def compute_return(first, close):
  return float(np.log(close / first))

def get_stock_returns():

  # Apply a tumbling 1 day window on each instrument
  window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)

  # apply sliding window and take first element
  stocks_df = spark.table(config['stock_table']) \
    .filter(F.col('close').isNotNull()) \
    .withColumn("first", F.first('close').over(window)) \
    .withColumn("return", compute_return('first', 'close')) \
    .select('date', 'ticker', 'return')
  
  return stocks_df

# COMMAND ----------

stocks_df = get_stock_returns().filter(F.col('date') < run_date)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create features
# MAGIC Risk models are complex and cannot really be expressed simply through the form of notebooks. This solution accelerator does not aim to build best financial model, but rather to walk someone through all processes to do so. The starting point to any good risk model will be to diligently study correlations between all different indicators (limited to 5 here for presentation purpose)

# COMMAND ----------

import seaborn as sns
import matplotlib.pyplot as plt

# we simply plot correlation matrix via pandas (market factors fit in memory)
# we assume market factors are not correlated (NASDAQ and SP500 are, so are OIL and TREASURY BONDS)
f_cor_pdf = market_pd.corr(method='spearman', min_periods=12)
sns.set(rc={'figure.figsize':(11,8)})
sns.heatmap(f_cor_pdf, annot=True)
plt.savefig('{}/factor_correlation.png'.format(temp_directory))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We join our market indicator data with stock returns to build an input dataset we can machine learn. We'll use `tempo` for this AS-OF join since our timestamps may be differents in real life, with intra day tick data.

# COMMAND ----------

from tempo import *
market_tsdf = TSDF(market_df.join(stocks_df.select('ticker').distinct()), ts_col="date", partition_cols=['ticker'])
stocks_tsdf = TSDF(stocks_df, ts_col="date", partition_cols=['ticker'])

# COMMAND ----------

features_df = (
  stocks_tsdf.asofJoin(market_tsdf).df
    .select(
      F.col('date'),
      F.col('ticker'),
      F.col('right_features').alias('features'),
      F.col('return')
    )
    .filter(F.col('features').isNotNull())
)

display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building models
# MAGIC We show how any function or model can be easily wrapped as a `mlflow.pyfunc` model and registered as such on ml registry. Real life VAR models are obviously more complex than a simple linear regression described here and are not necessarily out of the box sklearn or keras models. Still, they should follow same ML development standard and can easily benefit from ml-flow functionalities as long as one can express model I/O as a form of `pd.Series`, `pd.DataFrame` or `np.array`

# COMMAND ----------

# add non linear transformations as simple example on non linear returns
def non_linear_features(xs):
  import numpy as np
  fs = []
  for x in xs:
    fs.append(x)
    fs.append(np.sign(x) * x**2)
    fs.append(x**3)
    fs.append(np.sign(x) * np.sqrt(abs(x)))
  return fs

# COMMAND ----------

import statsmodels.api as sm
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType

# use pandas UDF to train multiple model (one for each instrument) in parallel
# the resulting dataframe will be the linear regression weights for each instrument
schema = StructType([
  StructField('ticker', StringType(), True), 
  StructField('weights', ArrayType(FloatType()), True)
])

# a model would also be much more complex than the below
@pandas_udf(schema, PandasUDFType.GROUPED_MAP)
def train_model(group, pdf):
  import pandas as pd
  # build market factor vectors
  # add a constant - the intercept term for each instrument i.
  X = [non_linear_features(row) for row in np.array(pdf['features'])]
  X = sm.add_constant(X, prepend=True) 
  y = np.array(pdf['return'])
  model = sm.OLS(y, X).fit()
  w_df = pd.DataFrame(data=[[model.params]], columns=['weights'])
  w_df['ticker'] = group[0]
  return w_df

# COMMAND ----------

# the resulting dataframe easily fits in memory and will be saved as our "uber model"
model_df = features_df.groupBy('ticker').apply(train_model).toPandas()
model_df.head(10)

# COMMAND ----------

# MAGIC %md
# MAGIC One can package an entire business logic (being statistical models or more AI models) as a simple `pyfunc`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel

class RiskMLFlowModel(PythonModel):
  
  import numpy as np
  import pandas as pd

  def __init__(self, model_df):
    self.weights = dict(zip(model_df.ticker, model_df.weights))
    
  def _non_linear_features(self, xs):
    fs = []
    for x in xs:
      fs.append(x)
      fs.append(np.sign(x) * x**2)
      fs.append(x**3)
      fs.append(np.sign(x) * np.sqrt(abs(x)))
    return fs
  
  def _predict_record(self, ticker, xs):
    ps = self.weights[ticker]
    fs = self._non_linear_features(xs)
    s = ps[0]
    for i, f in enumerate(fs):
      s = s + ps[i + 1] * f
    return float(s)
  
  def predict(self, context, model_input):
    predicted = model_input[['ticker','features']].apply(lambda x: self._predict_record(*x), axis=1)
    return predicted

# COMMAND ----------

# MAGIC %md
# MAGIC Such a model will be tracked, stored, registered and signature of the model enforced to prevent from data drift.

# COMMAND ----------

from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name='value-at-risk') as run:
  
  # get mlflow run ID
  run_id = run.info.run_id
  
  # create our pyfunc model
  python_model = RiskMLFlowModel(model_df)
  
  # Get model input and output signatures
  model_input_df  = features_df.select('ticker', 'features').limit(10).toPandas()
  model_output_df = python_model.predict(None, model_input_df)
  model_signature = infer_signature(model_input_df, model_output_df)
  
  # log our model to mlflow
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=python_model,
    signature=model_signature
  )
  
  # log additional artifacts
  mlflow.log_artifact("{}/factor_correlation.png".format(temp_directory))

# COMMAND ----------

model_udf = mlflow.pyfunc.spark_udf(model_uri='runs:/{}/model'.format(run_id), result_type='float', spark=spark)
prediction_df = features_df.withColumn('predicted', model_udf(F.struct('ticker', 'features')))
display(prediction_df)

# COMMAND ----------

from pyspark.sql.functions import udf

@udf("float")
def wsse_udf(p, a):
  return float((p - a)**2)

# compare expected vs. actual return
# sum mean square error per instrument
wsse_df = prediction_df \
  .withColumn('wsse', wsse_udf(F.col('predicted'), F.col('return'))) \
  .groupBy('ticker') \
  .agg(F.sum('wsse').alias('wsse'))

# get average wsse across portfolio
wsse = wsse_df.select(F.avg('wsse').alias('wsse')).toPandas().iloc[0].wsse

# plot mean square error as accuracy of our model for each instrument
ax = wsse_df.toPandas().plot.bar(x='ticker', y='wsse', rot=0, label=None, figsize=(24,5))
ax.get_legend().remove()
plt.title("Model WSSE for each instrument")
plt.xticks(rotation=45)
plt.ylabel("wsse")
plt.savefig("{}/model_wsse.png".format(temp_directory))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can update our previous experiment with results of our prediction model (sum square of error)

# COMMAND ----------

with mlflow.start_run(run_id=run_id) as run:
  mlflow.log_metric("wsse", wsse)
  mlflow.log_artifact("{}/model_wsse.png".format(temp_directory))

# COMMAND ----------

# MAGIC %md
# MAGIC The experiment captured now contains all libraries required to run in isolation and is linked to specific delta version to enable full reproducibility

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/var/images/var_experiments.png width="1000px">

# COMMAND ----------

# MAGIC %md
# MAGIC By registering our model to ML registry, we make it available to downstream processes and backend jobs such as our next notebook focused on monte carlo simulations

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/model".format(run_id)
result = mlflow.register_model(model_uri, config['model_name'])
version = result.version

# COMMAND ----------

# MAGIC %md
# MAGIC We can also promote our model to different stages programmatically. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact for our next notebook and programmatically transition previous runs back to Archive.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(config['model_name'])):
  if model.current_stage == 'Production':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=config['model_name'],
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=config['model_name'],
    version=version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Our model now production candidate, we can load our predictive logic as a simple user defined function and predict investment returns for every observed market condition

# COMMAND ----------

model_udf = mlflow.pyfunc.spark_udf(
  model_uri='models:/{}/production'.format(config['model_name']), 
  result_type='float', 
  spark=spark
)

# COMMAND ----------

plt.figure(figsize=(25,12))

prediction_df = features_df.withColumn('predicted', model_udf(F.struct("ticker", "features")))
df_past_1 = prediction_df.filter(F.col('ticker') == "EC").orderBy('date').toPandas()
df_past_2 = prediction_df.filter(F.col('ticker') == "EC").orderBy('date').toPandas()
plt.plot(df_past_1.date, df_past_1['return'])
plt.plot(df_past_2.date, df_past_2['predicted'], color='green', linestyle='--')

min_return = np.min(df_past_2['return'])
max_return = np.max(df_past_2['return'])

plt.ylim([min_return, max_return])
plt.title('Log return of EC')
plt.ylabel('log return')
plt.xlabel('date')
plt.show()

# COMMAND ----------


