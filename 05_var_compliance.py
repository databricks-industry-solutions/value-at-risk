# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # Compliance
# MAGIC The Basel Committee specified a methodology for backtesting VaR. The 1 day VaR 99 results are to
# MAGIC be compared against daily P&L’s. Backtests are to be performed quarterly using the most recent 250
# MAGIC days of data. Based on the number of exceedances experienced during that period, the VaR
# MAGIC measure is categorized as falling into one of three colored zones:
# MAGIC 
# MAGIC | Level   | Threshold                 | Results                       |
# MAGIC |---------|---------------------------|-------------------------------|
# MAGIC | Green   | Up to 4 exceedances       | No particular concerns raised |
# MAGIC | Yellow  | Up to 9 exceedances       | Monitoring required           |
# MAGIC | Red     | More than 10 exceedances  | VaR measure to be improved    |

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

# MAGIC %run ./utils/var_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ## Get investment returns
# MAGIC In order to detect possible investments breaches, one would need to overlay existing investments to latest value at risk calculations. We simply compute our investment returns using window partitioning function

# COMMAND ----------

import numpy as np
from pyspark.sql import Window
from pyspark.sql.functions import udf
from pyspark.sql import functions as F

@udf("double")
def compute_return(first, close):
  return float(np.log(close / first))

# Apply a tumbling 1 day window on each instrument
window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)

# apply sliding window and take first element
inv_returns_df = spark.table(config['stock_table']) \
  .filter(F.col('close').isNotNull()) \
  .withColumn("first", F.first('close').over(window)) \
  .withColumn("return", compute_return('first', 'close')) \
  .groupBy('date') \
  .agg(F.sum('return').alias('return'))
  
display(inv_returns_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Retrieve value at risk
# MAGIC As covered in our previous section, we can easily compute our value at risk against our entire history by aggregating trial vectors and finding a 99 percentile.

# COMMAND ----------

from pyspark.ml.stat import Summarizer

risk_exposure = (
  spark.read.table(config['trials_table'])
    .groupBy('date')
    .agg(Summarizer.sum(F.col('returns')).alias('returns'))
    .withColumn('var_99', var(F.col('returns'), F.lit(99)))
    .drop('returns')
    .orderBy('date')
)

# COMMAND ----------

# MAGIC %md
# MAGIC As previously, `tempo` is used as an efficient way to join those 2 series (investments and risk exposure) together.

# COMMAND ----------

from tempo import *
risk_exposure_tsdf = TSDF(risk_exposure, ts_col="date")
inv_returns_tsdf = TSDF(inv_returns_df, ts_col="date")

# COMMAND ----------

asof_df = (
  inv_returns_tsdf.asofJoin(risk_exposure_tsdf).df
    .na.drop()
    .orderBy('date')
    .select(
      F.col('date'),
      F.col('return'),
      F.col('right_var_99').alias('var_99')
    )
)

display(asof_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Extract breaches
# MAGIC We want to retrieve all investments done within a 250 days period that exceeded our Var99 threshold

# COMMAND ----------

# timestamp is interpreted as UNIX timestamp in seconds
days = lambda i: i * 86400
compliance_window = Window.orderBy(F.col("date").cast("long")).rangeBetween(-days(250), 0)

# COMMAND ----------

@udf('int')
def count_breaches(xs, var):
  breaches = len([x for x in xs if x <= var])
  if breaches <= 3:
    return 0
  elif breaches < 10:
    return 1
  else:
    return 2

compliance_df = (
  asof_df
    .withColumn('previous_return', F.collect_list('return').over(compliance_window))
    .withColumn('basel', count_breaches('previous_return', 'var_99'))
    .drop('previous_return')
    .toPandas()
    .set_index('date')
)

# COMMAND ----------

import pandas as pd
import numpy as np
idx = pd.date_range(np.min(compliance_df.index), np.max(compliance_df.index), freq='d')
compliance_df = compliance_df.reindex(idx, method='pad')

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we can represent our investments against our value at risk

# COMMAND ----------

import numpy as np
import matplotlib.pyplot as plt 

f, (a0, a1) = plt.subplots(2, 1, figsize=(20,8), gridspec_kw={'height_ratios': [10,1]})

a0.plot(compliance_df.index, compliance_df['return'], color='#86bf91', label='returns')
a0.plot(compliance_df.index, compliance_df['var_99'], label="var99", c='red', linestyle='--')
a0.axhline(y=0, linestyle='--', alpha=0.4, color='#86bf91', zorder=1)
a0.title.set_text('VAR99 compliance')
a0.set_ylabel('Daily log return')
a0.legend(loc="upper left")

colors = ['green', 'orange', 'red']
a1.bar(compliance_df.index, 1, color=[colors[i] for i in compliance_df['basel']], label='breaches', alpha=0.5, align='edge', width=1.0)
a1.get_yaxis().set_ticks([])
a1.set_xlabel('Date')

plt.subplots_adjust(wspace=0, hspace=0)

# COMMAND ----------

# MAGIC %md
# MAGIC #### Wall St banks’ trading risk surges to highest since 2011
# MAGIC 
# MAGIC [...] The top five Wall St banks’ aggregate “value-at-risk”, which measures their potential daily trading losses, soared to its highest level in 34 quarters during the first three months of the year, according to Financial Times analysis of the quarterly VaR high disclosed in banks’ regulatory filings
# MAGIC 
# MAGIC [https://on.ft.com/2SSqu8Q](https://on.ft.com/2SSqu8Q)
