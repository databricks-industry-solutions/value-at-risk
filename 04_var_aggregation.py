# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC # VaR Aggregation
# MAGIC In this notebook, we demonstrate the versatile nature of our model carlo simulation on **Delta Lake**. Stored in its most granular form, analysts have the flexibility to slice and dice their data to aggregate value-at-risk on demand via aggregated vector functions from **Spark ML**.

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

# MAGIC %run ./utils/var_utils

# COMMAND ----------

trials_df = spark.read.table(config['trials_table'])
portfolio_df = spark.read.table(config['portfolio_table'])
simulation_df = trials_df.join(portfolio_df, ['ticker'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Point in time VaR
# MAGIC With all our simulations stored with finest granularity, we can access a specific slice for a given day and retrieve the associated value at risk as a simple quantile function. We  aggregate trial vectors across our entire portfolio using the built in function of spark ML, `Summarizer`. 

# COMMAND ----------

from pyspark.sql import functions as F
min_date = trials_df.select(F.min('date').alias('date')).toPandas().iloc[0].date

# COMMAND ----------

from pyspark.ml.stat import Summarizer

point_in_time_vector = (
  simulation_df
    .filter(F.col('date') == min_date)
    .groupBy('date')
    .agg(Summarizer.sum(F.col('returns')).alias('returns'))
    .toPandas().iloc[0].returns.toArray()
)

# COMMAND ----------

plot_var(point_in_time_vector, 99)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evolution of risk exposure
# MAGIC The same can be achieved at scale, over our entire trading history. For each date, we aggregate all trial vectors and extract the worst 1% of events 

# COMMAND ----------

risk_exposure = (
  simulation_df
    .groupBy('date')
    .agg(Summarizer.sum(F.col('returns')).alias('returns'))
    .withColumn('var_99', var(F.col('returns'), F.lit(99)))
    .drop('returns')
    .orderBy('date')
    .toPandas()
)

# COMMAND ----------

import matplotlib.pyplot as plt
plt.figure(figsize=(20,8))
plt.plot(risk_exposure['date'], risk_exposure['var_99'])
plt.title('VaR across all portfolio')
plt.ylabel('value at risk')
plt.xlabel('date')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Slice and Dice
# MAGIC The main advantage of leaving monte carlo data in its finest granularity is the ability to slice and dice and visualize different segments, industries, countries. Using optimized delta tables, portfolio managers and risk analysts could efficiently run what-if scenario, adhoc analysis, such as value at risk aggregation by country of operation

# COMMAND ----------

risk_exposure_country = (
  simulation_df
    .groupBy('date', 'country')
    .agg(Summarizer.sum(F.col('returns')).alias('returns'))
    .withColumn('var_99', var(F.col('returns'), F.lit(99)))
    .drop('returns')
    .orderBy('date')
    .toPandas()
)

# COMMAND ----------

fig, ax = plt.subplots(figsize=(20,8))
for label, df in risk_exposure_country.groupby('country'):
    df.plot.line(x='date', y='var_99', ax=ax, label=label)

plt.title('VaR by country')
plt.ylabel('value at risk')
plt.xlabel('date')
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC The same can be translated as a risk contribution by industry for a given country. How much of my overall risk is linked to my investment in the mining industry? How would I reduce this exposure by rebalancing my portfolio?

# COMMAND ----------

risk_exposure_industry = (
  simulation_df
    .filter(F.col('country') == 'PERU')
    .groupBy('date', 'industry')
    .agg(Summarizer.sum(F.col('returns')).alias('returns'))
    .withColumn('var_99', var(F.col('returns'), F.lit(99)))
    .drop('returns')
    .orderBy('date')
    .toPandas()
)

# COMMAND ----------

import pandas as pd
import numpy as np
risk_contribution_country = pd.crosstab(risk_exposure_industry['date'], risk_exposure_industry['industry'], values=risk_exposure_industry['var_99'], aggfunc=np.sum)
risk_contribution_country = risk_contribution_country.div(risk_contribution_country.sum(axis=1), axis=0)
risk_contribution_country.plot.bar(figsize=(20,8), colormap="Pastel1", stacked=True, width=0.9)
