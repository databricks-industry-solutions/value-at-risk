# Databricks notebook source
# MAGIC %md 
# MAGIC You may find this series of notebooks at https://github.com/databricks-industry-solutions/value-at-risk on the `web-sync` branch. For more information about this solution accelerator, visit https://www.databricks.com/solutions/accelerators/market-risk.

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fs-lakehouse-logo.png width="600px">
# MAGIC 
# MAGIC [![DBU](https://img.shields.io/badge/DBU-XL-red)]()
# MAGIC [![COMPLEXITY](https://img.shields.io/badge/COMPLEXITY-401-red)]()
# MAGIC 
# MAGIC *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.* 
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %run ./config/var_config

# COMMAND ----------

tear_down()

# COMMAND ----------

# MAGIC %md
# MAGIC ## VAR 101
# MAGIC 
# MAGIC VaR is measure of potential loss at a specific confidence interval. A VAR statistic has three components: a time period, a confidence level and a loss amount (or loss percentage). What is the most I can - with a 95% or 99% level of confidence - expect to lose in dollars over the next month? There are 3 ways to compute Value at risk
# MAGIC # 
# MAGIC 
# MAGIC + **Historical Method**: The historical method simply re-organizes actual historical returns, putting them in order from worst to best.
# MAGIC + **The Variance-Covariance Method**: This method assumes that stock returns are normally distributed and use pdf instead of actual returns.
# MAGIC + **Monte Carlo Simulation**: This method involves developing a model for future stock price returns and running multiple hypothetical trials.
# MAGIC 
# MAGIC We report in below example a simple Value at risk calculation for a synthetic instrument, given a volatility (i.e. standard deviation of instrument returns) and a time horizon (300 days). **What is the most I could lose in 300 days with a 95% confidence?**

# COMMAND ----------

# time horizon
days = 300
dt = 1/float(days)

# volatility
sigma = 0.04 

# drift (average growth rate)
mu = 0.05  

# initial starting price
start_price = 10

# number of simulations
runs_gr = 500
runs_mc = 10000

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def generate_prices(start_price):
    shock = np.zeros(days)
    price = np.zeros(days)
    price[0] = start_price
    for i in range(1, days):
        shock[i] = np.random.normal(loc=mu * dt, scale=sigma * np.sqrt(dt))
        price[i] = max(0, price[i - 1] + shock[i] * price[i - 1])
    return price

plt.figure(figsize=(16,6))
for i in range(1, runs_gr):
    plt.plot(generate_prices(start_price))

plt.title('Simulated price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# COMMAND ----------

simulations = np.zeros(runs_mc)
for i in range(0, runs_mc):
    simulations[i] = generate_prices(start_price)[days - 1]

# COMMAND ----------

# MAGIC %run ./utils/var_utils

# COMMAND ----------

plot_var(simulations, 95)

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC &copy; 2021 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
# MAGIC | tempo                                  | Timeseries library      | Databricks | https://github.com/databrickslabs/tempo             |
