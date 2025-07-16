# Databricks notebook source
# MAGIC %md
# MAGIC <img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">
# MAGIC 
# MAGIC [![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC [![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
# MAGIC 
# MAGIC *Traditional banks relying on on-premises infrastructure can no longer effectively manage risk. Banks must abandon the computational inefficiencies of legacy technologies and build an agile Modern Risk Management practice capable of rapidly responding to market and economic volatility. Using value-at-risk use case, you will learn how Databricks is helping FSIs modernize their risk management practices, leverage Delta Lake, Apache Spark and MLFlow to adopt a more agile approach to risk management.* 
# MAGIC 
# MAGIC ___
# MAGIC <antoine.amend@databricks.com>

# COMMAND ----------

# MAGIC %md
# MAGIC <img src='https://raw.githubusercontent.com/databricks-industry-solutions/value-at-risk/master/images/reference_architecture.png' width=800>

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

# MAGIC %md
# MAGIC ## Modern Unity Catalog Setup
# MAGIC 
# MAGIC This modernized version uses Unity Catalog for data governance and modern MLflow for experiment tracking.

# COMMAND ----------

# Unity Catalog setup
import os

# Get parameters from DAB bundle
catalog_name = dbutils.widgets.get("catalog_name") if dbutils.widgets.get("catalog_name") else "dev_value_at_risk"
schema_name = dbutils.widgets.get("schema_name") if dbutils.widgets.get("schema_name") else "risk_management"
environment = dbutils.widgets.get("environment") if dbutils.widgets.get("environment") else "dev"

# Set Unity Catalog context
spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
spark.sql(f"USE CATALOG {catalog_name}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
spark.sql(f"USE SCHEMA {schema_name}")

print(f"✅ Unity Catalog configured: {catalog_name}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow 2.8+ Setup
# MAGIC 
# MAGIC Modern MLflow configuration for experiment tracking and model management.

# COMMAND ----------

import mlflow
import mlflow.sklearn

# Set environment variables for MLflow configuration
os.environ["MLFLOW_TRACKING_URI"] = "databricks"
os.environ["MLFLOW_REGISTRY_URI"] = "databricks-uc"

# Set MLflow tracking and registry URIs
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set MLflow experiment with Unity Catalog
# Use a simpler path that doesn't require pre-existing directories
experiment_name = f"/Shared/var_experiments_{catalog_name}_{schema_name}"
mlflow.set_experiment(experiment_name)

# Enable autologging for better experiment tracking
mlflow.sklearn.autolog()

print(f"✅ MLflow experiment: {experiment_name}")
print(f"✅ Using Unity Catalog model registry (databricks-uc)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## VaR Configuration Parameters

# COMMAND ----------

# time horizon
days = 300

# volatility
sigma = 0.04 

# drift (average growth rate)
mu = 0.05  

# initial starting price
start_price = 10

# COMMAND ----------

import matplotlib.pyplot as plt
from utils.var_utils import generate_prices

plt.figure(figsize=(16,6))
for i in range(1, 500):
    plt.plot(generate_prices(start_price, mu, sigma, days))

plt.title('Simulated price')
plt.xlabel("Time")
plt.ylabel("Price")
plt.show()

# COMMAND ----------

from utils.var_viz import plot_var
simulations = [generate_prices(start_price, mu, sigma, days)[-1] for i in range(10000)]
plot_var(simulations, 99)

# COMMAND ----------

# MAGIC %md
# MAGIC Expected shortfall is measure that produces better incentives for traders than VAR. This is also sometimes referred to as conditional VAR, or tail loss. Where VAR asks the question 'how bad can things get?', expected shortfall asks 'if things do get bad, what is our expected loss?'. 

# COMMAND ----------

from utils.var_utils import get_var
from utils.var_utils import get_shortfall

print('Var99: {}'.format(round(get_var(simulations, 99), 2)))
print('Shortfall: {}'.format(round(get_shortfall(simulations, 99), 2)))

# COMMAND ----------

# MAGIC %md
# MAGIC &copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC 
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
# MAGIC | tempo                                  | Timeseries library      | Databricks | https://github.com/databrickslabs/tempo             |
# MAGIC | PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
