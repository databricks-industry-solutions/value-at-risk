# Databricks notebook source
# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import yaml
with open('config/application.yaml', 'r') as f:
  config = yaml.safe_load(f)

# COMMAND ----------

dbutils.fs.mkdirs(config['database']['path'])
_ = sql("CREATE DATABASE IF NOT EXISTS {} LOCATION '{}'".format(
  config['database']['name'], 
  config['database']['path']
))

# COMMAND ----------

# use our newly created database by default
# each table will be created as a MANAGED table under this directory
_ = sql("USE {}".format(config['database']['name']))

# COMMAND ----------

import pandas as pd
portfolio_df = pd.read_json('config/portfolio.json', orient='records')

# COMMAND ----------

import json
with open('config/indicators.json', 'r') as f:
  market_indicators = json.load(f)

# COMMAND ----------

# FEATURE_DISABLED: Creation of experiments in jobs is not enabled. If using the Python fluent API, you can set an active experiment
import mlflow
mlflow.set_experiment(config['model']['name'])

# COMMAND ----------

import tempfile
tempfile.TemporaryDirectory()
