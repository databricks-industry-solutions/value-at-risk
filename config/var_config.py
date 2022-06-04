# Databricks notebook source
# MAGIC %pip install yfinance==0.1.70 dbl-tempo==0.1.17

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

# COMMAND ----------

import re
from pathlib import Path

# We ensure that all objects created in that notebooks will be registered in a user specific database. 
useremail = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
username = useremail.split('@')[0]

# Please replace this cell should you want to store data somewhere else.
database_name = '{}_var'.format(re.sub('\W', '_', username))
_ = sql("CREATE DATABASE IF NOT EXISTS {}".format(database_name))

# Similar to database, we will store actual content on a given path
home_directory = '/FileStore/{}/var'.format(username)
dbutils.fs.mkdirs(home_directory)

# Where we might stored temporary data on local disk
temp_directory = "/tmp/{}/var".format(username)
Path(temp_directory).mkdir(parents=True, exist_ok=True)

# COMMAND ----------

import re

config = {
  'portfolio_table'           : f'{database_name}.portfolio',
  'stock_table'               : f'{database_name}.stocks',
  'market_table'              : f'{database_name}.instruments',
  'volatility_table'          : f'{database_name}.volatility',
  'monte_carlo_table'         : f'{database_name}.monte_carlo',
  'trials_table'              : f'{database_name}.trials',
  'model_name'                : 'var_{}'.format(re.sub('\W', '_', username)),
  'feature_names'             : ['SP500', 'NYSE', 'OIL', 'TREASURY', 'DOWJONES'],
  'yfinance_start'            : '2018-05-01',
  'yfinance_stop'             : '2020-05-01',
  'model_training_date'       : '2019-09-01',
  'num_runs'                  : 32000,
  'past_volatility'           : 90,
  'num_executors'             : 20,
}

# COMMAND ----------

import mlflow
experiment_name = f"/Users/{useremail}/var"
mlflow.set_experiment(experiment_name) 

# COMMAND ----------

def tear_down():
  import shutil
  shutil.rmtree(temp_directory)
  dbutils.fs.rm(home_directory, True)
  _ = sql("DROP DATABASE IF EXISTS {} CASCADE".format(database_name))
