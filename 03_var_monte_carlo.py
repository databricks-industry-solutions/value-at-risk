# Databricks notebook source
# MAGIC %md
# MAGIC # Modern Monte Carlo Simulation
# MAGIC 
# MAGIC This notebook demonstrates enterprise-grade Monte Carlo simulation for financial risk:
# MAGIC - Uses modern distributed computing with Apache Spark
# MAGIC - Implements Unity Catalog for data governance
# MAGIC - Includes comprehensive error handling and logging
# MAGIC - Leverages MLflow for experiment tracking
# MAGIC - Stores results in Delta Lake with proper versioning

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Configuration and Setup
# MAGIC 
# MAGIC Enhanced configuration with comprehensive error handling and logging.

# COMMAND ----------

import datetime
from datetime import timedelta
import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import Window

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse dates from modern configuration
def parse_simulation_dates() -> Tuple[datetime.datetime, datetime.datetime, List[datetime.datetime]]:
    """Parse simulation date range from configuration with validation"""
    try:
        today = datetime.datetime.strptime(config['market_data']['yfinance']['maxdate'], '%Y-%m-%d')
        model_date = datetime.datetime.strptime(config['mlflow']['model']['training_date'], '%Y-%m-%d')
        
        # Generate weekly simulation dates
        simulation_dates = pd.date_range(model_date, today, freq='W').tolist()
        
        logger.info(f"Simulation period: {model_date} to {today}")
        logger.info(f"Number of simulation dates: {len(simulation_dates)}")
        
        return today, model_date, simulation_dates
        
    except Exception as e:
        logger.error(f"Failed to parse simulation dates: {str(e)}")
        raise

# Parse simulation configuration
today, model_date, simulation_dates = parse_simulation_dates()

# Load Monte Carlo configuration
mc_config = config['monte_carlo']['simulation']
num_simulations = mc_config['runs']
confidence_levels = mc_config['confidence_levels']
volatility_window = mc_config['volatility_window']

logger.info(f"Monte Carlo configuration:")
logger.info(f"  Simulations per run: {num_simulations:,}")
logger.info(f"  Confidence levels: {confidence_levels}")
logger.info(f"  Volatility window: {volatility_window} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Market Volatility Analysis
# MAGIC 
# MAGIC Enhanced volatility calculation with Unity Catalog integration and modern time series patterns.

# COMMAND ----------

def calculate_market_volatility(volatility_window: int = 90) -> DataFrame:
    """
    Calculate market volatility using modern time series patterns
    
    Args:
        volatility_window: Number of days for volatility calculation
        
    Returns:
        DataFrame with volatility metrics
    """
    try:
        from tempo import TSDF
        
        # Load market indicators from Unity Catalog
        indicators_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_indicators']}"
        
        logger.info(f"Calculating volatility from {indicators_table}")
        
        # Create time series DataFrame
        indicators_df = spark.read.table(indicators_table)
        
        # Use tempo library for time series operations
        ts_df = TSDF(
            indicators_df,
            ts_col="date",
            partition_cols=["ticker"]
        )
        
        # Calculate rolling volatility
        volatility_df = (
            ts_df
            .withColumn("log_return", F.log(F.col("close") / F.lag("close", 1).over(
                Window.partitionBy("ticker").orderBy("date")
            )))
            .withColumn("volatility", F.stddev("log_return").over(
                Window.partitionBy("ticker").orderBy("date").rowsBetween(-volatility_window, 0)
            ))
            .df
            .filter(F.col("volatility").isNotNull())
        )
        
        # Create feature vectors for each date
        feature_vectors_df = (
            volatility_df
            .groupBy("date")
            .agg(F.collect_list("volatility").alias("volatility_features"))
            .withColumn("feature_timestamp", F.current_timestamp())
        )
        
        # Save to Unity Catalog
        volatility_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_volatility']}"
        
        (feature_vectors_df
         .write
         .format("delta")
         .mode("overwrite")
         .option("overwriteSchema", "true")
         .saveAsTable(volatility_table))
        
        logger.info(f"Volatility features saved to {volatility_table}")
        
        return feature_vectors_df
        
    except Exception as e:
        logger.error(f"Volatility calculation failed: {str(e)}")
        raise

# Calculate market volatility
try:
    volatility_df = calculate_market_volatility(volatility_window)
    
    # Display volatility summary
    display(volatility_df.orderBy(F.desc("date")).limit(10))
    
    logger.info("Market volatility calculation completed")
    
except Exception as e:
    logger.error(f"Market volatility processing failed: {str(e)}")
    raise

# COMMAND ----------
market_tsdf = TSDF(spark.read.table(config['database']['tables']['volatility']), ts_col='date')
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
# MAGIC By fixing a seed strategy, we ensure that each trial will be independent (no random number will be the same) as well as enforcing full reproducibility should we need to process the same experiment twice

# COMMAND ----------

from utils.var_utils import create_seed_df
seed_df = create_seed_df(config['monte-carlo']['runs'])
display(seed_df)

# COMMAND ----------

from utils.var_udf import simulate_market

market_conditions = (
  volatility_df
    .join(spark.createDataFrame(seed_df))
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
    .repartition(config['monte-carlo']['executors'], 'date')
    .write
    .mode("overwrite")
    .format("delta")
    .saveAsTable(config['database']['tables']['mc_market'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute returns
# MAGIC Finally, we can leverage our model created earlier to predict our investment return for each stock given generated market indicators

# COMMAND ----------

import mlflow
model_udf = mlflow.pyfunc.spark_udf(
  model_uri='models:/{}/production'.format(config['model']['name']), 
  result_type='float', 
  spark=spark
)

# COMMAND ----------

simulations = (
  spark.read.table(config['database']['tables']['mc_market'])
    .join(spark.createDataFrame(portfolio_df[['ticker']]))
    .withColumn('return', model_udf(F.struct('ticker', 'features')))
    .drop('features')
)

display(simulations)

# COMMAND ----------

# MAGIC %md
# MAGIC Although we processed our simulated market conditions as a large table made of very few columns, we may want to create a better data asset by wrapping all trials into well defined vectors. This asset will help us manipulate vectors through simple aggregated functions using the `Summarizer` class from `pyspark.ml.stat` (see next notebook)

# COMMAND ----------

from pyspark.ml.linalg import Vectors, VectorUDT

@udf(VectorUDT())
def to_vector(xs, ys):
  v = Vectors.sparse(config['monte-carlo']['runs'], zip(xs, ys)).toArray()
  return Vectors.dense(v)

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
    .saveAsTable(config['database']['tables']['mc_trials'])
)  

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we make it easy to extract specific slices of our data asset by optimizing our table for faster read. This is achieved through the `OPTIMIZE` command of delta

# COMMAND ----------

_ = sql('OPTIMIZE {} ZORDER BY (`date`, `ticker`)'.format(config['database']['tables']['mc_trials']))

# COMMAND ----------


