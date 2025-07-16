# Databricks notebook source
# MAGIC %md
# MAGIC # Modern VaR Aggregation and Risk Metrics
# MAGIC 
# MAGIC This notebook demonstrates enterprise-grade risk aggregation capabilities:
# MAGIC - Modern Unity Catalog integration for risk data governance
# MAGIC - Scalable VaR calculation using distributed computing
# MAGIC - Real-time risk monitoring with configurable confidence levels
# MAGIC - Comprehensive error handling and logging
# MAGIC - MLflow integration for risk model tracking

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Risk Aggregation Setup
# MAGIC 
# MAGIC Enhanced setup with comprehensive logging and Unity Catalog integration.

# COMMAND ----------

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql import Window
from pyspark.ml.stat import Summarizer

import mlflow
import mlflow.sklearn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load risk management configuration
risk_config = config['risk_management']
confidence_levels = config['monte_carlo']['simulation']['confidence_levels']

logger.info(f"Risk aggregation configuration:")
logger.info(f"  Confidence levels: {confidence_levels}")
logger.info(f"  Portfolio rebalancing: {risk_config['portfolio']['rebalance_frequency']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Monte Carlo Data Loading
# MAGIC 
# MAGIC Load simulation results from Unity Catalog with proper validation and error handling.

# COMMAND ----------

def load_monte_carlo_trials() -> DataFrame:
    """
    Load Monte Carlo simulation results from Unity Catalog with validation
    
    Returns:
        DataFrame with Monte Carlo trial results
    """
    try:
        # Load trials from Unity Catalog
        trials_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['monte_carlo_trials']}"
        
        logger.info(f"Loading Monte Carlo trials from {trials_table}")
        
        trials_df = spark.read.table(trials_table)
        
        # Validate data
        trial_count = trials_df.count()
        if trial_count == 0:
            raise ValueError("No Monte Carlo trials found")
        
        # Get data summary
        date_range = trials_df.select(
            F.min('date').alias('min_date'),
            F.max('date').alias('max_date'),
            F.countDistinct('ticker').alias('unique_tickers')
        ).collect()[0]
        
        logger.info(f"Loaded {trial_count:,} Monte Carlo trials")
        logger.info(f"Date range: {date_range['min_date']} to {date_range['max_date']}")
        logger.info(f"Unique instruments: {date_range['unique_tickers']}")
        
        return trials_df
        
    except Exception as e:
        logger.error(f"Failed to load Monte Carlo trials: {str(e)}")
        raise

def create_weighted_simulation_df(trials_df: DataFrame) -> DataFrame:
    """
    Create weighted simulation DataFrame with portfolio weights
    
    Args:
        trials_df: DataFrame with Monte Carlo trials
        
    Returns:
        DataFrame with weighted returns
    """
    try:
        from utils.var_udf import weighted_returns
        
        # Join with portfolio weights
        simulation_df = (
            trials_df
            .join(portfolio_df, ['ticker'])
            .withColumn('weighted_returns', weighted_returns('returns', 'weight'))
            .withColumn('calculation_timestamp', F.current_timestamp())
        )
        
        logger.info("Created weighted simulation DataFrame")
        
        return simulation_df
        
    except Exception as e:
        logger.error(f"Failed to create weighted simulation: {str(e)}")
        raise

# Load and prepare simulation data
try:
    trials_df = load_monte_carlo_trials()
    simulation_df = create_weighted_simulation_df(trials_df)
    
    # Display sample data
    display(simulation_df.select(
        'date', 'ticker', 'returns', 'weight', 'weighted_returns'
    ).limit(10))
    
except Exception as e:
    logger.error(f"Simulation data loading failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Point-in-Time VaR Calculation
# MAGIC 
# MAGIC Enhanced VaR calculation with multiple confidence levels and comprehensive error handling.

# COMMAND ----------

def calculate_point_in_time_var(
    simulation_df: DataFrame, 
    target_date: str, 
    confidence_levels: List[float]
) -> Dict[str, Any]:
    """
    Calculate point-in-time VaR for multiple confidence levels
    
    Args:
        simulation_df: DataFrame with weighted simulation results
        target_date: Date for VaR calculation
        confidence_levels: List of confidence levels (e.g., [0.95, 0.99])
        
    Returns:
        Dictionary with VaR calculations for each confidence level
    """
    try:
        logger.info(f"Calculating point-in-time VaR for {target_date}")
        
        # Aggregate portfolio returns for the target date
        portfolio_returns_vector = (
            simulation_df
            .filter(F.col('date') == target_date)
            .groupBy('date')
            .agg(Summarizer.sum(F.col('weighted_returns')).alias('portfolio_returns'))
            .collect()
        )
        
        if not portfolio_returns_vector:
            raise ValueError(f"No simulation data found for date: {target_date}")
        
        # Extract return vector
        returns_array = portfolio_returns_vector[0]['portfolio_returns'].toArray()
        
        # Calculate VaR for each confidence level
        var_results = {}
        
        for confidence_level in confidence_levels:
            # Calculate VaR as quantile
            var_percentile = (1 - confidence_level) * 100
            var_value = np.percentile(returns_array, var_percentile)
            
            # Calculate additional risk metrics
            expected_shortfall = np.mean(returns_array[returns_array <= var_value])
            
            var_results[f'var_{int(confidence_level * 100)}'] = {
                'confidence_level': confidence_level,
                'var_value': float(var_value),
                'expected_shortfall': float(expected_shortfall),
                'simulation_count': len(returns_array),
                'portfolio_mean_return': float(np.mean(returns_array)),
                'portfolio_std_return': float(np.std(returns_array))
            }
        
        logger.info(f"VaR calculation completed for {len(confidence_levels)} confidence levels")
        
        return {
            'date': target_date,
            'var_metrics': var_results,
            'returns_vector': returns_array
        }
        
    except Exception as e:
        logger.error(f"Point-in-time VaR calculation failed: {str(e)}")
        raise

# Calculate point-in-time VaR
try:
    # Get the earliest date for demonstration
    min_date = trials_df.select(F.min('date').alias('date')).collect()[0]['date']
    
    # Calculate VaR for multiple confidence levels
    var_results = calculate_point_in_time_var(
        simulation_df, 
        min_date, 
        confidence_levels
    )
    
    # Display results
    logger.info("Point-in-time VaR Results:")
    for var_level, metrics in var_results['var_metrics'].items():
        logger.info(f"  {var_level}: VaR = {metrics['var_value']:.4f}, ES = {metrics['expected_shortfall']:.4f}")
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"point_in_time_var_{min_date}"):
        mlflow.log_param("calculation_date", min_date)
        mlflow.log_param("simulation_count", var_results['var_metrics']['var_99']['simulation_count'])
        
        for var_level, metrics in var_results['var_metrics'].items():
            mlflow.log_metric(f"{var_level}_value", metrics['var_value'])
            mlflow.log_metric(f"{var_level}_expected_shortfall", metrics['expected_shortfall'])
    
    logger.info("VaR results logged to MLflow")
    
except Exception as e:
    logger.error(f"Point-in-time VaR calculation failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern VaR Visualization
# MAGIC 
# MAGIC Enhanced visualization with modern plotting and error handling.

# COMMAND ----------

def create_modern_var_visualization(returns_vector: np.ndarray, confidence_levels: List[float]) -> None:
    """
    Create modern VaR visualization with multiple confidence levels
    
    Args:
        returns_vector: Array of portfolio returns
        confidence_levels: List of confidence levels to display
    """
    try:
        from utils.var_viz import plot_var
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Set modern plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create visualization for each confidence level
        fig, axes = plt.subplots(1, len(confidence_levels), figsize=(15, 5))
        
        if len(confidence_levels) == 1:
            axes = [axes]
        
        for i, confidence_level in enumerate(confidence_levels):
            plt.sca(axes[i])
            
            # Create VaR plot
            plot_var(returns_vector, int(confidence_level * 100))
            
            # Add title
            plt.title(f'VaR at {int(confidence_level * 100)}% Confidence Level', 
                     fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("VaR visualization created successfully")
        
    except Exception as e:
        logger.error(f"VaR visualization failed: {str(e)}")

# Create visualization
try:
    create_modern_var_visualization(
        var_results['returns_vector'], 
        confidence_levels
    )
    
except Exception as e:
    logger.error(f"VaR visualization failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Evolution of risk exposure
# MAGIC The same can be achieved at scale, over our entire trading history. For each date, we aggregate all trial vectors and extract the worst 1% of events 

# COMMAND ----------

from utils.var_udf import get_var_udf

risk_exposure = (
  simulation_df
    .groupBy('date')
    .agg(Summarizer.sum(F.col('weighted_returns')).alias('returns'))
    .withColumn('var_99', get_var_udf(F.col('returns'), F.lit(99)))
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
    .agg(Summarizer.sum(F.col('weighted_returns')).alias('returns'))
    .withColumn('var_99', get_var_udf(F.col('returns'), F.lit(99)))
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
    .agg(Summarizer.sum(F.col('weighted_returns')).alias('returns'))
    .withColumn('var_99', get_var_udf(F.col('returns'), F.lit(99)))
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

# COMMAND ----------


