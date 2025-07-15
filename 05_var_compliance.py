# Databricks notebook source
# MAGIC %md
# MAGIC # Modern Basel Committee Compliance and Risk Management
# MAGIC 
# MAGIC This notebook implements enterprise-grade Basel Committee compliance standards:
# MAGIC - Automated backtesting with Basel III traffic light system
# MAGIC - Real-time compliance monitoring and alerting
# MAGIC - Comprehensive risk reporting and audit trails
# MAGIC - Modern data governance with Unity Catalog integration
# MAGIC - MLflow tracking for regulatory compliance
# MAGIC 
# MAGIC ## Basel Committee Traffic Light System
# MAGIC 
# MAGIC | Zone   | Threshold                 | Results                       |
# MAGIC |---------|---------------------------|-------------------------------|
# MAGIC | Green   | Up to 4 exceedances       | No particular concerns raised |
# MAGIC | Yellow  | Up to 9 exceedances       | Monitoring required           |
# MAGIC | Red     | More than 10 exceedances  | VaR measure to be improved    |

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Basel Committee Compliance Framework
# MAGIC 
# MAGIC Enhanced compliance framework with comprehensive monitoring and reporting.

# COMMAND ----------

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime, timedelta
from enum import Enum

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

# Basel Committee compliance configuration
class ComplianceZone(Enum):
    """Basel Committee traffic light zones"""
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"

# Load compliance configuration
compliance_config = config['risk_management']['compliance']
basel_compliance = compliance_config['basel_compliance']
backtesting_config = compliance_config['backtesting']

# Basel thresholds
BASEL_THRESHOLDS = {
    ComplianceZone.GREEN: backtesting_config['thresholds']['green'],
    ComplianceZone.YELLOW: backtesting_config['thresholds']['yellow'],
    ComplianceZone.RED: backtesting_config['thresholds']['red']
}

logger.info(f"Basel Committee compliance configuration:")
logger.info(f"  Basel compliance enabled: {basel_compliance}")
logger.info(f"  Backtesting window: {backtesting_config['window']} days")
logger.info(f"  Compliance thresholds: {BASEL_THRESHOLDS}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Monte Carlo Data Loading
# MAGIC 
# MAGIC Load simulation and market data with comprehensive validation.

# COMMAND ----------

def load_compliance_data() -> Tuple[DataFrame, DataFrame]:
    """
    Load Monte Carlo trials and market data for compliance analysis
    
    Returns:
        Tuple of (simulation_df, investment_returns_df)
    """
    try:
        from utils.var_udf import weighted_returns, compute_return
        
        # Load Monte Carlo trials from Unity Catalog
        trials_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['monte_carlo_trials']}"
        
        logger.info(f"Loading Monte Carlo trials from {trials_table}")
        
        trials_df = spark.read.table(trials_table)
        
        # Create weighted simulation DataFrame
        simulation_df = (
            trials_df
            .join(portfolio_df, ['ticker'])
            .withColumn('weighted_returns', weighted_returns('returns', 'weight'))
            .withColumn('compliance_timestamp', F.current_timestamp())
        )
        
        logger.info(f"Created weighted simulation DataFrame with {simulation_df.count():,} records")
        
        # Load actual investment returns
        market_data_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_data']}"
        
        logger.info(f"Loading market data from {market_data_table}")
        
        # Calculate investment returns using window functions
        window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)
        
        investment_returns_df = (
            spark.read.table(market_data_table)
            .filter(F.col('close').isNotNull())
            .join(portfolio_df, ['ticker'])
            .withColumn("previous_close", F.lag('close', 1).over(window))
            .withColumn("return", compute_return('previous_close', 'close'))
            .withColumn('weighted_return', weighted_returns('return', 'weight'))
            .filter(F.col('return').isNotNull())
            .select('date', 'ticker', 'return', 'weight', 'weighted_return')
        )
        
        logger.info(f"Created investment returns DataFrame with {investment_returns_df.count():,} records")
        
        return simulation_df, investment_returns_df
        
    except Exception as e:
        logger.error(f"Failed to load compliance data: {str(e)}")
        raise

# Load compliance data
try:
    simulation_df, investment_returns_df = load_compliance_data()
    
    # Display sample data
    logger.info("Sample simulation data:")
    display(simulation_df.select('date', 'ticker', 'returns', 'weight', 'weighted_returns').limit(10))
    
    logger.info("Sample investment returns data:")
    display(investment_returns_df.limit(10))
    
except Exception as e:
    logger.error(f"Compliance data loading failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Portfolio Returns Calculation
# MAGIC 
# MAGIC Calculate actual portfolio returns for compliance testing.

# COMMAND ----------

def calculate_portfolio_returns(investment_returns_df: DataFrame) -> DataFrame:
    """
    Calculate daily portfolio returns for compliance testing
    
    Args:
        investment_returns_df: DataFrame with individual instrument returns
        
    Returns:
        DataFrame with daily portfolio returns
    """
    try:
        # Aggregate weighted returns by date
        portfolio_returns_df = (
            investment_returns_df
            .groupBy('date')
            .agg(
                F.sum('weighted_return').alias('portfolio_return'),
                F.count('ticker').alias('instruments_count'),
                F.current_timestamp().alias('calculation_timestamp')
            )
            .orderBy('date')
        )
        
        logger.info(f"Calculated portfolio returns for {portfolio_returns_df.count()} trading days")
        
        return portfolio_returns_df
        
    except Exception as e:
        logger.error(f"Portfolio returns calculation failed: {str(e)}")
        raise

# Calculate portfolio returns
try:
    portfolio_returns_df = calculate_portfolio_returns(investment_returns_df)
    
    # Display portfolio returns summary
    returns_summary = portfolio_returns_df.select(
        F.count('portfolio_return').alias('trading_days'),
        F.min('date').alias('start_date'),
        F.max('date').alias('end_date'),
        F.avg('portfolio_return').alias('avg_daily_return'),
        F.stddev('portfolio_return').alias('daily_volatility')
    ).collect()[0]
    
    logger.info("Portfolio returns summary:")
    for field in returns_summary.asDict():
        logger.info(f"  {field}: {returns_summary[field]}")
    
    # Display sample returns
    display(portfolio_returns_df.orderBy(F.desc('date')).limit(10))
    
except Exception as e:
    logger.error(f"Portfolio returns calculation failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern VaR Calculation for Compliance
# MAGIC 
# MAGIC Calculate Value at Risk using modern distributed computing patterns.

# COMMAND ----------

def calculate_var_timeseries(simulation_df: DataFrame, confidence_level: float = 0.99) -> DataFrame:
    """
    Calculate VaR time series for compliance backtesting
    
    Args:
        simulation_df: DataFrame with weighted Monte Carlo simulations
        confidence_level: Confidence level for VaR calculation
        
    Returns:
        DataFrame with VaR time series
    """
    try:
        from utils.var_udf import get_var_udf
        
        logger.info(f"Calculating VaR time series at {confidence_level*100}% confidence level")
        
        # Calculate VaR for each date
        var_timeseries_df = (
            simulation_df
            .groupBy('date')
            .agg(Summarizer.sum(F.col('weighted_returns')).alias('portfolio_returns'))
            .withColumn('var_99', get_var_udf(F.col('portfolio_returns'), F.lit(int(confidence_level*100))))
            .withColumn('confidence_level', F.lit(confidence_level))
            .withColumn('var_calculation_timestamp', F.current_timestamp())
            .select('date', 'var_99', 'confidence_level', 'var_calculation_timestamp')
            .orderBy('date')
        )
        
        logger.info(f"Calculated VaR for {var_timeseries_df.count()} dates")
        
        return var_timeseries_df
        
    except Exception as e:
        logger.error(f"VaR calculation failed: {str(e)}")
        raise

# Calculate VaR time series
try:
    var_timeseries_df = calculate_var_timeseries(simulation_df)
    
    # Display VaR summary
    var_summary = var_timeseries_df.select(
        F.count('var_99').alias('var_observations'),
        F.min('date').alias('start_date'),
        F.max('date').alias('end_date'),
        F.avg('var_99').alias('avg_var_99'),
        F.min('var_99').alias('min_var_99'),
        F.max('var_99').alias('max_var_99')
    ).collect()[0]
    
    logger.info("VaR time series summary:")
    for field in var_summary.asDict():
        logger.info(f"  {field}: {var_summary[field]}")
    
    # Display sample VaR values
    display(var_timeseries_df.orderBy(F.desc('date')).limit(10))
    
except Exception as e:
    logger.error(f"VaR calculation failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Basel Committee Backtesting
# MAGIC 
# MAGIC Implement Basel Committee backtesting with modern time series joins using tempo.

# COMMAND ----------

def perform_basel_backtesting(
    portfolio_returns_df: DataFrame,
    var_timeseries_df: DataFrame,
    backtesting_window: int = 250
) -> Dict[str, Any]:
    """
    Perform Basel Committee backtesting with modern patterns
    
    Args:
        portfolio_returns_df: DataFrame with actual portfolio returns
        var_timeseries_df: DataFrame with VaR time series
        backtesting_window: Number of days for backtesting (default: 250)
        
    Returns:
        Dictionary with backtesting results
    """
    try:
        from tempo import TSDF
        
        logger.info(f"Performing Basel Committee backtesting with {backtesting_window} day window")
        
        # Create time series DataFrames
        returns_tsdf = TSDF(portfolio_returns_df, ts_col="date")
        var_tsdf = TSDF(var_timeseries_df, ts_col="date")
        
        # Perform AS-OF join to match returns with VaR
        backtest_df = (
            returns_tsdf.asofJoin(var_tsdf)
            .df
            .na.drop()
            .orderBy('date')
            .select(
                F.col('date'),
                F.col('portfolio_return').alias('actual_return'),
                F.col('right_var_99').alias('var_99')
            )
        )
        
        # Calculate exceedances (actual loss > VaR)
        exceedances_df = (
            backtest_df
            .withColumn('exceedance', F.when(F.col('actual_return') < F.col('var_99'), 1).otherwise(0))
            .withColumn('loss_amount', F.when(F.col('actual_return') < F.col('var_99'), 
                                            F.col('actual_return') - F.col('var_99')).otherwise(0))
        )
        
        # Get recent data for backtesting
        recent_data = (
            exceedances_df
            .orderBy(F.desc('date'))
            .limit(backtesting_window)
        )
        
        # Calculate backtesting metrics
        backtest_metrics = recent_data.agg(
            F.count('*').alias('total_observations'),
            F.sum('exceedance').alias('total_exceedances'),
            F.avg('actual_return').alias('avg_actual_return'),
            F.avg('var_99').alias('avg_var_99'),
            F.sum('loss_amount').alias('total_excess_loss')
        ).collect()[0]
        
        # Determine Basel Committee zone
        exceedances_count = backtest_metrics['total_exceedances']
        
        if exceedances_count <= BASEL_THRESHOLDS[ComplianceZone.GREEN]:
            compliance_zone = ComplianceZone.GREEN
        elif exceedances_count <= BASEL_THRESHOLDS[ComplianceZone.YELLOW]:
            compliance_zone = ComplianceZone.YELLOW
        else:
            compliance_zone = ComplianceZone.RED
        
        # Calculate additional metrics
        exceedance_rate = exceedances_count / backtest_metrics['total_observations']
        expected_exceedances = backtest_metrics['total_observations'] * 0.01  # 1% for 99% VaR
        
        backtest_results = {
            'backtesting_window': backtesting_window,
            'total_observations': backtest_metrics['total_observations'],
            'total_exceedances': exceedances_count,
            'exceedance_rate': exceedance_rate,
            'expected_exceedances': expected_exceedances,
            'compliance_zone': compliance_zone.value,
            'avg_actual_return': backtest_metrics['avg_actual_return'],
            'avg_var_99': backtest_metrics['avg_var_99'],
            'total_excess_loss': backtest_metrics['total_excess_loss'],
            'backtesting_date': datetime.now().isoformat()
        }
        
        logger.info(f"Basel Committee backtesting results:")
        logger.info(f"  Compliance zone: {compliance_zone.value.upper()}")
        logger.info(f"  Exceedances: {exceedances_count}/{backtest_metrics['total_observations']} ({exceedance_rate:.2%})")
        logger.info(f"  Expected exceedances: {expected_exceedances:.1f}")
        
        return {
            'backtest_results': backtest_results,
            'exceedances_df': exceedances_df,
            'backtest_df': recent_data
        }
        
    except Exception as e:
        logger.error(f"Basel Committee backtesting failed: {str(e)}")
        raise

# Perform Basel Committee backtesting
try:
    backtesting_results = perform_basel_backtesting(
        portfolio_returns_df,
        var_timeseries_df,
        backtesting_config['window']
    )
    
    # Display backtesting results
    backtest_summary = backtesting_results['backtest_results']
    
    logger.info("Basel Committee Backtesting Summary:")
    logger.info(f"  📊 Total observations: {backtest_summary['total_observations']}")
    logger.info(f"  ⚠️  Total exceedances: {backtest_summary['total_exceedances']}")
    logger.info(f"  📈 Exceedance rate: {backtest_summary['exceedance_rate']:.2%}")
    logger.info(f"  🎯 Expected exceedances: {backtest_summary['expected_exceedances']:.1f}")
    logger.info(f"  🚦 Compliance zone: {backtest_summary['compliance_zone'].upper()}")
    
    # Display sample backtesting data
    display(backtesting_results['backtest_df'].orderBy(F.desc('date')).limit(20))
    
    # Log to MLflow
    with mlflow.start_run(run_name="basel_committee_backtesting"):
        mlflow.log_params({
            'backtesting_window': backtest_summary['backtesting_window'],
            'compliance_zone': backtest_summary['compliance_zone']
        })
        
        mlflow.log_metrics({
            'total_exceedances': backtest_summary['total_exceedances'],
            'exceedance_rate': backtest_summary['exceedance_rate'],
            'expected_exceedances': backtest_summary['expected_exceedances'],
            'avg_var_99': backtest_summary['avg_var_99']
        })
        
        # Save backtesting results to Unity Catalog
        compliance_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['compliance_reports']}"
        
        # Create compliance report DataFrame
        compliance_report_df = spark.createDataFrame([backtest_summary])
        
        (compliance_report_df
         .write
         .format("delta")
         .mode("append")
         .option("mergeSchema", "true")
         .saveAsTable(compliance_table))
        
        logger.info(f"Backtesting results saved to {compliance_table}")
    
    logger.info("Basel Committee backtesting completed and logged to MLflow")
    
except Exception as e:
    logger.error(f"Basel Committee backtesting failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Compliance Visualization
# MAGIC 
# MAGIC Create comprehensive compliance dashboards and visualizations.

# COMMAND ----------

def create_compliance_visualization(backtest_df: DataFrame, backtest_results: Dict[str, Any]) -> None:
    """
    Create modern compliance visualization dashboard
    
    Args:
        backtest_df: DataFrame with backtesting results
        backtest_results: Dictionary with backtesting metrics
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Convert to pandas for visualization
        backtest_pd = backtest_df.toPandas()
        
        # Set modern plotting style
        plt.style.use('seaborn-v0_8')
        
        # Create compliance dashboard
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Returns vs VaR over time
        axes[0, 0].plot(backtest_pd['date'], backtest_pd['actual_return'], 
                       label='Actual Returns', alpha=0.7)
        axes[0, 0].plot(backtest_pd['date'], backtest_pd['var_99'], 
                       label='VaR 99%', color='red', linewidth=2)
        axes[0, 0].fill_between(backtest_pd['date'], backtest_pd['var_99'], 
                               alpha=0.3, color='red')
        axes[0, 0].set_title('Portfolio Returns vs VaR 99%', fontweight='bold')
        axes[0, 0].set_xlabel('Date')
        axes[0, 0].set_ylabel('Returns')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Exceedances over time
        exceedances_pd = backtest_pd[backtest_pd['exceedance'] == 1]
        axes[0, 1].scatter(exceedances_pd['date'], exceedances_pd['actual_return'], 
                          color='red', s=50, label='Exceedances')
        axes[0, 1].plot(backtest_pd['date'], backtest_pd['var_99'], 
                       label='VaR 99%', color='red', linewidth=2)
        axes[0, 1].set_title('VaR Exceedances', fontweight='bold')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Returns distribution
        axes[1, 0].hist(backtest_pd['actual_return'], bins=50, alpha=0.7, 
                       density=True, label='Actual Returns')
        axes[1, 0].axvline(backtest_pd['var_99'].mean(), color='red', 
                          linestyle='--', linewidth=2, label='Avg VaR 99%')
        axes[1, 0].set_title('Returns Distribution', fontweight='bold')
        axes[1, 0].set_xlabel('Returns')
        axes[1, 0].set_ylabel('Density')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Compliance zone indicator
        compliance_zone = backtest_results['compliance_zone']
        zone_colors = {'green': 'green', 'yellow': 'orange', 'red': 'red'}
        
        axes[1, 1].bar(['Compliance Zone'], [1], 
                      color=zone_colors[compliance_zone], alpha=0.7)
        axes[1, 1].set_title(f'Basel Committee Zone: {compliance_zone.upper()}', 
                           fontweight='bold')
        axes[1, 1].set_ylim(0, 1.2)
        axes[1, 1].text(0, 0.5, f"{backtest_results['total_exceedances']} exceedances", 
                        ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        logger.info("Compliance visualization created successfully")
        
    except Exception as e:
        logger.error(f"Compliance visualization failed: {str(e)}")

# Create compliance visualization
try:
    create_compliance_visualization(
        backtesting_results['backtest_df'],
        backtesting_results['backtest_results']
    )
    
except Exception as e:
    logger.error(f"Compliance visualization failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compliance Summary
# MAGIC 
# MAGIC Generate final compliance report and recommendations.

# COMMAND ----------

def generate_compliance_report(backtest_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate comprehensive compliance report
    
    Args:
        backtest_results: Dictionary with backtesting results
        
    Returns:
        Dictionary with compliance report
    """
    try:
        results = backtest_results['backtest_results']
        
        # Generate recommendations based on compliance zone
        if results['compliance_zone'] == ComplianceZone.GREEN.value:
            recommendations = [
                "✅ VaR model is performing well within Basel Committee standards",
                "📊 Continue regular monitoring and quarterly backtesting",
                "🔄 Consider model validation and stress testing"
            ]
        elif results['compliance_zone'] == ComplianceZone.YELLOW.value:
            recommendations = [
                "⚠️  Increased monitoring required for VaR model",
                "🔍 Investigate causes of elevated exceedances",
                "📈 Consider model parameter adjustments",
                "📋 Prepare detailed analysis for regulators"
            ]
        else:  # RED zone
            recommendations = [
                "🚨 Immediate VaR model improvement required",
                "🔧 Comprehensive model revision needed",
                "📊 Implement enhanced risk management controls",
                "📋 Prepare regulatory explanation and remediation plan"
            ]
        
        compliance_report = {
            'report_date': datetime.now().isoformat(),
            'compliance_zone': results['compliance_zone'],
            'exceedances_count': results['total_exceedances'],
            'exceedance_rate': results['exceedance_rate'],
            'backtesting_window': results['backtesting_window'],
            'recommendations': recommendations,
            'model_performance': {
                'avg_var_99': results['avg_var_99'],
                'avg_actual_return': results['avg_actual_return'],
                'total_excess_loss': results['total_excess_loss']
            }
        }
        
        logger.info("📋 Basel Committee Compliance Report Generated")
        logger.info(f"🚦 Compliance Zone: {results['compliance_zone'].upper()}")
        logger.info("📝 Recommendations:")
        for rec in recommendations:
            logger.info(f"   {rec}")
        
        return compliance_report
        
    except Exception as e:
        logger.error(f"Compliance report generation failed: {str(e)}")
        raise

# Generate final compliance report
try:
    compliance_report = generate_compliance_report(backtesting_results)
    
    # Display compliance report
    print("=" * 60)
    print("🏛️  BASEL COMMITTEE COMPLIANCE REPORT")
    print("=" * 60)
    print(f"📅 Report Date: {compliance_report['report_date']}")
    print(f"🚦 Compliance Zone: {compliance_report['compliance_zone'].upper()}")
    print(f"⚠️  Exceedances: {compliance_report['exceedances_count']}/{compliance_report['backtesting_window']} ({compliance_report['exceedance_rate']:.2%})")
    print(f"🎯 Average VaR 99%: {compliance_report['model_performance']['avg_var_99']:.4f}")
    print(f"📊 Average Return: {compliance_report['model_performance']['avg_actual_return']:.4f}")
    print("\n📝 Recommendations:")
    for rec in compliance_report['recommendations']:
        print(f"   {rec}")
    print("=" * 60)
    
    logger.info("✅ Basel Committee compliance analysis completed successfully")
    
except Exception as e:
    logger.error(f"Compliance report generation failed: {str(e)}")
    raise

# COMMAND ----------