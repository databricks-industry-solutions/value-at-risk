# Databricks notebook source
# MAGIC %md
# MAGIC # Modern Market Data ETL
# MAGIC 
# MAGIC This notebook demonstrates modern data engineering practices for financial market data:
# MAGIC - Uses Unity Catalog for data governance
# MAGIC - Implements modern pandas UDF patterns
# MAGIC - Includes comprehensive error handling and logging
# MAGIC - Stores data in Delta tables with proper versioning

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Portfolio Configuration
# MAGIC 
# MAGIC Modern portfolio management with validation and Unity Catalog integration.

# COMMAND ----------

import logging
import datetime as dt
from typing import Optional, List, Dict, Any
from pyspark.sql import DataFrame
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, col, current_timestamp, lit
from pyspark.sql import functions as F
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Display portfolio with modern formatting
logger.info(f"Portfolio contains {portfolio_df.count()} instruments")
display(portfolio_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Market Data Download
# MAGIC 
# MAGIC Using updated pandas UDF patterns and comprehensive error handling.

# COMMAND ----------

def parse_date_config(date_str: str) -> dt.date:
    """Parse date configuration with error handling"""
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        logger.error(f"Invalid date format: {date_str}")
        raise

# Parse dates from modern configuration
y_min_date = parse_date_config(config['market_data']['yfinance']['mindate'])
y_max_date = parse_date_config(config['market_data']['yfinance']['maxdate'])

logger.info(f"Data range: {y_min_date} to {y_max_date}")

# COMMAND ----------

# Modern market data schema with additional metadata
market_data_schema = StructType([
    StructField('ticker', StringType(), False), 
    StructField('date', TimestampType(), False),
    StructField('open', DoubleType(), True),
    StructField('high', DoubleType(), True),
    StructField('low', DoubleType(), True),
    StructField('close', DoubleType(), False),
    StructField('volume', DoubleType(), True),
    StructField('adj_close', DoubleType(), True),
    StructField('data_source', StringType(), False),
    StructField('ingestion_timestamp', TimestampType(), False)
])

# Modern pandas UDF using current syntax
@pandas_udf(returnType=market_data_schema, functionType=pandas_udf.GROUPED_MAP)
def download_market_data_udf(group_key: pd.Series, pdf: pd.DataFrame) -> pd.DataFrame:
    """
    Modern pandas UDF for downloading market data with error handling
    
    Args:
        group_key: Series containing the ticker symbol
        pdf: DataFrame containing ticker information
        
    Returns:
        DataFrame with market data
    """
    from utils.var_utils import download_market_data
    
    ticker = group_key.iloc[0]
    
    try:
        logger.info(f"Downloading data for {ticker}")
        
        # Download market data with error handling
        market_data = download_market_data(ticker, y_min_date, y_max_date)
        
        if market_data.empty:
            logger.warning(f"No data found for ticker: {ticker}")
            return pd.DataFrame(columns=[field.name for field in market_data_schema.fields])
        
        # Add metadata columns
        market_data['data_source'] = 'yfinance'
        market_data['ingestion_timestamp'] = pd.Timestamp.now()
        
        # Ensure proper data types
        market_data = market_data.astype({
            'ticker': 'string',
            'open': 'float64',
            'high': 'float64', 
            'low': 'float64',
            'close': 'float64',
            'volume': 'float64',
            'adj_close': 'float64',
            'data_source': 'string'
        })
        
        logger.info(f"Successfully downloaded {len(market_data)} records for {ticker}")
        return market_data
        
    except Exception as e:
        logger.error(f"Error downloading data for {ticker}: {str(e)}")
        # Return empty DataFrame with correct schema
        return pd.DataFrame(columns=[field.name for field in market_data_schema.fields])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Unity Catalog Storage
# MAGIC 
# MAGIC Store market data in Unity Catalog with proper governance and versioning.

# COMMAND ----------

def save_market_data_to_unity_catalog(df: DataFrame, table_name: str) -> None:
    """
    Save market data to Unity Catalog with modern patterns
    
    Args:
        df: DataFrame to save
        table_name: Target table name in Unity Catalog
    """
    try:
        # Add audit columns
        df_with_audit = df.withColumn("created_at", current_timestamp()) \
                         .withColumn("created_by", lit(spark.sql("SELECT current_user()").collect()[0][0]))
        
        # Write to Unity Catalog with Delta format
        (df_with_audit
         .write
         .format('delta')
         .mode('overwrite')
         .option('overwriteSchema', 'true')
         .option('delta.autoOptimize.optimizeWrite', 'true')
         .option('delta.autoOptimize.autoCompact', 'true')
         .saveAsTable(f"{catalog_name}.{schema_name}.{table_name}"))
        
        logger.info(f"Successfully saved {df.count()} records to {catalog_name}.{schema_name}.{table_name}")
        
    except Exception as e:
        logger.error(f"Failed to save data to Unity Catalog: {str(e)}")
        raise

# Download market data using modern patterns
try:
    market_data_df = (
        portfolio_df
        .groupBy('ticker')
        .apply(download_market_data_udf)
        .filter(col('ticker').isNotNull())  # Filter out failed downloads
    )
    
    # Save to Unity Catalog
    save_market_data_to_unity_catalog(
        market_data_df, 
        config['unity_catalog']['tables']['market_data']
    )
    
except Exception as e:
    logger.error(f"Market data download failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Quality Validation
# MAGIC 
# MAGIC Modern data quality checks and validation.

# COMMAND ----------

def validate_market_data(table_name: str) -> Dict[str, Any]:
    """
    Validate market data quality with comprehensive checks
    
    Args:
        table_name: Table name to validate
        
    Returns:
        Dict containing validation results
    """
    try:
        df = spark.read.table(f"{catalog_name}.{schema_name}.{table_name}")
        
        validation_results = {
            'total_records': df.count(),
            'unique_tickers': df.select('ticker').distinct().count(),
            'date_range': {
                'min_date': df.select(F.min('date')).collect()[0][0],
                'max_date': df.select(F.max('date')).collect()[0][0]
            },
            'null_checks': {
                'ticker_nulls': df.filter(col('ticker').isNull()).count(),
                'date_nulls': df.filter(col('date').isNull()).count(),
                'close_nulls': df.filter(col('close').isNull()).count()
            },
            'data_quality': {
                'negative_prices': df.filter(col('close') < 0).count(),
                'zero_volume_days': df.filter(col('volume') == 0).count()
            }
        }
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

# Validate downloaded data
validation_results = validate_market_data(config['unity_catalog']['tables']['market_data'])

logger.info("Data Quality Validation Results:")
for category, results in validation_results.items():
    logger.info(f"{category}: {results}")

# Display validation summary
display(spark.sql(f"""
    SELECT 
        ticker,
        COUNT(*) as record_count,
        MIN(date) as min_date,
        MAX(date) as max_date,
        AVG(close) as avg_close_price,
        SUM(volume) as total_volume
    FROM {catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_data']}
    GROUP BY ticker
    ORDER BY ticker
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Data Visualization
# MAGIC 
# MAGIC Enhanced visualization with error handling and modern plotting libraries.

# COMMAND ----------

def get_sample_stock_data(ticker: Optional[str] = None) -> pd.DataFrame:
    """
    Get sample stock data for visualization
    
    Args:
        ticker: Specific ticker to retrieve, or None for first available
        
    Returns:
        Pandas DataFrame with stock data
    """
    try:
        if ticker is None:
            # Get first ticker from portfolio
            first_ticker = portfolio_df.select('ticker').first()['ticker']
        else:
            first_ticker = ticker
            
        stock_df = (
            spark.read.table(f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_data']}")
            .filter(col('ticker') == first_ticker)
            .orderBy(F.asc('date'))
            .toPandas()
        )
        
        logger.info(f"Retrieved {len(stock_df)} records for {first_ticker}")
        return stock_df
        
    except Exception as e:
        logger.error(f"Failed to retrieve stock data: {str(e)}")
        raise

# Create modern visualization
try:
    from utils.var_viz import plot_candlesticks
    
    sample_data = get_sample_stock_data()
    
    if not sample_data.empty:
        plot_candlesticks(sample_data)
        logger.info("Candlestick visualization created successfully")
    else:
        logger.warning("No data available for visualization")
        
except Exception as e:
    logger.error(f"Visualization failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Market Indicators Download
# MAGIC 
# MAGIC Enhanced market indicators processing with comprehensive error handling.

# COMMAND ----------

def download_market_indicators() -> DataFrame:
    """
    Download market indicators with modern error handling
    
    Returns:
        DataFrame with market indicators
    """
    try:
        from utils.var_utils import download_market_indicators
        
        logger.info("Downloading market indicators...")
        
        # Load indicator configuration
        with open('config/indicators.json', 'r') as f:
            import json
            indicators_config = json.load(f)
        
        # Download indicators using modern patterns
        indicators_df = download_market_indicators(
            indicators_config, 
            y_min_date, 
            y_max_date
        )
        
        if indicators_df.count() > 0:
            # Save to Unity Catalog
            save_market_data_to_unity_catalog(
                indicators_df,
                config['unity_catalog']['tables']['market_indicators']
            )
            
            logger.info(f"Successfully downloaded and saved market indicators")
            return indicators_df
        else:
            logger.warning("No market indicators data retrieved")
            return spark.createDataFrame([], schema=market_data_schema)
            
    except Exception as e:
        logger.error(f"Market indicators download failed: {str(e)}")
        raise

# Download and process market indicators
try:
    indicators_df = download_market_indicators()
    
    # Display indicator summary
    display(spark.sql(f"""
        SELECT 
            ticker as indicator,
            COUNT(*) as record_count,
            MIN(date) as min_date,
            MAX(date) as max_date,
            AVG(close) as avg_value
        FROM {catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_indicators']}
        GROUP BY ticker
        ORDER BY ticker
    """))
    
except Exception as e:
    logger.error(f"Market indicators processing failed: {str(e)}")

# COMMAND ----------

# Create a pandas dataframe where each column contain close index
market_indicators_df = pd.DataFrame()
for indicator in market_indicators.keys():    
    close_df = download_market_data(indicator, y_min_date, y_max_date)['close'].copy()
    market_indicators_df[market_indicators[indicator]] = close_df
        
# Pandas does not keep index (date) when converted into spark dataframe
market_indicators_df['date'] = market_indicators_df.index

# COMMAND ----------

_ = (
  spark
    .createDataFrame(market_indicators_df)
    .write
    .format("delta")
    .mode("overwrite")
    .saveAsTable(config['database']['tables']['indicators'])
)

# COMMAND ----------

display(spark.read.table(config['database']['tables']['indicators']))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute market volatility
# MAGIC As mentioned in the introduction, the whole concept of parametric VaR is to learn from past volatility. Instead of processing each day against its closest history sequentially, we can apply a simple window function to compute last X days' worth of market volatility at every single point in time, learning statistics behind those multi variate distributions

# COMMAND ----------

import numpy as np

def get_market_returns():
  
  f_ret_pdf = spark.table(config['database']['tables']['indicators']).orderBy('date').toPandas()

  # add date column as pandas index for sliding window
  f_ret_pdf.index = f_ret_pdf['date']
  f_ret_pdf = f_ret_pdf.drop(columns = ['date'])

  # compute daily log returns
  f_ret_pdf = np.log(f_ret_pdf.shift(1)/f_ret_pdf)

  # add date columns
  f_ret_pdf['date'] = f_ret_pdf.index
  f_ret_pdf = f_ret_pdf.dropna()
  
  return (
    spark
      .createDataFrame(f_ret_pdf)
      .select(F.array(list(market_indicators.values())).alias('features'), F.col('date'))
  )

# COMMAND ----------

# MAGIC %md
# MAGIC Instead of recursively querying our data, we can apply a window function so that each insert of our table is "joined" with last X days worth of observations. We can compute statistics of market volatility for each window using simple UDFs

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql import functions as F
from utils.var_udf import *

days = lambda i: i * 86400 
volatility_window = Window.orderBy(F.col('date').cast('long')).rangeBetween(-days(config['monte-carlo']['volatility']), 0)

volatility_df = (
  get_market_returns()
    .select(
      F.col('date'),
      F.col('features'),
      F.collect_list('features').over(volatility_window).alias('volatility')
    )
    .filter(F.size('volatility') > 1)
    .select(
      F.col('date'),
      F.col('features'),
      compute_avg(F.col('volatility')).alias('vol_avg'),
      compute_cov(F.col('volatility')).alias('vol_cov')
    )
)

# COMMAND ----------

volatility_df.write.format('delta').mode('overwrite').saveAsTable(config['database']['tables']['volatility'])

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, we now have access to up to date indicators at every single point in time. For each day, we know the average of returns and our covariance matrix. These statistics will be used to generate random market conditions in our next notebook.

# COMMAND ----------

display(spark.read.table(config['database']['tables']['volatility']))
