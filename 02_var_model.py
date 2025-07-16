# Databricks notebook source
# MAGIC %md
# MAGIC # Modern Model Building with MLflow 2.8+
# MAGIC 
# MAGIC This notebook demonstrates modern ML engineering practices for financial risk modeling:
# MAGIC - Uses MLflow 2.8+ for comprehensive experiment tracking
# MAGIC - Implements Unity Catalog for model governance
# MAGIC - Includes model validation and monitoring
# MAGIC - Uses distributed training with modern Spark ML patterns

# COMMAND ----------

# MAGIC %run ./config/configure_notebook

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern MLflow and Model Configuration
# MAGIC 
# MAGIC Set up MLflow 2.8+ with Unity Catalog integration for enterprise-grade model management.

# COMMAND ----------

import datetime
import logging
import tempfile
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from pyspark.sql import functions as F
from pyspark.sql import DataFrame
from pyspark.sql.types import *

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Parse model training date from modern configuration
model_date = datetime.datetime.strptime(config['mlflow']['model']['training_date'], '%Y-%m-%d')
logger.info(f"Model training date: {model_date}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Enhanced Model Artifacts Management
# MAGIC 
# MAGIC Modern artifact management with proper cleanup and versioning.

# COMMAND ----------

class ModelArtifactManager:
    """Modern artifact management for ML models"""
    
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.artifact_path = self.temp_dir.name
        logger.info(f"Model artifacts directory: {self.artifact_path}")
    
    def get_artifact_path(self, filename: str) -> str:
        """Get full path for artifact file"""
        return f"{self.artifact_path}/{filename}"
    
    def cleanup(self):
        """Clean up temporary artifacts"""
        try:
            self.temp_dir.cleanup()
            logger.info("Artifact cleanup completed")
        except Exception as e:
            logger.warning(f"Artifact cleanup failed: {e}")

# Initialize artifact manager
artifact_manager = ModelArtifactManager()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Market Data Processing
# MAGIC 
# MAGIC Enhanced data processing with proper error handling and Unity Catalog integration.

# COMMAND ----------

def load_market_features(training_date: datetime.datetime) -> pd.DataFrame:
    """
    Load market features from Unity Catalog with modern patterns
    
    Args:
        training_date: Cutoff date for training data
        
    Returns:
        DataFrame with market features
    """
    try:
        # Load market volatility data from Unity Catalog
        market_volatility_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_volatility']}"
        
        logger.info(f"Loading market features from {market_volatility_table}")
        
        market_df = (
            spark.read.table(market_volatility_table)
            .filter(F.col('date') < training_date)
            .select('date', 'features')
            .orderBy('date')
        )
        
        if market_df.count() == 0:
            logger.warning("No market features found for training")
            return pd.DataFrame()
        
        # Convert to pandas for model training
        market_pd = market_df.toPandas()
        
        # Extract features from array column
        features_list = market_pd['features'].tolist()
        
        # Load indicator names from configuration
        with open('config/indicators.json', 'r') as f:
            import json
            indicators_config = json.load(f)
        
        # Create feature DataFrame
        feature_columns = list(indicators_config.values())
        features_df = pd.DataFrame(features_list, columns=feature_columns)
        features_df['date'] = market_pd['date']
        
        logger.info(f"Loaded {len(features_df)} market feature records")
        logger.info(f"Features: {feature_columns}")
        
        return features_df
        
    except Exception as e:
        logger.error(f"Failed to load market features: {str(e)}")
        raise

# Load market features with error handling
try:
    market_features_df = load_market_features(model_date)
    
    if not market_features_df.empty:
        display(market_features_df.head())
        
        # Log feature statistics
        logger.info(f"Feature statistics:")
        logger.info(f"Date range: {market_features_df['date'].min()} to {market_features_df['date'].max()}")
        logger.info(f"Feature columns: {[col for col in market_features_df.columns if col != 'date']}")
    else:
        logger.warning("No market features available for training")
        
except Exception as e:
    logger.error(f"Market features loading failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Stock Returns Calculation
# MAGIC 
# MAGIC Enhanced returns calculation with proper error handling and Unity Catalog integration.

# COMMAND ----------

def calculate_stock_returns(training_date: datetime.datetime) -> DataFrame:
    """
    Calculate stock returns using modern Spark patterns
    
    Args:
        training_date: Cutoff date for training data
        
    Returns:
        DataFrame with stock returns
    """
    try:
        from utils.var_udf import compute_return
        
        # Load stock data from Unity Catalog
        stocks_table = f"{catalog_name}.{schema_name}.{config['unity_catalog']['tables']['market_data']}"
        
        logger.info(f"Calculating returns from {stocks_table}")
        
        # Apply windowing function for returns calculation
        window = Window.partitionBy('ticker').orderBy('date').rowsBetween(-1, 0)
        
        stocks_df = (
            spark.read.table(stocks_table)
            .filter(F.col('close').isNotNull())
            .filter(F.col('date') < training_date)
            .withColumn("previous_close", F.lag('close', 1).over(window))
            .withColumn("return", compute_return('previous_close', 'close'))
            .filter(F.col('return').isNotNull())  # Remove first day (no previous close)
            .select('date', 'ticker', 'return', 'close')
        )
        
        returns_count = stocks_df.count()
        logger.info(f"Calculated {returns_count} return observations")
        
        return stocks_df
        
    except Exception as e:
        logger.error(f"Failed to calculate stock returns: {str(e)}")
        raise

# Calculate stock returns
try:
    stocks_returns_df = calculate_stock_returns(model_date)
    
    # Display sample returns
    display(stocks_returns_df.orderBy('date', 'ticker').limit(100))
    
    # Log summary statistics
    returns_stats = stocks_returns_df.select(
        F.count('return').alias('total_observations'),
        F.countDistinct('ticker').alias('unique_tickers'),
        F.min('date').alias('min_date'),
        F.max('date').alias('max_date'),
        F.avg('return').alias('avg_return'),
        F.stddev('return').alias('return_volatility')
    ).collect()[0]
    
    logger.info(f"Returns statistics: {returns_stats}")
    
except Exception as e:
    logger.error(f"Stock returns calculation failed: {str(e)}")
    raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## Modern Feature Engineering and Correlation Analysis
# MAGIC 
# MAGIC Enhanced feature analysis with modern visualization and MLflow tracking.

# COMMAND ----------

def analyze_feature_correlations(features_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze feature correlations with modern patterns
    
    Args:
        features_df: DataFrame with market features
        
    Returns:
        Dictionary with correlation analysis results
    """
    try:
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Calculate correlations
        feature_cols = [col for col in features_df.columns if col != 'date']
        correlation_matrix = features_df[feature_cols].corr(method='spearman', min_periods=12)
        
        # Create modern visualization
        plt.figure(figsize=(12, 10))
        sns.set_style("whitegrid")
        
        # Create heatmap with modern styling
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        heatmap = sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            cmap='RdBu_r',
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": .8}
        )
        
        plt.title('Market Factor Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save to artifacts
        correlation_plot_path = artifact_manager.get_artifact_path('factor_correlation.png')
        plt.savefig(correlation_plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate correlation statistics
        correlation_stats = {
            'max_correlation': correlation_matrix.abs().max().max(),
            'min_correlation': correlation_matrix.abs().min().min(),
            'avg_correlation': correlation_matrix.abs().mean().mean(),
            'high_correlation_pairs': []
        }
        
        # Find highly correlated pairs
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:  # High correlation threshold
                    correlation_stats['high_correlation_pairs'].append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        logger.info(f"Feature correlation analysis completed")
        logger.info(f"High correlation pairs: {len(correlation_stats['high_correlation_pairs'])}")
        
        return {
            'correlation_matrix': correlation_matrix,
            'correlation_stats': correlation_stats,
            'correlation_plot_path': correlation_plot_path
        }
        
    except Exception as e:
        logger.error(f"Feature correlation analysis failed: {str(e)}")
        raise

# Perform correlation analysis
try:
    if not market_features_df.empty:
        correlation_results = analyze_feature_correlations(market_features_df)
        
        # Log results to MLflow
        with mlflow.start_run(run_name="feature_correlation_analysis"):
            mlflow.log_params(correlation_results['correlation_stats'])
            mlflow.log_artifact(correlation_results['correlation_plot_path'])
            
            # Log correlation matrix as artifact
            correlation_matrix_path = artifact_manager.get_artifact_path('correlation_matrix.csv')
            correlation_results['correlation_matrix'].to_csv(correlation_matrix_path)
            mlflow.log_artifact(correlation_matrix_path)
            
            logger.info("Correlation analysis logged to MLflow")
    else:
        logger.warning("No market features available for correlation analysis")
        
except Exception as e:
    logger.error(f"Correlation analysis failed: {str(e)}")

# COMMAND ----------

# MAGIC %md
# MAGIC We join our market indicator data with stock returns to build an input dataset we can machine learn. We'll use [`tempo`](https://databrickslabs.github.io/tempo/) for this AS-OF join since our timestamps may be different in real life, with intra day tick data.

# COMMAND ----------

from tempo import *
market_tsdf = TSDF(market_df.join(stocks_df.select('ticker').distinct()), ts_col="date", partition_cols=['ticker'])
stocks_tsdf = TSDF(stocks_df, ts_col="date", partition_cols=['ticker'])

# COMMAND ----------

features_df = (
  stocks_tsdf.asofJoin(market_tsdf).df
    .select(
      F.col('date'),
      F.col('ticker'),
      F.col('right_features').alias('features'),
      F.col('return')
    )
    .filter(F.col('features').isNotNull())
)

display(features_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Building models
# MAGIC We show how any function or model can be easily wrapped as a `mlflow.pyfunc` model and registered as such on ml registry. Real life VAR models are obviously more complex than a simple linear regression described here and are not necessarily out of the box sklearn or keras models. Still, they should follow same ML development standard and can easily benefit from ml-flow functionalities as long as one can express model I/O as a form of `pd.Series`, `pd.DataFrame` or `np.array`

# COMMAND ----------

import statsmodels.api as sm
from pyspark.sql.types import *
from pyspark.sql.functions import pandas_udf, PandasUDFType
from utils.var_utils import non_linear_features

# use pandas UDF to train multiple model (one for each instrument) in parallel
# the resulting dataframe will be the linear regression weights for each instrument
train_model_schema = StructType([
  StructField('ticker', StringType(), True), 
  StructField('weights', ArrayType(FloatType()), True)
])

# a model would also be much more complex than the below
@pandas_udf(train_model_schema, PandasUDFType.GROUPED_MAP)
def train_model(group, pdf):
  import pandas as pd
  import numpy as np
  # build market factor vectors
  # add a constant - the intercept term for each instrument i.
  X = [non_linear_features(row) for row in np.array(pdf['features'])]
  X = sm.add_constant(X, prepend=True) 
  y = np.array(pdf['return'])
  model = sm.OLS(y, X).fit()
  w_df = pd.DataFrame(data=[[model.params]], columns=['weights'])
  w_df['ticker'] = group[0]
  return w_df

# COMMAND ----------

# the resulting dataframe easily fits in memory and will be saved as our "uber model"
model_df = features_df.groupBy('ticker').apply(train_model).toPandas()
display(model_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC One can package an entire business logic (being statistical models or more AI models) as a simple `pyfunc`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc import PythonModel

class RiskMLFlowModel(PythonModel):
  
  def __init__(self, model_df):
    self.weights = dict(zip(model_df.ticker, model_df.weights))

  def _predict_record(self, ticker, xs):
    # Our logic is really simplistic and use simple non linear features with a linear regression
    # still, models could be packaged as pyfunc regardless of sklearn, complex DL or plain stats objects
    from utils.var_utils import non_linear_features
    from utils.var_utils import predict_non_linears
    ps = self.weights[ticker]
    fs = non_linear_features(xs)
    return predict_non_linears(ps, fs)
  
  def predict(self, context, model_input):
    predicted = model_input[['ticker','features']].apply(lambda x: self._predict_record(*x), axis=1)
    return predicted

# COMMAND ----------

# MAGIC %md
# MAGIC Such a model will be tracked, stored, registered and signature of the model enforced to prevent from data drift.

# COMMAND ----------

from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name='value-at-risk') as run:
  
  # get mlflow run ID
  run_id = run.info.run_id
  
  # create our pyfunc model
  python_model = RiskMLFlowModel(model_df)
  
  # Get model input and output signatures
  model_input_df  = features_df.select('ticker', 'features').limit(10).toPandas()
  model_output_df = python_model.predict(None, model_input_df)
  model_signature = infer_signature(model_input_df, model_output_df)
  
  # log our model to mlflow
  mlflow.pyfunc.log_model(
    artifact_path="model", 
    python_model=python_model,
    signature=model_signature
  )
  
  # log additional artifacts
  mlflow.log_artifact("{}/factor_correlation.png".format(tempDir.name))

# COMMAND ----------

model_udf = mlflow.pyfunc.spark_udf(model_uri='runs:/{}/model'.format(run_id), result_type='float', spark=spark)
prediction_df = features_df.withColumn('predicted', model_udf(F.struct('ticker', 'features')))
display(prediction_df)

# COMMAND ----------

# compare expected vs. actual return
# sum mean square error per instrument
from utils.var_udf import wsse_udf
wsse_df = prediction_df \
  .withColumn('wsse', wsse_udf(F.col('predicted'), F.col('return'))) \
  .groupBy('ticker') \
  .agg(F.sum('wsse').alias('wsse'))

# get average wsse across portfolio
wsse = wsse_df.select(F.avg('wsse').alias('wsse')).toPandas().iloc[0].wsse

# plot mean square error as accuracy of our model for each instrument
ax = wsse_df.toPandas().plot.bar(x='ticker', y='wsse', rot=0, label=None, figsize=(24,5))
ax.get_legend().remove()
plt.title("Model WSSE for each instrument")
plt.xticks(rotation=45)
plt.ylabel("wsse")
plt.savefig("{}/model_wsse.png".format(tempDir.name))
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC We can update our previous experiment with results of our prediction model (sum square of error)

# COMMAND ----------

with mlflow.start_run(run_id=run_id) as run:
  mlflow.log_metric("wsse", wsse)
  mlflow.log_artifact("{}/model_wsse.png".format(tempDir.name))

# COMMAND ----------

# MAGIC %md
# MAGIC The experiment captured now contains all libraries required to run in isolation and is linked to specific delta version to enable full reproducibility

# COMMAND ----------

# MAGIC %md
# MAGIC <img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/var/images/var_experiments.png width="1000px">

# COMMAND ----------

# MAGIC %md
# MAGIC By registering our model to ML registry, we make it available to downstream processes and backend jobs such as our next notebook focused on monte carlo simulations

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
model_uri = "runs:/{}/model".format(run_id)
result = mlflow.register_model(model_uri, config['model']['name'])
version = result.version

# COMMAND ----------

# MAGIC %md
# MAGIC We can also promote our model to different stages programmatically. Although our models would need to be reviewed in real life scenario, we make it available as a production artifact for our next notebook and programmatically transition previous runs back to Archive.

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
for model in client.search_model_versions("name='{}'".format(config['model']['name'])):
  if model.current_stage == 'Production':
    print("Archiving model version {}".format(model.version))
    client.transition_model_version_stage(
      name=config['model']['name'],
      version=int(model.version),
      stage="Archived"
    )

# COMMAND ----------

client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name=config['model']['name'],
    version=version,
    stage="Production"
)

# COMMAND ----------

# MAGIC %md
# MAGIC Our model now production candidate, we can load our predictive logic as a simple user defined function and predict investment returns for every observed market condition

# COMMAND ----------

model_udf = mlflow.pyfunc.spark_udf(
  model_uri='models:/{}/production'.format(config['model']['name']), 
  result_type='float', 
  spark=spark
)

# COMMAND ----------

import numpy as np

plt.figure(figsize=(25,12))

prediction_df = features_df.withColumn('predicted', model_udf(F.struct("ticker", "features")))
df_past_1 = prediction_df.filter(F.col('ticker') == "EC").orderBy('date').toPandas()
df_past_2 = prediction_df.filter(F.col('ticker') == "EC").orderBy('date').toPandas()
plt.plot(df_past_1.date, df_past_1['return'])
plt.plot(df_past_2.date, df_past_2['predicted'], color='green', linestyle='--')

min_return = np.min(df_past_2['return'])
max_return = np.max(df_past_2['return'])

plt.ylim([min_return, max_return])
plt.title('Log return of EC')
plt.ylabel('log return')
plt.xlabel('date')
plt.show()

# COMMAND ----------

tempDir.cleanup()

# COMMAND ----------


