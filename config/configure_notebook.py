# Databricks notebook source
# MAGIC %md
# MAGIC # Modern Configuration Setup
# MAGIC 
# MAGIC This notebook sets up the modern Unity Catalog environment and configuration
# MAGIC for the Value at Risk solution using industry best practices.

# COMMAND ----------

# MAGIC %pip install -r requirements.txt

# COMMAND ----------

import warnings
warnings.filterwarnings("ignore")

import yaml
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Loading
# MAGIC 
# MAGIC Load configuration with environment-specific overrides and DAB parameter integration.

# COMMAND ----------

def load_config():
    """Load configuration with environment-specific overrides"""
    try:
        with open('config/application.yaml', 'r') as f:
            config = yaml.safe_load(f)
        
        # Get environment from DAB bundle or default to dev
        environment = dbutils.widgets.get("environment") if dbutils.widgets.get("environment") else "dev"
        
        # Apply environment-specific overrides
        if environment in config.get('environments', {}):
            env_config = config['environments'][environment]
            # Deep merge environment config
            config = merge_configs(config, env_config)
        
        # Override with DAB bundle parameters
        catalog_name = dbutils.widgets.get("catalog_name") if dbutils.widgets.get("catalog_name") else "dev_value_at_risk"
        schema_name = dbutils.widgets.get("schema_name") if dbutils.widgets.get("schema_name") else "risk_management"
        
        # Update Unity Catalog configuration
        config['unity_catalog']['catalog_name'] = catalog_name
        config['unity_catalog']['schema_name'] = schema_name
        
        logger.info(f"Configuration loaded for environment: {environment}")
        logger.info(f"Unity Catalog: {catalog_name}.{schema_name}")
        
        return config
    
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        raise

def merge_configs(base_config, override_config):
    """Deep merge two configuration dictionaries"""
    for key, value in override_config.items():
        if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
            merge_configs(base_config[key], value)
        else:
            base_config[key] = value
    return base_config

# Load configuration
config = load_config()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Unity Catalog Setup
# MAGIC 
# MAGIC Modern Unity Catalog setup with proper governance and security.

# COMMAND ----------

def setup_unity_catalog():
    """Setup Unity Catalog with modern patterns"""
    try:
        catalog_name = config['unity_catalog']['catalog_name']
        schema_name = config['unity_catalog']['schema_name']
        
        # Create catalog if it doesn't exist
        spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog_name}")
        logger.info(f"✅ Catalog created/verified: {catalog_name}")
        
        # Use the catalog
        spark.sql(f"USE CATALOG {catalog_name}")
        
        # Create schema if it doesn't exist
        spark.sql(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        logger.info(f"✅ Schema created/verified: {schema_name}")
        
        # Use the schema
        spark.sql(f"USE SCHEMA {schema_name}")
        
        # Verify setup
        current_catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
        current_schema = spark.sql("SELECT current_schema()").collect()[0][0]
        
        logger.info(f"✅ Unity Catalog setup complete: {current_catalog}.{current_schema}")
        
        return current_catalog, current_schema
        
    except Exception as e:
        logger.error(f"Failed to setup Unity Catalog: {e}")
        raise

# Setup Unity Catalog
catalog_name, schema_name = setup_unity_catalog()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow 2.8+ Configuration
# MAGIC 
# MAGIC Modern MLflow setup with experiment tracking and model management.

# COMMAND ----------

import mlflow
import mlflow.sklearn

def setup_mlflow():
    """Setup MLflow 2.8+ with modern patterns"""
    try:
        # Set experiment with Unity Catalog integration
        experiment_name = f"/Shared/{catalog_name}/{schema_name}/{config['mlflow']['experiment']['name']}"
        mlflow.set_experiment(experiment_name)
        
        # Enable autologging
        mlflow.sklearn.autolog()
        
        # Set tracking URI to Unity Catalog
        mlflow.set_tracking_uri("databricks")
        
        # Set registry URI to Unity Catalog
        mlflow.set_registry_uri("databricks-uc")
        
        logger.info(f"✅ MLflow experiment configured: {experiment_name}")
        
        return experiment_name
        
    except Exception as e:
        logger.error(f"Failed to setup MLflow: {e}")
        raise

# Setup MLflow
experiment_name = setup_mlflow()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Portfolio Configuration
# MAGIC 
# MAGIC Load portfolio configuration with modern error handling.

# COMMAND ----------

def load_portfolio():
    """Load portfolio configuration with validation"""
    try:
        import json
        portfolio_file = config['risk_management']['portfolio']['config_file']
        
        with open(portfolio_file, 'r') as f:
            portfolio_data = json.load(f)
        
        # Validate portfolio data
        if not isinstance(portfolio_data, list):
            raise ValueError("Portfolio data must be a list")
        
        # Convert to DataFrame for easier handling
        portfolio_df = spark.createDataFrame(portfolio_data)
        
        logger.info(f"✅ Portfolio loaded: {portfolio_df.count()} instruments")
        
        return portfolio_df
        
    except Exception as e:
        logger.error(f"Failed to load portfolio: {e}")
        raise

# Load portfolio
portfolio_df = load_portfolio()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration Summary
# MAGIC 
# MAGIC Display configuration summary for verification.

# COMMAND ----------

print("🎯 Value at Risk - Modern Configuration Summary")
print("=" * 60)
print(f"Environment: {dbutils.widgets.get('environment') or 'dev'}")
print(f"Unity Catalog: {catalog_name}.{schema_name}")
print(f"MLflow Experiment: {experiment_name}")
print(f"Portfolio Instruments: {portfolio_df.count()}")
print(f"Monte Carlo Runs: {config['monte_carlo']['simulation']['runs']:,}")
print(f"Confidence Levels: {config['monte_carlo']['simulation']['confidence_levels']}")
print(f"Volatility Window: {config['monte_carlo']['simulation']['volatility_window']} days")
print(f"Time Horizon: {config['monte_carlo']['simulation']['time_horizon']} day(s)")
print("=" * 60)
print("✅ Configuration loaded successfully!")
