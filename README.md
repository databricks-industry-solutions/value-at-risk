# Value at Risk - Modern Risk Management Solution 📊

<img src=https://raw.githubusercontent.com/databricks-industry-solutions/.github/main/profile/solacc_logo.png width="600px">

[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://databricks.com/try-databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)
[![DAB](https://img.shields.io/badge/DAB-ENABLED-orange?style=for-the-badge)](https://docs.databricks.com/dev-tools/bundles/index.html)

## 🏦 Industry Use Case

**Modernized Value at Risk (VaR) calculation** for financial institutions using cutting-edge Databricks technologies. This solution demonstrates how banks can modernize their risk management practices by leveraging Unity Catalog, modern MLflow, and scalable Monte Carlo simulations.

### Key Features

- **🎯 Modern Architecture**: Built with Databricks Asset Bundles (DAB) for seamless deployment
- **🔐 Unity Catalog Integration**: Enterprise-grade data governance and security
- **🔬 MLflow 2.8+**: Advanced experiment tracking and model management
- **⚡ Serverless Compute**: Cost-effective, auto-scaling compute resources
- **🎲 Monte Carlo Simulations**: Distributed risk calculations using Apache Spark
- **📊 Real-time Monitoring**: Comprehensive backtesting and compliance reporting

<img src='https://raw.githubusercontent.com/databricks-industry-solutions/value-at-risk/master/images/reference_architecture.png' width=800>

## 🚀 Quick Start

### Option 1: One-Click Deployment
```bash
# Prerequisites
pip install databricks-cli

# Configure Databricks (if not already done)
databricks configure

# Deploy everything
./scripts/deploy.sh

# Clean up when done
./scripts/cleanup.sh
```

### Option 2: Manual Deployment
```bash
# Validate configuration
databricks bundle validate

# Deploy to development
databricks bundle deploy --target dev

# Run the VaR workflow
databricks bundle run value_at_risk_workflow --target dev
```

## 📁 Project Structure

```
├── databricks.yml              # DAB configuration
├── notebooks/
│   ├── 00_var_context.py       # Setup and configuration
│   ├── 01_var_market_etl.py    # Market data extraction
│   ├── 02_var_model.py         # Model training with MLflow
│   ├── 03_var_monte_carlo.py   # Monte Carlo simulations
│   ├── 04_var_aggregation.py   # Risk aggregation
│   └── 05_var_compliance.py    # Compliance reporting
├── .github/workflows/           # CI/CD automation
├── scripts/                     # Deployment utilities
├── utils/                       # Utility functions
├── config/                      # Configuration files
└── tests/                       # Unit tests
```

## 🔧 Configuration

### Environment Variables (.env)
```bash
DATABRICKS_HOST=https://your-workspace.cloud.databricks.com/
DATABRICKS_TOKEN=your-access-token
DATABRICKS_WAREHOUSE_ID=your-warehouse-id
CATALOG_NAME=dev_value_at_risk
SCHEMA_NAME=risk_management
ENVIRONMENT=dev
```

### Key Configuration Options
- **Catalog Name**: Unity Catalog name for data governance
- **Schema Name**: Database schema for risk management tables
- **Environment**: Deployment environment (dev/staging/prod)
- **Confidence Level**: VaR confidence level (default: 99%)
- **Monte Carlo Trials**: Number of simulation trials (default: 10,000)

## 🎯 Risk Management Pipeline

### 1. Context & Setup (`00_var_context.py`)
- Unity Catalog infrastructure setup
- MLflow experiment configuration
- Modern configuration management

### 2. Market Data ETL (`01_var_market_etl.py`)
- Yahoo Finance data extraction
- Data quality validation
- Delta Lake storage with versioning

### 3. Model Training (`02_var_model.py`)
- Predictive model development
- MLflow experiment tracking
- Model versioning and registry

### 4. Monte Carlo Simulation (`03_var_monte_carlo.py`)
- Distributed risk simulations
- Parallel computation using Spark
- Results storage in Delta tables

### 5. Risk Aggregation (`04_var_aggregation.py`)
- Portfolio-level risk calculations
- On-demand VaR aggregation
- Historical backtesting

### 6. Compliance Reporting (`05_var_compliance.py`)
- Basel Committee compliance
- Backtesting validation
- Automated reporting

## 📊 What's New in 2025

### Modernization Features
- ✅ **DAB Structure**: Asset Bundle deployment for reproducible environments
- ✅ **Unity Catalog**: Enterprise data governance and security
- ✅ **MLflow 2.8+**: Advanced experiment tracking and model management
- ✅ **Serverless Compute**: Cost-effective, auto-scaling infrastructure
- ✅ **Modern Dependencies**: Latest Python packages and Databricks features
- ✅ **CI/CD Pipeline**: Automated testing and deployment workflows
- ✅ **Enhanced Documentation**: Comprehensive setup and usage guides

### Technical Improvements
- 🔄 **Runtime Agnostic**: Works with latest Databricks runtime versions
- 🔐 **Security Enhanced**: Modern authentication and authorization patterns
- 📈 **Performance Optimized**: Efficient data processing and storage
- 🧪 **Testing Framework**: Comprehensive unit and integration tests
- 📋 **Monitoring**: Enhanced observability and error handling

## 🛠️ Development

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/tests_spark.py -v
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Validate bundle locally
databricks bundle validate

# Deploy to dev environment
databricks bundle deploy --target dev
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `databricks bundle validate`
5. Submit a pull request

## 📚 Dependencies

| Library | Description | License | Source |
|---------|-------------|---------|--------|
| yfinance | Yahoo Finance API | Apache2 | https://github.com/ranaroussi/yfinance |
| dbl-tempo | Time series library | Databricks | https://github.com/databrickslabs/tempo |
| mlflow | ML lifecycle management | Apache2 | https://github.com/mlflow/mlflow |
| pandas | Data manipulation | BSD | https://github.com/pandas-dev/pandas |
| numpy | Numerical computing | BSD | https://github.com/numpy/numpy |
| scikit-learn | Machine learning | BSD | https://github.com/scikit-learn/scikit-learn |

## 📄 License

This project is licensed under the Databricks License - see the [LICENSE.md](LICENSE.md) file for details.

## 🔗 Resources

- [Databricks Asset Bundles Guide](https://docs.databricks.com/dev-tools/bundles/index.html)
- [Unity Catalog Documentation](https://docs.databricks.com/data-governance/unity-catalog/index.html)
- [MLflow 2.8+ Documentation](https://mlflow.org/docs/latest/index.html)
- [Value at Risk Best Practices](https://www.bis.org/publ/bcbs75.htm)

---

**Built with ❤️ using Databricks Asset Bundles, Unity Catalog, and MLflow 2.8+**

*Modernized for 2025 with latest Databricks features and industry best practices*