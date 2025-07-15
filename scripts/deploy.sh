#!/bin/bash

# Value at Risk - Deployment Script
# This script deploys the modernized VaR solution using Databricks Asset Bundles

set -e

echo "🚀 Starting Value at Risk deployment..."

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    echo "❌ Databricks CLI not found. Please install it:"
    echo "pip install databricks-cli"
    exit 1
fi

# Check if authenticated
if ! databricks auth describe &> /dev/null; then
    echo "❌ Databricks CLI not authenticated. Please run:"
    echo "databricks configure"
    exit 1
fi

# Set deployment environment (default to dev)
ENVIRONMENT=${1:-dev}
echo "📦 Deploying to environment: $ENVIRONMENT"

# Validate bundle configuration
echo "🔍 Validating bundle configuration..."
databricks bundle validate --var="environment=$ENVIRONMENT"

# Deploy the bundle
echo "🚀 Deploying bundle..."
databricks bundle deploy --target $ENVIRONMENT --var="environment=$ENVIRONMENT"

# Show deployment summary
echo "📋 Deployment summary:"
databricks bundle summary --target $ENVIRONMENT --var="environment=$ENVIRONMENT"

echo "✅ Deployment completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Run the workflow: databricks bundle run value_at_risk_workflow --target $ENVIRONMENT"
echo "2. Monitor execution in your Databricks workspace"
echo "3. Review results in Unity Catalog: ${CATALOG_NAME:-dev_value_at_risk}.${SCHEMA_NAME:-risk_management}"
echo ""
echo "📊 Access your VaR dashboard at: ${DATABRICKS_HOST:-your-workspace}/sql/dashboards"