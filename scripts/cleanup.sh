#!/bin/bash

# Value at Risk - Cleanup Script
# This script removes all deployed resources from the specified environment

set -e

echo "🧹 Starting Value at Risk cleanup..."

# Set environment (default to dev)
ENVIRONMENT=${1:-dev}

# Confirmation prompt
echo "⚠️  This will remove ALL resources from the '$ENVIRONMENT' environment."
read -p "Are you sure you want to continue? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

# Destroy the bundle
echo "🗑️  Destroying bundle resources..."
databricks bundle destroy --target $ENVIRONMENT --var="environment=$ENVIRONMENT"

echo "✅ Cleanup completed successfully!"
echo "📋 All resources have been removed from the '$ENVIRONMENT' environment."