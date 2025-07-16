#!/bin/bash

# Value at Risk - Deployment Script
# This script deploys the modernized VaR solution using Databricks Asset Bundles

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging function
log() {
    echo -e "${BLUE}[$(date '+%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Value at Risk - Deployment Script

USAGE:
    $0 [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT    Target environment (dev, staging, prod). Default: dev

OPTIONS:
    -h, --help     Show this help message
    -v, --verbose  Enable verbose output
    --dry-run      Validate configuration without deploying
    --force        Skip confirmation prompts

EXAMPLES:
    $0                    # Deploy to dev environment
    $0 staging           # Deploy to staging environment
    $0 prod --force      # Deploy to prod without confirmation

REQUIRED CONFIGURATION:
    1. Databricks CLI installed and authenticated
    2. warehouse_id configured (see env.example)
    3. Appropriate permissions for target environment

For more information, see: README.md
EOF
}

# Parse command line arguments
ENVIRONMENT="dev"
VERBOSE=false
DRY_RUN=false
FORCE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        -*)
            error "Unknown option $1"
            show_help
            exit 1
            ;;
        *)
            ENVIRONMENT="$1"
            shift
            ;;
    esac
done

# Validate environment
if [[ ! "$ENVIRONMENT" =~ ^(dev|staging|prod)$ ]]; then
    error "Invalid environment: $ENVIRONMENT. Must be one of: dev, staging, prod"
    exit 1
fi

log "🚀 Starting Value at Risk deployment to $ENVIRONMENT environment..."

# Check prerequisites
log "🔍 Checking prerequisites..."

# Check if databricks CLI is installed
if ! command -v databricks &> /dev/null; then
    error "Databricks CLI not found. Please install it:"
    echo "   pip install databricks-cli"
    exit 1
fi

# Check if authenticated
if ! databricks auth describe &> /dev/null; then
    error "Databricks CLI not authenticated. Please run:"
    echo "   databricks configure"
    exit 1
fi

# Check for required configuration
if [[ -f ".env" ]]; then
    log "Loading environment configuration from .env file..."
    set -a
    source .env
    set +a
fi

# Validate warehouse configuration
if [[ -z "$DATABRICKS_WAREHOUSE_ID" ]]; then
    warning "DATABRICKS_WAREHOUSE_ID not set. You may need to provide it during deployment."
    echo "   Find warehouse ID: Databricks -> SQL Warehouses -> Copy warehouse ID"
fi

# Production deployment confirmation
if [[ "$ENVIRONMENT" == "prod" && "$FORCE" != true ]]; then
    warning "You are about to deploy to PRODUCTION environment!"
    read -p "Are you sure you want to continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log "Deployment cancelled."
        exit 0
    fi
fi

# Validate bundle configuration
log "🔍 Validating bundle configuration..."
if [[ "$VERBOSE" == true ]]; then
    databricks bundle validate --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT"
else
    databricks bundle validate --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" > /dev/null
fi

if [[ $? -ne 0 ]]; then
    error "Bundle validation failed. Please check your configuration."
    exit 1
fi

success "Bundle validation passed!"

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    log "🧪 Dry run mode - validation complete, skipping actual deployment"
    exit 0
fi

# Deploy the bundle
log "🚀 Deploying bundle to $ENVIRONMENT environment..."
if [[ "$VERBOSE" == true ]]; then
    databricks bundle deploy --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT"
else
    databricks bundle deploy --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" > /dev/null
fi

if [[ $? -ne 0 ]]; then
    error "Deployment failed. Please check the logs above."
    exit 1
fi

# Show deployment summary
log "📋 Deployment summary:"
databricks bundle summary --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT"

success "Deployment completed successfully!"

# Next steps
cat << EOF

🎯 Next steps:
1. Run the workflow:
   databricks bundle run value_at_risk_workflow --target $ENVIRONMENT

2. Monitor execution in your Databricks workspace

3. Review results in Unity Catalog:
   ${CATALOG_NAME:-dev_value_at_risk}.${SCHEMA_NAME:-risk_management}

📊 Access your workspace at: ${DATABRICKS_HOST:-your-workspace-url}

For troubleshooting, see: README.md
EOF