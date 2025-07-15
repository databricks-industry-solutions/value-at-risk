#!/bin/bash

# Value at Risk - Cleanup Script
# This script removes all deployed resources from the specified environment

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
Value at Risk - Cleanup Script

USAGE:
    $0 [ENVIRONMENT] [OPTIONS]

ARGUMENTS:
    ENVIRONMENT    Target environment (dev, staging, prod). Default: dev

OPTIONS:
    -h, --help     Show this help message
    -v, --verbose  Enable verbose output
    --force        Skip confirmation prompts
    --dry-run      Show what would be destroyed without actually doing it

EXAMPLES:
    $0                    # Cleanup dev environment
    $0 staging           # Cleanup staging environment
    $0 prod --force      # Cleanup prod without confirmation

WARNING:
    This will permanently delete ALL resources including:
    - Databricks workflows and jobs
    - Unity Catalog tables (if configured for cleanup)
    - MLflow experiments and models
    - Any associated compute clusters

For more information, see: README.md
EOF
}

# Parse command line arguments
ENVIRONMENT="dev"
VERBOSE=false
FORCE=false
DRY_RUN=false

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
        --force)
            FORCE=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
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

log "🧹 Starting Value at Risk cleanup for $ENVIRONMENT environment..."

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

# Load environment configuration if available
if [[ -f ".env" ]]; then
    log "Loading environment configuration from .env file..."
    set -a
    source .env
    set +a
fi

# Dry run mode
if [[ "$DRY_RUN" == true ]]; then
    log "🧪 Dry run mode - showing what would be destroyed..."
    echo "The following resources would be destroyed:"
    databricks bundle summary --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" 2>/dev/null || true
    log "Dry run complete. No resources were actually destroyed."
    exit 0
fi

# Confirmation prompt
if [[ "$FORCE" != true ]]; then
    warning "⚠️  This will permanently delete ALL resources from the '$ENVIRONMENT' environment!"
    echo ""
    echo "Resources that will be destroyed:"
    databricks bundle summary --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" 2>/dev/null || echo "  (Unable to fetch resource summary)"
    echo ""
    
    if [[ "$ENVIRONMENT" == "prod" ]]; then
        warning "🚨 You are about to cleanup the PRODUCTION environment!"
        read -p "Type 'DELETE' to confirm production cleanup: " -r
        if [[ "$REPLY" != "DELETE" ]]; then
            log "Cleanup cancelled."
            exit 0
        fi
    else
        read -p "Are you sure you want to continue? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log "Cleanup cancelled."
            exit 0
        fi
    fi
fi

# Destroy the bundle
log "🗑️  Destroying bundle resources..."
if [[ "$VERBOSE" == true ]]; then
    databricks bundle destroy --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" --auto-approve
else
    databricks bundle destroy --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" --auto-approve > /dev/null
fi

if [[ $? -ne 0 ]]; then
    error "Cleanup failed. Some resources may still exist."
    exit 1
fi

success "Cleanup completed successfully!"
log "📋 All resources have been removed from the '$ENVIRONMENT' environment."

# Cleanup verification
log "🔍 Verifying cleanup..."
REMAINING_RESOURCES=$(databricks bundle summary --target "$ENVIRONMENT" --var="environment=$ENVIRONMENT" 2>/dev/null || echo "")
if [[ -n "$REMAINING_RESOURCES" ]]; then
    warning "Some resources may still exist. Please check your workspace manually."
else
    success "All resources have been successfully removed."
fi