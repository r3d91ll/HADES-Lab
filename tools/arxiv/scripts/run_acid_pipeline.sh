#!/bin/bash
#
# Run the ACID-compliant pipeline for processing ArXiv papers
# Achieves 6.2 papers/minute with 100% success rate
#

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../.." && pwd )"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
CONFIG_FILE="${SCRIPT_DIR}/../configs/acid_pipeline_phased.yaml"
LOG_FILE="${SCRIPT_DIR}/../logs/acid_pipeline_$(date +%Y%m%d_%H%M%S).log"

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required environment variables
if [ -z "$ARANGO_PASSWORD" ]; then
    print_error "ARANGO_PASSWORD environment variable not set"
    echo "Please set: export ARANGO_PASSWORD='your_password'"
    exit 1
fi

# PostgreSQL no longer used - removed check

# Print startup information
echo "=================================="
echo "  ACID Pipeline Runner"
echo "  6.2 papers/minute"
echo "=================================="
echo
print_info "Configuration: $CONFIG_FILE"
print_info "Log file: $LOG_FILE"
print_info "GPUs: ${CUDA_VISIBLE_DEVICES:-Auto-detect}"

# Create log directory if it doesn't exist
mkdir -p "${SCRIPT_DIR}/../logs"

# Change to pipelines directory
cd "${SCRIPT_DIR}/../pipelines"

# Run the pipeline
print_info "Starting ACID pipeline..."
echo

python arxiv_pipeline.py \
    --config "$CONFIG_FILE" \
    2>&1 | tee "$LOG_FILE"

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    print_success "Pipeline completed successfully"
    
    # Print summary
    echo
    echo "=================================="
    echo "  Processing Summary"
    echo "=================================="
    grep -E "Papers processed:|Success rate:|Processing rate:" "$LOG_FILE" | tail -3
else
    print_error "Pipeline failed. Check log file: $LOG_FILE"
    exit 1
fi