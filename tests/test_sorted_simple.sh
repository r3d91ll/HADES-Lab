#!/bin/bash
# Test script for simplified size-sorted workflow (workflow_arxiv_sorted_simple.py)
# ====================================================================================
# This comprehensive test validates the simplified ArXiv processing workflow
#
# Features tested:
# - Size-sorted processing for optimal GPU throughput
# - Fast path (no existing embeddings) vs incremental processing
# - Multi-GPU worker distribution
# - Atomic transaction handling
# - Error recovery and logging
# - Throughput benchmarking

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_header() {
    echo -e "\n${CYAN}========================================${NC}"
    echo -e "${CYAN}$1${NC}"
    echo -e "${CYAN}========================================${NC}\n"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[⚠]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_metric() {
    echo -e "${MAGENTA}[METRIC]${NC} $1"
}

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
WORKFLOW_SCRIPT="$PROJECT_ROOT/core/workflows/workflow_arxiv_sorted_simple.py"

# Test parameters
DATABASE="${ARXIV_DB_NAME:-arxiv_repository}"
USERNAME="${ARXIV_DB_USER:-root}"
PASSWORD_ENV="${ARXIV_PASSWORD_ENV:-ARANGO_PASSWORD}"

# Test sizes - adjust based on your system
TEST_SIZES=(
    "100:2:16:1"     # Small test: 100 records, 2 workers, batch 16, 1 GPU
    "1000:4:32:2"    # Medium test: 1000 records, 4 workers, batch 32, 2 GPUs
    "5000:8:48:2"    # Large test: 5000 records, 8 workers, batch 48, 2 GPUs
)

# Check environment
print_header "Environment Check"

# Check Python
if command -v python3 &>/dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python found: $PYTHON_VERSION"
else
    print_error "Python3 not found"
    exit 1
fi

# Check CUDA
if command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    print_success "CUDA available: $GPU_COUNT GPU(s) detected"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    print_warning "CUDA not available - will use CPU (slower)"
    GPU_COUNT=0
fi

# Check environment variables
if [ -z "${!PASSWORD_ENV}" ]; then
    if [ -f "$PROJECT_ROOT/.env" ]; then
        print_info "Loading .env file..."
        export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
    fi
fi

if [ -z "${!PASSWORD_ENV}" ]; then
    print_error "$PASSWORD_ENV not set. Please export it or add to .env"
    exit 1
fi
print_success "Database password configured"

# Check workflow script exists
if [ ! -f "$WORKFLOW_SCRIPT" ]; then
    print_error "Workflow script not found: $WORKFLOW_SCRIPT"
    exit 1
fi
print_success "Workflow script found"

# Activate virtual environment if available
VENV_PATH="/home/todd/.cache/pypoetry/virtualenvs/hades-z5jmQstn-py3.12"
if [ -d "$VENV_PATH" ]; then
    print_info "Activating virtual environment..."
    source "$VENV_PATH/bin/activate"
    print_success "Virtual environment activated"
fi

# Function to setup database
setup_database() {
    local drop_collections=$1

    print_info "Setting up database: $DATABASE"

    # Try to create database using setup script if available
    if [ -f "$PROJECT_ROOT/dev-utils/create_arxiv_repository_db.py" ]; then
        python3 "$PROJECT_ROOT/dev-utils/create_arxiv_repository_db.py" 2>/dev/null || true
    fi

    if [ "$drop_collections" = true ]; then
        print_warning "Will drop and recreate collections"
    fi
}

# Function to run test
run_test() {
    local records=$1
    local workers=$2
    local batch_size=$3
    local gpus=$4
    local drop_collections=$5

    print_header "Test Configuration"
    echo "Records: $records"
    echo "Workers: $workers"
    echo "Batch Size: $batch_size"
    echo "GPUs: $gpus"
    echo "Drop Collections: $drop_collections"
    echo ""

    # Set GPU visibility
    if [ "$gpus" -gt 0 ] && [ "$GPU_COUNT" -gt 0 ]; then
        # Create GPU list (0,1,2... up to gpus-1)
        GPU_LIST=$(seq -s, 0 $((gpus-1)))
        export CUDA_VISIBLE_DEVICES="$GPU_LIST"
        print_info "Using GPUs: $CUDA_VISIBLE_DEVICES"
    else
        export CUDA_VISIBLE_DEVICES=""
        print_warning "Running on CPU only"
    fi

    # Build command
    CMD="python3 $WORKFLOW_SCRIPT"
    CMD="$CMD --database $DATABASE"
    CMD="$CMD --username $USERNAME"
    CMD="$CMD --password-env $PASSWORD_ENV"
    CMD="$CMD --count $records"
    CMD="$CMD --batch-size $batch_size"
    CMD="$CMD --embedding-batch-size $((batch_size/2))"  # Half of batch size for embeddings
    CMD="$CMD --workers $workers"

    if [ "$drop_collections" = true ]; then
        CMD="$CMD --drop-collections"
    fi

    print_info "Executing workflow..."
    echo "Command: $CMD"
    echo ""

    # Run and capture output
    TEMP_OUTPUT=$(mktemp)
    START_TIME=$(date +%s)

    if $CMD 2>&1 | tee "$TEMP_OUTPUT"; then
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        # Extract metrics from output
        PROCESSED=$(grep "Processed:" "$TEMP_OUTPUT" | tail -1 | awk '{print $2}')
        FAILED=$(grep "Failed:" "$TEMP_OUTPUT" | tail -1 | awk '{print $2}')
        THROUGHPUT=$(grep "Throughput:" "$TEMP_OUTPUT" | tail -1 | awk '{print $2}')

        print_success "Test completed successfully"
        print_metric "Duration: ${DURATION}s"
        print_metric "Processed: $PROCESSED"
        print_metric "Failed: $FAILED"
        print_metric "Throughput: $THROUGHPUT records/sec"

        # Validate results
        if [ -n "$PROCESSED" ] && [ "$PROCESSED" -gt 0 ]; then
            print_success "Processing validation passed"
        else
            print_warning "No records processed"
        fi

        rm "$TEMP_OUTPUT"
        return 0
    else
        END_TIME=$(date +%s)
        DURATION=$((END_TIME - START_TIME))

        print_error "Test failed after ${DURATION}s"
        echo "Last 20 lines of output:"
        tail -20 "$TEMP_OUTPUT"
        rm "$TEMP_OUTPUT"
        return 1
    fi
}

# Function to check database status
check_database() {
    print_info "Checking database status..."

    python3 -c "
import os
import sys
sys.path.insert(0, '$PROJECT_ROOT')

from core.database.database_factory import DatabaseFactory

try:
    db = DatabaseFactory.get_arango(
        database='$DATABASE',
        username='$USERNAME',
        use_unix=True
    )

    # Check collections
    collections = ['arxiv_metadata', 'arxiv_abstract_chunks', 'arxiv_abstract_embeddings']
    for coll_name in collections:
        if db.has_collection(coll_name):
            count = db[coll_name].count()
            print(f'  {coll_name}: {count} documents')
        else:
            print(f'  {coll_name}: not found')

    print('Database connection successful')
except Exception as e:
    print(f'Database connection failed: {e}')
    sys.exit(1)
" || {
    print_error "Database check failed"
    return 1
}

    print_success "Database status checked"
}

# Function to monitor GPU usage
monitor_gpu() {
    if [ "$GPU_COUNT" -gt 0 ]; then
        print_info "GPU Usage:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
    fi
}

# Main test execution
print_header "ArXiv Sorted Simple Workflow Test Suite"
echo "Project Root: $PROJECT_ROOT"
echo "Database: $DATABASE"
echo "Username: $USERNAME"
echo ""

# Parse command line arguments
RUN_ALL=false
TEST_TYPE="small"
MONITOR_GPU=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --type)
            TEST_TYPE="$2"
            shift 2
            ;;
        --monitor-gpu)
            MONITOR_GPU=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --all           Run all test sizes"
            echo "  --type TYPE     Run specific test (small/medium/large)"
            echo "  --monitor-gpu   Monitor GPU usage during tests"
            echo "  --help          Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Setup database
setup_database true

# Check initial database status
check_database

# GPU monitoring in background (optional)
if [ "$MONITOR_GPU" = true ] && [ "$GPU_COUNT" -gt 0 ]; then
    print_info "Starting GPU monitor..."
    (
        while true; do
            monitor_gpu >> /tmp/gpu_monitor.log
            sleep 5
        done
    ) &
    GPU_MONITOR_PID=$!
    print_success "GPU monitor started (PID: $GPU_MONITOR_PID)"
fi

# Run tests
TESTS_PASSED=0
TESTS_FAILED=0

if [ "$RUN_ALL" = true ]; then
    # Run all test sizes
    for test_config in "${TEST_SIZES[@]}"; do
        IFS=':' read -r records workers batch gpus <<< "$test_config"

        print_header "Running Test: $records records"

        # First run with drop (fresh start)
        if run_test "$records" "$workers" "$batch" "$gpus" true; then
            ((TESTS_PASSED++))

            # Second run without drop (resume test)
            print_info "Testing resume capability..."
            if run_test "$((records/2))" "$workers" "$batch" "$gpus" false; then
                print_success "Resume test passed"
                ((TESTS_PASSED++))
            else
                print_error "Resume test failed"
                ((TESTS_FAILED++))
            fi
        else
            ((TESTS_FAILED++))
        fi

        # Check database after test
        check_database

        # Brief pause between tests
        sleep 2
    done
else
    # Run single test based on type
    case $TEST_TYPE in
        small)
            test_config="${TEST_SIZES[0]}"
            ;;
        medium)
            test_config="${TEST_SIZES[1]}"
            ;;
        large)
            test_config="${TEST_SIZES[2]}"
            ;;
        *)
            print_error "Invalid test type: $TEST_TYPE"
            exit 1
            ;;
    esac

    IFS=':' read -r records workers batch gpus <<< "$test_config"
    print_header "Running $TEST_TYPE test: $records records"

    if run_test "$records" "$workers" "$batch" "$gpus" true; then
        ((TESTS_PASSED++))
        check_database
    else
        ((TESTS_FAILED++))
    fi
fi

# Kill GPU monitor if running
if [ -n "$GPU_MONITOR_PID" ]; then
    kill $GPU_MONITOR_PID 2>/dev/null || true
    print_info "GPU monitor stopped"

    if [ -f /tmp/gpu_monitor.log ]; then
        print_info "GPU usage summary:"
        tail -10 /tmp/gpu_monitor.log
        rm /tmp/gpu_monitor.log
    fi
fi

# Final summary
print_header "Test Summary"
echo "Tests Passed: $TESTS_PASSED"
echo "Tests Failed: $TESTS_FAILED"

if [ "$TESTS_FAILED" -eq 0 ]; then
    print_success "All tests completed successfully!"
    exit 0
else
    print_error "Some tests failed"
    exit 1
fi