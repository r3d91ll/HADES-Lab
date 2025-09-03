#!/bin/bash
# Large-Scale ArXiv Processing Test Runner
# =========================================
# Runs the complete test suite for 5000+ papers

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
ARXIV_TOOLS="$PROJECT_ROOT/tools/arxiv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Large-Scale ArXiv Processing Test Suite${NC}"
echo -e "${GREEN}========================================${NC}"

# Check environment
echo -e "\n${YELLOW}Checking environment...${NC}"

# Check ARANGO_PASSWORD
if [ -z "$ARANGO_PASSWORD" ]; then
    echo -e "${RED}Error: ARANGO_PASSWORD environment variable not set${NC}"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU acceleration may not work.${NC}"
else
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
fi

# Check Python environment
echo -e "\n${YELLOW}Python environment:${NC}"
python --version
echo "Working directory: $PROJECT_ROOT"

# Step 1: Collect papers from ArXiv
echo -e "\n${GREEN}Step 1: Collecting papers from ArXiv API${NC}"
echo "This will search for papers on AI, RAG, LLMs, and Actor Network Theory"

cd "$ARXIV_TOOLS/scripts"

if [ ! -f "data/arxiv_collections/arxiv_ids_*.txt" ]; then
    echo "Running paper collection script..."
    python collect_ai_papers.py
    
    # Check if we got enough papers
    PAPER_COUNT=$(wc -l < data/arxiv_collections/arxiv_ids_*.txt | head -1)
    echo -e "${GREEN}Collected $PAPER_COUNT unique papers${NC}"
    
    if [ "$PAPER_COUNT" -lt 5000 ]; then
        echo -e "${YELLOW}Warning: Only collected $PAPER_COUNT papers (target: 5000+)${NC}"
        echo "Consider adjusting search queries or date ranges"
    fi
else
    echo "Using existing paper collection"
    PAPER_FILE=$(ls -t data/arxiv_collections/arxiv_ids_*.txt | head -1)
    PAPER_COUNT=$(wc -l < "$PAPER_FILE")
    echo -e "${GREEN}Found $PAPER_COUNT papers in $PAPER_FILE${NC}"
fi

# Get the latest paper list file
PAPER_LIST=$(ls -t data/arxiv_collections/arxiv_ids_*.txt 2>/dev/null | head -1)

if [ -z "$PAPER_LIST" ]; then
    echo -e "${RED}Error: No paper list found${NC}"
    exit 1
fi

# Step 2: Setup test environment
echo -e "\n${GREEN}Step 2: Setting up test environment${NC}"

# Create staging directory if needed
STAGING_DIR="/dev/shm/large_scale_staging"
if [ ! -d "$STAGING_DIR" ]; then
    echo "Creating staging directory: $STAGING_DIR"
    mkdir -p "$STAGING_DIR"
fi

# Clear any previous test data
echo "Clearing previous staging data..."
rm -rf "$STAGING_DIR"/*

# Create log directory
LOG_DIR="$ARXIV_TOOLS/logs"
mkdir -p "$LOG_DIR"

# Step 3: Run database setup check
echo -e "\n${GREEN}Step 3: Checking database connection${NC}"

cd "$ARXIV_TOOLS/utils"
python check_db_status.py --detailed

# Step 4: Run small test batch first
echo -e "\n${GREEN}Step 4: Running small test batch (100 papers)${NC}"
echo "This validates the pipeline before full-scale processing"

cd "$ARXIV_TOOLS/tests"

# Run with limited papers first
python test_large_scale_processing.py \
    --config ../configs/large_scale_test.yaml \
    --papers "$PAPER_LIST" \
    --limit 100

# Check if small batch succeeded
if [ $? -ne 0 ]; then
    echo -e "${RED}Small batch test failed. Aborting full test.${NC}"
    exit 1
fi

echo -e "${GREEN}Small batch test successful!${NC}"

# Step 5: Ask for confirmation before full run
echo -e "\n${YELLOW}Ready to process all $PAPER_COUNT papers${NC}"
echo "This will take several hours depending on your hardware."
echo -n "Continue with full processing? (y/n): "
read -r response

if [[ ! "$response" =~ ^[Yy]$ ]]; then
    echo "Test aborted by user"
    exit 0
fi

# Step 6: Run full-scale test
echo -e "\n${GREEN}Step 6: Running full-scale processing${NC}"
echo "Starting at: $(date)"

# Run the full test suite
python test_large_scale_processing.py \
    --config ../configs/large_scale_test.yaml \
    --papers "$PAPER_LIST" \
    --arango-password "$ARANGO_PASSWORD" 2>&1 | tee "$LOG_DIR/large_scale_test_$(date +%Y%m%d_%H%M%S).log"

TEST_EXIT_CODE=$?

# Step 7: Generate report
echo -e "\n${GREEN}Step 7: Generating test report${NC}"

# Find the latest test results file
RESULTS_FILE=$(ls -t test_results_*.json 2>/dev/null | head -1)

if [ -n "$RESULTS_FILE" ]; then
    echo "Test results saved to: $RESULTS_FILE"
    
    # Extract key metrics
    echo -e "\n${GREEN}Key Metrics:${NC}"
    python -c "
import json
with open('$RESULTS_FILE') as f:
    results = json.load(f)
    if 'tests' in results:
        for test_name, test_data in results['tests'].items():
            print(f'\n{test_name}:')
            if isinstance(test_data, dict):
                for key, value in test_data.items():
                    if isinstance(value, (int, float)):
                        print(f'  {key}: {value:.2f}' if isinstance(value, float) else f'  {key}: {value}')
    "
fi

# Step 8: Cleanup
echo -e "\n${GREEN}Step 8: Cleanup${NC}"

# Clear staging directory
echo "Clearing staging directory..."
rm -rf "$STAGING_DIR"/*

# Step 9: Final summary
echo -e "\n${GREEN}========================================${NC}"
echo -e "${GREEN}Test Suite Complete${NC}"
echo -e "${GREEN}========================================${NC}"
echo "Ended at: $(date)"

if [ $TEST_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ All tests passed successfully${NC}"
else
    echo -e "${RED}❌ Some tests failed. Check logs for details.${NC}"
fi

echo -e "\nLogs available in: $LOG_DIR"
echo "Test results: $RESULTS_FILE"

exit $TEST_EXIT_CODE