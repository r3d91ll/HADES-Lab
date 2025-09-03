#!/bin/bash
#
# Weekend Test Runner - 15,000 Paper Processing
# ==============================================
#
# This script sets up and runs the weekend test with 15,000 papers.
# Designed to run unattended over a holiday weekend.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Weekend Test Runner - 15,000 Papers  ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check environment
if [ -z "$ARANGO_PASSWORD" ]; then
    echo -e "${RED}ERROR: ARANGO_PASSWORD not set${NC}"
    echo "Please set: export ARANGO_PASSWORD='your-password'"
    exit 1
fi

# Set paths
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/../../../" && pwd )"
ARXIV_DIR="$PROJECT_ROOT/tools/arxiv"
PIPELINE_DIR="$ARXIV_DIR/pipelines"
CONFIG_FILE="$ARXIV_DIR/configs/weekend_test_15k.yaml"
LOG_DIR="$ARXIV_DIR/logs"

# Create log directory if needed
mkdir -p "$LOG_DIR"

echo "📁 Project root: $PROJECT_ROOT"
echo "📁 ArXiv directory: $ARXIV_DIR"
echo "📋 Config file: $CONFIG_FILE"
echo ""

# Step 1: Collect papers if needed
echo -e "${YELLOW}Step 1: Paper Collection${NC}"
echo "Checking for existing paper collection..."

COLLECTION_DIR="$ARXIV_DIR/scripts/data/arxiv_collections"
if [ -d "$COLLECTION_DIR" ]; then
    PAPER_COUNT=$(find "$COLLECTION_DIR" -name "arxiv_ids_*.txt" -exec wc -l {} \; | awk '{sum+=$1} END {print sum}')
    echo "Found $PAPER_COUNT papers in existing collections"
    
    if [ "$PAPER_COUNT" -lt 20000 ]; then
        echo "Collection has less than 20,000 papers. Running extended collector..."
        cd "$ARXIV_DIR/scripts"
        python collect_ai_papers_extended.py
    else
        echo -e "${GREEN}✅ Sufficient papers available ($PAPER_COUNT)${NC}"
    fi
else
    echo "No collection found. Running initial collector..."
    cd "$ARXIV_DIR/scripts"
    python collect_ai_papers.py
    echo "Running extended collector to reach 20,000..."
    python collect_ai_papers_extended.py
fi

# Find the most recent paper list
PAPER_LIST=$(ls -t "$COLLECTION_DIR"/arxiv_ids*.txt 2>/dev/null | head -1)
if [ -z "$PAPER_LIST" ]; then
    echo -e "${RED}ERROR: No paper list found${NC}"
    exit 1
fi

TOTAL_PAPERS=$(wc -l < "$PAPER_LIST")
echo ""
echo -e "${GREEN}📄 Using paper list: $PAPER_LIST${NC}"
echo -e "${GREEN}📊 Total papers available: $TOTAL_PAPERS${NC}"
echo ""

# Step 2: Clean up staging area
echo -e "${YELLOW}Step 2: Cleanup${NC}"
echo "Cleaning staging directory..."
STAGING_DIR="/dev/shm/weekend_staging"
if [ -d "$STAGING_DIR" ]; then
    rm -rf "$STAGING_DIR"/*
    echo "✅ Staging directory cleaned"
else
    mkdir -p "$STAGING_DIR"
    echo "✅ Staging directory created"
fi

# Clear GPU memory
echo "Clearing GPU memory..."
python3 -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true
echo "✅ GPU memory cleared"
echo ""

# Step 3: Verify environment
echo -e "${YELLOW}Step 3: Environment Check${NC}"

# Check GPUs
echo -n "GPUs available: "
nvidia-smi --query-gpu=name --format=csv,noheader | wc -l

# Check memory
echo -n "System memory: "
free -h | grep "^Mem:" | awk '{print $2}'

# Check disk space
echo -n "Bulk store space: "
df -h /bulk-store | tail -1 | awk '{print $4 " available"}'

# Check ArangoDB
echo -n "ArangoDB status: "
if curl -s "http://192.168.1.69:8529/_api/version" > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Connected${NC}"
else
    echo -e "${RED}❌ Not reachable${NC}"
    echo "Please ensure ArangoDB is running"
    exit 1
fi
echo ""

# Step 4: Start pipeline
echo -e "${YELLOW}Step 4: Starting Pipeline${NC}"
echo "Configuration: weekend_test_15k.yaml"
echo "Target: 15,000 papers"
echo "Estimated runtime: ~22 hours @ 11.3 papers/minute"
echo ""

# Create startup timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="$LOG_DIR/weekend_test_${TIMESTAMP}.log"

echo "Log file: $LOG_FILE"
echo ""

# Ask for confirmation
echo -e "${YELLOW}Ready to start processing 15,000 papers.${NC}"
echo "This will run for approximately 22 hours."
echo ""
read -p "Press ENTER to start, or Ctrl+C to cancel... "

# Start the pipeline
cd "$PIPELINE_DIR"
echo ""
echo -e "${GREEN}🚀 Starting pipeline...${NC}"
echo "You can monitor progress with:"
echo "  1. tail -f $LOG_FILE"
echo "  2. cd $ARXIV_DIR/monitoring && python weekend_monitor.py"
echo ""

# Run pipeline with nohup for background execution
nohup python arxiv_pipeline.py \
    --config "$CONFIG_FILE" \
    --count 15000 \
    --paper-list "$PAPER_LIST" \
    --arango-password "$ARANGO_PASSWORD" \
    > "$LOG_FILE" 2>&1 &

PIPELINE_PID=$!
echo "Pipeline started with PID: $PIPELINE_PID"
echo ""

# Save PID for later reference
echo $PIPELINE_PID > "$ARXIV_DIR/weekend_test.pid"

# Wait a moment and check if it's still running
sleep 5
if ps -p $PIPELINE_PID > /dev/null; then
    echo -e "${GREEN}✅ Pipeline is running successfully!${NC}"
    echo ""
    echo "Monitor with:"
    echo "  cd $ARXIV_DIR/monitoring && python weekend_monitor.py"
    echo ""
    echo "Stop pipeline with:"
    echo "  kill $PIPELINE_PID"
    echo ""
    echo "Have a great holiday weekend! 🎉"
else
    echo -e "${RED}❌ Pipeline failed to start${NC}"
    echo "Check the log file for errors:"
    echo "  tail -100 $LOG_FILE"
    exit 1
fi