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

cd "$ARXIV_TOOLS/utils"

# Use compgen to safely check for matching files
# Use array assignment with proper quoting
mapfile -t PAPER_LIST_FILES < <(find "../../../data/arxiv_collections" -maxdepth 1 -name "arxiv_ids_*.txt" -type f 2>/dev/null)
if [ ${#PAPER_LIST_FILES[@]} -eq 0 ]; then
    echo "No existing paper lists found. You need to create a paper list first."
    echo -e "${YELLOW}Note: Use the lifecycle manager to process papers:${NC}"
    echo "python lifecycle.py batch <paper_list.txt>"
    exit 1
else
    # Use the most recent file by modification time
    PAPER_LIST=$(ls -t "${PAPER_LIST_FILES[@]}" | head -1)
    PAPER_COUNT=$(wc -l < "$PAPER_LIST")
    echo -e "${GREEN}Found $PAPER_COUNT papers in $PAPER_LIST${NC}"
    
    if [ "$PAPER_COUNT" -lt 100 ]; then
        echo -e "${YELLOW}Warning: Only found $PAPER_COUNT papers (recommended: 100+)${NC}"
        echo "Consider creating a larger paper list for comprehensive testing"
    fi
fi

# Validate paper list was properly set
if [ -z "$PAPER_LIST" ] || [ ! -f "$PAPER_LIST" ]; then
    echo -e "${RED}Error: No valid paper list found${NC}"
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
# Clear any previous test data
if [ -z "$STAGING_DIR" ] || [ "$STAGING_DIR" = "/" ]; then
    echo -e "${RED}Error: Invalid staging directory${NC}"
    exit 1
fi
echo "Clearing previous staging data..."
find "$STAGING_DIR" -mindepth 1 -delete 2>/dev/null || true

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
SMALL_BATCH_EXIT_CODE=$?

# Check if small batch succeeded
if [ $SMALL_BATCH_EXIT_CODE -ne 0 ]; then
    echo -e "${RED}Small batch test failed. Aborting full test.${NC}"
    exit 1
fi

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

    python3 - "$RESULTS_FILE" <<'EOF'
import json
import sys
with open(sys.argv[1]) as f:
    results = json.load(f)
    if 'tests' in results:
        for test_name, test_data in results['tests'].items():
            print(f'\n{test_name}:')
            if isinstance(test_data, dict):
                for key, value in test_data.items():
                    if isinstance(value, (int, float)):
                        print(f'  {key}: {value:.2f}' if isinstance(value, float) else f'  {key}: {value}')
EOF
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

    # Compute and display HADES Conveyance Scorecard
    echo -e "\n${GREEN}HADES Conveyance Scorecard:${NC}"
    python3 - "$RESULTS_FILE" <<'EOF'
import json
import sys
import math

try:
    with open(sys.argv[1]) as f:
        results = json.load(f)

    # Extract or default variables for Conveyance Framework
    # W = What (signal/content quality) - map from success rates
    W = 0.0
    if 'tests' in results:
        success_count = 0
        total_count = 0
        for test_name, test_data in results['tests'].items():
            if isinstance(test_data, dict):
                if 'success_rate' in test_data:
                    W += test_data['success_rate']
                    success_count += 1
                elif 'papers_processed' in test_data and 'successful' in test_data:
                    if test_data['papers_processed'] > 0:
                        W += (test_data['successful'] / test_data['papers_processed'])
                        success_count += 1
        if success_count > 0:
            W = W / success_count / 100.0 if W > 1 else W  # Normalize to 0-1
    W = max(0.01, min(1.0, W)) if W > 0 else 0.5  # Default to 0.5 if no data

    # R = Where (relational positioning) - map from database integrity
    R = 1.0  # Default to perfect if no issues
    if 'tests' in results and 'database_integrity' in results['tests']:
        integrity = results['tests']['database_integrity']
        if isinstance(integrity, dict):
            # Count any integrity issues
            issue_count = 0
            if 'orphaned_chunks' in integrity:
                issue_count += integrity['orphaned_chunks']
            if 'missing_embeddings' in integrity:
                issue_count += integrity['missing_embeddings']
            if 'inconsistent_status' in integrity:
                issue_count += integrity['inconsistent_status']
            # Map to 0-1 score (1 = no issues, 0 = many issues)
            R = 1.0 / (1.0 + issue_count * 0.1) if issue_count > 0 else 1.0

    # H = Who (agent capability) - map from error recovery success
    H = 0.5  # Default capability
    if 'tests' in results and 'error_recovery' in results['tests']:
        recovery = results['tests']['error_recovery']
        if isinstance(recovery, dict):
            if 'successful_recoveries' in recovery and 'papers_tested' in recovery:
                if recovery['papers_tested'] > 0:
                    H = recovery['successful_recoveries'] / recovery['papers_tested']
    H = max(0.1, min(1.0, H))

    # T = Time (latency/cost) - from performance benchmarks
    T = 1.0  # Default to 1 second if no data
    if 'tests' in results and 'performance_benchmarks' in results['tests']:
        perf = results['tests']['performance_benchmarks']
        if isinstance(perf, dict):
            if 'mean_total_time' in perf:
                T = perf['mean_total_time']
            elif 'overall_rate' in perf and perf['overall_rate'] > 0:
                T = 1.0 / perf['overall_rate']  # Convert rate to time
    T = max(0.01, T)  # Ensure T > 0 to avoid division by zero

    # Context components (default weights of 0.25 each)
    L = results.get('context', {}).get('local_coherence', 0.8)  # Default 0.8
    I = results.get('context', {}).get('instruction_fit', 0.7)   # Default 0.7
    A = results.get('context', {}).get('actionability', 0.6)    # Default 0.6
    G = results.get('context', {}).get('grounding', 0.9)         # Default 0.9

    # Context weights (default to equal weighting)
    wL = results.get('context_weights', {}).get('wL', 0.25)
    wI = results.get('context_weights', {}).get('wI', 0.25)
    wA = results.get('context_weights', {}).get('wA', 0.25)
    wG = results.get('context_weights', {}).get('wG', 0.25)

    # Compute combined context
    Ctx = wL * L + wI * I + wA * A + wG * G
    Ctx = max(0.01, min(1.0, Ctx))  # Ensure 0 < Ctx <= 1

    # Alpha exponent (default to 1.75, middle of recommended range)
    alpha = results.get('alpha', 1.75)
    alpha = max(1.5, min(2.0, alpha))  # Constrain to [1.5, 2.0]

    # Compute Conveyance using efficiency view
    # C = (W * R * H / T) * Ctx^α
    if W > 0 and R > 0 and H > 0 and T > 0:
        C = (W * R * H / T) * math.pow(Ctx, alpha)
    else:
        C = 0.0  # Zero-propagation gate

    # Print scorecard
    print(f"  Component Values:")
    print(f"    W (What/Quality):     {W:.3f}")
    print(f"    R (Where/Topology):   {R:.3f}")
    print(f"    H (Who/Capability):   {H:.3f}")
    print(f"    T (Time/Latency):     {T:.3f}s")
    print(f"  Context Factors:")
    print(f"    L (Local Coherence):  {L:.3f} (weight: {wL:.2f})")
    print(f"    I (Instruction Fit):  {I:.3f} (weight: {wI:.2f})")
    print(f"    A (Actionability):    {A:.3f} (weight: {wA:.2f})")
    print(f"    G (Grounding):        {G:.3f} (weight: {wG:.2f})")
    print(f"    Ctx (Combined):       {Ctx:.3f}")
    print(f"    α (Exponent):         {alpha:.2f}")
    print(f"  ─────────────────────────────")
    print(f"  Conveyance Score (C): {C:.4f}")
    print(f"  Formula: C = (W·R·H/T)·Ctx^α")

except Exception as e:
    print(f"  Error computing Conveyance Score: {e}")
    print(f"  Using default values for demonstration")
EOF
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