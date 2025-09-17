#!/bin/bash
# Production test script for size-sorted workflow with proper database practices

set -e  # Exit on error

echo "=========================================="
echo "ArXiv Size-Sorted Workflow Production Test"
echo "=========================================="
echo ""

# Check if root password is set
if [ -z "$ARANGO_PASSWORD" ]; then
    echo "ERROR: ARANGO_PASSWORD not set"
    echo "Please set: export ARANGO_PASSWORD='your-root-password'"
    exit 1
fi

# Step 1: Set up database with proper users
echo "Step 1: Setting up database with proper security..."
echo "----------------------------------------------"
python dev-utils/setup_arxiv_database.py --drop-existing

echo ""
echo "Step 2: Loading credentials..."
echo "----------------------------------------------"

# Load the generated credentials
if [ -f "core/config/arxiv_repository.env" ]; then
    source core/config/arxiv_repository.env
    echo "✅ Loaded credentials from core/config/arxiv_repository.env"
else
    echo "❌ ERROR: core/config/arxiv_repository.env not found"
    echo "Please run: python dev-utils/setup_arxiv_database.py"
    exit 1
fi

# Verify writer password is loaded
if [ -z "$ARXIV_WRITER_PASSWORD" ]; then
    echo "❌ ERROR: ARXIV_WRITER_PASSWORD not loaded"
    exit 1
fi

echo ""
echo "Step 3: Running sorted workflow test..."
echo "----------------------------------------------"
echo "Database: arxiv_repository"
echo "Username: arxiv_writer"
echo "Test size: 10,000 records"
echo ""

# Set GPU configuration
export CUDA_VISIBLE_DEVICES="0,1"

# Run the sorted workflow with proper credentials
python -m core.workflows.workflow_arxiv_sorted \
    --database arxiv_repository \
    --username arxiv_writer \
    --password-env ARXIV_WRITER_PASSWORD \
    --count 10000 \
    --batch-size 100 \
    --embedding-batch-size 32 \
    --workers 2 \
    --drop-collections

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "To monitor progress (in another terminal):"
echo "  source core/config/arxiv_repository.env"
echo "  python dev-utils/simple_monitor.py \\"
echo "    --database arxiv_repository \\"
echo "    --collection arxiv_abstract_embeddings \\"
echo "    --username arxiv_reader \\"
echo "    --password-env ARXIV_READER_PASSWORD"
echo ""
echo "For full production run:"
echo "  source core/config/arxiv_repository.env"
echo "  python -m core.workflows.workflow_arxiv_sorted \\"
echo "    --database arxiv_repository \\"
echo "    --username arxiv_writer \\"
echo "    --batch-size 1000 \\"
echo "    --embedding-batch-size 128 \\"
echo "    --workers 2"