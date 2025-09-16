#!/bin/bash
# Test script for size-sorted workflow

echo "Testing ArXiv Size-Sorted Workflow"
echo "==================================="

# Set environment
export ARANGO_PASSWORD="${ARANGO_PASSWORD}"
export CUDA_VISIBLE_DEVICES="0,1"

# Create the new database
echo "Setting up arxiv_repository database..."
python dev-utils/create_arxiv_repository_db.py

echo ""
echo "Running sorted workflow with 10,000 records..."
echo ""

# Run with small test set first
python -m core.workflows.workflow_arxiv_sorted \
    --database arxiv_repository \
    --count 10000 \
    --batch-size 100 \
    --embedding-batch-size 32 \
    --workers 2 \
    --drop-collections

echo ""
echo "Test complete!"
echo ""
echo "To monitor progress in another terminal:"
echo "python dev-utils/simple_monitor.py --interval 30 --collection arxiv_abstract_embeddings --database arxiv_repository"