#!/bin/bash
# Run the complete graph build with GPU optimization

cd /home/todd/olympus/HADES-Lab/core/graph/builders

echo "========================================================================"
echo "COMPLETE GRAPH BUILD (Fresh Start)"
echo "========================================================================"
echo "This will build:"
echo "  1. same_field edges (~4M edges, ~2 minutes)"
echo "  2. temporal_proximity edges (~52M edges, ~15 minutes)"
echo "  3. keyword_similarity edges (GPU accelerated, ~10 minutes)"
echo ""
echo "Using: 48 CPU workers + GPU acceleration"
echo ""
echo "Monitor progress in another terminal with:"
echo "  tail -f /home/todd/olympus/HADES-Lab/logs/complete_build.log"
echo "========================================================================"
echo ""

# Run complete build with 48 workers and log output
python build_complete_graph.py --workers 48 2>&1 | tee /home/todd/olympus/HADES-Lab/logs/complete_build.log