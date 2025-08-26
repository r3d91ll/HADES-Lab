#!/bin/bash
# Test MCP environment readiness

echo "========================================="
echo "Testing MCP Environment Setup"
echo "========================================="

echo -e "\n1. Testing uv installation:"
if command -v uv &> /dev/null; then
    echo "✓ uv is installed: $(uv --version)"
else
    echo "✗ uv not found"
    exit 1
fi

echo -e "\n2. Testing Poetry installation:"
if command -v poetry &> /dev/null; then
    echo "✓ Poetry is installed: $(poetry --version)"
else
    echo "✗ Poetry not found"
    exit 1
fi

echo -e "\n3. Testing Python with uv:"
echo "Running: uv run python --version"
uv run python --version
if [ $? -eq 0 ]; then
    echo "✓ uv can run Python"
else
    echo "✗ uv cannot run Python"
fi

echo -e "\n4. Testing package imports with uv:"
echo "Testing key imports..."
uv run python -c "
import sys
print(f'Python: {sys.version}')
try:
    import torch
    print('✓ PyTorch imported')
except ImportError:
    print('✗ PyTorch not available')
    
try:
    import arango
    print('✓ ArangoDB client imported')
except ImportError:
    print('✗ ArangoDB client not available')
    
try:
    import pydantic
    print('✓ Pydantic imported')
except ImportError:
    print('✗ Pydantic not available')
"

echo -e "\n========================================="
echo "MCP Environment Test Complete"
echo "========================================="
echo ""
echo "To use MCP servers:"
echo "  uv run python your_mcp_server.py"
echo ""
echo "To enter Poetry shell:"
echo "  poetry shell"
echo ""
echo "To run ACID pipeline:"
echo "  poetry run python arxiv/pipelines/arxiv_pipeline.py --config arxiv/configs/acid_pipeline_phased.yaml"