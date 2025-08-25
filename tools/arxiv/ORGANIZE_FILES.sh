#!/bin/bash

# Script to organize the tools/arxiv directory structure
# This will move files into logical subdirectories

echo "="*60
echo "ORGANIZING TOOLS/ARXIV DIRECTORY"
echo "="*60
echo ""

# Create timestamp for Acheron moves
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")

# Create new directory structure
echo "Creating directory structure..."
mkdir -p pipelines
mkdir -p monitoring
mkdir -p scripts
mkdir -p database
mkdir -p deprecated
mkdir -p configs
mkdir -p logs
mkdir -p checkpoints

# Move pipeline files
echo ""
echo "Moving pipeline files..."
mv arxiv_pipeline.py pipelines/ 2>/dev/null
mv arxiv_pipeline_unified_hysteresis.py pipelines/ 2>/dev/null
mv hybrid_search.py pipelines/ 2>/dev/null
echo "  ✓ Moved main pipeline scripts to pipelines/"

# Move monitoring tools
echo ""
echo "Moving monitoring tools..."
mv monitor_*.py monitoring/ 2>/dev/null
mv monitor_*.sh monitoring/ 2>/dev/null
mv MONITOR_README.md monitoring/ 2>/dev/null
mv demo_queue_monitoring.py monitoring/ 2>/dev/null
mv test_queue_monitoring.py monitoring/ 2>/dev/null
mv pipeline_status_reporter.py monitoring/ 2>/dev/null
echo "  ✓ Moved monitoring tools to monitoring/"

# Move utility scripts
echo ""
echo "Moving utility scripts..."
mv reset_databases.py scripts/ 2>/dev/null
mv conveyance_measurement.py scripts/ 2>/dev/null
mv run_*.sh scripts/ 2>/dev/null
echo "  ✓ Moved utility scripts to scripts/"

# Move test files to deprecated
echo ""
echo "Moving test/temporary files..."
mv test_status.json deprecated/ 2>/dev/null
mv test_status.tmp deprecated/ 2>/dev/null
mv pipeline_status.json deprecated/ 2>/dev/null
echo "  ✓ Moved test files to deprecated/"

# Move logs
echo ""
echo "Moving log files..."
mv *.log logs/ 2>/dev/null
echo "  ✓ Moved log files to logs/"

# Keep important files in root
echo ""
echo "Files keeping in root:"
echo "  - README.md"
echo "  - CLAUDE.md"
echo "  - __init__.py"

echo ""
echo "="*60
echo "ORGANIZATION COMPLETE"
echo "="*60
echo ""
echo "New structure:"
echo "  pipelines/    - Main processing pipelines"
echo "  monitoring/   - Monitoring and status tools"
echo "  scripts/      - Utility and helper scripts"
echo "  database/     - Database setup and management"
echo "  utils/        - Check and verification utilities"
echo "  configs/      - Configuration files"
echo "  logs/         - Log files"
echo "  checkpoints/  - Checkpoint files"
echo "  deprecated/   - Temporary and test files"
echo ""