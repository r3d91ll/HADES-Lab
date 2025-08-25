#!/bin/bash

# Live Pipeline Monitor Launcher
# =============================
# 
# Launches the live monitoring dashboard with environment variables.
# Usage: ./monitor_live.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔍 Pipeline Live Monitor${NC}"
echo -e "${BLUE}========================${NC}"

# Check if we're in the right directory
if [ ! -f "monitor_pipeline_live.py" ]; then
    echo -e "${RED}❌ Error: monitor_pipeline_live.py not found${NC}"
    echo "Please run this script from the tools/arxiv directory"
    exit 1
fi

# Check for required environment variables
if [ -z "$PGPASSWORD" ]; then
    echo -e "${RED}❌ Error: PGPASSWORD environment variable not set${NC}"
    echo "Please set: export PGPASSWORD='your_password'"
    exit 1
fi

if [ -z "$ARANGO_PASSWORD" ]; then
    echo -e "${RED}❌ Error: ARANGO_PASSWORD environment variable not set${NC}" 
    echo "Please set: export ARANGO_PASSWORD='your_password'"
    exit 1
fi

# Check for required Python packages
echo -e "${YELLOW}📦 Checking Python dependencies...${NC}"
python3 -c "import psycopg2, arango, psutil, GPUtil" 2>/dev/null || {
    echo -e "${RED}❌ Missing Python packages${NC}"
    echo "Please install: pip3 install psycopg2-binary python-arango psutil GPUtil"
    exit 1
}

# Check database connectivity
echo -e "${YELLOW}🔌 Testing database connections...${NC}"

# Test PostgreSQL
PGPASSWORD=$PGPASSWORD psql -h localhost -U postgres -d Avernus -c "SELECT 1;" >/dev/null 2>&1 || {
    echo -e "${RED}❌ Cannot connect to PostgreSQL${NC}"
    echo "Please check that PostgreSQL is running and credentials are correct"
    exit 1
}

# Test ArangoDB
curl -s -u root:$ARANGO_PASSWORD http://192.168.1.69:8529/_api/version >/dev/null || {
    echo -e "${RED}❌ Cannot connect to ArangoDB${NC}"
    echo "Please check that ArangoDB is running at 192.168.1.69:8529"
    exit 1
}

echo -e "${GREEN}✅ All checks passed${NC}"
echo -e "${GREEN}🚀 Starting live monitoring dashboard...${NC}"
echo ""
echo -e "${YELLOW}Controls:${NC}"
echo "  - Ctrl+C to exit"
echo "  - Updates every 10 seconds"
echo "  - Resize terminal for better display"
echo ""

# Launch the monitor
python3 monitor_pipeline_live.py \
    --pg-password "$PGPASSWORD" \
    --arango-password "$ARANGO_PASSWORD"