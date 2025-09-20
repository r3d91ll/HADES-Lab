#!/bin/bash
# Setup ArXiv Repository Database with Production Best Practices
# ==============================================================
# This script sets up the arxiv_repository database with proper user management
# It integrates with the existing HADES-Lab environment setup

set -e  # Exit on error

# ==============================================================
# CONFIGURATION - Modify these values as needed
# ==============================================================

# Database Configuration
DB_NAME="${ARXIV_DB_NAME:-arxiv_repository}"

# Connection Configuration
# Prioritize Unix socket for local connections (much faster, more secure)
UNIX_SOCKET="${ARANGO_UNIX_SOCKET:-/run/arangodb3/arangodb.sock}"
USE_UNIX_SOCKET="${USE_UNIX_SOCKET:-true}"

# Fallback to HTTP if Unix socket not available
DB_HOST="${ARXIV_DB_HOST:-http://localhost:8529}"

# User Configuration
DB_ADMIN_USER="${ARXIV_ADMIN_USER:-arxiv_admin}"
DB_WRITER_USER="${ARXIV_WRITER_USER:-arxiv_writer}"
DB_READER_USER="${ARXIV_READER_USER:-arxiv_reader}"

# Password Configuration
# If not set, passwords will be auto-generated
DB_ADMIN_PASSWORD="${ARANGODB_PASSWORD:-}"
DB_WRITER_PASSWORD="${ARANGODB_PASSWORD:-}"
DB_READER_PASSWORD="${ARANGODB_PASSWORD:-}"

# Root password (required - from environment or .env)
ROOT_PASSWORD="${ARANGO_PASSWORD:-}"

# Collections to create
COLLECTIONS=(
    "arxiv_papers"
    "arxiv_chunks"
    "arxiv_embeddings"
    "arxiv_structures"
    "arxiv_processing_order"
    "arxiv_processing_stats"
)

# Roles and Permissions
# rw = read-write, ro = read-only
ADMIN_PERMISSIONS="rw"
WRITER_PERMISSIONS="rw"
READER_PERMISSIONS="ro"

# Configuration Directory
CONFIG_DIR="${CONFIG_DIR:-config}"

# ==============================================================
# END CONFIGURATION
# ==============================================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Print header
echo "============================================"
echo "  ArXiv Repository Database Setup"
echo "  Production-grade with User Management"
echo "============================================"
echo

# Check for root password
if [ -z "$ROOT_PASSWORD" ]; then
    # Try to load from .env file
    if [ -f ".env" ]; then
        print_info "Loading credentials from .env file..."
        set -a
        # shellcheck disable=SC1091
        . ./.env
        set +a
        ROOT_PASSWORD="${ARANGO_PASSWORD:-}"
    fi

    # Check again after loading .env
    if [ -z "$ROOT_PASSWORD" ]; then
        print_error "ARANGO_PASSWORD not set"
        echo "Please either:"
        echo "  1. Export it: export ARANGO_PASSWORD='your_password'"
        echo "  2. Add it to .env file"
        exit 1
    fi
fi

print_success "Root password found"

# Check for Unix socket availability
if [ "$USE_UNIX_SOCKET" = "true" ] && [ -S "$UNIX_SOCKET" ]; then
    print_success "Unix socket found at $UNIX_SOCKET"
    print_info "Using Unix socket for optimal performance"
    CONNECTION_URL="http+unix://${UNIX_SOCKET//\//%2F}"
else
    if [ "$USE_UNIX_SOCKET" = "true" ]; then
        print_warning "Unix socket not found at $UNIX_SOCKET"
        print_info "Falling back to HTTP connection at $DB_HOST"
    else
        print_info "Using HTTP connection at $DB_HOST"
    fi
    CONNECTION_URL="$DB_HOST"
fi

print_info "Database name: $DB_NAME"
print_info "Admin user: $DB_ADMIN_USER"
print_info "Writer user: $DB_WRITER_USER"
print_info "Reader user: $DB_READER_USER"

# Parse command line arguments
DROP_EXISTING=false
AUTO_MODE=false
QUIET_MODE=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --drop-existing)
            DROP_EXISTING=true
            shift
            ;;
        --auto)
            AUTO_MODE=true
            shift
            ;;
        --quiet)
            QUIET_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --drop-existing  Drop and recreate database if it exists"
            echo "  --auto          Run in automatic mode (no prompts)"
            echo "  --quiet         Minimal output"
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

# Check Python and dependencies
print_info "Checking Python environment..."
if ! python3 -c "import arango" 2>/dev/null; then
    print_warning "python-arango not installed"
    print_info "Installing with pip..."
    pip install python-arango
fi

# Generate passwords if not provided
if [ -z "$DB_ADMIN_PASSWORD" ]; then
    DB_ADMIN_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated admin password"
fi
if [ -z "$DB_WRITER_PASSWORD" ]; then
    DB_WRITER_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated writer password"
fi
if [ -z "$DB_READER_PASSWORD" ]; then
    DB_READER_PASSWORD=$(openssl rand -base64 24)
    print_info "Generated reader password"
fi

# Set up database
print_info "Setting up $DB_NAME database..."

# Export configuration for Python script
export DB_NAME
export DB_HOST
export CONNECTION_URL
export UNIX_SOCKET
export USE_UNIX_SOCKET
export ROOT_PASSWORD
export DB_ADMIN_USER
export DB_ADMIN_PASSWORD
export DB_WRITER_USER
export DB_WRITER_PASSWORD
export DB_READER_USER
export DB_READER_PASSWORD
export ADMIN_PERMISSIONS
export WRITER_PERMISSIONS
export READER_PERMISSIONS
export CONFIG_DIR

# Convert collections array to JSON for Python
COLLECTIONS_JSON=$(printf '%s\n' "${COLLECTIONS[@]}" | jq -R . | jq -s .)
export COLLECTIONS_JSON

# Prepare Python command arguments
PYTHON_ARGS=()
if [ "$DROP_EXISTING" = true ]; then
    PYTHON_ARGS+=("--drop-existing")
fi
if [ "$AUTO_MODE" = true ]; then
    PYTHON_ARGS+=("--auto")
fi
if [ "$QUIET_MODE" = true ]; then
    PYTHON_ARGS+=("--quiet")
fi

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Run the simplified Python setup script from project root
if python3 "$PROJECT_ROOT/dev-utils/setup_arxiv_database_simple.py" "${PYTHON_ARGS[@]}"; then
    print_success "Database setup completed successfully"
else
    print_error "Database setup failed"
    exit 1
fi

# Load the generated credentials
if [ -f "$CONFIG_DIR/$DB_NAME.env" ]; then
    print_info "Loading generated credentials..."
    source "$CONFIG_DIR/$DB_NAME.env"

    # Update .env file if it exists
    if [ -f ".env" ]; then
        print_info "Updating .env file with new credentials..."

        # Backup existing .env
        cp .env .env.backup

        # Add or update database credentials
        if grep -q "ARXIV_DB_NAME" .env; then
            # Update existing entries
            sed -i "s/^ARXIV_DB_NAME=.*/ARXIV_DB_NAME=$ARXIV_DB_NAME/" .env
            sed -i "s/^ARXIV_WRITER_PASSWORD=.*/ARXIV_WRITER_PASSWORD=$ARXIV_WRITER_PASSWORD/" .env
            sed -i "s/^ARXIV_READER_PASSWORD=.*/ARXIV_READER_PASSWORD=$ARXIV_READER_PASSWORD/" .env
            sed -i "s/^ARXIV_ADMIN_PASSWORD=.*/ARXIV_ADMIN_PASSWORD=$ARXIV_ADMIN_PASSWORD/" .env
        else
            # Add new entries
            echo "" >> .env
            echo "# ArXiv Repository Database" >> .env
            echo "ARXIV_DB_NAME=$ARXIV_DB_NAME" >> .env
            echo "ARXIV_WRITER_PASSWORD=$ARXIV_WRITER_PASSWORD" >> .env
            echo "ARXIV_READER_PASSWORD=$ARXIV_READER_PASSWORD" >> .env
            echo "ARXIV_ADMIN_PASSWORD=$ARXIV_ADMIN_PASSWORD" >> .env
        fi

        print_success "Updated .env file with database credentials"
    fi
else
    print_warning "Could not load generated credentials"
fi

# Test connection with new credentials
print_info "Testing database connection..."

# First test if we can import the arango_unix_client module
if python3 -c "import sys; sys.path.insert(0, '.'); from core.database.arango_unix_client import get_database_for_workflow" 2>/dev/null; then
    print_success "Using optimized arango_unix_client module"

    # Test with the optimized client
    python3 -c "
import os
import sys
sys.path.insert(0, '.')
from core.database.arango_unix_client import get_optimized_client, get_database_for_workflow

db_name = os.environ.get('DB_NAME', 'arxiv_repository')
writer_user = os.environ.get('DB_WRITER_USER', 'arxiv_writer')
writer_password = os.environ.get('DB_WRITER_PASSWORD', '')

try:
    # Get optimized client
    client = get_optimized_client(prefer_unix=True)
    info = client.get_connection_info()

    print(f'Connection configured:')
    print(f'  Type: {\"Unix socket\" if info[\"using_unix\"] else \"HTTP\"}')
    if info['using_unix']:
        print(f'  Socket: {info[\"unix_socket\"]}')
    else:
        print(f'  Host: {info[\"http_host\"]}')

    # Get database connection
    db = get_database_for_workflow(
        db_name=db_name,
        username=writer_user,
        password=writer_password,
        prefer_unix=True
    )

    # Test query execution
    result = list(db.aql.execute('RETURN 1'))
    assert result == [1], 'Query test failed'

    # Get collections
    collections = db.collections()
    user_collections = [c for c in collections if not c['name'].startswith('_')]

    print(f'✓ Successfully connected as {writer_user}')
    print(f'  Collections: {len(user_collections)}')

    # Test write permissions
    test_collection = 'arxiv_papers'
    if test_collection in [c['name'] for c in user_collections]:
        try:
            test_doc = db[test_collection].insert({'_key': '_test_permission', 'test': True})
            db[test_collection].delete('_test_permission')
            print(f'  Write permissions: ✓ Verified')
        except:
            print(f'  Write permissions: Read-only')

    # Performance comparison if Unix socket is available
    if info['using_unix']:
        print(f'\\n📊 Performance Note:')
        print(f'  Unix sockets provide ~40% better throughput than TCP')
        print(f'  This is optimal for local model datastores')

except Exception as e:
    print(f'✗ Connection failed: {e}')
    exit(1)
"
else
    print_warning "arango_unix_client module not found, using basic connection"

    # Fallback to basic connection test
    python3 -c "
import os
from arango import ArangoClient
from pathlib import Path

db_name = os.environ.get('DB_NAME', 'arxiv_repository')
writer_user = os.environ.get('DB_WRITER_USER', 'arxiv_writer')
writer_password = os.environ.get('DB_WRITER_PASSWORD', '')

# Check for Unix socket
unix_socket = os.environ.get('UNIX_SOCKET', '/tmp/arangodb.sock')
use_unix = os.environ.get('USE_UNIX_SOCKET', 'true').lower() == 'true'

# Determine connection URL
if use_unix and os.path.exists(unix_socket) and Path(unix_socket).is_socket():
    connection_url = f\"http+unix://{unix_socket.replace('/', '%2F')}\"
    print(f'Testing Unix socket connection: {unix_socket}')
else:
    connection_url = os.environ.get('DB_HOST', 'http://localhost:8529')
    print(f'Testing HTTP connection: {connection_url}')

client = ArangoClient(hosts=connection_url)
try:
    # Test with writer credentials
    db = client.db(db_name,
                   username=writer_user,
                   password=writer_password)

    # Test query execution
    result = list(db.aql.execute('RETURN 1'))
    assert result == [1], 'Query test failed'

    # Get collections
    collections = db.collections()
    user_collections = [c for c in collections if not c['name'].startswith('_')]

    print(f'✓ Successfully connected as {writer_user}')
    print(f'  Connection type: {\"Unix socket\" if \"unix\" in connection_url else \"HTTP\"}')
    print(f'  Collections: {len(user_collections)}')

    # Test write permissions
    test_collection = 'arxiv_papers'
    if test_collection in [c['name'] for c in user_collections]:
        # Try a simple insert/delete to verify write permissions
        try:
            test_doc = db[test_collection].insert({'_key': '_test_permission', 'test': True})
            db[test_collection].delete('_test_permission')
            print(f'  Write permissions: ✓ Verified')
        except:
            print(f'  Write permissions: Read-only')

except Exception as e:
    print(f'✗ Connection failed: {e}')
    exit(1)
"
fi

if [ $? -eq 0 ]; then
    print_success "Database connection test passed"
else
    print_error "Database connection test failed"
    exit 1
fi

# Summary
echo
echo "============================================"
echo "  Database Setup Complete!"
echo "============================================"
echo
echo "Database: $DB_NAME"
echo "Host: $DB_HOST"
echo
echo "Users created:"
echo "  • $DB_ADMIN_USER  - Full admin access"
echo "  • $DB_WRITER_USER - Read/write for processing"
echo "  • $DB_READER_USER - Read-only for monitoring"
echo
echo "Collections created:"
for collection in "${COLLECTIONS[@]}"; do
    echo "  • $collection"
done
echo
echo "Credentials saved to:"
echo "  • $CONFIG_DIR/$DB_NAME.env"
if [ -f ".env" ]; then
    echo "  • .env (updated)"
fi
echo
echo "To use in workflows:"
echo "  source $CONFIG_DIR/$DB_NAME.env"
echo "  python -m core.workflows.workflow_arxiv_sorted \\"
echo "    --database $DB_NAME \\"
echo "    --username $DB_WRITER_USER"
echo
print_success "Database ready for production use!"
