#!/usr/bin/env python3
"""
Test Unix Socket Connection for ArangoDB
========================================

This script tests the Unix socket connection to ArangoDB and compares
performance against HTTP connections.

Theory Connection (Conveyance Framework):
Optimizes T (time) dimension by eliminating network overhead.
Unix sockets provide ~40% better throughput than TCP for local connections.
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_unix_socket_connection():
    """Test Unix socket connection to ArangoDB."""
    print("\n" + "=" * 60)
    print("Testing Unix Socket Connection to ArangoDB")
    print("=" * 60)

    try:
        from core.database.arango_unix_client import get_optimized_client, get_database_for_workflow

        # Test 1: Check client initialization
        print("\n1. Testing client initialization...")
        client = get_optimized_client(prefer_unix=True)
        info = client.get_connection_info()

        print(f"   Connection type: {'Unix socket' if info['using_unix'] else 'HTTP'}")
        if info['using_unix']:
            print(f"   Socket path: {info['unix_socket']}")
            print(f"   ✅ Unix socket connection available")
        else:
            print(f"   HTTP host: {info['http_host']}")
            print(f"   ⚠️  Using HTTP fallback")

        # Test 2: Database connection
        print("\n2. Testing database connection...")
        db_name = os.environ.get('ARXIV_DB_NAME', 'arxiv_repository')
        username = os.environ.get('ARXIV_WRITER_USER', 'arxiv_writer')
        password = os.environ.get('ARXIV_WRITER_PASSWORD')

        if not password:
            print("   ❌ ARXIV_WRITER_PASSWORD not set")
            print("   Run: source config/arxiv_repository.env")
            return False

        db = get_database_for_workflow(
            db_name=db_name,
            username=username,
            password=password,
            prefer_unix=True
        )

        # Test 3: Query execution
        print("\n3. Testing query execution...")
        result = list(db.aql.execute("RETURN 1"))
        if result == [1]:
            print("   ✅ Query execution successful")
        else:
            print(f"   ❌ Query returned unexpected result: {result}")
            return False

        # Test 4: Collection access
        print("\n4. Testing collection access...")
        collections = db.collections()
        user_collections = [c for c in collections if not c['name'].startswith('_')]
        print(f"   Found {len(user_collections)} user collections:")
        for col in user_collections:
            print(f"     • {col['name']}")

        # Test 5: Performance comparison (if Unix socket is available)
        if info['using_unix']:
            print("\n5. Performance comparison (Unix socket vs HTTP)...")

            # Test Unix socket performance
            start = time.time()
            for _ in range(100):
                list(db.aql.execute("RETURN 1"))
            unix_time = time.time() - start
            unix_qps = 100 / unix_time

            print(f"   Unix socket: {unix_qps:.1f} queries/sec")

            # Test HTTP performance
            try:
                from arango import ArangoClient
                http_client = ArangoClient(hosts=f"http://localhost:8529")
                http_db = http_client.db(db_name, username=username, password=password)

                start = time.time()
                for _ in range(100):
                    list(http_db.aql.execute("RETURN 1"))
                http_time = time.time() - start
                http_qps = 100 / http_time

                print(f"   HTTP:        {http_qps:.1f} queries/sec")

                improvement = ((unix_qps - http_qps) / http_qps) * 100
                print(f"   📊 Unix socket is {improvement:.1f}% faster")

            except Exception as e:
                print(f"   Could not test HTTP performance: {e}")

        print("\n" + "=" * 60)
        print("✅ All tests passed successfully!")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure the arango_unix_client module is available")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_socket_file():
    """Check if the Unix socket file exists."""
    socket_path = os.environ.get('ARANGO_UNIX_SOCKET', '/tmp/arangodb.sock')
    socket_file = Path(socket_path)

    print(f"\nChecking Unix socket at: {socket_path}")

    if socket_file.exists():
        if socket_file.is_socket():
            print(f"✅ Socket file exists and is a valid socket")

            # Check permissions
            import stat
            mode = socket_file.stat().st_mode
            print(f"   Permissions: {oct(stat.S_IMODE(mode))}")
            print(f"   Owner: {socket_file.owner()}")

            return True
        else:
            print(f"⚠️  File exists but is not a socket")
            return False
    else:
        print(f"❌ Socket file does not exist")
        print("\nTo enable Unix socket in ArangoDB:")
        print("1. Edit /etc/arangodb3/arangod.conf")
        print("2. Add under [server] section:")
        print("   endpoint = unix:///tmp/arangodb.sock")
        print("3. Restart ArangoDB:")
        print("   sudo systemctl restart arangodb3")
        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Test Unix socket connection to ArangoDB")
    parser.add_argument('--check-socket', action='store_true',
                        help='Only check if socket file exists')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose logging')
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check environment
    print("\nEnvironment Configuration:")
    print(f"  ARXIV_DB_NAME: {os.environ.get('ARXIV_DB_NAME', 'Not set (using default: arxiv_repository)')}")
    print(f"  ARXIV_WRITER_USER: {os.environ.get('ARXIV_WRITER_USER', 'Not set (using default: arxiv_writer)')}")
    print(f"  ARXIV_WRITER_PASSWORD: {'Set' if os.environ.get('ARXIV_WRITER_PASSWORD') else 'Not set'}")
    print(f"  ARANGO_UNIX_SOCKET: {os.environ.get('ARANGO_UNIX_SOCKET', 'Not set (using default: /tmp/arangodb.sock)')}")

    if args.check_socket:
        success = check_socket_file()
    else:
        # First check if socket exists
        socket_exists = check_socket_file()

        if not socket_exists:
            print("\n⚠️  Unix socket not available, tests will use HTTP fallback")

        # Run connection tests
        success = test_unix_socket_connection()

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())