#!/usr/bin/env python3
"""
Complete graph builder using HTTP connection (no Unix socket).
Simple wrapper to avoid permission issues.
"""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Temporarily disable Unix socket
os.environ['DISABLE_UNIX_SOCKET'] = '1'

# Import and run the main builder
from build_complete_graph import main

if __name__ == "__main__":
    main()