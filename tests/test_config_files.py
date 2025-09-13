#!/usr/bin/env python3
"""
Test Config File Handling
==========================

Test how Jina v4 handles config files with minimal metadata.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from core.extractors import CodeExtractor
import json
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_config_files():
    """Test config file extraction with Tree-sitter."""
    
    # Create test config files
    test_dir = Path("/tmp/config_test")
    test_dir.mkdir(exist_ok=True)
    
    # Create a test JSON file
    json_file = test_dir / "config.json"
    json_content = {
        "database": {
            "host": "localhost",
            "port": 5432,
            "credentials": {
                "username": "admin",
                "password": "secret"
            }
        },
        "features": {
            "authentication": True,
            "logging": {
                "level": "debug",
                "file": "/var/log/app.log"
            }
        }
    }
    json_file.write_text(json.dumps(json_content, indent=2))
    
    # Create a test YAML file  
    yaml_file = test_dir / "config.yaml"
    yaml_content = """
database:
  host: localhost
  port: 5432
  credentials:
    username: admin
    password: secret

features:
  authentication: true
  logging:
    level: debug
    file: /var/log/app.log
    
servers:
  - name: primary
    host: server1.example.com
  - name: backup
    host: server2.example.com
"""
    yaml_file.write_text(yaml_content)
    
    # Create a test TOML file
    toml_file = test_dir / "config.toml"
    toml_content = """
[database]
host = "localhost"
port = 5432

[database.credentials]
username = "admin"
password = "secret"

[features]
authentication = true

[features.logging]
level = "debug"
file = "/var/log/app.log"
"""
    toml_file.write_text(toml_content)
    
    # Test extraction
    extractor = CodeExtractor(use_tree_sitter=True)
    
    for config_file in [json_file, yaml_file, toml_file]:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing: {config_file.name}")
        
        result = extractor.extract(str(config_file))
        
        if result:
            # Check what Tree-sitter extracted
            if result.get('symbols'):
                symbols = result['symbols']
                logger.info(f"Config type: {symbols.get('config_type', 'unknown')}")
                
                if symbols.get('structure_info'):
                    info = symbols['structure_info'][0]
                    logger.info("Structure metadata:")
                    logger.info(f"  Format: {info.get('format')}")
                    logger.info(f"  Key count: {info.get('key_count')}")
                    logger.info(f"  Max nesting: {info.get('max_nesting_depth')}")
                    logger.info(f"  Valid syntax: {info.get('is_valid_syntax')}")
            
            # Show first 200 chars of content (what Jina will see)
            logger.info("\nContent preview (first 200 chars):")
            logger.info(result.get('text', '')[:200])
            
            # Check metadata
            if result.get('metadata'):
                logger.info(f"\nFile metadata:")
                logger.info(f"  Lines: {result['metadata'].get('line_count')}")
                logger.info(f"  Size: {result.get('file_size')} bytes")
    
    # Test with invalid JSON to show syntax validation
    logger.info(f"\n{'='*60}")
    logger.info("Testing invalid JSON:")
    
    invalid_json = test_dir / "invalid.json"
    invalid_json.write_text('{"key": "value", invalid}')
    
    result = extractor.extract(str(invalid_json))
    if result and result.get('symbols'):
        symbols = result['symbols']
        if symbols.get('structure_info'):
            info = symbols['structure_info'][0]
            logger.info(f"Valid syntax: {info.get('is_valid_syntax')}")
    
    # Clean up
    import shutil
    shutil.rmtree(test_dir)
    
    logger.info("\n" + "="*60)
    logger.info("Summary:")
    logger.info("- Config files pass full content to Jina v4")
    logger.info("- Tree-sitter provides minimal structural metadata")
    logger.info("- No semantic interpretation - that's Jina's job")
    logger.info("- Syntax validation ensures files are well-formed")


if __name__ == "__main__":
    test_config_files()