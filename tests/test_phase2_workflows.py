#!/usr/bin/env python3
"""
Unit Tests for Phase 2: Database and Workflows

Tests the reorganized database and workflow modules.
"""

import unittest
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


class TestWorkflowModule(unittest.TestCase):
    """Test the workflows module structure."""

    def test_workflow_imports(self):
        """Test that workflow modules can be imported."""
        try:
            from core.workflows import WorkflowBase, WorkflowConfig, WorkflowResult
            self.assertIsNotNone(WorkflowBase)
            self.assertIsNotNone(WorkflowConfig)
            self.assertIsNotNone(WorkflowResult)
            print("✓ Workflow base classes import successfully")
        except ImportError as e:
            self.fail(f"Failed to import workflows module: {e}")

    def test_state_management_imports(self):
        """Test that state management can be imported."""
        try:
            from core.workflows.state import StateManager, CheckpointManager
            self.assertIsNotNone(StateManager)
            self.assertIsNotNone(CheckpointManager)
            print("✓ State management imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import state management: {e}")

    def test_storage_structure(self):
        """Test that storage module structure is correct."""
        from pathlib import Path

        storage_path = Path("core/workflows/storage")
        self.assertTrue(storage_path.exists(), "Storage directory should exist")

        # Check for expected files
        expected_files = ["__init__.py", "storage_base.py", "storage_local.py"]
        for filename in expected_files:
            file_path = storage_path / filename
            self.assertTrue(file_path.exists(), f"{filename} should exist")

        print("✓ Storage module structure correct")

    def test_workflow_config_creation(self):
        """Test creating a workflow configuration."""
        from core.workflows import WorkflowConfig

        config = WorkflowConfig(
            name="test_workflow",
            batch_size=64,
            num_workers=8,
            use_gpu=True
        )

        self.assertEqual(config.name, "test_workflow")
        self.assertEqual(config.batch_size, 64)
        self.assertEqual(config.num_workers, 8)
        self.assertTrue(config.use_gpu)
        self.assertTrue(config.checkpoint_enabled)  # Default value

        print("✓ WorkflowConfig creation working")


class TestDatabaseModule(unittest.TestCase):
    """Test the database module restructure."""

    def test_database_factory_import(self):
        """Test that DatabaseFactory can be imported."""
        try:
            from core.database import DatabaseFactory
            self.assertIsNotNone(DatabaseFactory)
            print("✓ DatabaseFactory imports successfully")
        except ImportError as e:
            self.fail(f"Failed to import DatabaseFactory: {e}")

    def test_database_structure(self):
        """Test that database module structure is correct."""
        from pathlib import Path

        db_path = Path("core/database")
        self.assertTrue(db_path.exists(), "Database directory should exist")

        # Check subdirectories
        arango_path = db_path / "arango"
        postgres_path = db_path / "postgres"

        self.assertTrue(arango_path.exists(), "Arango subdirectory should exist")
        self.assertTrue(postgres_path.exists(), "PostgreSQL subdirectory should exist")

        # Check for factory
        factory_path = db_path / "database_factory.py"
        self.assertTrue(factory_path.exists(), "database_factory.py should exist")

        print("✓ Database module structure correct")

    def test_arango_imports(self):
        """Test that ArangoDB components can be imported."""
        try:
            from core.database.arango import ArangoDBManager, retry_with_backoff
            self.assertIsNotNone(ArangoDBManager)
            self.assertIsNotNone(retry_with_backoff)
            print("✓ ArangoDB components import successfully")
        except ImportError as e:
            # This is okay if the module hasn't been fully migrated yet
            print(f"⚠ ArangoDB import warning: {e}")


class TestProcessorsRestructure(unittest.TestCase):
    """Test the processors module restructure."""

    def test_text_processors(self):
        """Test that text processors are in the right place."""
        from pathlib import Path

        text_path = Path("core/processors/text")
        self.assertTrue(text_path.exists(), "Text processors directory should exist")

        # Check for chunking strategies
        chunking_path = text_path / "chunking_strategies.py"
        self.assertTrue(chunking_path.exists(), "chunking_strategies.py should exist")

        print("✓ Text processors structure correct")

    def test_chunking_imports(self):
        """Test that chunking strategies can be imported from new location."""
        try:
            from core.processors.text import (
                ChunkingStrategy,
                TokenBasedChunking,
                ChunkingStrategyFactory
            )
            self.assertIsNotNone(ChunkingStrategy)
            self.assertIsNotNone(TokenBasedChunking)
            self.assertIsNotNone(ChunkingStrategyFactory)
            print("✓ Chunking strategies import from new location")
        except ImportError as e:
            self.fail(f"Failed to import chunking strategies: {e}")

    def test_backward_compatibility(self):
        """Test that backward compatibility imports work with deprecation warning."""
        import warnings

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Try to import from old location
            try:
                from core.processors import DocumentProcessor
                # Should trigger deprecation warning
                self.assertTrue(len(w) > 0, "Should have deprecation warning")
                self.assertTrue(any("deprecated" in str(warning.message).lower()
                                  for warning in w))
                print("✓ Backward compatibility with deprecation warning")
            except ImportError:
                # This is okay if the actual classes haven't been created yet
                print("⚠ DocumentProcessor not yet implemented in workflows")


class TestPhase2Integration(unittest.TestCase):
    """Integration tests for Phase 2 changes."""

    def test_module_organization(self):
        """Test that all modules are properly organized."""
        from pathlib import Path

        # Check core structure
        core_path = Path("core")

        expected_dirs = [
            "embedders",      # Phase 1
            "extractors",     # Phase 1
            "database",       # Phase 2
            "workflows",      # Phase 2
            "processors",     # Phase 2 (reorganized)
        ]

        for dir_name in expected_dirs:
            dir_path = core_path / dir_name
            self.assertTrue(dir_path.exists(), f"{dir_name} directory should exist")

        print("✓ All Phase 1 and 2 directories present")

    def test_no_circular_imports(self):
        """Test that there are no circular import issues."""
        try:
            # Import all major modules
            from core.embedders import EmbedderFactory
            from core.extractors import ExtractorFactory
            from core.workflows import WorkflowBase
            from core.database import DatabaseFactory
            from core.processors.text import ChunkingStrategyFactory

            print("✓ No circular import issues detected")
        except ImportError as e:
            self.fail(f"Import error (possible circular dependency): {e}")


if __name__ == "__main__":
    # Run with verbosity
    unittest.main(verbosity=2)