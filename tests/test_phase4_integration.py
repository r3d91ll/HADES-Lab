#!/usr/bin/env python3
"""
Phase 4 Integration Tests

Comprehensive tests to validate the core restructure integration.
Ensures all modules work together correctly with the new structure.
"""

import pytest
import warnings
from pathlib import Path
import tempfile
import json
from typing import Dict, Any

# Test new imports work
def test_new_imports():
    """Test that all new module imports work correctly."""
    # Embedders
    from core.embedders import JinaV4Embedder, EmbedderFactory
    assert JinaV4Embedder is not None
    assert EmbedderFactory is not None

    # Extractors
    from core.extractors import DoclingExtractor, ExtractorFactory
    assert DoclingExtractor is not None
    assert ExtractorFactory is not None

    # Workflows
    from core.workflows.workflow_pdf import DocumentProcessor, ProcessingConfig
    assert DocumentProcessor is not None
    assert ProcessingConfig is not None

    # Database
    from core.database import DatabaseFactory
    assert DatabaseFactory is not None

    # Config
    from core.config import ConfigManager, BaseConfig
    assert ConfigManager is not None
    assert BaseConfig is not None

    # Monitoring
    from core.monitoring import PerformanceMonitor, ProgressTracker
    assert PerformanceMonitor is not None
    assert ProgressTracker is not None


def test_backward_compatibility():
    """Test that old imports still work with deprecation warnings."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # Old framework imports
        from core.framework.extractors import DoclingExtractor
        from core.framework.embedders import JinaV4Embedder

        # Check deprecation warnings were issued
        assert len(w) >= 2
        assert any("deprecated" in str(warning.message).lower() for warning in w)

        # Verify classes are still accessible
        assert DoclingExtractor is not None
        assert JinaV4Embedder is not None


def test_factory_patterns():
    """Test that all factory patterns work correctly."""
    from core.embedders import EmbedderFactory
    from core.extractors import ExtractorFactory
    from core.config import ConfigManager

    # Test embedder factory
    embedder_types = EmbedderFactory.list_available()
    assert "jina_v4" in embedder_types

    # Test extractor factory
    extractor_types = ExtractorFactory.list_available()
    assert "docling" in extractor_types

    # Test config manager singleton
    manager = ConfigManager()
    assert manager is not None


def test_workflow_integration():
    """Test that workflows integrate properly with new structure."""
    from core.workflows.workflow_pdf import DocumentProcessor, ProcessingConfig
    from core.workflows.workflow_base import WorkflowConfig, WorkflowBase

    # Create a processing config
    config = ProcessingConfig(
        extraction_method="auto",
        enable_ocr=False,
        extract_images=True
    )

    # Verify config attributes
    assert config.extraction_method == "auto"
    assert config.enable_ocr == False
    assert config.extract_images == True

    # Test workflow config
    workflow_config = WorkflowConfig(name="test_workflow")
    assert workflow_config.name == "test_workflow"
    assert workflow_config.batch_size == 32  # Default value


def test_config_module():
    """Test configuration module functionality."""
    from core.config import BaseConfig, ConfigLoader, ConfigManager
    from core.config.config_base import ProcessingConfig as ConfigProcessing

    # Test base config
    class TestConfig(BaseConfig):
        test_field: str = "test_value"

        def validate_semantics(self):
            """Implement required abstract method."""
            return []

    config = TestConfig()
    assert config.test_field == "test_value"

    # Test context scoring
    context_score = config.calculate_context_score()
    assert 0 <= context_score <= 1

    # Test config manager singleton
    manager1 = ConfigManager()
    manager2 = ConfigManager()
    assert manager1 is manager2  # Singleton pattern


def test_monitoring_integration():
    """Test monitoring module integration."""
    from core.monitoring import (
        PerformanceMonitor,
        ProgressTracker,
        MetricsCollector
    )
    from core.monitoring.metrics_base import MetricSeries

    # Test performance monitor
    monitor = PerformanceMonitor(
        interval=1.0,
        track_gpu=False  # Don't require GPU for tests
    )

    # Test progress tracker
    tracker = ProgressTracker(
        total_steps=100,
        description="Test Progress"
    )
    tracker.update(10)
    assert tracker.current_step == 10

    # Test metrics
    series = MetricSeries(name="test_metric")
    series.add(1.0)
    series.add(2.0)
    series.add(3.0)
    assert series.mean() == 2.0


def test_database_factory():
    """Test database factory functionality."""
    from core.database import DatabaseFactory

    # Test that factory methods exist
    assert hasattr(DatabaseFactory, 'get_arango')
    assert hasattr(DatabaseFactory, 'get_postgres')
    assert hasattr(DatabaseFactory, 'get_redis')
    assert hasattr(DatabaseFactory, 'create_pool')


def test_conveyance_framework_integration():
    """Test that Conveyance Framework is properly integrated."""
    from core.config import BaseConfig
    from core.monitoring.metrics_base import MetricsCollector

    # Test config has Context scoring
    class TestConfig(BaseConfig):
        test_value: int = 42

        def validate_semantics(self):
            """Implement required abstract method."""
            return []

    config = TestConfig()
    context = config.calculate_context_score()
    assert 0 <= context <= 1

    # Test metrics collector has Conveyance calculation
    class TestCollector(MetricsCollector):
        def collect(self) -> Dict[str, Any]:
            return {"test": 1.0}

    collector = TestCollector(name="test")
    conveyance = collector.calculate_conveyance(
        what=0.8,
        where=0.7,
        who=0.9,
        time=1.0
    )
    assert conveyance > 0


def test_processor_chunking():
    """Test that processor chunking strategies work."""
    from core.processors.text import (
        ChunkingStrategy,
        TokenBasedChunking,
        ChunkingStrategyFactory
    )

    # Test factory can create strategies
    strategy = ChunkingStrategyFactory.create("token_based", chunk_size=100)
    assert strategy is not None

    # Test token-based chunking
    chunker = TokenBasedChunking(chunk_size=100, overlap=10)
    text = "This is a test text. " * 50
    chunks = chunker.chunk(text)
    assert len(chunks) > 0
    assert all(isinstance(chunk, str) for chunk in chunks)


def test_end_to_end_pipeline():
    """Test a simplified end-to-end pipeline with new structure."""
    from core.config import ConfigManager, BaseConfig
    from core.monitoring import ProgressTracker
    from core.workflows.workflow_base import WorkflowConfig

    # Setup config
    class PipelineConfig(BaseConfig):
        batch_size: int = 10
        use_gpu: bool = False

        def validate_semantics(self):
            """Implement required abstract method."""
            return []

    config = PipelineConfig()

    # Setup monitoring
    tracker = ProgressTracker(total_steps=3, description="Pipeline Test")

    # Step 1: Initialize
    tracker.update(1, "Initializing")
    assert tracker.current_step == 1

    # Step 2: Process
    tracker.update(2, "Processing")
    assert tracker.current_step == 2

    # Step 3: Complete
    tracker.update(3, "Complete")
    assert tracker.current_step == 3

    # Verify context score
    context = config.calculate_context_score()
    assert context > 0


def test_import_coverage():
    """Ensure all expected modules and classes are importable."""
    import_tests = [
        # Embedders
        ("core.embedders", ["JinaV4Embedder", "EmbedderFactory", "EmbedderBase"]),
        # Extractors
        ("core.extractors", ["DoclingExtractor", "ExtractorFactory"]),
        # Workflows
        ("core.workflows", ["WorkflowBase"]),
        ("core.workflows.workflow_pdf", ["DocumentProcessor", "ProcessingConfig"]),
        # Database
        ("core.database", ["DatabaseFactory"]),
        # Config
        ("core.config", ["ConfigManager", "ConfigLoader", "BaseConfig"]),
        # Monitoring
        ("core.monitoring", ["PerformanceMonitor", "ProgressTracker", "MetricsCollector"]),
        # Processors
        ("core.processors", ["ChunkingStrategy", "ChunkingStrategyFactory"]),
    ]

    for module_path, expected_attrs in import_tests:
        module = __import__(module_path, fromlist=expected_attrs)
        for attr in expected_attrs:
            assert hasattr(module, attr), f"{module_path} missing {attr}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "--tb=short"])