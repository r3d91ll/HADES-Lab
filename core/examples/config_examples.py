"""
Core Configuration Usage Examples
=================================

Theory Connection - Information Reconstructionism:
These examples demonstrate hierarchical configuration management that optimizes
the Conveyance Framework through structured Context coherence. Configuration
acts as the foundational WHERE positioning that enables consistent semantic
relationships across distributed system components.

From Actor-Network Theory: Configuration examples serve as "immutable mobiles"
that maintain stable parameter relationships while enabling local adaptation
through hierarchical override patterns and validation mechanisms.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Import core configuration components
from core.config import (
    BaseConfig,
    ProcessingConfig,
    StorageConfig,
    ConfigLoader,
    ConfigManager,
    config_manager,
    get_config,
    register_config,
    ConfigScope,
    ConfigError,
    ConfigValidationError
)


class CustomProcessorConfig(BaseConfig):
    """
    Example custom configuration class.

    Theory Connection: Demonstrates how to extend base configuration
    while maintaining Context coherence through validation and
    semantic relationship preservation.
    """

    # Processing parameters
    model_name: str = "jinaai/jina-embeddings-v4"
    chunk_size: int = 1024
    chunk_overlap: int = 128
    use_gpu: bool = True

    # Quality thresholds
    min_text_length: int = 100
    quality_threshold: float = 0.8

    # Performance settings
    batch_size: int = 16
    max_workers: int = 8

    def validate_semantics(self) -> list[str]:
        """
        Validate custom configuration semantics.

        Theory Connection: Ensures Context components maintain coherent
        relationships that support exponential amplification (Ctx^α).

        Returns:
            List of validation error messages
        """
        errors = []

        # Validate chunk relationships
        if self.chunk_overlap >= self.chunk_size:
            errors.append("Chunk overlap must be smaller than chunk size")

        if self.chunk_size < 256:
            errors.append("Chunk size should be at least 256 for quality embeddings")

        # Validate performance settings
        if self.use_gpu and self.batch_size > 64:
            errors.append("GPU batch size should not exceed 64 to avoid memory issues")

        if self.max_workers > 32:
            errors.append("Worker count above 32 may cause resource contention")

        # Validate quality thresholds
        if not (0.0 <= self.quality_threshold <= 1.0):
            errors.append("Quality threshold must be between 0.0 and 1.0")

        return errors


def example_basic_configuration():
    """
    Example 1: Basic Configuration Usage

    Theory Connection: Demonstrates fundamental WHERE positioning through
    configuration hierarchy and Context coherence validation.
    """
    print("Example 1: Basic Configuration Usage")
    print("=" * 40)

    # Use built-in configuration types
    processing_config = get_config('processing')
    print(f"Default processing workers: {processing_config.workers}")
    print(f"Default batch size: {processing_config.batch_size}")

    storage_config = get_config('storage')
    print(f"Default database host: {storage_config.host}")
    print(f"Default database port: {storage_config.port}")

    # Override with runtime parameters
    custom_processing = get_config('processing', workers=16, batch_size=32)
    print(f"Custom processing workers: {custom_processing.workers}")
    print(f"Custom batch size: {custom_processing.batch_size}")

    print("Basic configuration example completed.\n")


def example_custom_configuration():
    """
    Example 2: Custom Configuration Classes

    Theory Connection: Shows how to extend configuration system while
    maintaining Context validation and semantic coherence.
    """
    print("Example 2: Custom Configuration Classes")
    print("=" * 40)

    # Register custom configuration type
    register_config(
        'custom_processor',
        CustomProcessorConfig,
        ConfigScope.COMPONENT
    )

    # Use custom configuration
    try:
        custom_config = get_config('custom_processor')
        print(f"Model name: {custom_config.model_name}")
        print(f"Chunk size: {custom_config.chunk_size}")
        print(f"GPU enabled: {custom_config.use_gpu}")
        print(f"Context score: {custom_config.get_context_score():.3f}")

        # Test validation by creating invalid configuration
        try:
            invalid_config = CustomProcessorConfig(
                chunk_size=512,
                chunk_overlap=1024,  # Invalid: overlap > chunk_size
                quality_threshold=1.5  # Invalid: > 1.0
            )
        except ConfigValidationError as e:
            print(f"Validation caught invalid config: {len(e.errors)} errors")

    except ConfigError as e:
        print(f"Configuration error: {e}")

    print("Custom configuration example completed.\n")


def example_hierarchical_loading():
    """
    Example 3: Hierarchical Configuration Loading

    Theory Connection: Demonstrates WHERE dimension optimization through
    source priority resolution, enabling Context adaptation while
    maintaining base coherence.
    """
    print("Example 3: Hierarchical Configuration Loading")
    print("=" * 40)

    # Create test configuration files
    config_dir = Path("example_configs")
    config_dir.mkdir(exist_ok=True)

    # Base configuration
    base_config = {
        "workers": 4,
        "batch_size": 16,
        "timeout_seconds": 300,
        "use_gpu": False
    }

    with open(config_dir / "processing.yaml", "w") as f:
        f.write("# Base processing configuration\n")
        f.write(f"workers: {base_config['workers']}\n")
        f.write(f"batch_size: {base_config['batch_size']}\n")
        f.write(f"timeout_seconds: {base_config['timeout_seconds']}\n")
        f.write(f"use_gpu: {base_config['use_gpu']}\n")

    # Environment-specific override
    env_config = {
        "workers": 8,
        "use_gpu": True
    }

    with open(config_dir / "processing_production.yaml", "w") as f:
        f.write("# Production overrides\n")
        f.write(f"workers: {env_config['workers']}\n")
        f.write(f"use_gpu: {env_config['use_gpu']}\n")

    # Set environment variable for additional override
    os.environ['HADES_PROCESSING_BATCH_SIZE'] = '32'

    try:
        # Create loader with custom base directory
        loader = ConfigLoader(base_dir=config_dir)

        # Load with hierarchy: base -> env-specific -> environment variables
        config = loader.load_hierarchical(
            'processing',
            ProcessingConfig
        )

        print("Hierarchical configuration loaded:")
        print(f"  Workers: {config.workers} (from production override)")
        print(f"  Batch size: {config.batch_size} (from environment variable)")
        print(f"  GPU enabled: {config.use_gpu} (from production override)")
        print(f"  Timeout: {config.timeout_seconds} (from base config)")

        # Show source information
        print(f"  Configuration source: {config._source or 'merged'}")
        print(f"  Context score: {config.get_context_score():.3f}")

    except Exception as e:
        print(f"Hierarchical loading failed: {e}")

    finally:
        # Cleanup
        if 'HADES_PROCESSING_BATCH_SIZE' in os.environ:
            del os.environ['HADES_PROCESSING_BATCH_SIZE']

    print("Hierarchical configuration example completed.\n")


def example_configuration_validation():
    """
    Example 4: Configuration Validation and Error Handling

    Theory Connection: Shows Context coherence enforcement through
    validation, preventing configurations that would degrade
    system performance or cause zero-propagation conditions.
    """
    print("Example 4: Configuration Validation and Error Handling")
    print("=" * 40)

    # Test semantic validation
    print("Testing semantic validation...")

    try:
        # Valid configuration
        valid_config = CustomProcessorConfig(
            chunk_size=1024,
            chunk_overlap=256,
            quality_threshold=0.85,
            batch_size=32,
            max_workers=8
        )
        print(f"Valid config context score: {valid_config.get_context_score():.3f}")

    except ConfigValidationError as e:
        print(f"Unexpected validation error: {e}")

    # Test invalid configurations
    invalid_configs = [
        {
            "name": "Invalid chunk relationship",
            "config": {
                "chunk_size": 512,
                "chunk_overlap": 1024,  # > chunk_size
            }
        },
        {
            "name": "Invalid quality threshold",
            "config": {
                "quality_threshold": 1.5  # > 1.0
            }
        },
        {
            "name": "Excessive batch size for GPU",
            "config": {
                "use_gpu": True,
                "batch_size": 128  # Too large for GPU
            }
        }
    ]

    for test_case in invalid_configs:
        try:
            config = CustomProcessorConfig(**test_case["config"])
            config.validate_full()
            print(f"ERROR: {test_case['name']} should have failed validation!")

        except ConfigValidationError as e:
            print(f"✓ Correctly caught: {test_case['name']} ({len(e.errors)} errors)")

    print("Configuration validation example completed.\n")


def example_configuration_persistence():
    """
    Example 5: Configuration Persistence and Versioning

    Theory Connection: Demonstrates Context preservation through
    configuration state management, enabling system reproducibility
    and coherence tracking over time.
    """
    print("Example 5: Configuration Persistence and Versioning")
    print("=" * 40)

    # Create configuration with metadata
    config = CustomProcessorConfig(
        model_name="jinaai/jina-embeddings-v4",
        chunk_size=2048,
        use_gpu=True,
        batch_size=24,
        config_version="2.0",
        source="example_creation"
    )

    print(f"Created configuration:")
    print(f"  Version: {config.config_version}")
    print(f"  Source: {config.source}")
    print(f"  Context score: {config.get_context_score():.3f}")

    # Save to file
    config_file = Path("saved_config.json")
    try:
        config.save_to_file(config_file)
        print(f"Configuration saved to: {config_file}")

        # Load from file
        loaded_config = CustomProcessorConfig.from_file(config_file)
        print(f"Configuration loaded from file:")
        print(f"  Model: {loaded_config.model_name}")
        print(f"  Chunk size: {loaded_config.chunk_size}")
        print(f"  Context score: {loaded_config.get_context_score():.3f}")

        # Verify they match
        original_dict = config.to_dict()
        loaded_dict = loaded_config.to_dict()

        # Remove timestamps for comparison
        for d in [original_dict, loaded_dict]:
            d.pop('created_at', None)

        if original_dict == loaded_dict:
            print("✓ Configuration persistence verified")
        else:
            print("✗ Configuration persistence failed")

    except Exception as e:
        print(f"Persistence error: {e}")

    finally:
        # Cleanup
        if config_file.exists():
            config_file.unlink()

    print("Configuration persistence example completed.\n")


def example_configuration_merging():
    """
    Example 6: Configuration Merging and Override Patterns

    Theory Connection: Shows Context composition through configuration
    merging, enabling hierarchical Context coherence while maintaining
    local adaptation capabilities.
    """
    print("Example 6: Configuration Merging and Override Patterns")
    print("=" * 40)

    # Base configuration (development environment)
    base_config = CustomProcessorConfig(
        model_name="base-model",
        chunk_size=512,
        batch_size=8,
        max_workers=4,
        use_gpu=False
    )

    # Production overrides
    production_overrides = CustomProcessorConfig(
        model_name="jinaai/jina-embeddings-v4",
        batch_size=32,
        max_workers=16,
        use_gpu=True
    )

    print("Base configuration:")
    print(f"  Model: {base_config.model_name}")
    print(f"  Batch size: {base_config.batch_size}")
    print(f"  GPU: {base_config.use_gpu}")
    print(f"  Context score: {base_config.get_context_score():.3f}")

    print("\nProduction overrides:")
    print(f"  Model: {production_overrides.model_name}")
    print(f"  Batch size: {production_overrides.batch_size}")
    print(f"  GPU: {production_overrides.use_gpu}")

    # Merge configurations
    merged_config = base_config.merge(production_overrides)

    print("\nMerged configuration:")
    print(f"  Model: {merged_config.model_name} (from production)")
    print(f"  Chunk size: {merged_config.chunk_size} (from base)")
    print(f"  Batch size: {merged_config.batch_size} (from production)")
    print(f"  GPU: {merged_config.use_gpu} (from production)")
    print(f"  Context score: {merged_config.get_context_score():.3f}")

    # Verify base configs unchanged
    print("\nVerifying original configs unchanged:")
    print(f"  Base batch size: {base_config.batch_size} (should be 8)")
    print(f"  Production chunk size: {production_overrides.chunk_size} (should be 1024 - default)")

    print("Configuration merging example completed.\n")


def run_all_examples():
    """Run all configuration examples in sequence."""
    print("HADES Core Configuration Examples")
    print("=" * 50)
    print("Demonstrating hierarchical Context coherence through")
    print("configuration management and validation.\n")

    examples = [
        example_basic_configuration,
        example_custom_configuration,
        example_hierarchical_loading,
        example_configuration_validation,
        example_configuration_persistence,
        example_configuration_merging
    ]

    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"Example failed: {e}\n")

    # Cleanup
    cleanup_example_files()

    print("All configuration examples completed!")
    print("\nKey Benefits Demonstrated:")
    print("• Hierarchical configuration loading with validation")
    print("• Context coherence enforcement through semantic validation")
    print("• Configuration persistence and versioning")
    print("• Flexible merging and override patterns")
    print("• Type-safe configuration with automatic error detection")


def cleanup_example_files():
    """Clean up example configuration files."""
    cleanup_paths = [
        Path("example_configs"),
        Path("saved_config.json")
    ]

    for path in cleanup_paths:
        try:
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                import shutil
                shutil.rmtree(path)
        except Exception as e:
            print(f"Warning: Could not cleanup {path}: {e}")


if __name__ == "__main__":
    run_all_examples()