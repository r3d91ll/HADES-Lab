#!/usr/bin/env python3
"""
ArXiv Metadata Processing Configuration
========================================

Configuration for processing the complete ArXiv metadata dataset.
Extends BaseConfig with proper validation and semantic checks.

Theory Connection (Conveyance Framework):
Configuration ensures Context (Ctx) coherence through validation,
preventing zero-propagation by maintaining all dimensions > 0.
"""

from typing import Optional, List
from pathlib import Path
from pydantic import Field, validator
from core.config.config_base import BaseConfig


class ArxivMetadataConfig(BaseConfig):
    """
    Configuration for ArXiv metadata processing workflow.

    Implements semantic validation to ensure all Conveyance dimensions
    remain positive (W, R, H > 0, T < ∞) to prevent zero-propagation.
    """

    # Data source configuration
    metadata_file: Path = Field(
        default=Path("/bulk-store/arxiv-data/metadata/arxiv-kaggle-latest.json"),
        description="Path to ArXiv metadata JSON file"
    )
    max_records: Optional[int] = Field(
        default=None,
        description="Maximum records to process (None for all)"
    )

    # Processing configuration
    batch_size: int = Field(
        default=1000,
        ge=1,
        le=10000,
        description="Records per batch for processing"
    )
    embedding_batch_size: int = Field(
        default=128,  # Optimal for A6000 GPUs
        ge=1,
        le=512,  # Allow higher batch sizes for better throughput
        description="Batch size for embedder (adjust based on GPU memory)"
    )
    num_workers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Number of parallel workers for embedding generation"
    )
    worker_batch_size: int = Field(
        default=500,
        ge=1,
        description="Number of records each worker processes"
    )

    # Staging configuration (required by WorkflowBase)
    staging_path: Optional[Path] = Field(
        default=None,
        description="Path for staging temporary files"
    )

    # Database configuration
    drop_collections: bool = Field(
        default=False,
        description="Drop existing collections before processing"
    )
    arango_host: str = Field(
        default="http://192.168.1.69:8529",
        description="ArangoDB host URL"
    )
    arango_database: str = Field(
        default="academy_store",
        description="ArangoDB database name"
    )
    arango_username: str = Field(
        default="root",
        description="ArangoDB username"
    )

    # Collection names
    metadata_collection: str = Field(
        default="arxiv_metadata",
        description="Collection for metadata documents"
    )
    chunks_collection: str = Field(
        default="arxiv_abstract_chunks",
        description="Collection for abstract chunks"
    )
    embeddings_collection: str = Field(
        default="arxiv_abstract_embeddings",
        description="Collection for embeddings"
    )

    # Embedder configuration
    embedder_model: str = Field(
        default="jinaai/jina-embeddings-v4",
        description="Embedding model to use"
    )
    use_fp16: bool = Field(
        default=True,
        description="Use FP16 precision for embeddings"
    )
    chunk_size_tokens: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Token size for chunks"
    )
    chunk_overlap_tokens: int = Field(
        default=128,
        ge=0,
        le=512,
        description="Token overlap between chunks"
    )

    # GPU configuration
    use_gpu: bool = Field(
        default=True,
        description="Use GPU for embedding"
    )
    gpu_device: str = Field(
        default="cuda:0",
        description="GPU device to use"
    )

    # Checkpoint configuration
    checkpoint_interval: int = Field(
        default=10000,
        ge=100,
        description="Save checkpoint every N records"
    )
    resume_from_checkpoint: bool = Field(
        default=True,
        description="Resume from last checkpoint if available"
    )
    checkpoint_file: Path = Field(
        default=Path("/tmp/arxiv_metadata_checkpoint.json"),
        description="Path to checkpoint file"
    )
    state_file: Path = Field(
        default=Path("/tmp/arxiv_metadata_state.json"),
        description="Path to state file"
    )

    # Performance monitoring
    monitor_interval: int = Field(
        default=100,
        ge=10,
        description="Report progress every N batches"
    )
    target_throughput: float = Field(
        default=48.0,
        gt=0,
        description="Target papers per second"
    )

    # Validation
    validate_embeddings: bool = Field(
        default=True,
        description="Validate embedding dimensions"
    )
    expected_embedding_dim: int = Field(
        default=2048,
        description="Expected embedding dimension for Jina v4"
    )

    @validator('chunk_overlap_tokens')
    def validate_overlap(cls, v, values):
        """
        Validate that the chunk overlap is strictly less than the chunk size.
        
        Parameters:
            cls: The validator's class (unused).
            v (int): Proposed value for `chunk_overlap_tokens`.
            values (dict): Other field values; expects `chunk_size_tokens` may be present.
        
        Returns:
            int: The validated overlap value `v`.
        
        Raises:
            ValueError: If `v` is greater than or equal to `chunk_size_tokens`.
        """
        chunk_size = values.get('chunk_size_tokens', 512)
        if v >= chunk_size:
            raise ValueError(f"Overlap ({v}) must be less than chunk size ({chunk_size})")
        return v

    @validator('metadata_file')
    def validate_metadata_file(cls, v):
        """
        Validate that the provided metadata file path exists.
        
        Parameters:
            v (Path): Path to the ArXiv metadata file.
        
        Returns:
            Path: The same path if it exists.
        
        Raises:
            ValueError: If the file at `v` does not exist.
        """
        if not v.exists():
            raise ValueError(f"Metadata file not found: {v}")
        return v

    def validate_semantics(self) -> List[str]:
        """
        Validate semantic consistency of the configuration and return any problems found.
        
        Performs lightweight, actionable checks and returns a list of human-readable error messages (empty if configuration is semantically valid). Checks performed:
        - Embedder model compatibility: warns if `embedder_model` does not contain `"jina"` (case-insensitive).
        - GPU/device consistency: if `use_gpu` is True, requires `gpu_device` to start with `"cuda"`.
        - Batch-size relationship: ensures `embedding_batch_size` <= `batch_size`.
        - Checkpoint interval: flags excessively large `checkpoint_interval` (> 100000).
        - Throughput feasibility: flags unrealistic throughput when `batch_size` < 100 and `target_throughput` > 10.
        - Collection name uniqueness: ensures `metadata_collection`, `chunks_collection`, and `embeddings_collection` are all distinct.
        
        Returns:
            List[str]: A list of validation error messages; empty when no semantic issues are detected.
        """
        errors = []

        # Validate embedder model compatibility
        if "jina" not in self.embedder_model.lower():
            errors.append(f"Embedder model {self.embedder_model} may not support late chunking")

        # Validate GPU settings consistency
        if self.use_gpu and not self.gpu_device.startswith("cuda"):
            errors.append(f"GPU enabled but device {self.gpu_device} is not CUDA")

        # Validate batch size relationships
        if self.embedding_batch_size > self.batch_size:
            errors.append(f"Embedding batch size ({self.embedding_batch_size}) exceeds record batch size ({self.batch_size})")

        # Validate checkpoint interval
        if self.checkpoint_interval > 100000:
            errors.append(f"Checkpoint interval ({self.checkpoint_interval}) may be too large for recovery")

        # Validate throughput feasibility
        if self.batch_size < 100 and self.target_throughput > 10:
            errors.append(f"Batch size ({self.batch_size}) too small for target throughput ({self.target_throughput} papers/sec)")

        # Validate collection names are different
        collections = [self.metadata_collection, self.chunks_collection, self.embeddings_collection]
        if len(set(collections)) != len(collections):
            errors.append("Collection names must be unique")

        return errors

    def to_workflow_config(self) -> dict:
        """
        Return a workflow-ready configuration dictionary derived from this ArxivMetadataConfig.
        
        The returned dict contains a compact subset of runtime settings used to initialize or run
        the workflow orchestrator.
        
        Returns:
            dict: Workflow configuration with the following keys:
                - name (str): Workflow name.
                - batch_size (int): Number of records per processing batch.
                - num_workers (int): Number of worker processes to run.
                - use_gpu (bool): Whether to enable GPU usage.
                - checkpoint_enabled (bool): Whether workflow checkpoints/resume are enabled.
                - checkpoint_interval (int): Records between checkpoints.
                - staging_path (Path): Local staging directory for temporary files.
                - timeout_seconds (int): Workflow timeout in seconds.
        """
        return {
            "name": "arxiv_metadata_workflow",
            "batch_size": self.batch_size,
            "num_workers": 1,  # Single-threaded for streaming
            "use_gpu": self.use_gpu,
            "checkpoint_enabled": self.resume_from_checkpoint,
            "checkpoint_interval": self.checkpoint_interval,
            "staging_path": Path("/tmp/arxiv_metadata_staging"),
            "timeout_seconds": 3600  # 1 hour timeout
        }

    @classmethod
    def from_yaml_with_defaults(cls, override_path: Optional[Path] = None) -> "ArxivMetadataConfig":
        """
        Create an ArxivMetadataConfig by loading defaults from the package YAML and optionally merging an override YAML.
        
        Loads defaults from core/config/workflows/arxiv_metadata_default.yaml (located relative to this module). If override_path is provided and exists, its mappings are shallow-merged on top of the defaults (keys in the override replace defaults). Returns an instance of ArxivMetadataConfig constructed from the merged configuration.
        """
        import yaml

        # Load default configuration
        default_path = Path(__file__).parent.parent.parent / "core/config/workflows/arxiv_metadata_default.yaml"

        with open(default_path, 'r') as f:
            config_dict = yaml.safe_load(f)

        # Apply overrides if provided
        if override_path and override_path.exists():
            with open(override_path, 'r') as f:
                overrides = yaml.safe_load(f)
                if overrides:
                    config_dict.update(overrides)

        return cls(**config_dict)