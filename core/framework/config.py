"""
Configuration Management
========================

Hierarchical configuration system with support for:
- Base configuration
- Processor-specific configuration
- Environment variables
- Runtime overrides
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load .env file if it exists
load_dotenv()


class DatabaseConfig(BaseModel):
    """Database configuration."""
    host: str = Field(default="localhost", description="Database host")
    port: int = Field(default=8529, description="Database port")
    username: str = Field(default="root", description="Database username")
    password: str = Field(default="", description="Database password")
    database: str = Field(default="academy_store", description="Database name")


class ProcessorConfig(BaseModel):
    """Processor-specific configuration."""
    chunk_size: int = Field(default=1024, description="Text chunk size")
    chunk_overlap: int = Field(default=128, description="Chunk overlap size")
    embedding_batch_size: int = Field(default=10, description="Batch size for embeddings")
    embedding_model: str = Field(default="jinaai/jina-embeddings-v4", description="Embedding model")
    max_file_size: int = Field(default=10_485_760, description="Maximum file size (10MB)")
    
    # Allow extra fields for processor-specific settings
    class Config:
        extra = "allow"


class Config(BaseModel):
    """Main configuration object."""
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processor: ProcessorConfig = Field(default_factory=ProcessorConfig)
    logging_level: str = Field(default="INFO", description="Logging level")
    metrics_enabled: bool = Field(default=True, description="Enable metrics collection")


class ConfigManager:
    """
    Configuration hierarchy (highest to lowest priority):
    1. Environment variables (HADES_*)
    2. .env file
    3. Processor-specific config (configs/processors/{name}.yaml)
    4. Base config (configs/base.yaml)
    """
    
    @staticmethod
    def _load_yaml(file_path: Path) -> Dict[str, Any]:
        """Load YAML configuration file."""
        if not file_path.exists():
            return {}
        
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    @staticmethod
    def _load_base() -> Dict[str, Any]:
        """Load base configuration."""
        base_path = Path(__file__).parent.parent / "configs" / "base.yaml"
        return ConfigManager._load_yaml(base_path)
    
    @staticmethod
    def _load_processor(processor_name: str) -> Dict[str, Any]:
        """Load processor-specific configuration."""
        processor_path = Path(__file__).parent.parent / "configs" / "processors" / f"{processor_name}.yaml"
        return ConfigManager._load_yaml(processor_path)
    
    @staticmethod
    def _load_env() -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}
        
        # Database configuration from env
        if os.getenv("HADES_DB_HOST"):
            config.setdefault("database", {})["host"] = os.getenv("HADES_DB_HOST")
        if os.getenv("HADES_DB_PORT"):
            config.setdefault("database", {})["port"] = int(os.getenv("HADES_DB_PORT"))
        if os.getenv("HADES_DB_USERNAME"):
            config.setdefault("database", {})["username"] = os.getenv("HADES_DB_USERNAME")
        if os.getenv("HADES_DB_PASSWORD"):
            config.setdefault("database", {})["password"] = os.getenv("HADES_DB_PASSWORD")
        if os.getenv("HADES_DB_NAME"):
            config.setdefault("database", {})["database"] = os.getenv("HADES_DB_NAME")
        
        # Also support ARANGO_ prefix for compatibility
        if os.getenv("ARANGO_HOST"):
            config.setdefault("database", {})["host"] = os.getenv("ARANGO_HOST")
        if os.getenv("ARANGO_PORT"):
            config.setdefault("database", {})["port"] = int(os.getenv("ARANGO_PORT"))
        if os.getenv("ARANGO_USERNAME"):
            config.setdefault("database", {})["username"] = os.getenv("ARANGO_USERNAME")
        if os.getenv("ARANGO_PASSWORD"):
            config.setdefault("database", {})["password"] = os.getenv("ARANGO_PASSWORD")
        if os.getenv("ARANGO_DATABASE"):
            config.setdefault("database", {})["database"] = os.getenv("ARANGO_DATABASE")
        
        # Processor configuration from env
        if os.getenv("HADES_CHUNK_SIZE"):
            config.setdefault("processor", {})["chunk_size"] = int(os.getenv("HADES_CHUNK_SIZE"))
        if os.getenv("HADES_CHUNK_OVERLAP"):
            config.setdefault("processor", {})["chunk_overlap"] = int(os.getenv("HADES_CHUNK_OVERLAP"))
        if os.getenv("HADES_EMBEDDING_BATCH_SIZE"):
            config.setdefault("processor", {})["embedding_batch_size"] = int(os.getenv("HADES_EMBEDDING_BATCH_SIZE"))
        
        # Logging and metrics
        if os.getenv("HADES_LOG_LEVEL"):
            config["logging_level"] = os.getenv("HADES_LOG_LEVEL")
        if os.getenv("HADES_METRICS_ENABLED"):
            config["metrics_enabled"] = os.getenv("HADES_METRICS_ENABLED").lower() == "true"
        
        return config
    
    @staticmethod
    def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    @staticmethod
    def load(processor_name: str, override: Optional[Dict] = None) -> Config:
        """
        Load configuration with hierarchy.
        
        Args:
            processor_name: Name of the processor
            override: Runtime configuration overrides
            
        Returns:
            Loaded configuration object
        """
        # Start with base config
        config = ConfigManager._load_base()
        
        # Overlay processor-specific config
        processor_config = ConfigManager._load_processor(processor_name)
        config = ConfigManager._deep_merge(config, processor_config)
        
        # Overlay environment variables
        env_config = ConfigManager._load_env()
        config = ConfigManager._deep_merge(config, env_config)
        
        # Apply any runtime overrides
        if override:
            config = ConfigManager._deep_merge(config, override)
        
        return Config(**config)