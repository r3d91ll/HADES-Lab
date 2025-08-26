# Issue #4: Create Docker Setup for HADES-Lab

## Overview

Create comprehensive Docker containerization for HADES-Lab to ensure consistent deployment across different environments and simplify setup for new developers.

## Background

Current setup requires manual installation of dependencies, GPU drivers, and database connections. Docker would provide:
- Reproducible environments
- Simplified onboarding
- Consistent dependency versions
- Easy scaling and deployment

## Requirements

### Core Components

1. **Base Images**
   - GPU-enabled PyTorch base (nvidia/cuda)
   - Python 3.11+ with Poetry
   - CUDA 12.1+ for GPU acceleration

2. **Service Containers**
   - **hades-processor**: Main processing pipeline
   - **arangodb**: Database (optional, can use external)
   - **staging**: RamFS volume for inter-phase data

3. **Development Features**
   - Hot reload for code changes
   - Volume mounts for data persistence
   - GPU passthrough for CUDA
   - Debug ports exposed

## Implementation Plan

### Dockerfile Structure

```dockerfile
# Dockerfile
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock ./

# Install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Copy application code
COPY . .

# Entry point
CMD ["python", "tools/arxiv/pipelines/arxiv_pipeline.py"]
```

### Docker Compose Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  hades-processor:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - ARANGO_PASSWORD=${ARANGO_PASSWORD}
      - ARANGO_HOST=arangodb
    volumes:
      - /bulk-store/arxiv-data:/bulk-store/arxiv-data:ro
      - ./experiments:/app/experiments
      - /dev/shm:/dev/shm
    tmpfs:
      - /dev/shm:size=32G
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]

  arangodb:
    image: arangodb:3.11
    environment:
      - ARANGO_ROOT_PASSWORD=${ARANGO_PASSWORD}
    ports:
      - "8529:8529"
    volumes:
      - arangodb_data:/var/lib/arangodb3

volumes:
  arangodb_data:
```

### Development Setup

```yaml
# docker-compose.dev.yml
version: '3.8'

services:
  hades-dev:
    extends:
      file: docker-compose.yml
      service: hades-processor
    volumes:
      - .:/app
      - /app/.venv  # Exclude venv from mount
    environment:
      - PYTHONDONTWRITEBYTECODE=1
      - PYTHONUNBUFFERED=1
    command: tail -f /dev/null  # Keep container running
```

## Benefits

1. **Consistency**: Same environment across all deployments
2. **Simplicity**: One command to start entire stack
3. **Scalability**: Easy to deploy multiple workers
4. **Isolation**: No system-wide dependency conflicts
5. **Portability**: Runs on any Docker-enabled system with GPU

## Challenges

1. **GPU Support**: Requires nvidia-docker runtime
2. **Large Images**: ML dependencies create multi-GB images
3. **Data Volumes**: Need to mount large PDF datasets
4. **Performance**: Some overhead vs native execution

## Testing Requirements

- [ ] GPU acceleration working
- [ ] ArangoDB connectivity
- [ ] RamFS staging functional
- [ ] Performance within 5% of native
- [ ] Multi-GPU support
- [ ] Development hot-reload

## Documentation Needed

1. **README.docker.md**: Setup instructions
2. **GPU configuration guide**
3. **Volume mounting best practices**
4. **Performance tuning tips**

## Related Files

- `setup_local.sh` - Current setup script
- `pyproject.toml` - Dependencies
- `.env.template` - Environment variables

## Priority

**Low-Medium** - Current setup works, but Docker would improve onboarding

## Estimated Effort

- Initial Dockerfile: 1 day
- Docker Compose setup: 1 day
- GPU integration testing: 2 days
- Documentation: 1 day
- Total: ~1 week

## Labels

`enhancement`, `docker`, `devops`, `infrastructure`

## Success Criteria

- New developer can run pipeline with 3 commands:
  ```bash
  git clone <repo>
  docker-compose up -d
  docker-compose exec hades-dev python tools/arxiv/pipelines/arxiv_pipeline.py
  ```