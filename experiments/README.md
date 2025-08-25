# HADES Experiments

This directory contains experimental code and research that uses the HADES infrastructure but is separate from the core system.

## Directory Structure

```
experiments/
├── README.md                    # This file
├── experiment_template/         # Template for new experiments
│   ├── README.md               # Experiment description
│   ├── config/                 # Configuration files
│   ├── src/                    # Experiment code
│   ├── data/                   # Input data (small files only)
│   ├── results/                # Output/results (gitignored)
│   └── analysis/               # Analysis notebooks/scripts
├── datasets/                    # Shared datasets for experiments
│   ├── cs_papers.json          # Computer Science papers
│   ├── graph_papers.json       # Graph theory papers
│   ├── ml_ai_papers.json       # ML/AI papers
│   └── ...                     # Other curated datasets
├── documentation/              # Experiment-specific documentation
│   └── experiments/            # Analysis and research notes
└── experiment_*/               # Individual experiments
```

## Creating a New Experiment

1. **Copy the template**:
   ```bash
   cp -r experiment_template experiment_semantic_bridge
   ```

2. **Update the README** in your experiment directory with:
   - Experiment name and description
   - Hypothesis being tested
   - Infrastructure dependencies
   - How to run the experiment

3. **Configure your experiment**:
   - Edit `config/experiment_config.yaml`
   - Specify which infrastructure components you need
   - Set data paths and parameters

4. **Write experiment code** in `src/`:
   - Import infrastructure: `from core.framework.embedders import JinaV4Embedder`
   - Keep experiment-specific logic separate
   - Use configuration files for parameters

## Separation Rules

### What Goes in Infrastructure (`/core/`, `/tools/`)
- Reusable components (embedders, extractors, storage)
- Production pipelines (ACID pipeline)
- Database management tools
- MCP servers and APIs
- Shared utilities

### What Goes in Experiments (`/experiments/`)
- Research hypotheses testing
- One-off analysis scripts
- Experimental algorithms not ready for production
- Visualization and analysis notebooks
- Research-specific configurations

## Using Infrastructure Components

Experiments can import and use any infrastructure component:

```python
# Import core framework components
from core.framework.embedders import JinaV4Embedder
from core.framework.extractors.docling_extractor import DoclingExtractor
from core.framework.storage import ArangoDBManager

# Import tools
from tools.arxiv.pipelines.arxiv_pipeline import ProcessingTask

# Use in your experiment
embedder = JinaV4Embedder()
embeddings = embedder.embed_documents(documents)
```

## Shared Datasets

The `datasets/` directory contains curated datasets for experiments:

- **cs_papers.json**: Computer Science ArXiv papers
- **graph_papers.json**: Graph theory papers  
- **ml_ai_papers.json**: Machine Learning and AI papers
- **gears_dataset.json**: GEARS framework related papers
- **graphsage_dataset.json**: GraphSAGE related papers
- **combined_experiments.json**: Combined dataset for multi-domain experiments
- **sample_10k.json**: Small sample for quick testing

Load datasets in your experiments:

```python
import json

with open('../datasets/cs_papers.json', 'r') as f:
    papers = json.load(f)
```

## Best Practices

1. **Keep experiments self-contained**: Each experiment should have its own README and be runnable independently

2. **Use configuration files**: Don't hardcode parameters, use YAML configs

3. **Document dependencies**: Clearly list which infrastructure components are used

4. **Version your experiments**: Use git tags or branches for different experiment versions

5. **Clean up resources**: Experiments should clean up GPU memory and close connections

6. **Use relative paths**: Reference datasets and infrastructure using relative imports

7. **Archive completed experiments**: Move finished experiments to an archive branch or tag

## Running Experiments

Most experiments follow this pattern:

```bash
cd experiments/experiment_name
pip install -r requirements.txt  # If experiment has special deps
python src/run_experiment.py --config config/experiment_config.yaml
```

## Output Management

Experiment outputs are gitignored by default:
- `results/` - Experimental results
- `output/` - Generated files
- `logs/` - Log files
- `.cache/` - Cached data

To preserve results, either:
1. Create a summary in the experiment's README
2. Move important results to `documentation/experiments/`
3. Use git-lfs for large result files that must be versioned

## Contributing

When adding a new experiment:

1. Follow the template structure
2. Document your hypothesis and methodology
3. Include a requirements.txt if you need special packages
4. Add a summary to this README when complete
5. Consider writing up findings in `documentation/experiments/`

## Current Experiments

### experiment_1
- **Status**: In Development
- **Purpose**: TBD
- **Owner**: TBD

### documentation/experiments
Contains analysis and research documentation from various experiments:
- `author_qualitative_data/`: Author research and tracking
- `citation_analysis/`: Citation network analysis
- `paper_analysis/`: Individual paper deep-dives (Conveyance, GEARS, GraphSAGE)

---

Remember: Experiments are for exploration and research. Once an approach is validated and ready for production, refactor it into the appropriate infrastructure component.