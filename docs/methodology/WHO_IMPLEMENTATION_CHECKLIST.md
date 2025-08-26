# WHO Dimension Implementation Checklist

## Phase 1: Infrastructure Setup ⏳

### Database Preparation
- [ ] Create PostgreSQL tables for experimental data
  ```sql
  CREATE TABLE who_experiments (
      run_id SERIAL PRIMARY KEY,
      task_id VARCHAR(50) NOT NULL,
      task_type VARCHAR(20) NOT NULL,
      model VARCHAR(50) NOT NULL,
      timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
  );
  
  CREATE TABLE who_traces (
      trace_id SERIAL PRIMARY KEY,
      run_id INTEGER REFERENCES who_experiments(run_id),
      compressed_trace BYTEA,
      compression_method VARCHAR(20)
  );
  
  CREATE TABLE who_metrics (
      metric_id SERIAL PRIMARY KEY,
      run_id INTEGER REFERENCES who_experiments(run_id),
      h_external FLOAT NOT NULL,
      h_transfer FLOAT NOT NULL,
      h_internal FLOAT NOT NULL,
      h_effective FLOAT NOT NULL,
      ctx_raw FLOAT NOT NULL,
      ctx_transformed FLOAT NOT NULL,
      conveyance FLOAT NOT NULL,
      response_time FLOAT NOT NULL
  );
  ```

- [ ] Create ArangoDB collections for caching
  ```javascript
  db._create("who_cached_retrievals");
  db._create("who_task_embeddings");
  db.who_cached_retrievals.ensureIndex({ 
    type: "hash", 
    fields: ["cache_key"] 
  });
  ```

### Environment Configuration
- [ ] Configure GPU allocation for trace collection
- [ ] Set up Redis for intermediate caching
- [ ] Configure batch processing parameters
  ```yaml
  # configs/who_experiment.yaml
  experiment:
    batch_size: 10
    max_parallel: 4
    trace_collection: true
    compression: pca
    compression_target: 0.95  # variance retained
  ```

## Phase 2: Component Measurement Tools 🔧

### H_external Measurement
- [ ] Implement offline capability scorer
  ```python
  def measure_h_external(model_name: str) -> float:
      """
      Measure external model capability (offline).
      Returns value in [0,1] based on:
      - Context window size (normalized)
      - Model parameter count (log-scaled)
      - Benchmark scores (if available)
      """
      # Implementation here
  ```

- [ ] Create benchmark suite for models
  - [ ] Context window utilization test
  - [ ] Instruction following capability
  - [ ] Retrieval integration ability

### H_transfer Measurement
- [ ] Implement MCP boundary analyzer
  ```python
  def measure_h_transfer(mcp_config: dict, trace: dict) -> float:
      """
      Measure transfer efficiency at MCP boundary.
      Analyzes:
      - Tool call success rate
      - Parameter marshaling efficiency
      - Response parsing accuracy
      """
      # Implementation here
  ```

- [ ] Create transfer efficiency metrics
  - [ ] Serialization overhead
  - [ ] Tool invocation latency
  - [ ] Error rate at boundary

### H_internal Measurement
- [ ] Implement RAG system profiler
  ```python
  def measure_h_internal(
      retrieval_trace: dict,
      ranking_trace: dict
  ) -> float:
      """
      Measure internal RAG capability.
      Combines:
      - Retrieval recall@k
      - Ranking precision
      - Reranking effectiveness
      """
      # Implementation here
  ```

## Phase 3: Experimental Protocol 🧪

### Task Preparation
- [ ] Create 30 diverse tasks across 6 categories
  ```python
  TASK_CATEGORIES = {
      "factual_qa": 5,      # Simple fact retrieval
      "multi_hop": 5,       # Requires multiple retrievals
      "temporal": 5,        # Time-sensitive queries
      "analytical": 5,      # Requires reasoning
      "creative": 5,        # Generation with grounding
      "adversarial": 5      # Edge cases
  }
  ```

- [ ] Generate ground truth for evaluation
- [ ] Create relevance judgments for retrieval

### Component Isolation Protocol
- [ ] Implement retrieval caching mechanism
  ```python
  class RetrievalCache:
      def cache_retrieval(self, task_id: str, docs: List[Doc]):
          """Cache retrieval results for component isolation."""
          
      def get_cached(self, task_id: str) -> Optional[List[Doc]]:
          """Retrieve cached results to fix R component."""
  ```

- [ ] Create isolation experimental runs
  1. **Baseline**: Full system, all components variable
  2. **Fixed R**: Cache retrievals, vary H_internal only
  3. **Fixed H_int**: Fix RAG config, vary H_external only
  4. **Fixed H_ext**: Same model, vary H_transfer only

### Data Collection Pipeline
- [ ] Implement trace collector with compression
  ```python
  class TraceCollector:
      def __init__(self, compression='pca', target_variance=0.95):
          self.compression = compression
          self.target_variance = target_variance
          
      def collect_and_compress(self, trace: dict) -> bytes:
          """Collect trace and compress for storage."""
          # PCA/SVD compression implementation
  ```

- [ ] Create metric extraction pipeline
  - [ ] Parse model responses
  - [ ] Calculate component metrics
  - [ ] Compute derived metrics (AMI, PEI, TCR)

## Phase 4: Statistical Analysis 📊

### Model Fitting
- [ ] Implement ridge regression for H_eff model
  ```python
  from sklearn.linear_model import RidgeCV
  
  def fit_h_effective_model(data: pd.DataFrame):
      """
      Fit: H_eff = (H_ext·H_int)^θ · H_trans^φ · (1 + χ·H_ext·H_int)
      Using log-transform and ridge regression
      """
      # Transform to log-linear
      # Fit with cross-validation
      # Return θ, φ, χ estimates
  ```

- [ ] Implement bootstrap validation
  ```python
  def bootstrap_validate(data, n_bootstrap=1000):
      """Bootstrap validation for small sample."""
      # Resample with replacement
      # Fit model
      # Collect parameter distributions
  ```

### Hypothesis Testing
- [ ] Implement Holm-Bonferroni correction
  ```python
  def holm_bonferroni(p_values: List[float], alpha: float = 0.05):
      """Apply Holm-Bonferroni correction."""
      # Sort p-values
      # Apply sequential rejection
      # Return adjusted p-values
  ```

- [ ] Create mixed-effects model for clustering
  ```python
  import statsmodels.formula.api as smf
  
  def fit_mixed_model(data):
      """Account for task clustering."""
      model = smf.mixedlm(
          "conveyance ~ h_external + h_transfer + h_internal",
          data,
          groups=data["task_type"],
          re_formula="~1"  # Random intercepts
      )
      return model.fit()
  ```

## Phase 5: Validation & Testing ✅

### Unit Tests
- [ ] Test metric calculations
  ```python
  def test_ami_calculation():
      """Test Agency Magnification Index."""
      assert calculate_ami(h_eff=0.8, h_ext=0.6) == pytest.approx(1.33, 0.01)
  ```

- [ ] Test zero-propagation
  ```python
  def test_zero_propagation():
      """Verify zero-propagation gate."""
      assert calculate_h_eff(h_ext=0, h_trans=0.5, h_int=0.8) == 0
      assert calculate_h_eff(h_ext=0.5, h_trans=0, h_int=0.8) == 0
  ```

### Integration Tests
- [ ] Test full experimental pipeline
- [ ] Verify trace compression/decompression
- [ ] Test component isolation mechanism

### Performance Benchmarks
- [ ] Measure trace collection overhead (<5% impact)
- [ ] Verify compression ratios (>20x)
- [ ] Test query performance with caching

## Phase 6: Production Deployment 🚀

### Monitoring Setup
- [ ] Create Grafana dashboards
  - [ ] Component metrics over time
  - [ ] Conveyance distribution
  - [ ] Error rates and latencies

- [ ] Set up alerts
  - [ ] H_transfer degradation (boundary issues)
  - [ ] H_internal failures (RAG problems)
  - [ ] Anomalous conveyance values

### Documentation
- [ ] Write experimental protocol guide
- [ ] Document metric definitions
- [ ] Create troubleshooting guide
- [ ] Generate API documentation

### Incremental Rollout
- [ ] Deploy to staging environment
- [ ] Run pilot with 10 tasks
- [ ] Analyze results and refine
- [ ] Scale to full 300 tasks
- [ ] Generate final report

## Phase 7: Analysis & Reporting 📈

### Data Analysis
- [ ] Generate descriptive statistics
- [ ] Create correlation matrices
- [ ] Fit final models
- [ ] Validate predictions

### Visualization
- [ ] Component distribution plots
- [ ] Interaction effects visualization
- [ ] Time series of metrics
- [ ] Task category comparisons

### Scientific Output
- [ ] Prepare results section
- [ ] Create figures and tables
- [ ] Write discussion of findings
- [ ] Address limitations explicitly
- [ ] Suggest future work

## Success Criteria 🎯

### Minimum Viable Experiment
- ✓ 30 tasks completed (5 per category)
- ✓ All component metrics collected
- ✓ Basic statistical analysis completed
- ✓ Trace compression working (>20x)

### Full Success
- ✓ 300 tasks completed (if resources permit)
- ✓ All 8 hypotheses tested with corrections
- ✓ Bootstrap validation completed
- ✓ Production deployment ready
- ✓ Scientific paper drafted

## Risk Mitigation 🛡️

### Technical Risks
- **GPU memory overflow**: Use gradient checkpointing
- **Trace storage explosion**: Increase compression ratio
- **MCP boundary failures**: Implement retry logic
- **Database deadlocks**: Use proper transaction isolation

### Statistical Risks
- **Low power**: Acknowledge as exploratory research
- **Model overfitting**: Use ridge regression
- **Multiple comparisons**: Apply Holm-Bonferroni
- **Clustering effects**: Use mixed-effects models

### Operational Risks
- **Resource constraints**: Start with minimum viable
- **Time overruns**: Parallelize where possible
- **Data loss**: Implement checkpointing
- **Integration issues**: Test incrementally

---

## Quick Start Commands

```bash
# Setup environment
cd /home/todd/olympus/HADES
poetry install
poetry shell

# Initialize databases
python tools/who/init_databases.py

# Run pilot experiment
python tools/who/run_experiment.py \
  --config configs/who_experiment.yaml \
  --tasks 10 \
  --parallel 4

# Analyze results
python tools/who/analyze_results.py \
  --input results/who_pilot.json \
  --output analysis/who_pilot_report.html

# Full experiment (when ready)
python tools/who/run_experiment.py \
  --config configs/who_experiment.yaml \
  --tasks 300 \
  --parallel 8 \
  --checkpoint
```

---

*Implementation checklist for "The WHO Dimension: Agency Distribution Across LLM-RAG Boundaries" v2.5*