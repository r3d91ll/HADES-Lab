# The WHO Dimension: Agency, Access, and Capability in Information Systems

**Date:** 2025-01-21  
**Status:** Working Paper (v3.0)  
**Authors:** Todd & Claude  
**Version:** 3.0 (Unified Agency Framework with MCP Gatekeeper Theory)

## Abstract

The WHO dimension (H) represents agency, capability, and access patterns in the Information Reconstructionism framework. This document formalizes how agents - whether external LLMs, internal processes, or boundary protocols - determine information accessibility and transformation potential. We present a unified theory where the MCP endpoint serves dual roles: as a gatekeeper limiting external agency AND as an internal agent with transferred capabilities operating within the RAG system. This duality resolves the circular dependencies between W, R, and H dimensions through a two-phase calculation approach.

We present two complementary views of conveyance:

**Efficiency View (Default):** (Equation 1)
$$C = \left(\frac{W \cdot R \cdot H}{T}\right) \cdot \text{Ctx}^{\alpha}$$

**Capability View (Fixed Time):** (Equation 2)
$$C_{\text{cap}} = (W \cdot R \cdot H) \cdot \text{Ctx}^{\alpha}$$

Where:

- **W**: Work quality/signal factor (0–1)
- **R**: Retrieval relevance/recall factor (0–1)  
- **H**: Agency/capability factor (distributed; 0–1)
- **T**: Makespan (critical-path wall time) of the tool-call DAG; per-stage times logged for diagnostics
- **Ctx**: Context quality (normalized to [0,1])
- **α**: Context amplification exponent ∈ [1.5, 2.0], applied **only** to Ctx

**Context Transformation**: To achieve smooth amplification while preserving mathematical properties, we apply a sigmoid transformation:

$$\text{Ctx}^* = \frac{1}{2}\left(1 + \tanh(s(\text{Ctx} - c_0))\right), \quad \text{Ctx}^* \in [0,1]$$

```python
def ctx_transform(ctx, s=5.0, c0=0.5):
    """
    Smooth sigmoid transformation for context amplification.
    Maps [0,1] → [0,1] with steeper growth above c0.
    
    Args:
        ctx: Raw context quality ∈ [0,1]
        s: Steepness parameter (default 5.0)
        c0: Inflection point (default 0.5)
    """
    import numpy as np
    return 0.5 * (1 + np.tanh(s * (ctx - c0)))
```

This creates a C¹ smooth function where context effects plateau naturally beyond optimal levels. The transformed Ctx* is then used in conveyance calculations, avoiding double-counting with H interactions.

**Key Insights:** 
1. The MCP endpoint exhibits **agency duality**: It acts as a gatekeeper from outside (limiting H_external) while simultaneously being an agent inside the RAG (defining H_internal through its access patterns)
2. This duality resolves circular dependencies: W depends on embeddings accessed by H, while H depends on the quality W of what it accesses
3. The WHO dimension is fundamentally about **transformation capability** - who can access, process, and transform information within the system

## 1. The WHO Dimension: Foundational Theory

### 1.1 Definition and Scope

The WHO dimension (H) quantifies the **agency and capability** of actors within an information system. Unlike W (semantic quality) and R (relational positioning), H captures:

- **Access rights**: What information can be reached
- **Processing capability**: What transformations can be applied
- **Tool availability**: What operations are permitted
- **Context utilization**: How effectively context amplifies capability

### 1.2 Operational Definition

```
H = H_access × H_capability × H_tools × H_context
```

Where:
- **H_access** ∈ [0,1]: Authorization and reachability (can the agent access the data?)
- **H_capability** ∈ [0,1]: Processing power and sophistication (can the agent understand it?)
- **H_tools** ∈ [0,1]: Available operations and transformations (can the agent act on it?)
- **H_context** ∈ [0,1]: Context window utilization (can the agent maintain coherence?)

### 1.3 Zero-Propagation Through Agency

**Fundamental Axiom**: If H = 0, then C = 0 regardless of other dimensions.

Examples of H = 0:
- No access permissions (H_access = 0)
- Insufficient processing capability (H_capability = 0)
- No available tools (H_tools = 0)
- Zero context window (H_context = 0)

### 1.4 Breaking Circular Dependencies

The circular dependency between W, R, and H is resolved through a **two-phase calculation**:

**Phase 1: Static Measurement**
- Measure W_base from raw content (before embedding)
- Measure R_base from graph structure (topology only)
- Measure H_base from agent capabilities (intrinsic properties)

**Phase 2: Dynamic Interaction**
- W_effective = W_base × f(H_access, R_proximity)
- R_effective = R_base × g(W_similarity, H_tools)
- H_effective = H_base × h(W_quality, R_reachability)

This two-phase approach ensures each dimension has an independent baseline before interaction effects are calculated.

## 2. The Agency Architecture: MCP as Dual Agent

### The Three-Layer Agency Model

```
External Agent     ←→    MCP Gatekeeper    ←→    Internal Black Box
(Claude/LLM)              (Boundary)              (HADES RAG)
     ↑                        ↑                        ↑
Unbounded Agency        Agency Transfer         Bounded Agency
H_external              H_transfer              H_internal
```

```mermaid
flowchart LR
  A[External Agent (LLM)] <--> B[MCP Gatekeeper]
  B <--> C[Internal Black Box (RAG)]
  A:::u B:::t C:::b
  classDef u fill:#eaf6ff,stroke:#1d7afc;
  classDef t fill:#fff3e0,stroke:#ff9800;
  classDef b fill:#e8f5e9,stroke:#4caf50;
```


### Agency Transfer Mechanism

The WHO dimension is not located in any single component but emerges from the **multiplicative interaction** across the boundary:

**H_effective = H_external × H_transfer × H_internal**

All H terms are normalized to [0,1].

#### Formal Definitions (Measurable Components)

- **H_external**: Planning/decomposition competence of the LLM
  - *Measurement*: Tool-intent prediction and plan-program synthesis vs gold under fixed offline gate (no runtime). Report pass@k and edit distance.
  - *Knob E*: Change models, restrict tool-use, adjust temperature, limit function depth

- **H_transfer**: Protocol expressivity × authorization × compositionality (NOT including runtime overhead)
  - *Measurement*: AMI (Actionable Mutual Information), PEI (Protocol Expressivity Index), TCR (Tool Coverage Ratio)
  - *Knob G*: Alter tool alphabet, restrict schemas, modify authorization
  - *Note*: Runtime effects (serialization, retries, rate limiting) are attributed to T, not H_transfer

- **H_internal**: Action set power of the RAG/KB at fixed retrieval quality R
  - *Measurement*: BAS (Bounded Agency Score) measured by replaying cached retrievals (fixed R) while ablating operators
  - *Knob I*: Cap retrieval depth, disable operators, remove indices, freeze caches

#### Alternative Models Under Investigation

**Pure Multiplicative (Default):** (Equation 3)
$$H_{\text{eff}} = H_{\text{external}} \times H_{\text{transfer}} \times H_{\text{internal}}$$

**Exponentiated Model (Preserves Zero-Propagation):** (Equation 4)
$$H_{\text{eff}} = (H_{\text{external}} \cdot H_{\text{internal}})^{\theta} \cdot H_{\text{transfer}}^{\phi}$$

Where θ and φ are **independent elasticities** that capture differential importance. Constraints: θ, φ ∈ (0,1] to maintain concavity and zero-propagation. No sum constraint needed - these are elasticities, not probabilities. In log form: log H_eff = θ(log H_ext + log H_int) + φ log H_trans.

**Component Dependence Model (Zero-Gate Preserved):** (Equation 5)
$$H_{\text{eff}} = (H_{\text{ext}} \cdot H_{\text{int}})^{\theta} \cdot H_{\text{trans}}^{\phi} \cdot (1 + \kappa \cdot H_{\text{ext}} \cdot H_{\text{int}})$$

Where θ, φ ∈ (0,1] and κ ≥ 0. This preserves zero-propagation: if any essential factor → 0, then H_eff → 0. The κ term captures positive co-competence when both H_ext and H_int are strong. Include κ only if ΔBIC ≤ -6 vs Equation 4 and likelihood ratio test p < 0.01, with ≥ 5% median C gain and no p95 T regression.

**Critical Insight:** Agency is **distributed and multiplicative** for essential components. Any factor approaching 0 collapses total agency (zero-propagation gate).

## The MCP Gatekeeper Function

### 2.1 Inside the RAG: MCP as Internal Agent

From the RAG's internal perspective, the MCP endpoint is an **agent with transferred authority** that:

1. **Navigates the graph structure** (uses R dimension)
2. **Accesses embeddings and content** (reads W dimension)
3. **Executes transformations** (exercises H dimension)
4. **Returns processed information** (creates new W)

The MCP's agency inside the RAG is bounded by:

```python
class MCPInternalAgent:
    # What the MCP can do INSIDE the RAG
    capabilities = {
        'graph_traversal': ['BFS', 'DFS', 'shortest_path', 'similarity_search'],
        'embedding_access': ['read', 'compare', 'rank'],
        'content_retrieval': ['fetch', 'filter', 'aggregate'],
        'transformation': ['summarize', 'extract', 'combine']
    }
    
    # What limits the MCP's internal agency
    constraints = {
        'traversal_depth': 5,          # How far in the graph
        'result_limit': 100,            # How many results
        'time_budget': 10000,           # Milliseconds
        'memory_limit': 1024 * 1024     # Bytes
    }
```

This internal agency directly determines what information can be accessed and how it can be transformed, making the MCP the primary determinant of H_internal.

### The Black Box Property

From the external agent's perspective, the RAG system is a **true black box**:

```python
class HADESBlackBox:
    # OBSERVABLE (through MCP gate)
    observable = {
        'inputs': list[str],            # accepted request schemas
        'outputs': list[str],           # response schemas
        'available_tools': list[str],   # tool identifiers exposed via MCP
        'response_patterns': dict       # e.g., status codes, latency bands
    }
    
    # HIDDEN (inside the black box)
    hidden = {
        'exact_algorithms': object,     # embedding/distance functions, etc.
        'internal_state': object,       # caches, indices, runtime config
        'processing_steps': list[str],  # dynamic execution paths
        'optimization_choices': dict    # planner/optimizer heuristics
    }
```

### The Gatekeeper's Control Mechanisms

The MCP endpoint exercises control through four critical functions:

1. **Stimulus Filtering**: What external inputs are allowed to enter the system
2. **Translation Protocol**: How external intents are converted to internal operations
3. **Process Triggering**: Which internal capabilities can be activated
4. **Response Formatting**: How internal results are presented externally

```python
class MCPGatekeeper:
    def gate_control(self, external_stimulus, requesting_agent):
        # 1. Access Control
        if not self.is_authorized(requesting_agent):
            return {"ok": False, "code": "ACCESS_DENIED", "message": "Unauthorized"}
        
        # 2. Stimulus Translation
        internal_operation = self.translate_stimulus(external_stimulus)
        
        # 3. Capability Binding
        if not self.capability_available(internal_operation):
            return {"ok": False, "code": "CAPABILITY_NOT_AVAILABLE", "message": f"{internal_operation} unavailable"}
        
        # 4. Process Execution (BLACK BOX)
        result = self.execute_internal_process(internal_operation)
        
        # 5. Response Control
        return {"ok": True, "code": "OK", "data": self.format_response(result, requesting_agent)}
```

## Behavioral vs Mechanistic Understanding

### The Fundamental Epistemological Boundary

External agents can only develop **behavioral models** of the RAG system, never **mechanistic models**:

| Behavioral Model (Observable) | Mechanistic Model (Hidden) |
|------------------------------|---------------------------|
| "Semantic queries return similar documents" | "Jina v4 embeddings computed via cosine similarity" |
| "Graph traversals follow connections" | "ArangoDB using specific traversal algorithms" |
| "Context improves results" | "Late chunking preserves semantic coherence" |
| "Some queries are faster than others" | "Vector index optimization and caching strategies" |

**Implication:** The external agent (Claude) operates through **pattern recognition** and **behavioral conditioning**, not through **direct mechanical control**.

### The Skinner Box Analogy

The relationship between Claude and HADES resembles a sophisticated Skinner Box:

- **Stimuli**: Queries, commands, tool calls
- **Responses**: Results, data, status codes
- **Reinforcement**: Successful vs failed interactions
- **Learning**: Pattern recognition of what works
- **Black Box**: Internal mechanisms remain hidden

## Context Effects Architecture

### Unified Context Amplification

To avoid double-counting, all context effects are captured through the transformed Ctx* in the main conveyance equation:

**Primary Conveyance with Context:** (Equation 6)
$$C = \left(\frac{W \cdot R \cdot H}{T}\right) \cdot (\text{Ctx}^*)^{\alpha}$$

Where Ctx* = ctx_transform(Ctx) using the smooth sigmoid function defined above.

### Optional Interaction Term

For capturing context-agency synergy without double-counting: (Equation 7)

$$\log C_{\text{cap}} = \beta_W \log W + \beta_R \log R + \beta_H \log H + \alpha \log \text{Ctx}^* + \delta \cdot (\log H)(\log \text{Ctx}^*)$$

Where δ captures synergistic effects (e.g., tools more useful with good examples). Note: H remains context-independent; all context modulation happens through Ctx*.

### Context Saturation Points

Ctx_optimal values to be determined empirically by domain through cross-validation. The sigmoid transformation naturally plateaus beyond optimal levels, preventing unbounded growth while maintaining smooth derivatives.

## The Agency Paradox

### The Paradox Defined

High-capability external agents (Claude with H=0.9) depend on low-capability internal systems (HADES RAG with H=0.3) to accomplish complex tasks. However, the **gatekeeper controls what is possible**, creating this paradox:

**The most intelligent component has the least direct control over the actual mechanisms.**

### Resolution Through Behavioral Agency

The paradox resolves when we understand that **effective agency** emerges from:

1. **Pattern Recognition**: Learning what stimuli produce desired responses
2. **Strategic Interaction**: Crafting inputs to maximize output utility
3. **Compositional Reasoning**: Combining simple operations into complex workflows
4. **Meta-Learning**: Understanding the boundary conditions of the gatekeeper

## Iterative Gatekeeper Synthesis

### The Architectural Strategy

Rather than building MCP strictly first or last, we advocate for **iterative gatekeeper synthesis** - a scaffold-early, freeze-late approach that balances empirical discovery with systematic design.

### 1. Scaffold MCP (Early Phase)

Start with an **introspective MCP** that provides:

- **Self-describing tool registry** with hot-reload capability
- **Telemetry-by-default** for collecting AMI, TCR, p95 latency
- **Introspection endpoints** (list tools, schemas, rate limits)
- **Versioned tool grammar** with compatibility guards

This scaffold enables rapid learning about:

- Protocol Expressivity Index (PEI)
- Tool Coverage Ratio (TCR)
- Actionable Mutual Information (AMI)

### 2. Empirical Discovery (Middle Phase)

With the scaffold in place, empirically discover:

- **Optimal patterns**: What tool compositions work well
- **Performance profiles**: Where bottlenecks actually occur
- **Usage patterns**: How the system is most effectively used
- **Capability gaps**: What tools are missing or poorly designed

### 3. Interface Freezing (Late Phase)

Only after empirical validation, freeze the public interface with:

- **Optimal granularity** based on observed compositions
- **Efficient protocols** minimizing measured T_translation
- **Maximum expressivity** achieving high AMI/PEI
- **Back-compat layers** for evolution

### 4. Design Rules for Success

- **Hot-swap providers**: Don't hardcode RAG operations
- **Capacity & cost tracking**: Monitor serialization overhead (γ), rate limits (ρ)
- **Protocol efficiency**: H_transfer = PEI(A,L) × η(ρ,γ) × π(authz) × χ(composability)
- **Time attribution**: Runtime overhead → T, expressivity limits → H_transfer

## The HADES Self-Analysis Case Study

### Why HADES as First Repository

Processing HADES itself as the first repository perfectly demonstrates these principles:

```python
# The recursive agency challenge
def hades_analyzes_hades():
    """
    When HADES analyzes itself through the MCP gateway:
    """
    
    # External stimulus (Claude's query)
    query = "What components depend on JinaV4Embedder?"
    
    # MCP translation
    mcp_operation = "semantic_search + dependency_graph_traversal"
    
    # Internal processing (black box)
    results = [
        "tools/arxiv/arxiv_pipeline.py",
        "tools/github/github_hybrid_pipeline.py", 
        "core/framework/embedders.py"
    ]
    
    # The insight: HADES provides DATA about itself
    # Claude provides UNDERSTANDING of that data
    # The MCP gatekeeper controls what CAN BE KNOWN
```

### The Self-Awareness Boundary

HADES cannot truly be "self-aware" in the conscious sense, but it can provide **structured access to information about itself**. The "self-awareness" emerges from:

1. **Claude's queries** about HADES (external intelligence)
2. **MCP's translations** of those queries (gatekeeper function)
3. **HADES' retrievals** from its own embeddings (internal capability)
4. **Claude's interpretations** of the results (external understanding)

## Measurement Protocols

### Core Metrics (Efficiency View)

- **W**: Task quality measured by pass@k, human rubric scores
- **R**: Retrieval quality via nDCG@k, Recall@k, Coverage metrics
- **H_external**: Tool-intent prediction and plan-program synthesis vs gold under fixed offline gate. Pass@k, edit distance.
- **H_transfer**: AMI, PEI, TCR and TCR_TA (protocol expressivity and faithfulness, NOT runtime)
- **H_internal**: BAS measured by replaying cached retrievals (fixed R) while ablating operators
- **T**: p50/p95 makespan; per-stage latency; serialization/validation; rate-limit waits; retries
- **Ctx**: Weighted sum w_L·L + w_I·I + w_A·A + w_G·G with normalized weights

**Context Weight Normalization:**
```python
def normalize_context_weights(w_L, w_I, w_A, w_G):
    """Ensure context weights sum to 1."""
    total = w_L + w_I + w_A + w_G
    if total == 0:
        # Equal weights if all zero
        return 0.25, 0.25, 0.25, 0.25
    # Normalize to sum to 1
    return w_L/total, w_I/total, w_A/total, w_G/total

def calculate_context(L, I, A, G, w_L=0.25, w_I=0.25, w_A=0.25, w_G=0.25):
    """Calculate context with normalized weights."""
    # Normalize weights
    w_L, w_I, w_A, w_G = normalize_context_weights(w_L, w_I, w_A, w_G)
    
    # Calculate weighted sum
    Ctx = w_L * L + w_I * I + w_A * A + w_G * G
    
    # Apply sigmoid transformation
    Ctx_star = ctx_transform(Ctx)
    
    return Ctx_star
```

### Key Performance Indicators

#### Actionable Mutual Information (AMI)

$$\text{AMI} = \frac{I(\text{External plans}; \text{Internal ops})}{H(\text{Internal ops})}$$

Measures how well the MCP gate channels external intent into internal operations. Higher AMI indicates better H_transfer.

#### Protocol Expressivity Index (PEI)

$$\text{PEI} = \log|\mathcal{L}(A)_{\leq L}|$$

Where 𝓛(A) is the set of valid tool programs of length ≤L. Approximates the size of reachable action space.

**Estimation when enumeration explodes:** Use Monte-Carlo program sampling with coverage extrapolation (Good-Turing/Chao estimators). Sample N random valid programs, measure unique patterns seen, extrapolate total space.

#### Tool Coverage Ratio (TCR)

$$\text{TCR} = \frac{\text{# distinct tools used in solved tasks}}{\text{# tools exposed}}$$

$$\text{TCR}_{\text{TA}} = \frac{\text{# tools used in solved tasks}}{\text{# tools plausibly applicable to task family}}$$

Report both metrics. Low TCR with high PEI suggests H_external bottleneck; low PEI suggests H_transfer bottleneck.

#### Bounded Agency Score (BAS)

$$\text{BAS} = \frac{\text{tasks solved with programs length} \leq L}{\text{all tasks}}$$

Measured at fixed R, Ctx, T to isolate H_internal.

### Multiple Comparisons Correction

#### Family-Wise Error Rate Control

For the factorial experiment with multiple hypothesis tests:

**Primary Method: Holm-Bonferroni**
1. Order p-values: p₁ ≤ p₂ ≤ ... ≤ pₘ
2. For i-th hypothesis, reject if pᵢ ≤ α/(m-i+1)
3. Stop at first non-rejection

**Test Families**:
- **Component Effects Family**: Tests for H_ext, H_trans, H_int main effects (3 tests)
- **Interaction Family**: All two-way and three-way interactions (4 tests)
- **Model Selection Family**: Multiplicative vs additive vs exponentiated (3 comparisons)

Apply correction within each family separately with α_family = 0.05.

**Power Analysis (Primary)**:
- Target effect size: Cohen's d = 0.5
- Per-test α after Holm-Bonferroni ≈ 0.017 (worst case)
- **Mixed-effects simulation**: Targeting 80% power at N ≥ 120 under corrections and ICC from pilot
- **Clustering adjustment**: Effective sample size n_eff = n/(1 + (m-1)·ICC)
  - With ICC = 0.2 and m = 20 tasks/cluster: n_eff ≈ 120/4.8 = 25
  - Mitigated through mixed-effects models with task cluster as random effect
- **Sample size**: 120 runs minimum; increase to 300 if resources permit for robust power
- (An uncorrected IID analysis suggests ~93% power at N=120; reported in appendix for context only)

### Sample Size Considerations

**Effective Sample Size Challenge**:
With clustering (ICC = 0.2, m = 20), effective N ≈ 25 for 120 runs. This provides limited power (~40-50%) for 10 hypothesis tests.

**Mitigation Strategies**:
1. **Primary analyses only**: Focus on main effects (H_ext, H_trans, H_int) and Equation 4 vs 3
2. **Mixed-effects models**: Account for clustering structure explicitly
3. **Resource-permitting expansion**: Increase to 300 runs for adequate power (n_eff ≈ 62)
4. **Sequential testing**: Run initial 120, analyze, then adaptive sampling if needed

**External Validity Trade-offs**:
- Temperature = 0 and cached retrievals create controlled but artificial conditions
- Consider supplementary online retrieval runs with temperature > 0 for generalization
- Balance: 80% controlled conditions, 20% production-like for external validity

### Experimental Design

#### Task Complexity Pilot Protocol

**Phase 1: Continuous Scoring (30 tasks)**

- Collect complexity ratings from 3 annotators (0=trivial to 1=expert-level)
- Compute inter-rater reliability (ICC > 0.7 required)

**Phase 2: Natural Clustering**

- Apply k-means clustering on complexity scores
- Optimal k via elbow method (expect 3-4 clusters)

**Phase 3: Stratified Sampling**

- 20 tasks per identified cluster
- Random selection within clusters

#### Minimum Viable Experiment (120 runs - Reviewer Recommended)

- **Task Stratification**: Use clusters from pilot protocol
- **Sample Size**: 20 tasks per complexity cluster × conditions = 120 runs minimum
- **Power Analysis**: Detects Cohen's d = 0.5 with 93% power (verified via power.t.test)
- **Conditions**:
  - Baseline: H_ext=0.8, H_trans=0.5, H_int=0.5
  - Ablations: Systematic reduction of each component
- **Include C=0 observations**: Use Tobit regression for censored data

#### Robust Design (2×2 Factorial)

- Transfer (low/high) × Internal (low/high)
- 20-30 tasks per cell, 3 random seeds
- Total: 120-180 runs
- Model selection via BIC (penalizes complexity appropriately for sample size)
- **Effect Size Threshold**: Cohen's d < 0.2 considered negligible
- **Tie Resolution**: When ΔBIC < 2, select model with fewer parameters

### Component Isolation Protocol

**Key Principle**: Use cached/replayed retrievals to fix R while varying H_internal

1. **Baseline**: Full system measurement with retrieval logging
2. **Fix R, Vary H_int**: 
   - Cache retrieval results from baseline
   - Replay same retrievals with reduced operators (H_internal → reduced)
   - This isolates H_internal effects without confounding R changes
3. **Ablation H_trans**: Restrict protocol alphabet (H_transfer → reduced)
4. **Ablation H_ext**: Weaker model with same prompts (H_external → reduced)
5. **Temperature Control**: Fix temperature = 0 for all H_external measurements
6. **Null H_int**: Disable all operators while keeping retrieval interface (H_internal → 0)

### Component Independence Decision Tree

```
Measure partial correlation |r| between H_ext and H_int
├─ |r| < 0.3: Use Equation 3 (pure multiplicative)
├─ 0.3 ≤ |r| < 0.5: Use Equation 4 (exponentiated with free θ, φ)
└─ |r| ≥ 0.5: Use Equation 4 with free (θ, φ); 
    add Equation 5 (κ > 0) only if ΔBIC ≤ -6
    Avoid additive forms that violate zero-propagation
```

Continue experiment regardless; model selection post-hoc based on BIC and effect sizes.

### Component Variation Settings

**H_external variations**:

- High: GPT-4, temperature=0, full context window
- Low: GPT-3.5, temperature=0, limited context

**H_transfer variations**:

- High: Full tool alphabet, 100 req/min, all schemas
- Low: Restricted tools (10 tools), 10 req/min, basic schemas only

**H_internal variations**:

- High: k=20, all operators enabled, full indices
- Low: k=5, basic operators only, limited indices

### Model Fitting Recipe

**Parameter Estimation**:
- Fit θ, φ via constrained regression on log C with ridge λ = 10^(-3)
- Constraints: θ, φ ∈ (0,1]
- Use clustered standard errors by task_cluster
- Mixed-effects model: lmer(log(C) ~ log(H_ext*H_int) + log(H_trans) + (1|task_cluster))

### Model Selection Criteria

**Decision Hierarchy**:

1. **ΔBIC < 2**: Models equivalent, select simpler (fewer parameters)
2. **ΔBIC ≥ 2**: Select lower BIC (penalizes complexity)
3. **ΔBIC ≥ 6**: Reject complex model decisively
4. **AIC only for pure prediction**: Not for parameter interpretation

**Secondary Criteria (when BIC inconclusive)**:

- **Structural axioms**: Must satisfy zero-propagation
- **Practical gain**: Require ≥10% improvement in C (or ≥5% if p95 T unchanged)
- **Parameter stability**: Narrow CIs, low collinearity

**Interpretability**: Not explicitly weighted - BIC complexity penalty sufficient

### Implementation Recipe for AMI

1. **Log traces**: (P_t, O_t, S_t) for each task
2. **Discretize** to symbol sequences or n-grams:
   - **Vocabulary size**: 100 symbols
   - **Bin method**: Equal-frequency quantiles
   - **Smoothing**: Laplace (α = 1)
3. **Estimate distributions** with Dirichlet smoothing
4. **Compute** I(P;O) and H(O) via plug-in estimators
5. **Report**: AMI, NMI = I(P;O)/min{H(P),H(O)}, AMI|success
6. **Gaming prevention**: Report AMI together with H(O) and program length L distributions
   - Any AMI increase with collapsing H(O) or degenerate L triggers red-flag audit
7. **Confidence Intervals**: Bootstrap with 1000 resamples, percentile method for skewed C distributions

## Practical Implications

### For System Architecture

1. **Build core capabilities first**: PostgreSQL schemas, embedding pipelines, graph structures
2. **Develop processing tools second**: ArXiv pipeline, GitHub pipeline, web scrapers
3. **Create MCP interface last**: After understanding optimal interaction patterns

### For Agency Design

1. **Maximize H_internal**: Build robust, capable RAG systems
2. **Optimize H_transfer**: Design efficient, expressive MCP protocols
3. **Leverage H_external**: Create tools that amplify LLM capabilities

### For Tool Development

1. **Focus on behavioral patterns**: What works well in practice
2. **Profile real usage**: How the system is actually used
3. **Design for composition**: Simple tools that combine powerfully
4. **Optimize for iteration**: Fast feedback loops for development

## Falsifiable Predictions

### Model Discrimination Tests

**P1 (Zero-Propagation Gate):** If any of {W, R, H} = 0 or T → ∞, then C = 0

**P2 (Exponent Sensitivities):** Component degradation effects depend on elasticities θ, φ

- Halving H_trans → C₂/C₁ = 2^(-φ)
- Halving (H_ext·H_int) → C₂/C₁ = 2^(-θ)

**P3 (Pure Multiplicative Baseline):** With θ = φ = 1 (Equation 3)

- 50% degradation in any component → 50% conveyance loss

**P4 (Dependence Bonus):** With Equation 5 and κ > 0

- Co-increase: Δlog C ≈ θ·Δlog(H_ext·H_int) + φ·Δlog H_trans + log(1 + κ·H_ext·H_int)

**P5 (Context Plateau):** Performance plateaus beyond Ctx_optimal

- Observable: Marginal gains diminish as Ctx approaches saturation
- Measurable: Response latency may increase while conveyance gains plateau

### Self-Reference Efficiency Test

**Experimental Design**: Domain ∈ {Self, External} as factorial factor

**Structure**:

- Factorial: Domain × (Transfer low/high) × (Internal low/high)
- **Block randomization**: Interleave run order to avoid drift
- Block 1: External tasks (80% of trials)
- Block 2: Self-reference tasks (20% of trials)
- Randomize block order across experimental sessions

**Control for "Knows Itself Better"**:

- **Equalize R**: Replay same retrieved sets for self vs matched external tasks
- **Matched pairs**: External repos matched by size, code/language mix, graph density
- **Blindfold condition**: Self with obfuscated identifiers to remove trivial shortcuts

**Metrics**: Report both ratio and absolutes

- Hypothesis: C_self/C_external will differ from 1.0
- **Difference-in-differences**: (C_self - C_self,blind) - (C_ext - C_ext,blind)
- **Guardrail**: Require |ΔnDCG@k| < 0.03 between self and external before comparing C
- Report: C, C_cap, R, H-decomposition, p95 T for both conditions

### HRM Integration Readiness

**Data Collection for Training**:

- **Supervised targets**: Next-op prediction, argument-schema prediction, program validity
- **Gold programs**: Expert curation for subset, weak labeling from successful traces
- **Retrieval Outcome Prediction (ROP)**: Model predicts nDCG@k before calling tools
- **Scoring**: Brier score, ECE, AUROC for metacognitive calibration

**Trace Requirements**:

- Canonical alphabets for Plans P and Ops O (versioned)
- Persist tool_grammar_hash, index_state_hash, model_hash per run
- Include failures/timeouts (not just successes)
- Record makespan T, step times, and Ctx components (L/I/A/G)

## Time Attribution Rules

### Component Breakdown

**Operational Time**: T is the **makespan** (critical-path wall time) of the tool-call DAG, not the sum of components.

**Diagnostic Logging**: Still track per-stage times for analysis:

- T_request, T_translation, T_retrieval, T_response
- DAG depth and effective concurrency
- Total steps vs critical path length

### Attribution Guidelines

**Assign to H_transfer:**

- What you can say/do (reachability, privilege, composability)
- Protocol expressivity limits (AMI, PEI, TCR)
- Authorization constraints
- **NOT runtime overhead** (that goes to T)

**Assign to T:**

- How fast it runs (serialization, validation, marshaling)
- Network latency, rate limiting delays, retry overhead
- **Report empirical p50 and p95 makespan**
- **Monte Carlo settings**: 1,000 samples per scenario (pilot), 5,000 (final)
- Report p50/p95 of makespan estimate; seed logged for reproducibility

### Parallelism Handling

**Primary Metric**: Empirical p95 makespan from traces (also show p50)
**Secondary**: Monte Carlo estimates for conditional branches
**Always Log**: Per-stage times, DAG depth, effective parallelism efficiency

## Theoretical Implications

### For Information Reconstructionism

This analysis clarifies the **WHO** dimension in our fundamental equation:

**C = (W·R·H)/T · Ctx^α**

- **H** is not a simple capability measure but a **distributed agency function**
- **Agency transfers** across boundaries but remains **bounded by interfaces**
- **Information existence** depends on the **stimulus-response landscape** defined by gatekeepers

### For Actor-Network Theory

The MCP gatekeeper exemplifies a **boundary object** with mathematical connections to our framework:

- **Boundary object** → affects **R** (relational positioning between networks)
- **Obligatory passage point** → defines **H_transfer** (all agency must pass through)
- **Translation** → manifests in **T_translation** (time cost of protocol conversion)
- **Enrollment** → captured by **TCR** (Tool Coverage Ratio)

### For Cybernetic Systems

This demonstrates **requisite variety** with quantifiable constraints:

- **Requisite variety** → constrains **H_transfer upper bound**
- **Stimulus-response landscape** → defines **W × R product space**
- **Feedback loops** → measured by **AMI** (mutual information)
- **Control capacity** → limited by **PEI** (expressivity index)

## Conclusion

The WHO dimension completes the Information Reconstructionism framework by defining how agency, capability, and access patterns determine information transformation potential. Our unified theory provides:

### Theoretical Contributions

1. **Dual Agency Model**: The MCP endpoint serves as both gatekeeper (from outside) and agent (from inside), resolving the apparent paradox of distributed agency
2. **Circular Dependency Resolution**: Two-phase calculation separates static measurements from dynamic interactions
3. **Zero-Propagation Guarantee**: Multiplicative formulation ensures C=0 when H=0
4. **Measurable Components**: H_access, H_capability, H_tools, H_context with operational definitions

### Integration with W and R Dimensions

The three dimensions interact multiplicatively:

```
C = (W × R × H) / T × Ctx^α
```

Where:
- **W** (WHAT): Semantic quality accessed by agents with capability H
- **R** (WHERE): Topological positioning navigated by agents with access H  
- **H** (WHO): Agency determining what W can be accessed and how R can be traversed
- **T** (Time): Latency cost of agency operations
- **Ctx^α**: Super-linear context amplification

### Practical Implications

1. **System Design**: Build capabilities inside the RAG before exposing through MCP
2. **Performance Optimization**: H_internal bottlenecks limit overall conveyance
3. **Security Model**: H_access gates provide zero-propagation security
4. **Scalability**: Distributed H across multiple agents requires coordination

### Falsifiable Predictions

1. **Zero-Propagation**: Setting any H component to 0 results in C=0
2. **Multiplicative Decay**: 50% reduction in H causes 50% reduction in C
3. **Context Amplification**: High H with high Ctx shows super-linear gains
4. **Agency Duality**: MCP measurements from inside vs outside will differ predictably

The key insight is that **information exists only through agency** - without an agent to access, process, and transform it, even perfect semantic content (W=1) in optimal position (R=1) conveys nothing (C=0).

## 2.5 Temporal Dependencies of Agency

### 2.5.1 Agency Requires Time to Manifest

The WHO dimension is fundamentally temporal - agency cannot exist in zero time:

**H(T) Dependency Function:**
```
H(T) = H_max × (1 - exp(-T/τ_H))
```

Where:
- T = available time for agent operations
- τ_H = time constant for agency (typically 10-100ms for LLMs)
- H_max = maximum agency capability given infinite time

As T→0, H→0 because:
- **Tool execution requires time**: Each MCP tool call has latency
- **Context processing requires time**: Token generation is sequential
- **Decision making requires time**: Agent must evaluate options

### 2.5.2 MCP Tool Execution Times

Each component of H has characteristic timescales:

**H_external (Gateway Operations):**
- Authentication: 5-50ms
- Permission checking: 1-10ms
- Rate limiting: 1-5ms
- Total gateway overhead: T_gateway ≈ 10-100ms

**H_transfer (Protocol Operations):**
- Serialization: 1-10ms per MB
- Network transfer: 10-100ms (local) or 100-1000ms (remote)
- Deserialization: 1-10ms per MB
- Total transfer time: T_transfer ≈ 20-200ms

**H_internal (Processing Operations):**
- Embedding generation: 100-500ms
- Database query: 10-1000ms
- Similarity search: 50-500ms
- Total processing: T_internal ≈ 200-2000ms

**Minimum Observable Agency:**
```
T_min_H = T_gateway + T_transfer + T_internal ≈ 230-2300ms
```

Below T_min_H, agency cannot meaningfully operate, and H effectively becomes 0.

### 2.5.3 Parallel vs Sequential Time

MCP operations can be parallelized, affecting makespan:

**Sequential Execution:**
```
T_sequential = Σᵢ T_tool_i
H_sequential = f(T_sequential)
```

**Parallel Execution (Makespan):**
```
T_parallel = max(T_tool_i) + T_coordination
H_parallel = f(T_parallel) × efficiency_factor
```

Where efficiency_factor ∈ [0.6, 0.9] accounts for coordination overhead.

### 2.5.4 Time Budget Allocation

Given total time budget T_total, optimal allocation:

```
T_total = T_access + T_process + T_transform

Optimal allocation (empirically derived):
- T_access ≈ 0.2 × T_total (finding relevant information)
- T_process ≈ 0.5 × T_total (understanding/embedding)
- T_transform ≈ 0.3 × T_total (generating response)
```

### 2.5.5 Temporal Decay of Agency

Agency effectiveness decays over extended time periods:

```
H_effective(T) = H_peak × exp(-(T-T_optimal)²/2σ_T²)
```

Where:
- T_optimal ≈ 1-10 seconds for interactive queries
- σ_T ≈ 5 seconds (tolerance window)
- Beyond T_optimal, diminishing returns set in

This captures that:
- Too little time: Agent can't complete operations
- Optimal time: Agent operates at peak efficiency
- Too much time: Context staleness, resource exhaustion

## 3. Validation Protocols

### 3.1 Component Measurement Validation

Each H component requires specific validation:

**H_access Validation**:
- Test with varying permission levels
- Measure information retrieval success rate
- Validate against ground truth access logs
- Statistical test: Chi-square for access pattern independence

**H_capability Validation**:
- Compare models of different sizes (GPT-3.5 vs GPT-4)
- Measure task completion rates
- Control for task complexity
- Statistical test: ANOVA with model as factor

**H_tools Validation**:
- Systematically ablate tool availability
- Measure conveyance degradation
- Ensure zero-propagation when H_tools = 0
- Statistical test: Regression discontinuity at H_tools = 0

**H_context Validation**:
- Vary context window sizes
- Measure coherence maintenance
- Test with long-document tasks
- Statistical test: Exponential decay model for context degradation

### 3.2 Statistical Power Analysis

For the factorial experiment with H components:

**Sample Size Calculation**:
```python
from statsmodels.stats.power import FTestAnovaPower
power_analysis = FTestAnovaPower()
sample_size = power_analysis.solve_power(
    effect_size=0.25,  # Medium effect (Cohen's f)
    alpha=0.05,
    power=0.8,
    k_groups=8  # 2^3 factorial design
)
# Result: n ≈ 128 per condition
```

**Multiple Comparisons**:
- Apply Holm-Bonferroni within component families
- Component family: 4 tests (H_access, H_capability, H_tools, H_context)
- Interaction family: 6 two-way + 4 three-way + 1 four-way = 11 tests
- Critical α after correction ≈ 0.0125 for component tests

## 4. Sensitivity Analysis

### 4.1 Parameter Sensitivity

**Robustness Testing Protocol**:

1. **Baseline Parameters**:
   - H_access = 0.8, H_capability = 0.7, H_tools = 0.6, H_context = 0.9
   
2. **Perturbation Analysis**:
   - Vary each parameter ±20% while holding others constant
   - Measure ∂C/∂H_i for each component
   - Calculate elasticity: ε_i = (∂C/∂H_i) × (H_i/C)

3. **Joint Sensitivity**:
   - Latin Hypercube Sampling with 1000 samples
   - Sobol indices for variance decomposition
   - Identify parameter interactions

### 4.2 Model Robustness

**Alternative Formulations**:

1. **Additive with Threshold** (violates zero-propagation but tested for comparison):
   ```
   H = Θ(H_min) × (w₁H_access + w₂H_capability + w₃H_tools + w₄H_context)
   ```
   Where Θ is a step function requiring all H_i > H_min

2. **Geometric Mean** (preserves zero-propagation):
   ```
   H = (H_access × H_capability × H_tools × H_context)^(1/4)
   ```

3. **Minimum Gating** (strongest zero-propagation):
   ```
   H = min(H_access, H_capability, H_tools, H_context) × H_combined
   ```

**Model Selection Criteria**:
- BIC for complexity penalty
- Cross-validation with 5 folds
- Requirement: Must satisfy zero-propagation axiom

## Implementation Priorities

### Immediate Actions (Phase 1)

1. **Implement Context Transformation**: Sigmoid function with empirical parameter fitting
2. **Deploy Measurement Infrastructure**: Database schema for tracking H components with proper indices
3. **Build Scaffold MCP**: Introspective interface with telemetry collection

### Experimental Validation (Phase 2)

1. **Minimal Viable Experiment**: 120 runs minimum with mixed-effects models
2. **Self-Reference Testing**: Measure C_self/C_external ratio
3. **Context Optimization**: Identify Ctx_optimal for different task domains

### System Optimization (Phase 3)

1. **AMI Implementation**: Build logging pipeline for mutual information calculation
2. **Time Attribution**: Instrument each component with timestamp collection
3. **Model Selection**: Use AIC/BIC to select between multiplicative, additive, mixed

### Iterative Development Guidelines

**Re-estimation Frequency**:

- **Event-based**: Re-estimate when {E/G/I} changes materially (model swap, grammar change, new operator)
- **Time-based**: Weekly during scaffold phase, monthly after freeze
- **Drift-based**: Trigger when AMI or NMI drifts >15%, or when Ctx distribution shifts (KS test p<0.01)
- **TTLs**: Expire α and δ after 4 weeks or after grammar change, whichever comes first

**Early Simplified Protocol**:

- Minimal tool set, fixed schemas, strict rate limits
- Single-tool calls only to reduce H_transfer variance
- This tight baseline makes early parameter estimates cleaner

**Validity Maintenance**:

- **Two harnesses**: Frozen Benchmark Harness (immutable) and Dev Harness (fast iteration)
- **Run registry**: Persist all hashes (tool_grammar, index_state, model, cache_policy)
- **Version lock**: Run all conditions on identical codebase
- **Statistical software**: Use R packages 'lme4' (mixed models), 'censReg' (Tobit), 'BayesFactor' (model comparison)

### Trace Storage & Compression Strategy

#### Compression Approach (20-50x reduction)

**For Attention Weights**:
1. Apply PCA/SVD, keep top-k components explaining 95% variance
2. Store compressed representation (BYTEA) + reconstruction error
3. Keep summary statistics: mean, std, percentiles per head/layer
4. Retain 10% raw samples for validation

**For Embeddings**:
1. Quantize to int8 or float16 
2. Store top-k nearest neighbors instead of full vectors
3. Keep similarity distribution statistics
4. Use product quantization for dense storage

**Storage Tiers**:
- **Hot (1 week)**: Full fidelity, all fields
- **Warm (1 month)**: Compressed representations + stats
- **Cold (permanent)**: Statistics only + 10% sampled raw

**Implementation**:
```python
def compress_attention(weights, compression_ratio=0.05):
    """Compress attention weights to 5% of original size."""
    U, S, Vt = np.linalg.svd(weights, full_matrices=False)
    k = int(len(S) * compression_ratio)
    compressed = {
        'U_k': U[:, :k],
        'S_k': S[:k],
        'Vt_k': Vt[:k, :],
        'stats': {
            'mean': np.mean(weights),
            'std': np.std(weights),
            'percentiles': np.percentile(weights, [25, 50, 75, 95])
        }
    }
    return compressed
```

## 5. Database Schema Specifications

### 5.1 Core H Dimension Tables

```sql
-- Primary table for H component measurements
CREATE TABLE h_dimension_measurements (
    measurement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- H Components (all bounded [0,1])
    h_access FLOAT NOT NULL CHECK (h_access BETWEEN 0 AND 1),
    h_capability FLOAT NOT NULL CHECK (h_capability BETWEEN 0 AND 1),
    h_tools FLOAT NOT NULL CHECK (h_tools BETWEEN 0 AND 1),
    h_context FLOAT NOT NULL CHECK (h_context BETWEEN 0 AND 1),
    
    -- Computed H values
    h_multiplicative FLOAT GENERATED ALWAYS AS 
        (h_access * h_capability * h_tools * h_context) STORED,
    h_geometric FLOAT GENERATED ALWAYS AS 
        (POWER(h_access * h_capability * h_tools * h_context, 0.25)) STORED,
    h_minimum FLOAT GENERATED ALWAYS AS 
        (LEAST(h_access, h_capability, h_tools, h_context)) STORED,
    
    -- Context for measurement
    agent_type VARCHAR(50) NOT NULL,  -- 'external_llm', 'mcp_endpoint', 'internal_process'
    agent_id VARCHAR(100) NOT NULL,
    task_id VARCHAR(100),
    
    -- Foreign keys to other dimensions (to be added when those tables exist)
    -- w_measurement_id UUID REFERENCES w_dimension_measurements(measurement_id),
    -- r_measurement_id UUID REFERENCES r_dimension_measurements(measurement_id),
    
    -- Metadata
    measurement_context JSONB,
    
    CONSTRAINT pk_h_measurements PRIMARY KEY (measurement_id),
    INDEX idx_h_timestamp (timestamp),
    INDEX idx_h_agent (agent_type, agent_id),
    INDEX idx_h_task (task_id)
);

-- MCP-specific agency measurements
CREATE TABLE mcp_agency_measurements (
    mcp_measurement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID NOT NULL REFERENCES h_dimension_measurements(measurement_id),
    
    -- External perspective (gatekeeper role)
    h_external FLOAT NOT NULL CHECK (h_external BETWEEN 0 AND 1),
    h_transfer FLOAT NOT NULL CHECK (h_transfer BETWEEN 0 AND 1),
    
    -- Internal perspective (agent role)
    h_internal FLOAT NOT NULL CHECK (h_internal BETWEEN 0 AND 1),
    
    -- Effective agency
    h_effective FLOAT GENERATED ALWAYS AS 
        (h_external * h_transfer * h_internal) STORED,
    
    -- Protocol metrics
    ami FLOAT,  -- Actionable Mutual Information
    pei FLOAT,  -- Protocol Expressivity Index
    tcr FLOAT,  -- Tool Coverage Ratio
    bas FLOAT,  -- Bounded Agency Score
    
    -- Constraints and capabilities
    traversal_depth INTEGER,
    result_limit INTEGER,
    time_budget_ms INTEGER,
    memory_limit_bytes BIGINT,
    
    CONSTRAINT fk_mcp_h_measurement 
        FOREIGN KEY (measurement_id) 
        REFERENCES h_dimension_measurements(measurement_id)
        ON DELETE CASCADE
);

-- Agency interaction effects
CREATE TABLE agency_interactions (
    interaction_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    measurement_id UUID NOT NULL REFERENCES h_dimension_measurements(measurement_id),
    
    -- Two-phase calculation results
    h_base FLOAT NOT NULL CHECK (h_base BETWEEN 0 AND 1),
    w_base FLOAT NOT NULL CHECK (w_base BETWEEN 0 AND 1),
    r_base FLOAT NOT NULL CHECK (r_base BETWEEN 0 AND 1),
    
    h_effective FLOAT NOT NULL CHECK (h_effective BETWEEN 0 AND 1),
    w_effective FLOAT NOT NULL CHECK (w_effective BETWEEN 0 AND 1),
    r_effective FLOAT NOT NULL CHECK (r_effective BETWEEN 0 AND 1),
    
    -- Interaction coefficients
    h_w_interaction FLOAT,  -- How H affects W
    h_r_interaction FLOAT,  -- How H affects R
    w_h_interaction FLOAT,  -- How W affects H
    r_h_interaction FLOAT,  -- How R affects H
    
    -- Conveyance calculation
    conveyance_observed FLOAT,
    conveyance_predicted FLOAT,
    
    CONSTRAINT check_phase_values CHECK (
        h_effective <= h_base * 1.5 AND  -- Can't amplify more than 50%
        w_effective <= w_base * 1.5 AND
        r_effective <= r_base * 1.5
    )
);

CREATE TABLE agency_experiments (
    -- Component measurements
    h_external_theoretical FLOAT,
    h_external_measured FLOAT,
    h_transfer_theoretical FLOAT,
    h_transfer_measured FLOAT,
    h_internal_theoretical FLOAT,
    h_internal_measured FLOAT,
    
    -- Context parameters
    ctx_level FLOAT,
    ctx_optimal_estimated FLOAT,
    ctx_components JSONB,  -- {L,I,A,G}
    
    -- Time breakdown
    t_component_breakdown JSONB,
    makespan_ms INTEGER,
    dag_depth INTEGER,
    dag_parallelism_eff FLOAT,
    
    -- Model predictions
    c_observed FLOAT,
    c_predicted_multiplicative FLOAT,
    c_predicted_additive FLOAT,
    c_predicted_exponentiated FLOAT,
    
    -- Metrics
    ami FLOAT,
    nmi FLOAT,
    pei FLOAT,
    tcr FLOAT,
    
    -- Versioning
    tool_grammar_hash TEXT,
    index_state_hash TEXT,
    model_hash TEXT,
    plan_alphabet_version TEXT,
    op_alphabet_version TEXT,
    
    -- Metadata
    behavioral_pattern_id VARCHAR,
    mechanistic_ground_truth VARCHAR,
    task_complexity FLOAT
);

-- Retrieval caching for component isolation
CREATE TABLE cached_retrievals (
    cache_key VARCHAR(255) PRIMARY KEY,
    task_id VARCHAR(50) NOT NULL,
    query_embedding FLOAT[],
    retrieved_docs JSONB NOT NULL,  -- [{doc_id, score, content}, ...]
    retrieval_params JSONB,  -- {k: 20, method: 'cosine', index: 'hnsw'}
    ndcg_at_k FLOAT,
    recall_at_k FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_cached_task ON cached_retrievals(task_id);
CREATE INDEX idx_cached_created ON cached_retrievals(created_at);

CREATE TABLE experimental_traces (
    trace_id UUID PRIMARY KEY,
    experiment_version VARCHAR(10) NOT NULL,
    task_id VARCHAR(50) NOT NULL,
    task_complexity FLOAT NOT NULL CHECK (task_complexity BETWEEN 0 AND 1),
    task_cluster VARCHAR(20),  -- Cluster assignment from pilot
    model_name VARCHAR(50) NOT NULL,
    temperature FLOAT NOT NULL DEFAULT 0,
    
    -- Component measurements
    h_external_setting VARCHAR(50) NOT NULL,
    h_external_measured FLOAT CHECK (h_external_measured BETWEEN 0 AND 1),
    h_transfer_setting VARCHAR(50) NOT NULL,
    h_transfer_measured FLOAT CHECK (h_transfer_measured BETWEEN 0 AND 1),
    h_internal_setting VARCHAR(50) NOT NULL,
    h_internal_measured FLOAT CHECK (h_internal_measured BETWEEN 0 AND 1),
    
    -- HRM fields (compressed storage)
    gradient_norm FLOAT,
    attention_weights_compressed BYTEA,  -- PCA/SVD compressed
    attention_weights_stats JSONB,       -- mean, std, percentiles
    retrieval_embeddings_compressed BYTEA,  -- Top-k components only
    retrieval_embeddings_stats JSONB,    -- similarity distribution stats
    retrieval_predictions JSONB,
    actual_retrievals JSONB,
    reward_signal FLOAT,
    raw_trace_sample_rate FLOAT DEFAULT 0.1,  -- 10% keep raw
    
    -- Outcomes
    conveyance_score FLOAT NOT NULL,
    intermediate_states JSONB,
    
    -- Metadata
    protocol_version VARCHAR(10) NOT NULL,
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX idx_traces_task ON experimental_traces(task_id);
CREATE INDEX idx_traces_timestamp ON experimental_traces(timestamp);
CREATE INDEX idx_traces_cluster ON experimental_traces(task_cluster);

-- Partitioning strategy for time-series data
-- Partition by month for hot/warm/cold storage tiers
ALTER TABLE experimental_traces PARTITION BY RANGE (timestamp);

-- W Dimension Measurements Table (Required for Foreign Keys)
CREATE TABLE w_dimension_measurements (
    measurement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- W Components (all bounded [0,1])
    w_semantic FLOAT NOT NULL CHECK (w_semantic BETWEEN 0 AND 1),
    w_density FLOAT NOT NULL CHECK (w_density BETWEEN 0 AND 1),
    w_fidelity FLOAT NOT NULL CHECK (w_fidelity BETWEEN 0 AND 1),
    w_phase FLOAT NOT NULL CHECK (w_phase BETWEEN 0 AND 1),
    
    -- Computed W values
    w_multiplicative FLOAT GENERATED ALWAYS AS 
        (w_semantic * w_density * w_fidelity * w_phase) STORED,
    
    -- Context for measurement
    content_type VARCHAR(50) NOT NULL,  -- 'text', 'equation', 'table', 'image'
    document_id VARCHAR(100) NOT NULL,
    modality VARCHAR(50),
    
    -- Temporal dependency
    processing_time_ms INTEGER NOT NULL,
    time_to_quality JSONB,  -- W(T) curve data
    
    -- Metadata
    measurement_context JSONB,
    
    INDEX idx_w_timestamp (timestamp),
    INDEX idx_w_document (document_id),
    INDEX idx_w_modality (modality)
);

-- R Dimension Measurements Table (Required for Foreign Keys)
CREATE TABLE r_dimension_measurements (
    measurement_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    
    -- R Components (all bounded [0,1])
    r_distance FLOAT NOT NULL CHECK (r_distance BETWEEN 0 AND 1),
    r_centrality FLOAT NOT NULL CHECK (r_centrality BETWEEN 0 AND 1),
    r_connectivity FLOAT NOT NULL CHECK (r_connectivity BETWEEN 0 AND 1),
    r_reachability FLOAT NOT NULL CHECK (r_reachability BETWEEN 0 AND 1),
    
    -- Computed R values
    r_multiplicative FLOAT GENERATED ALWAYS AS 
        (r_distance * r_centrality * r_connectivity * r_reachability) STORED,
    
    -- Graph context
    node_id VARCHAR(100) NOT NULL,
    graph_type VARCHAR(50) NOT NULL,  -- 'citation', 'author', 'semantic', 'filesystem'
    topological_distance INTEGER,
    boundary_crossings INTEGER,
    
    -- Temporal dependency
    traversal_time_ms INTEGER NOT NULL,
    time_to_position JSONB,  -- R(T) curve data
    
    -- Path information
    path_length INTEGER,
    path_redundancy FLOAT,
    
    -- Metadata
    measurement_context JSONB,
    
    INDEX idx_r_timestamp (timestamp),
    INDEX idx_r_node (node_id),
    INDEX idx_r_graph_type (graph_type)
);

-- Now we can uncomment the foreign keys in h_dimension_measurements
-- Run this after creating the above tables:
-- ALTER TABLE h_dimension_measurements 
--   ADD COLUMN w_measurement_id UUID REFERENCES w_dimension_measurements(measurement_id),
--   ADD COLUMN r_measurement_id UUID REFERENCES r_dimension_measurements(measurement_id);
```

---

**Version Notes (v2.5):**

This version incorporates final refinements for mathematical consistency and statistical rigor:

**Symbol Disambiguation**:
- **Context steepness**: k → s (avoiding collision with retrieval depth)
- **Composability**: κ → χ (reserving κ for synergy term in Eq. 5)
- **Context weights**: Fixed notation to w_L·L + w_I·I + w_A·A + w_G·G

**Prediction Corrections**:
- **P2/P3/P4**: Parameterized by elasticities θ, φ rather than assuming unit values
- **Exponent sensitivities**: Halving H_trans → C₂/C₁ = 2^(-φ)
- **Dependence bonus**: Δlog C includes log(1 + κ·H_ext·H_int) term

**Statistical Enhancements**:
- **Power statement**: Unified at 80% with N=120 (93% uncorrected moved to appendix)
- **Sample size discussion**: Acknowledged n_eff ≈ 25 limitation, proposed 300-run expansion
- **External validity**: Added 80/20 split between controlled and production-like conditions
- **Model fitting**: Added ridge regression recipe with clustered SEs

**Measurement Refinements**:
- **H_internal**: Removed precision/recall (which are R metrics)
- **TCR_TA**: Added task-adjusted version accounting for applicable tools
- **AMI gaming**: Added H(O) and L distribution monitoring
- **Monte Carlo**: Specified 1000/5000 samples for pilot/final

**Implementation Clarifications**:
- **Mixed-effects model**: lmer(log(C) ~ ... + (1|task_cluster))
- **Time attribution**: Runtime strictly to T, expressivity to H_transfer
- **Reconstruction error**: Acknowledged PCA information loss needs quantification
- **Sequential testing**: Initial 120 runs with adaptive sampling option

**Keywords:** Agency Transfer, MCP Protocol, Black Box Systems, Behavioral Modeling, Distributed Intelligence, Boundary Objects, Information Reconstructionism, Actionable Mutual Information, Protocol Expressivity Index, Zero-Propagation Gate, Tobit Regression, Component Ablation
