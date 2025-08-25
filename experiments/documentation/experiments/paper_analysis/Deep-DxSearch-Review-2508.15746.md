# HADES Paper Review: Deep-DxSearch (ArXiv 2508.15746)

## Paper Metadata

- **Title**: End-to-End Agentic RAG System Training for Traceable Diagnostic Reasoning
- **Authors**: Qiaoyu Zheng, Yuze Sun, Chaoyi Wu, et al.
- **Venue/Year**: ArXiv preprint, August 2025
- **Paper Link**: <https://arxiv.org/abs/2508.15746>
- **Repository**: <https://github.com/MAGIC-AI4Med/Deep-DxSearch>
- **License**: Apache-2.0

## TL;DR Summary

- **Key Claim**: End-to-end reinforcement learning can train agentic RAG systems for traceable medical diagnostic reasoning
- **Core Method**: Frames LLM as agent in retrieval environment with multi-component reward signals (format, retrieval, reasoning, accuracy)
- **HADES Relevance**: Directly applicable to HADES's agentic RAG architecture with strong conveyance through available implementation

## Conveyance Score

### Variable Mapping

- **W (What - Semantic Quality)**: 0.85
  - *Justification*: Strong theoretical foundation with empirical validation across multiple medical datasets
- **R (Where - Relevance)**: 0.90
  - *Justification*: Directly addresses RAG, agents, and reasoning - core HADES modules
- **H (Who - Implementation Quality)**: 0.80
  - *Justification*: Open-source implementation with documentation, though medical-specific adaptations needed
- **T (Time/Cost)**: 1.5
  - *Justification*: Requires GPU training but feasible on single A6000-class hardware

### Context Components

- **L (Language)**: 0.85 - Clear technical writing with structured presentation
- **I (Intent)**: 0.75 - Medical focus requires adaptation but core principles align with HADES
- **A (Actionability)**: 0.90 - GitHub repository with step-by-step implementation
- **G (Grounding)**: 0.85 - Code available, environment specs provided, active repository

### Calculation

```
Ctx = 0.25×0.85 + 0.25×0.75 + 0.25×0.90 + 0.25×0.85 = 0.8375
C = (0.85 × 0.90 × 0.80 / 1.5) × 0.8375^1.67
C = (0.612 / 1.5) × 0.735
C = 0.408 × 0.735 = 0.30
```

**Final Conveyance Score: C = 0.30**

### Zero Gate Check

- W = 0.85 > 0 ✓
- R = 0.90 > 0 ✓
- H = 0.80 > 0 ✓
- T = 1.5 (finite) ✓
- **No zero gates triggered**

## Technical Summary

The paper introduces Deep-DxSearch, an end-to-end trainable agentic RAG system specifically designed for medical diagnostic reasoning. The key innovation is framing the LLM as a reinforcement learning agent operating within a retrieval environment, where the agent learns to perform traceable retrieval-augmented reasoning through multi-component reward signals. The system constructs a large-scale medical retrieval corpus and trains the agent using rewards for format adherence, retrieval quality, reasoning structure, and diagnostic accuracy. Empirical evaluation demonstrates consistent improvements over prompt-engineering approaches, training-free RAG methods, and strong baselines like GPT-4o and DeepSeek-R1, with particular strength in handling both common and rare disease diagnoses. The approach provides a blueprint for training domain-specific agentic RAG systems with traceable reasoning paths.

## Contextual Placement (R = 0.90)

### Nearest Prior Work

- **RAG Systems**: Extends traditional RAG with agent-based architecture and RL training
- **Medical AI**: Builds on diagnostic AI systems but adds traceable reasoning
- **Agent Training**: Advances beyond prompt engineering to end-to-end RL optimization

### True Delta/Innovation

- **End-to-end RL training** for RAG agents (not just prompting)
- **Multi-component reward design** balancing format, retrieval, reasoning, and accuracy
- **Traceable reasoning paths** enabling diagnostic transparency

### Composability with HADES Modules

- **RAG Module**: Direct enhancement through agent-based retrieval
- **Agents Module**: RL training methodology transferable to HADES agents
- **GraphSAGE Integration**: Could enhance retrieval corpus with graph-structured medical knowledge
- **Embedding Layer**: Compatible with HADES's Jina v4 embeddings for corpus representation

## Evidence Quality (W = 0.85, H = 0.80)

### Datasets Used

- MIMIC-IV-note (clinical notes)
- PMC-Patients (patient cases)
- MedDialog (medical conversations)
- RareArena (rare disease cases)

### Baselines Compared

- GPT-4o
- DeepSeek-R1
- Training-free RAG methods
- Prompt-engineering approaches

### Ablation Studies

- Reward component analysis
- Retrieval corpus impact
- Model size effects (7B vs 14B)

### Statistical Significance

- Multiple evaluation metrics reported
- Consistent improvements across datasets
- Both in-distribution and out-of-distribution testing

### Documented Failure Modes

- Limited discussion of failure cases
- Medical domain specificity may limit generalization

## Reproducibility & Cost (T = 1.5, G = 0.85)

### Code/Data Availability

- **GitHub**: <https://github.com/MAGIC-AI4Med/Deep-DxSearch> (Apache-2.0)
- **Models**: Qwen2.5 series (7B, 14B) available
- **Datasets**: Public medical datasets listed with access instructions

### Environment Specifications

- Python 3.10
- CUDA ≥12.1
- PyTorch 2.4.0
- Flash Attention, Transformers
- Optional: Faiss-GPU, FastAPI, SGLang

### Compute Footprint

- GPU-based training required
- Recommends high-memory GPUs (A6000-class)
- Batch size and model parallelism adjustable

### Expected Integration Effort

- 2-3 weeks for domain adaptation from medical to general
- 1 week for basic integration with existing RAG system
- Additional time for RL training infrastructure setup

### Time-to-First-Result Estimate

- 2-3 days for environment setup and initial testing
- 1 week for basic RAG enhancement
- 2-4 weeks for full RL training implementation

## Actionability Checklist (A = 0.90)

- [x] Clear algorithm description - RL training loop well-specified
- [x] Pseudocode or equations - Reward formulations provided
- [x] Implementation details - GitHub repository with working code
- [x] Hyperparameters specified - Training configurations documented
- [x] Evaluation protocol - Multiple datasets and metrics described

**All critical elements present: A = 0.90 justified**

## Risk Assessment

### Internal Validity Threats

- **Medical domain bias**: Results may not generalize beyond healthcare
- **Reward engineering**: Heavy dependence on hand-crafted reward components
- **Data leakage**: Potential overlap between training corpus and evaluation

### External Validity Concerns

- **Domain specificity**: Medical focus requires significant adaptation for HADES
- **Computational requirements**: RL training adds complexity and compute needs
- **Corpus construction**: Quality heavily dependent on retrieval corpus

### Ethical Considerations

- Medical AI requires careful validation before deployment
- Transparency in diagnostic reasoning addresses some concerns
- Need to ensure diverse training data to avoid bias

### Proposed Mitigations

1. Start with small-scale pilot on HADES-specific domain
2. Adapt reward structure for code/paper understanding tasks
3. Leverage existing HADES corpus instead of building from scratch
4. Implement gradual rollout with human oversight

## HADES Integration Plan

### Module Placement

- **Primary**: RAG Module enhancement with agent-based retrieval
- **Secondary**: Agents Module for RL training infrastructure
- **Supporting**: Embedding layer for corpus representation

### MVE Steps (Minimum Viable Enhancement)

1. **Extract core RL training loop** from Deep-DxSearch repository
2. **Adapt reward functions** for code/paper understanding domain
3. **Integrate with existing HADES RAG** using ArXiv corpus subset
4. **Implement traceable reasoning paths** for paper-to-code bridges
5. **Benchmark against current HADES RAG** performance

### Key Metrics to Track

- Retrieval precision/recall
- Reasoning path quality (human evaluation)
- End-to-end task completion rate
- Training convergence speed
- Inference latency impact

### Success Thresholds

- 15% improvement in retrieval relevance
- 80% human-rated reasoning path quality
- <2x inference latency increase
- Convergence within 100K training steps

### Owner and Timeline Recommendation

- **Owner**: RAG/Agent team collaboration
- **Phase 1** (2 weeks): Environment setup and code adaptation
- **Phase 2** (3 weeks): Reward engineering and corpus preparation
- **Phase 3** (2 weeks): Training and initial evaluation
- **Phase 4** (1 week): Integration testing and benchmarking
- **Total Timeline**: 8 weeks to production-ready enhancement

## Decision Block

### Priority: **MEDIUM**

### Verdict: **ADAPT**

### Justification

Strong technical approach with available implementation, but requires significant domain adaptation from medical to code/paper understanding. The RL training methodology and traceable reasoning are valuable additions to HADES.

### Required Follow-ups

1. Conduct feasibility study on reward function adaptation
2. Prototype with small ArXiv subset (1000 papers)
3. Evaluate computational cost-benefit tradeoff
4. Design evaluation protocol for code/paper domain
5. Establish baseline metrics before implementation

## Additional Analysis for HADES

### Theoretical Alignment with Information Reconstructionism

The paper's approach aligns with HADES's conveyance framework in several key ways:

1. **Multi-dimensional Information**: The reward structure (format, retrieval, reasoning, accuracy) maps to HADES's dimensional model
2. **Context Amplification**: The RL training discovers optimal context usage patterns (α parameter)
3. **Observer Effect**: The agent creates reality through retrieval choices (FRAME boundary crossing)
4. **Zero Propagation**: Poor retrieval (WHERE=0) or reasoning (CONVEYANCE=0) leads to diagnostic failure

### Specific HADES Advantages

1. **Existing Infrastructure**: HADES already has:
   - ArangoDB for graph-structured retrieval corpus
   - Jina v4 embeddings for semantic representation
   - ACID pipeline for processing papers

2. **Domain Advantages**: Code/paper domain may be easier than medical:
   - More structured content (code has syntax)
   - Clearer correctness criteria (code runs or doesn't)
   - Existing evaluation benchmarks

3. **Synergistic Enhancements**:
   - GraphSAGE could improve retrieval through citation networks
   - HERMES could preprocess papers for better corpus quality
   - Ladon could monitor RL training metrics

### Implementation Risks Specific to HADES

1. **Corpus Quality**: HADES's 2.79M ArXiv papers may need curation
2. **Reward Engineering**: Defining "correct" for research papers is harder than medical diagnosis
3. **Computational Scale**: RL training on full corpus may be prohibitive
4. **Evaluation Challenges**: No clear ground truth for paper understanding

### Recommended Pilot Approach

1. **Start Small**: Use 10K paper subset with known theory-practice bridges
2. **Simple Rewards**: Begin with binary retrieval relevance and code executability
3. **Incremental Complexity**: Add reasoning and format rewards after baseline
4. **Human-in-the-Loop**: Use expert annotations for reward signal refinement

## Conclusion

Deep-DxSearch presents a compelling approach to agentic RAG that could significantly enhance HADES's capability to bridge theory and practice. While the medical domain focus requires adaptation, the core RL training methodology and traceable reasoning framework are directly applicable. The conveyance score of 0.30 reflects moderate actionability given the domain translation effort required, but the strong technical foundation (W=0.85) and high relevance (R=0.90) justify investment in adaptation. The ADAPT verdict with MEDIUM priority reflects the balance between potential value and implementation complexity. Success depends on effective reward engineering for the code/paper domain and careful management of computational costs during RL training.
