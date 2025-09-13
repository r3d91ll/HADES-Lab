# Master Thesis Ideas

> A living document for capturing potential thesis topics, paper ideas, and research directions emerging from the HADES-Lab work.

---

## Thesis Idea 1: Dual Graph Analysis of Knowledge Production

### Working Title

"The Epistemic-Social Gap: Comparing Knowledge Structure and Social Structure in Scientific Production Through Complementary Graph Topologies"

### Core Argument

The divergence between paper-centric (epistemic) and author-centric (social) network structures reveals critical sites of innovation, interdisciplinary breakthrough, and paradigm formation in scientific knowledge production.

### Research Questions

1. How do knowledge structures (topic/method based) differ from social structures (collaboration based) in scientific production?
2. Where do these structures diverge most significantly, and what does this reveal about innovation?
3. Can we identify "bridge papers" that connect communities epistemically but not socially (or vice versa)?
4. How do these patterns vary across disciplines and over time?

### Methodology

**Phase 1: Paper-Centric Graph (Current Work)**

- Nodes: 2.8M arXiv papers
- Edges: Temporal proximity, category similarity, keyword overlap
- "Death of the author" approach - treats papers as autonomous knowledge units
- GraphSAGE embeddings to learn latent knowledge structure

**Phase 2: Author-Centric Graph (Future Work)**

- Nodes: Disambiguated authors
- Edges: Co-authorship, citation patterns, institutional affiliations
- Social network analysis of collaboration patterns
- Community detection to identify "invisible colleges"

**Phase 3: Comparative Analysis**

- Identify papers/authors that bridge in one graph but not the other
- Measure "epistemic-social divergence" scores
- Temporal analysis of how gaps open/close over time
- Case studies of high-divergence areas

### Theoretical Framework

- Actor-Network Theory (ANT) - treating papers as actants
- Sociology of Scientific Knowledge (SSK)
- Philosophy of Science (Kuhn, Lakatos on paradigm shifts)
- Information Theory (conveyance framework from HADES)

### Expected Contributions

1. **Methodological**: New framework for dual-graph analysis of knowledge production
2. **Empirical**: Map of epistemic-social gaps across scientific disciplines
3. **Theoretical**: Understanding of how social and knowledge structures co-evolve
4. **Practical**: Identify emerging interdisciplinary opportunities

### Potential Funding Sources

- NSF Science of Science and Innovation Policy (SciSIP)
- NSF Social, Behavioral and Economic Sciences (SBE)
- Sloan Foundation - Science of Science
- European Research Council (ERC) - Social Sciences and Humanities

### Key Literature to Engage

- Newman, M. (2001). "The structure of scientific collaboration networks"
- Uzzi et al. (2013). "Atypical combinations and scientific impact"
- Evans & Foster (2011). "Metaknowledge"
- Fortunato et al. (2018). "Science of science"
- Sinatra et al. (2016). "Quantifying the evolution of individual scientific impact"

### Potential Criticisms & Responses

**Criticism**: "You're ignoring the social dimension of knowledge production!"
**Response**: "Phase 2 explicitly addresses this through author disambiguation and social network analysis. The comparison between phases is the key contribution."

**Criticism**: "Author disambiguation at this scale is intractable"
**Response**: "Modern techniques (deep learning, institutional affiliation matching) make this feasible. Even 80% accuracy would reveal meaningful patterns."

### Related Paper Ideas

1. "Temporal Dynamics of Epistemic Communities in Machine Learning" (focused study on cs.LG)
2. "Keywords as Boundary Objects: How Terminology Bridges Disciplines"
3. "The Death of the Author in Scientific Publishing: A Network Analysis"

Absolutely! Let’s outline that first article so you have a clear structure to build from.

---

## Article Outline: **“Death of the Author: Framing Scale and the First Layer of Analysis”**

1. **Introduction: Setting the Stage**

   - Open with the concept of “death of the author” as a way to strip away all author identifiers and focus purely on documents as independent agents.
   - Briefly explain why this approach is necessary: the sheer complexity of disambiguating names like “Yang” or “Smith” in a dataset of millions of papers.

2. **The Scale of the Dataset**

   - Describe the initial layer: just abstracts and metadata.
   - Provide a concrete example: show how large a single document object is at this minimal level, and multiply that by 3 million to give a sense of the total size.
   - Mention the JSON file size and give a download link for readers to see for themselves.

3. **Scaling Up: Adding Full Text and Embeddings**

   - Transition to the next layer of complexity: adding the full text of the paper and embedding it.
   - Explain how this drastically increases the size of each document object and what that means in terms of total storage and compute.
   - Again, offer a sample download link so readers can compare the “light” and “heavy” versions of the same document.

4. **The Hardware and Environment**

   - Lay out the hardware environment needed: describe the workstation, the GPUs, and the reasoning behind needing something like the Blackwell A6000 Pro for future work.
   - Without explicitly naming the conveyance framework, structure the explanation around the “who, what, where, when” of the project so that it naturally flows from the problem to the solution.

5. **Conclusion: Setting Up the Next Steps**

   - Wrap up by hinting at the next phase: once the dataset is fully embedded and local, that’s when you can start the deeper analysis and eventually bring the authors back into the picture for the next layer of the project.

---

## Thesis Idea 2: [Title]

### Working Title

### Core Argument

### Research Questions

### Methodology

### Theoretical Framework

### Expected Contributions

### Potential Funding Sources

### Key Literature to Engage

### Potential Criticisms & Responses

### Related Paper Ideas

---

## Thesis Idea 3: [Title]

[Template continues...]

---

## Quick Paper Ideas (Undeveloped)

### From Technical Work

- [ ] "GraphSAGE at Scale: Lessons from 150M Edge Academic Graphs"
- [ ] "Conveyance Theory Applied to Scientific Knowledge Diffusion"
- [ ] "Memory-Efficient Pipeline Parallelism for Graph Neural Networks"

### From Theoretical Insights

- [ ] "Post-Structuralist Approaches to Citation Networks"
- [ ] "Information Reconstructionism as Research Methodology"
- [ ] "The Anthropology of ArXiv: Cultural Patterns in Preprint Posting"

### From Methodological Innovations

- [ ] "Late Chunking for Academic Document Embeddings"
- [ ] "Temporal Graph Construction from Publication Metadata"
- [ ] "Multi-Modal Knowledge Graphs: Combining Text, Code, and Mathematics"

---

## Grant Proposal Ideas

### Small Grants (< $50K)

- AWS Cloud Credits for Author Disambiguation Pipeline
- Digital Humanities seed grant for visualization tools
- University internal grant for GPU cluster time

### Medium Grants ($50K - $500K)

- NSF CISE IIS Small: "Scalable Dual-Graph Analysis for Science of Science"
- Sloan Foundation: "Mapping Interdisciplinary Knowledge Transfer"
- Chan Zuckerberg Initiative: "Open Infrastructure for Scientific Collaboration Analysis"

### Large Grants (> $500K)

- NSF CAREER: "Understanding Knowledge Production Through Multi-Layer Networks"
- ERC Starting Grant: "EPISOC - Epistemic and Social Structures in Science"
- DARPA?: "Predictive Models of Scientific Innovation" (check eligibility)

---

## Notes on Timing

- **Year 1**: Complete paper-centric analysis, write methodology paper
- **Year 2**: Implement author disambiguation, begin comparative analysis
- **Year 3**: Full comparative study, write thesis
- **Publications Timeline**:
  - Conference paper at ISSI or Science of Science Conference (Year 1)
  - Journal article in Scientometrics or JASIST (Year 2)
  - Thesis defense (Year 3)

---

## Resources Needed

### Technical

- [ ] Access to full arXiv corpus with metadata
- [ ] GPU cluster for training (currently have 2x A6000)
- [ ] Storage for graph data (currently ~2TB)
- [ ] Author disambiguation service/API

### Collaborative

- [ ] Committee member with STS/Philosophy of Science background
- [ ] Potential co-authors with domain expertise
- [ ] Industry partner for validation (Google Scholar? Semantic Scholar?)

### Data

- [ ] ORCID database for author validation
- [ ] Institutional affiliation databases
- [ ] Conference/journal metadata for venue analysis

---

*Last Updated: 2025-09-10*
*Next Review: After current GraphSAGE training completes*
