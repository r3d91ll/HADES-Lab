# Conveyance Paper-Review Checklist 

please review the following paper {arxiv_link_here}. use the template posted below for the structure of your review. Provide frank and honest critique of the highest academic caliber

---

## 0) Core Definitions (authoritative)

**Efficiency view (default):**

```math
C = \frac{W \cdot R \cdot H}{T} \cdot \mathrm{Ctx}^{\alpha}
```

* **W**: What (signal/content quality)
* **R**: Where (relational/topological positioning)
* **H**: Who (agent/model capability & access patterns)
* **T**: Time to converge (latency/cost)
* **Ctx**: Context quality = `wL·L + wI·I + wA·A + wG·G`

  * **L**: Local coherence, **I**: Instruction fit, **A**: Actionability, **G**: Grounding
* **α** ∈ \[1.5, 2.0] — super‑linear amplification from context quality only.

**Capability view (fixed/controlled time):**

```math
C_{cap} = (W \cdot R \cdot H) \cdot \mathrm{Ctx}^{\alpha}
```

**Self‑optimization (solve for H):**

```math
H = \frac{C_{target} \cdot T}{W \cdot R \cdot \mathrm{Ctx}^{\alpha}}
```

**Zero‑propagation gate:** if any of {W, R, H} = 0 or T → ∞ ⇒ C = 0.

> **When to use which view**: Use **efficiency** when latency matters; use **capability** for apples‑to‑apples comparisons at a fixed time budget.

---

## 1) Rapid Mapping Checklist

Use this table to map a paper’s variables to the framework. Fill one row per item.

| Framework                      | What in this paper? | Notes / how measured |
| ------------------------------ | ------------------- | -------------------- |
| **C** (primary outcome)        |                     |                      |
| **W** (content/quality)        |                     |                      |
| **R** (relational/bridging)    |                     |                      |
| **H** (agent/model capability) |                     |                      |
| **T** (time/steps/latency)     |                     |                      |
| **L** (local coherence)        |                     |                      |
| **I** (instruction fit)        |                     |                      |
| **A** (actionability)          |                     |                      |
| **G** (grounding)              |                     |                      |

> Compute `Ctx = wL·L + wI·I + wA·A + wG·G` (defaults 0.25 each unless the paper implies other weights).

---

## 2) Amplification (α) Identifiability

* **Question:** Does the paper vary **Ctx** while holding **W/R/H/T** fixed or measured?
* **If YES**: estimate α using log–log slope

```text
α̂ = Δ log(C) / Δ log(Ctx)
```

* Prefer within‑item contrasts (same task, different context treatments).

* Report CI (bootstrap or regression SEs) and whether time was fixed or controlled.

* **If NO**: explain the confounders (e.g., time co‑varies with context; model changes; different retrieval graphs).

---

## 3) Expertise Sharing → **R** Evidence

Summarize how the paper quantifies cross‑silo transfer or graph‑distance collapse (e.g., convergence across roles, cross‑domain competency, community bridging, neighborhood mixing). State clearly why this is **R↑**.

---

## 4) AI Capabilities → **H** Inventory

List all model/agent knobs:

* Model/version/checkpoint; decoding; temperature; max tokens; tools/plugins; RAG; halting/steps; training or prompt‑training; usage intensity (#prompts, retention); planning vs execution modules; controller policies.
* Mark which knobs were **manipulated** vs **held constant**.

---

## 5) Time Stance & Results

Produce both, if possible:

* **Efficiency result (uses T):**

  * Equation: `C = (W·R·H / T) · Ctx^α`
  * Summary: <insert short interpretation>

* **Capability result (fixed T):**

  * Equation: `C_cap = (W·R·H) · Ctx^α`
  * Summary: <insert short interpretation>

* **Confounds/Notes:** \<e.g., time inside scaffolding cost, retrieval latency, outer‑loop effects>

# How to treat time vs compute (within the framework)

* Keep the core metric:

  ```
  C_time = (W · R · H / T_latency) · Ctx^α
  ```

* Also report a **compute-efficiency** companion:

  ```
  C_comp = (W · R · H / B_compute) · Ctx^α
  ```

  where **B\_compute** can be FLOPs, tokens×passes, or another architecture-neutral budget.
  (α still applies only to **Ctx**.)

* If you must collapse to one denominator, define a **composite cost**:

  ```
  T_eff = w_t·T_latency + w_f·(FLOPs) + w_e·(energy) + w_$·(dollars)
  C = (W · R · H / T_eff) · Ctx^α
  ```

  but this hides trade-offs; I prefer reporting **both C\_time and C\_comp** and presenting a Pareto frontier.

---

## 6) Reporting Template (copy‑fill)

```
# Mapping to Conveyance (Corrected)

## Core equation (efficiency view)
C = (W·R·H / T) · Ctx^α ,   Ctx = wL·L + wI·I + wA·A + wG·G ,  α∈[1.5,2.0]

## 1) Metric Mapping
- C: <paper primary outcome>
- W: <content/quality metrics>
- R: <bridging/graph position>
- H: <model/agent capability + usage>
- T: <time/latency/steps>
- L/I/A/G: <measures enabling Ctx>

## 2) Amplification (α)
Identifiable? <yes/no + reason>
If yes: α̂ = <value> (95% CI: <..>)  — method: ΔlogC / ΔlogCtx (controls: W,R,H,T)

## 3) Expertise Sharing → R
Evidence: <how measured> ⇒ R↑

## 4) AI Capabilities → H
List: <models, tools, training, usage intensity, planning/halting, etc.>

## 5) Time stance
- Efficiency result (uses T): <summary>
- Capability result (fixed T): <summary>
Confounds/notes: <...>
```

---

## 7) Notes on Using α

* Interpret α as the **non‑linear lift from context quality** only (do not exponentiate time).
* Typical operating bands: light 1.5, standard 1.7, heavy learning 1.8–2.0, critical 2.0.
* For time‑sweeps (anytime inference), keep **Ctx fixed** and analyze `∂ log C / ∂ log T` separately to avoid double‑counting.

---

## 8) Minimal Logging (so every review is estimable)

For each result or ablation:

* **Outcome**: C (EM/F1/pass\@k/quality)
* **Explanatory**: W, R, H (or κ₀ = W·R·H), L, I, A, G (→ Ctx), T
* **Protocol**: model/decoding, retrieval policy, steps/halting, dataset split

---




### Version

* v1.0 (corrected time handling; α applies to Ctx only)