# GEARS Spatio‑Temporal Implementation Spec (v1.0)

**Date:** 2025‑08‑12
**Owner:** HADES / PathRAG team
**Audience:** Build engineers & researchers
**Goal:** Turn the math from GEARS‑style canonicalization + dual attention into a production‑ready module with unit tests, ablations, and hooks to the Conveyance framework.

---

## 1) Scope & Outcomes

**Deliverables**

* PyTorch reference implementation of a spatio‑temporal model for articulated hands (generalizable to other kinematic graphs).
* Data pipeline for canonicalization and template‑frame features.
* Dual attention (spatial across joints; temporal across frames) with gated fusion.
* Displacement head + FK (forward kinematics) update loop.
* Full training loss (init, displacement, pose regularizers, temporal smoothness, acceleration).
* Invariance unit tests, benchmarks, ablation harness, and ConveyanceScorer integration.

**Non‑goals**

* Object‑specific contact dynamics and differentiable physics (future work).

---

## 2) Notation & Conventions

* Rotations: $R\in SO(3)$, translations $t\in\mathbb{R}^3$, homogeneous transform $T=\begin{bmatrix}R & t\\ 0 & 1\end{bmatrix}$.
* Points: $p\in\mathbb{R}^3$, point clouds $P^t\in\mathbb{R}^{N\times 3}$, normals $N^t\in\mathbb{R}^{N\times 3}$.
* Joints: index set $\mathcal{J}$, joint position $j_k^t\in\mathbb{R}^3$, parent function $pa(k)$, ancestor set $A(k)$.
* Feature dimension $d$; attention head dimension $l$ (typically $l=d/h$ for $h$ heads).
* Time steps $t\in\{1,\dots,T\}$.

**Shape conventions** (per batch):

* Canonicalized points: $\tilde{P}^t\in\mathbb{R}^{N\times 3}$, normals $\tilde{N}^t\in\mathbb{R}^{N\times 3}$.
* Per‑joint feature tensor per frame: $X_S^t\in\mathbb{R}^{|\mathcal{J}|\times d}$.
* Per‑joint temporal sequence: $X_T^k\in\mathbb{R}^{T\times d}$.

---

## 3) Kinematics & Canonicalization

### 3.1 Canonical frame transformation

Given global hand pose at time $t$: rotation $R_H^t$ and reference position $w^t$, define

$$
\tilde{P}^t = (R_H^t)^\top\,(P^t - \mathbf{1}\, (w^t)^\top)\,.
$$

This is invariant to global hand rotation/translation.

### 3.2 Template (kinematic) frame

Define local joint transforms as homogeneous matrices

$$
\mathcal{T}_k = \prod_{i\in A(k)} \begin{bmatrix} R_{i,pa(i)} & t_i \\ 0 & 1 \end{bmatrix},
\quad R_{i,pa(i)}\in SO(3),\; t_i\in\mathbb{R}^3.
$$

Let $R_k=\mathrm{rot}(\mathcal{T}_k)$ and $o_k=\mathrm{trans}(\mathcal{T}_k)$.

### 3.3 Feature‑space transform (positions vs normals)

Positions are rotated into joint frames and centered at joint origins; normals rotate only:

$$
\bar{p}_{k} = R_k^\top (p - j_k), \qquad \bar{n}_{k} = R_k^\top n.
$$

Aggregate per‑joint features as concatenation

$$
\bar{F}_k = [\,\bar{p}_{k} \;\Vert\; \bar{n}_{k}\,] \in \mathbb{R}^{6} \quad (\text{or projected to } d_0).
$$

### 3.4 Joint embedding

Embed joint coordinates in a consistent root frame (not the joint’s own frame), e.g.

$$
 e_k = g_{\text{embed}}\big(\mathcal{T}_{\text{root}}^{-1}[\,j_k;1\,]\big)\in\mathbb{R}^{d_e}.
$$

Final per‑joint input feature: $x_k = \phi([\bar{F}_k\,\Vert\, e_k])\in\mathbb{R}^d$.

---

## 4) Dual Attention & Fusion

### 4.1 Spatial self‑attention (within a frame)

For frame $t$ with joint features $X_S^t\in\mathbb{R}^{J\times d}$:

$$
\tilde{X}_S^t = \mathrm{softmax}\!\left(\frac{Q_S^t (K_S^t)^\top}{\sqrt{l}}\right) V_S^t,
\quad Q_S^t=X_S^t W_Q,\; K_S^t=X_S^t W_K,\; V_S^t=X_S^t W_V.
$$

Mask non‑existent joints with $-\infty$ before softmax.

### 4.2 Temporal self‑attention (across frames per joint)

For joint $k$ with sequence $X_T^k\in\mathbb{R}^{T\times d}$:

$$
\tilde{X}_T^k = \mathrm{softmax}\!\left(\frac{Q_T^k (K_T^k)^\top}{\sqrt{l}}\right) V_T^k.
$$

### 4.3 Fusion

Per joint & frame, fuse spatial and temporal contexts:

$$
Y_{k}^t = \sigma(\gamma)\,\tilde{X}_{S,k}^t + (1-\sigma(\gamma))\,\tilde{X}_{T,k}^t,
$$

where $\gamma$ can be a learned scalar, per‑joint, or produced by a small gating MLP on $[\tilde{X}_{S,k}^t \Vert \tilde{X}_{T,k}^t]$.

**Note on dynamic graphs:** softmax couples all nodes. For huge/dynamic graphs consider local softmax (windowed neighborhoods), sparse attention, or $\alpha$‑entmax; see §11.

---

## 5) Displacement Head, Back‑Transform, and FK

### 5.1 Local displacement prediction

Head predicts local displacements in joint frame:

$$
\bar{d}_k^t = h(Y_k^t) \in \mathbb{R}^3.
$$

### 5.2 Back‑transform to root/global

Let the rotation chain from root to $k$ be

$$
R_{k\leftarrow \text{root}}^t = \prod_{i\in A(k)} R_{i,pa(i)}^t,\qquad d_k^t = (R_{k\leftarrow \text{root}}^t)\,\bar{d}_k^t.
$$

### 5.3 Forward kinematics update

Recursive FK with local offsets $\bar{t}_k$ (bone vectors in parent frames):

$$
 j_k^t = j_{pa(k)}^t + R_{k,pa(k)}^t\,\bar{t}_k + d_k^t,\quad j_{\text{root}}^t = o_{\text{root}}^t.
$$

---

## 6) Losses & Objective

### 6.1 Initialization & displacement losses

$$
 L_{\text{init}} = \sum_{k}\| j_{k,\text{init}} - j_{k,\text{gt}} \|_2^2,\qquad
 L_{\text{disp}} = \sum_{t,k}\| j_{k,\text{disp}}^t + d_k^t - j_{k,\text{gt}}^t \|_2^2.
$$

### 6.2 Pose & temporal regularizers

$$
 \mathcal{L}_{\text{reg}} = w_1\|\beta\|_2^2 + w_2\sum_t\|\theta^t\|_2^2
 + w_3\sum_{t=1}^{T-1} \|\theta^{t+1}-\theta^t\|_2^2
 + w_4\sum_{t=2}^{T-1} \sum_{i\in\mathcal{J}} \|\ddot{j}_i^t\|_2^2.
$$

Finite differences: $\dot{j}^t=j^{t+1}-j^t$, $\ddot{j}^t=j^{t+1}-2j^t+j^{t-1}$.

### 6.3 Overall objective

$$
\mathcal{L}(\beta,\theta) = \sum_{t,k} \|J_k(H(\beta,\theta^t)) - j_k^t\|_2^2 + \mathcal{L}_{\text{reg}}(\beta,\theta).
$$

### 6.4 Metrics

Report MPJPE (mm), PA‑MPJPE (Procrustes‑aligned), and velocity/acceleration errors.

---

## 7) Interpolation for Data Synthesis

* **Linear** for positions: $p(t)=(1-t)p^0+tp^T$.
* **SLERP** for rotations using unit quaternions $q$: $q(t)=\operatorname{SLERP}(q^0,q^T,t)$, then $R(t)=\mathcal{R}(q(t))$.

---

## 8) Expected Empirical Patterns

* Canonicalization + dual attention generally reduces error vs. raw global inputs.
* Fusion yields superlinear gains vs. either attention in isolation.
* Shared per‑joint processing improves generalization.

(Ablations will quantify these; see §13.)

---

## 9) Implementation Blueprint (PyTorch)

### 9.1 Data structures

```python
class KinematicTree(NamedTuple):
    parents: List[int]          # len J, -1 for root
    local_offsets: Tensor       # (J, 3)  \bar{t}_k

class FrameInputs(NamedTuple):
    P: Tensor       # (N, 3)
    N: Tensor       # (N, 3)
    joints: Tensor  # (J, 3)  j_k^t
    R_H: Tensor     # (3, 3)
    w: Tensor       # (3,)
```

### 9.2 Modules (signatures)

```python
class Canonicalizer(nn.Module):
    def forward(self, P, N, R_H, w): ...  # (N,3),(N,3),(3,3),(3,) -> (N,3),(N,3)

class TemplateProjector(nn.Module):
    def forward(self, P_c, N_c, joints, T_list): ...  # -> per-joint features (J,d)

class SpatialAttention(nn.Module):
    def forward(self, X_S): ...  # (B,T,J,d) -> (B,T,J,d)

class TemporalAttention(nn.Module):
    def forward(self, X_T): ...  # (B,J,T,d) -> (B,J,T,d)

class FusionGate(nn.Module):
    def forward(self, Xs, Xt): ...  # -> Y (B,T,J,d)

class DisplacementHead(nn.Module):
    def forward(self, Y): ...  # -> dbar (B,T,J,3)

class ForwardKinematics(nn.Module):
    def forward(self, dbar, kin): ...  # -> joints_pred (B,T,J,3)
```

### 9.3 End‑to‑end pseudocode

```python
P_c, N_c = Canonicalizer()(P, N, R_H, w)
X_S = TemplateProjector()(P_c, N_c, joints, T_list)   # (B,T,J,d)
Xs = SpatialAttention()(X_S)                          # (B,T,J,d)
Xt = TemporalAttention()(rearrange(X_S,'b t j d -> b j t d'))
Xt = rearrange(Xt,'b j t d -> b t j d')
Y  = FusionGate()(Xs, Xt)
dbar = DisplacementHead()(Y)                          # (B,T,J,3)
j_pred = ForwardKinematics()(dbar, kin)
loss = criterion(j_pred, j_gt, theta, beta)
```

---

## 10) Numeric Stability & Invariances

* Use log‑sum‑exp softmax; mask padding joints/frames.
* Normalize normals; handle degenerate normals with fallbacks.
* **Unit tests**:

  1. **Translation invariance**: add constant to inputs → MPJPE unchanged.
  2. **Rotation covariance**: premultiply inputs & GT by same $R$ → predictions rotate accordingly.
  3. **Bone length stability**: $\|j_k-j_{pa(k)}\|$ within tolerance.
  4. **Quaternion continuity**: enforce hemisphere alignment before SLERP.

---

## 11) Large / Dynamic Graph Options

* **Local softmax**: windowed neighborhoods; block‑diag attention.
* **Sparse attention**: kNN over kinematic distance.
* **$\alpha$\*\*\*\*\*\*\*\*\*\*\*\*-entmax** for sparse probabilities (softmax alternative).
* **Hierarchical pooling**: per‑finger clusters → hand‑level attention.

---

## 12) Training Recipe (reference)

* Optimizer: AdamW (lr=3e‑4, wd=0.05).
* Scheduler: cosine decay with warmup (5% of steps).
* Dropout: 0.1 in attention & MLPs.
* Pose repr: axis‑angle or 6D; keep quats for SLERP utilities.
* Mixed precision: bf16 preferred.
* Augmentations: small global rotations, Gaussian jitter to points, occasional occlusions.

---

## 13) Ablation Plan

1. Remove canonicalization.
2. Spatial‑only; temporal‑only; both.
3. No joint embedding $e_k$.
4. Fusion: sum vs gated.
5. Local vs global attention.
6. Displacement head vs direct positions.

Report MPJPE/PA‑MPJPE, velocity/accel errors, params/FLOPs.

---

## 14) Conveyance Framework Hook

**Core relation**

`(What × Where × Who) / Time = Context^α`

**Variable mapping (this module → Conveyance terms)**

* **What**: Canonicalized sensory signals + per‑joint embeddings x\_k.
* **Where**: Kinematic/graph structure (ancestor sets A(k), transforms T\_k).
* **Who**: Model capacity/config (heads, feature dim d, fusion gate params, iteration budget).
* **Time**: Temporal compute/latency budget (temporal attention depth, smoothing horizon).
* **Context^α**: The **result** of the interaction above — not a multiplier here. It captures effective context amplification; α is fitted/held stable per deployment.

**Operating modes**

1. **Query mode (solve for Context^α)** `Context^α = (What × Where × Who) / Time`

   Implementation: measure proxies for What/Where/Who/Time on each query batch; emit the computed Context^α alongside predictions.

2. **Sleep‑cycle mode (solve for Who; Context^α is configured)** Choose a target `Context^α_target ∈ [1.5, 20]`. Then `Who = (Context^α_target × Time) / (What × Where)`

   Use this to auto‑tune capacity: adjust number of heads, feature width d, fusion/gating strength, or allowed temporal iterations until the measured Who matches the required value within tolerance.

**Estimating the terms** (reference proxies)

* **What**: normalized signal quality (SNR of points/normals), embedding entropy, coverage.
* **Where**: graph locality/conditioning (avg kinematic distance, cluster conductance, attention sparsity over the tree).
* **Who**: effective capacity = f(params, FLOPs, gate activations) — e.g., Who = FLOPs × utilization × gate openness.
* **Time**: wall‑clock or fixed iteration budget for the temporal stack.

**On α**

* Fit α once from ablations (context richness vs. accuracy), then **freeze** for comparability across runs; log it with model artifacts.

### 14.1 Concrete metrics (v1, deprecated—kept for comparison)

```python
import torch
from datetime import datetime

def compute_what_metric(P_c, N_c, X_S, joints, coverage_radius=0.1):
    """\
    What = signal_quality × embedding_richness × coverage
    P_c: (N,3) canonicalized points
    N_c: (N,3) canonicalized normals (not necessarily unit)
    X_S: (J,d) per‑joint embedded features for the frame
    joints: (J,3) joint positions in canonical/global as used for coverage
    """
    # 1) Signal quality (scale‑normalized). Higher when points vary and normals are unit‑length.
    point_var = torch.var(P_c, dim=0).mean()            # ≥0
    point_score = (point_var / (point_var + 1.0)).clamp(0, 1)
    norms = torch.linalg.vector_norm(N_c, dim=1)
    normal_consistency = (1.0 - (norms - 1.0).abs().mean()).clamp(0, 1)
    signal_quality = point_score * normal_consistency

    # 2) Embedding richness via normalized entropy per joint, then averaged.
    #    Richness ∈ (0,1], higher ⇒ more information content.
    if X_S.ndim == 2:
        J, d = X_S.shape
        probs = torch.softmax(X_S, dim=-1)              # (J,d)
        ent = -(probs * (probs + 1e-8).log()).sum(dim=-1)  # (J,)
        max_ent = torch.log(torch.tensor(float(d)))
        richness = (ent / (max_ent + 1e-8)).mean()
    else:
        # fallback for unexpected shapes
        flat = X_S.flatten()
        probs = torch.softmax(flat, dim=0)
        ent = -(probs * (probs + 1e-8).log()).sum()
        max_ent = torch.log(torch.tensor(float(flat.numel())))
        richness = ent / (max_ent + 1e-8)

    # 3) Coverage: fraction of points within radius around each joint, averaged.
    dists = torch.cdist(joints, P_c)                    # (J,N)
    cov_per_joint = (dists < coverage_radius).float().mean(dim=1)
    coverage = cov_per_joint.mean()

    what = (signal_quality * richness * coverage).clamp(min=0)
    return what


def compute_where_metric(kinematic_tree, attention_weights=None):
    """\
    Where = connectivity × kinematic_locality × attention_focus
    kinematic_tree.parents: list[int] of length J (−1 for root)
    attention_weights: optional attention over joints.
      Accepts shapes: (J,J), (T,J,J), (B,T,J,J); will average appropriately.
    """
    parents = kinematic_tree.parents
    J = len(parents)

    # 1) Connectivity: for a tree this is 1.0; keeps formula general.
    edges = sum(1 for p in parents if p != -1)
    max_edges = max(J - 1, 1)
    connectivity = edges / max_edges

    # 2) Kinematic locality: inverse of average path length over all pairs.
    #    Build parent chains to root for each node once.
    chains = []
    for i in range(J):
        c = [i]
        while c[-1] != -1:
            c.append(parents[c[-1]])
        chains.append(c)

    def path_len(i, j):
        si, sj = set(chains[i]), set(chains[j])
        common = si & sj
        if not common:
            return float('inf')
        # distance = index to LCA along each chain
        # Use total steps (discrete kinematic edges)
        li = chains[i]
        lj = chains[j]
        best = None
        best_sum = 10**9
        for x in common:
            di = li.index(x)
            dj = lj.index(x)
            if di + dj < best_sum:
                best_sum = di + dj
                best = x
        return float(best_sum)

    tot = 0.0
    cnt = 0
    for i in range(J):
        for j in range(i + 1, J):
            d = path_len(i, j)
            if d != float('inf'):
                tot += d
                cnt += 1
    avg_dist = (tot / cnt) if cnt > 0 else 1.0
    kinematic_locality = 1.0 / (1.0 + avg_dist)        # ∈ (0,1]

    # 3) Attention focus: higher when attention is concentrated (low entropy).
    if attention_weights is not None:
        A = attention_weights
        # reduce to (..., J, J)
        if A.ndim == 4:   # (B,T,J,J)
            A = A.mean(dim=(0,1))
        elif A.ndim == 3: # (T,J,J)
            A = A.mean(dim=0)
        # normalize rows to sum to 1 (safety)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        ent = -(A * (A + 1e-8).log()).sum(dim=-1).mean()   # mean row entropy
        max_ent = torch.log(torch.tensor(float(J)))
        attention_focus = 1.0 - (ent / (max_ent + 1e-8))   # ∈ [0,1]
    else:
        attention_focus = torch.tensor(1.0)

    where = (connectivity * kinematic_locality * attention_focus).clamp(min=0)
    return where


def compute_who_metric(model, Y, gate_weights=None, grads_available=False):
    """\
    Who = param_utilization × computational_intensity × gate_activation × feature_utilization
    model: torch.nn.Module
    Y: (B,T,J,d) fused features
    gate_weights: tensor/scalar in [0,1] (openness); optional
    grads_available: set True when called during training *after* backward()
    """
    # 1) Parameter utilization
    total_params = sum(p.numel() for p in model.parameters()) + 1e-8
    if grads_available and any(p.grad is not None for p in model.parameters()):
        active_params = sum((p.grad is not None) * p.numel() for p in model.parameters())
    else:
        active_params = sum(p.requires_grad * p.numel() for p in model.parameters())
    param_util = torch.tensor(active_params / total_params, dtype=torch.float32).clamp(0, 1)

    # 2) Computational intensity (proxy FLOPs)
    B, T, J, d = Y.shape
    spatial_flops = B * T * (J * J * d)
    temporal_flops = B * J * (T * T * d)
    fusion_flops = B * T * J * d * 2
    total_flops = torch.tensor(spatial_flops + temporal_flops + fusion_flops, dtype=torch.float32)
    per_elem = total_flops / max(B * T * J, 1)
    comp_intensity = (per_elem.clamp(min=1.0).log() / 10.0).clamp(0, 1)

    # 3) Gate activation
    if gate_weights is None:
        gate_act = torch.tensor(1.0)
    else:
        g = gate_weights
        if not torch.is_tensor(g):
            g = torch.tensor(float(g))
        gate_act = (2.0 * (g - 0.5).abs()).mean().clamp(0, 1)

    # 4) Feature utilization
    feat_std = Y.std(dim=-1).mean()                      # avg over feature dim
    feat_util = torch.tanh(feat_std * 2.0).clamp(0, 1)

    who = (param_util * comp_intensity * gate_act * feat_util).clamp(min=0)
    return who


class ConveyanceScorer:
    def __init__(self, alpha=1.5):
        self.alpha = float(alpha)

    def compute(self, model, batch, outputs, mode='query'):
        """Return dict: What, Where, Who, Time, Context_alpha, mode, timestamp"""
        P_c = batch['P_canonicalized']      # (B,T,N,3) or (N,3); handle per‑frame outside
        N_c = batch['N_canonicalized']
        joints = batch['joints']            # (B,T,J,3) or (J,3)
        kin = batch['kinematic_tree']
        X_S = outputs['spatial_features']   # (B,T,J,d) or (J,d)
        Y   = outputs['fused_features']     # (B,T,J,d)
        attn = outputs.get('attention_weights')  # optional
        gates = outputs.get('gate_weights')      # optional
        time_s = float(outputs.get('inference_time', 1.0))

        # Assume single frame for metric example; extend via batching loop in trainer.
        if P_c.ndim == 3: P_c = P_c.reshape(-1, P_c.shape[-1])  # (N,3)
        if N_c.ndim == 3: N_c = N_c.reshape(-1, N_c.shape[-1])  # (N,3)
        if joints.ndim == 4: joints = joints.reshape(-1, joints.shape[-2], 3)[0]
        if X_S.ndim == 4: X_S = X_S[0,0]                        # (J,d)
        W_what  = compute_what_metric(P_c, N_c, X_S, joints)
        W_where = compute_where_metric(kin, attn)
        W_who   = compute_who_metric(model, Y if Y.ndim==4 else Y[None,None], gates,
                                     grads_available=outputs.get('grads_available', False))

        if mode == 'query':
            context_alpha = (W_what * W_where * W_who) / max(time_s, 1e-6)
        elif mode == 'sleep':
            context_alpha = torch.tensor(self.alpha)
        else:
            raise ValueError("mode must be 'query' or 'sleep'")

        return {
            'What': float(W_what.item()),
            'Where': float(W_where.item()),
            'Who': float(W_who.item()),
            'Time': time_s,
            'Context_alpha': float(context_alpha.item()),
            'mode': mode,
            'timestamp': datetime.now().isoformat(),
        }
```

### 14.1a Principle: Quantify first, then qualify

To avoid meaning drift, we **define dimensionless, computable metrics first (quantification)** and only then interpret or use them for control (qualification). Each term below is normalized to $[0,1]$ and carries no physical units. Time is normalized by a reference budget.

**Normalization references (config):**

* `time_ref` (s): baseline latency for the task/dataset.
* `flops_ref` (ops): baseline compute cost (e.g., current model’s mean forward FLOPs).
* `eta_cov` (unitless): coverage radius scale vs. median bone length.

### 14.1b Quantified metrics (v2, dimensionless)

```python
import torch

def _effective_rank(X: torch.Tensor, eps: float = 1e-8):
    """Normalized effective rank in (0,1]: exp(H)/min(m,n) for X (m×n)."""
    # Center features per column
    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD on small dims is cheap (J,d are modest here)
    s = torch.linalg.svdvals(Xc)
    p = s / (s.sum() + eps)
    H = -(p * (p + eps).log()).sum()
    erank = torch.exp(H)
    return (erank / float(min(X.shape[0], X.shape[1]))).clamp(0, 1)


def compute_what_v2(P_c, N_c, X_S, joints, bone_lengths, eta_cov=0.5):
    """
    WHAT (dimensionless) = normal_consistency × embedding_erank × coverage
    - normal_consistency: 1 - mean(|‖n̂‖-1|)
    - embedding_erank: normalized effective rank of per‑joint embeddings (J×d)
    - coverage: fraction of points within r = eta_cov * median_bone_length around each joint
    """
    # Normalize normals and measure consistency
    norms = torch.linalg.vector_norm(N_c, dim=-1, keepdim=True) + 1e-8
    N_hat = N_c / norms
    normal_consistency = (1.0 - (torch.linalg.vector_norm(N_hat, dim=-1) - 1.0).abs().mean()).clamp(0, 1)

    # Embedding richness via normalized effective rank
    if X_S.ndim == 2:
        embedding_erank = _effective_rank(X_S)
    else:
        embedding_erank = _effective_rank(X_S.reshape(-1, X_S.shape[-1]))

    # Scale‑aware coverage
    med_bone = torch.median(bone_lengths) if torch.is_tensor(bone_lengths) else torch.tensor(float(bone_lengths))
    r = float(eta_cov) * float(med_bone + 1e-8)
    dists = torch.cdist(joints, P_c)                    # (J,N)
    coverage = (dists < r).float().mean(dim=1).mean()

    return (normal_consistency * embedding_erank * coverage).clamp(0, 1)


def compute_where_v2(pairwise_path_lengths, attention_weights=None):
    """
    WHERE (dimensionless) = path_compactness × attention_locality × attention_focus
    - path_compactness: 1 / (1 + mean path length) using precomputed all‑pairs distances L (J×J)
    - attention_locality: 1 − E_L[distance] / max_distance, weighted by attention rows
    - attention_focus: 1 − normalized row entropy of attention
    """
    L = pairwise_path_lengths  # (J,J) symmetric, zeros on diagonal
    J = L.shape[0]

    # 1) Path compactness
    tri_idx = torch.triu_indices(J, J, offset=1)
    mean_path = L[tri_idx[0], tri_idx[1]].mean() if J > 1 else torch.tensor(0.0)
    path_compactness = (1.0 / (1.0 + mean_path)).clamp(0, 1)

    # 2) Attention locality (optional)
    if attention_weights is not None:
        A = attention_weights
        if A.ndim == 4:   # (B,T,J,J)
            A = A.mean(dim=(0,1))
        elif A.ndim == 3: # (T,J,J)
            A = A.mean(dim=0)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-8)
        # expected distance from each joint
        exp_d = (A * L).sum(dim=-1)                      # (J,)
        max_d = L.max()
        attention_locality = (1.0 - (exp_d.mean() / (max_d + 1e-8))).clamp(0, 1)
        # focus via entropy
        ent = -(A * (A + 1e-8).log()).sum(dim=-1).mean()
        max_ent = torch.log(torch.tensor(float(J)))
        attention_focus = (1.0 - (ent / (max_ent + 1e-8))).clamp(0, 1)
    else:
        attention_locality = torch.tensor(1.0)
        attention_focus = torch.tensor(1.0)

    return (path_compactness * attention_locality * attention_focus).clamp(0, 1)


def compute_who_v2(model_cfg, Y, gate_weights=None, flops_ref=None):
    """
    WHO (dimensionless) = param_fraction × comp_intensity × gate_openness × feature_erank
    - param_fraction: trainable_params / total_params from model_cfg
    - comp_intensity: min(1, FLOPs_current / flops_ref)
    - gate_openness: mean(2*|g-0.5|)
    - feature_erank: normalized effective rank over features (J×d or flattened)
    """
    total_params = float(model_cfg.get('total_params', 1.0))
    trainable_params = float(model_cfg.get('trainable_params', total_params))
    param_fraction = min(1.0, trainable_params / max(total_params, 1.0))

    # FLOPs
    flops_curr = float(model_cfg.get('flops', 0.0))
    if flops_ref is None or flops_ref <= 0:
        comp_intensity = 1.0 if flops_curr > 0 else 0.0
    else:
        comp_intensity = min(1.0, flops_curr / float(flops_ref))

    # Gates
    if gate_weights is None:
        gate_openness = 1.0
    else:
        g = gate_weights if torch.is_tensor(gate_weights) else torch.tensor(gate_weights)
        gate_openness = float((2.0 * (g - 0.5).abs()).mean().clamp(0, 1).item())

    # Feature effective rank
    if Y.ndim == 4:  # (B,T,J,d) -> (J,d)
        J, d = Y.shape[-2], Y.shape[-1]
        Yf = Y.reshape(-1, d)
    else:
        Yf = Y.reshape(-1, Y.shape[-1])
    feature_erank = float(_effective_rank(Yf).item())

    return torch.tensor(param_fraction * comp_intensity * gate_openness * feature_erank).clamp(0, 1)


def normalize_time(time_s: float, time_ref: float):
    """TIME (dimensionless) = time_s / time_ref, clipped to [1e-6, +∞)."""
    t = max(time_s, 1e-6)
    return max(t / max(time_ref, 1e-6), 1e-6)
```

**Equation (dimensionless form, used in query mode):**

`Context^α = (What × Where × Who) / Time_norm`

Where `Time_norm = normalize_time(time_s, time_ref)`.

### 14.1c Qualification & validation (after quantification)

* **Dimensional sanity:** All terms are unitless in $[0,1]$; multiplication is legitimate.
* **Hyperparameters, not constants:** `eta_cov`, `time_ref`, `flops_ref` are explicit and must be specified per dataset/model.
* **Empirical check:** At the end of each epoch, compute Spearman/Pearson correlation between `Context^α` and accuracy (−MPJPE). If corr < 0.3 for 3 consecutive epochs, **disable** Conveyance control and log a warning.
* **Ablation tie‑in:** Track monotonicity: when increasing temporal depth (holding others fixed), `Who` should not decrease; when masking attention neighborhoods, `Where` should not increase.
* **Cost disclosure:** Log wall‑clock overhead of metrics (>+5% triggers a warning and reduces evaluation frequency).

### 14.2 Integration in training/inference

* **Training step**: after forward pass (and optionally after backward to enable `grads_available=True`), compute metrics and log them with predictions.
* **Sleep cycle**: set `ConveyanceScorer(alpha=target)` and rearrange capacity (heads, d, temporal depth) until `Who ≈ (Context^α_target × Time)/(What × Where)`.
* **Batching**: for sequences, compute metrics per‑frame and average, or compute on a diagnostic key‑frame.

---

## 15) Logging & Instrumentation

* Per‑batch: MPJPE, PA‑MPJPE, velocity/accel errors, gate weights statistics.
* Attention diagnostics: sparsity, entropy, neighborhood sizes.
* ConveyanceScorer JSON blob (What/Where/Who/Time proxies; Context^α — computed in query mode or target in sleep mode; mode flag: `query`|`sleep`).

### 15.1 PyTorch Lightning Callback: ConveyanceLogger

The callback wires the ConveyanceScorer into training/validation without touching the core module. It validates keys/shapes, logs timestamps, and avoids silent failures.

```python
import time
import pytorch_lightning as pl

REQUIRED_KEYS = ["spatial_features", "fused_features"]

class ConveyanceLightningCallback(pl.Callback):
    def __init__(self, scorer, mode: str = 'query', every_n_steps: int = 50):
        assert mode in {'query','sleep'}
        self.scorer = scorer
        self.mode = mode
        self.every_n_steps = int(every_n_steps)

    def _validate_outputs(self, outputs):
        missing = [k for k in REQUIRED_KEYS if k not in outputs]
        return missing

    def _log(self, trainer, pl_module, batch, outputs, tag: str):
        # Basic schema check
        missing = self._validate_outputs(outputs)
        if missing:
            pl_module.log(f"conv/{tag}/missing_keys", float(len(missing)))
            return
        # Compute and log metrics
        try:
            metrics = self.scorer.compute(pl_module, batch, outputs, mode=self.mode)
        except Exception as e:
            pl_module.log(f"conv/{tag}/error", 1.0)
            # Also surface as a trainer log message for visibility
            trainer.logger.log_metrics({f"conv/{tag}/exception": 1.0}, step=trainer.global_step)
            return
        # Numeric metrics only
        payload = {f"conv/{tag}/{k}": v for k, v in metrics.items() if isinstance(v, (int, float))}
        payload[f"conv/{tag}/timestamp_unix"] = time.time()
        payload["conv/mode_is_sleep"] = 1.0 if metrics.get("mode") == "sleep" else 0.0
        trainer.logger.log_metrics(payload, step=trainer.global_step)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.every_n_steps and (trainer.global_step % self.every_n_steps) != 0:
            return
        self._log(trainer, pl_module, batch, outputs, tag="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if self.every_n_steps and (trainer.global_step % self.every_n_steps) != 0:
            return
        self._log(trainer, pl_module, batch, outputs, tag="val")
```

**LightningModule contract**

* `training_step` and `validation_step` should **return** a dict that includes:

  * `spatial_features` (B,T,J,d), `fused_features` (B,T,J,d)
  * optionally `attention_weights` (B,T,J,J), `gate_weights` (B,T,J)
  * optionally `inference_time` (float seconds), `grads_available` (bool)
* If returning a different structure, adapt the callback or add a small adapter in your module to expose these tensors.

### 15.2 Trainer wiring (example)

```python
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger

scorer = ConveyanceScorer(alpha=1.5)  # or target value in sleep mode
conv_cb = ConveyanceLightningCallback(scorer, mode='query', every_n_steps=50)

loggers = [
    CSVLogger(save_dir="logs", name="gears"),
    TensorBoardLogger(save_dir="tb", name="gears"),
]

trainer = Trainer(
    max_epochs=100,
    logger=loggers,
    callbacks=[conv_cb],
    precision="bf16-mixed",
)

trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
```

**Notes**

* In **sleep mode**, instantiate `ConveyanceScorer(alpha=Context_alpha_target)` and set `mode='sleep'`. Use the logged `Who` to drive an outer loop that adjusts capacity (heads/width/temporal depth) until `Who ≈ (Context^α_target × Time)/(What × Where)`.
* CSVLogger will automatically write a row per `log_metrics` call. TensorBoardLogger provides per‑metric scalars at `conv/train/*` and `conv/val/*` namespaces.

### 15.3 Optional CSV schema (for downstream analysis)

Fields (all floats unless noted):

* `step` (int), `epoch` (int)
* `conv/train/What`, `conv/train/Where`, `conv/train/Who`, `conv/train/Time`, `conv/train/Context_alpha`, `conv/train/timestamp_unix`
* `conv/val/What`, `conv/val/Where`, `conv/val/Who`, `conv/val/Time`, `conv/val/Context_alpha`, `conv/val/timestamp_unix`
* `conv/mode_is_sleep` (0 or 1)

**Mode label (string):** if you want a human‑readable label, log via the TB logger’s `add_text` in a separate callback to avoid breaking numeric CSV logs.
