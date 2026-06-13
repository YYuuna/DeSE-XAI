# DeSE Model Architecture - Comprehensive Documentation

## 📋 Module Overview

**File:** `model.py`  
**Purpose:** Implements the Deep Structural Entropy (DeSE) model for hierarchical graph clustering  
**Key Innovation:** Combines structural graph information with embedding-based feature information in a multi-level hierarchy

---

## 🎯 Core Concept: Deep Structural Entropy

### What is DeSE?

DeSE is a **hierarchical graph clustering algorithm** that views the clustering problem through TWO complementary lenses:

1. **Structural View (Graph Topology)**
   - How nodes are connected in the original graph
   - Captures community structure through network topology
   - Measured via: Adjacency matrix A

2. **Embedding View (Feature Space)**
   - How similar nodes are in their feature representations
   - Helps identify nodes that should cluster together despite being distant
   - Measured via: KNN graph of learned embeddings

### Why Two Information Sources?

- **Structure alone:** May miss clusters when nodes aren't directly connected but are feature-similar
- **Features alone:** May ignore actual network connections
- **Combined:** Captures both aspects, leading to better clustering

---

## 🏗️ Architecture Components

### 1. **MLP (Multi-Layer Perceptron)**

```
Input Features (F_dim) → FC1 → Activation → Dropout → FC2 → Embeddings (embed_dim)
```

**Purpose:** Transform raw node features into a fixed-dimensional embedding space

**Why?** Provides uniform representation for all datasets regardless of feature dimensions

**Code Reference:** `class MLP`

**Parameters:**
- `input_dim`: Original feature dimension (varies by dataset)
- `hidden_dim`: Intermediate layer dimension
- `output_dim`: Final embedding dimension (typically 16-64)
- `dropout`: Regularization (prevents overfitting)
- `activation`: Non-linearity (ReLU, ELU, etc.)

---

### 2. **GCN Layer (Graph Convolutional Network)**

```
Input: Node embeddings + Graph structure
       ↓
Linear Transformation
       ↓
Message Passing: Aggregate neighbor information
       ↓
Optional Attention: Weight contributions by importance
       ↓
Output: Updated embeddings
```

**Purpose:** Propagate information across the graph, allowing nodes to incorporate neighbor information

**Two Variants:**
- **GCN_layer:** Standard DGL-based implementation with optional attention
- **GCN_layer1:** Optimized version using direct tensor operations

**How it works:**
1. Each node transforms its features linearly
2. Information flows from neighbors to each node
3. Aggregated messages are normalized (mean or attention-weighted)
4. Activation function introduces non-linearity

**Why Attention?** Learns which neighbors are most important to listen to

---

### 3. **Assign Layer (Cluster Assignment)**

```
Input: Embeddings + Graph structure
       ↓
GCN for refinement
       ↓
GCN for assignment: Output num_clusters values per node
       ↓
Softmax: Convert to probability distribution
       ↓
Output: Soft assignment matrix S (num_nodes × num_clusters)
```

**Purpose:** Assigns each node to clusters with learned probabilities

**Key Operation - Graph Coarsening:**
```
new_adjacency = S^T @ original_adjacency @ S
```

This transforms the graph from node-level to cluster-level representation:
- Nodes become clusters
- Edges become inter-cluster connections
- Weights represent connection strength between clusters

---

### 4. **ASS Layer (Assignment-Sigmoid-Structure)**

```
Input: Node embeddings + Structural adjacency + Features
       ↓
Compute Soft Assignments (structure-based)
       ↓
Compute KNN Graph (feature-based similarity)
       ↓
Mix both adjacencies: adj = struct_adj + β·feature_adj
       ↓
Output: Coarsened embeddings and adjacency for next level
```

**Purpose:** Bridges structural and embedding-based clustering at each hierarchy level

**Why both adjacencies?**
- **Structural:** Captures actual connections
- **Feature-based:** Captures learned similarity between node representations

**Mixing Parameter β (beta_f):**
- β = 0: Pure graph topology
- β = 0.5: Balanced topology and features
- β = 1: Pure feature similarity

---

### 5. **KNN Graph Construction**

**Standard KNN:**
```
For each node:
  - Find k nearest neighbors in embedding space
  - Create edges to these neighbors
  - Result: Additional adjacency matrix based on feature similarity
```

**Dynamic KNN:**
```
For each node with degree d:
  - Compute k = ceil(log₂(d + 1))
  - Higher degree nodes connect to more distant neighbors in embedding space
  - Adapts neighborhood size to node importance
```

**Purpose:** Regularizes clustering with feature-space information

---

## 🔗 Hierarchical Clustering Architecture

### Multi-Level Hierarchy Concept

```
Level h (Top):     1 supernode (all nodes clustered into it)
                   ↓ (coarsen via soft assignments)
Level h-1:         c_{h-1} clusters (supernodes)
                   ↓
Level h-2:         c_{h-2} clusters
                   ↓
...
Level 2:           c_2 clusters
                   ↓
Level 1 (Bottom):  c_1 clusters (final output)
                   ↓
Original Nodes:    All original nodes with their cluster IDs
```

### Why Hierarchical?

1. **Multi-scale Structure:** Captures communities at different granularities
2. **Computational Efficiency:** Coarsening reduces computation at higher levels
3. **Interpretability:** Can explain clustering at each level
4. **Stability:** Multiple levels reduce sensitivity to initialization

### Flow Example (height=3)

```
Forward Pass:
  Level 3: Process original graph, output 100 clusters
  Level 2: Process coarsened graph (100 nodes), output 20 clusters  
  Level 1: Process further coarsened graph (20 nodes), output 5 clusters

Backward Pass:
  Accumulate assignments: Level 1 clusters ← Level 2 clusters ← Level 3 clusters ← Original nodes
  Final output: Each original node assigned to one of 5 final clusters
```

---

## 💡 Loss Functions

### 1. Structural Entropy (SE) Loss

**Purpose:** Measures clustering quality using information-theoretic principles

**Intuition:**
- Better clustering = fewer edges cross cluster boundaries
- SE loss penalizes large cuts relative to cluster size

**Formula:**
```
SE_loss = -1/vol(G) * Σ_levels Σ_clusters [
    (volume_cut_c) * log₂(volume_cut_c / volume_parent_c)
]
```

**Key Terms:**
- `volume_cut_c`: Total weight of edges leaving cluster c
- `volume_parent_c`: Total weight in parent partition
- `vol(G)`: Total graph volume (sum of all degrees)

**Minimizing SE:**
- Small cuts = small volume_cut = smaller loss
- Clusters balanced in size = better generalization
- Normalized by parent volume = scale-invariant

**Code:** `calculate_se_loss1()` method

---

### 2. Link Prediction (LP) Loss

**Purpose:** Ensures learned embeddings preserve graph structure

**Intuition:**
- Nodes connected by edges should have similar embeddings
- Non-connected nodes should be dissimilar
- Binary classification: edge exists or not

**Algorithm:**
```
1. For each edge (u, v) in graph:
   - Compute similarity: sim = sigmoid((2 - ||emb_u - emb_v||₂) / 1)
   - Target: 1 (edge exists)

2. For each non-edge sampled:
   - Compute same similarity
   - Target: 0 (edge doesn't exist)

3. Loss = BCE(predictions, targets)
```

**Why this formulation?**
- Distance 0 (identical embeddings) → similarity ≈ 0.88
- Distance 2 (moderately far) → similarity ≈ 0.50
- Distance > 2 (very far) → similarity < 0.50

**Code:** `calculate_lp_loss()` method

---

### 3. Combined Loss

```
Total Loss = se_lamda * SE_loss + lp_lamda * LP_loss
```

**Balancing:**
- **High se_lamda:** Emphasize structure quality
- **High lp_lamda:** Emphasize embedding fidelity
- Typical: se_lamda=0.01, lp_lamda=1.0 (structure via SE, embeddings more important)

---

## 🔄 Forward Pass Detailed Flow

```
Input: adj_g (sparse adjacency), features, degrees

Step 1: Compute KNN Feature Graph
  ├─ MLP transforms features → embeddings'
  └─ KNN finds k nearest neighbors
     Result: adj_f (feature-based adjacency)

Step 2: Mix Adjacencies
  adj = adj_g + β·adj_f

Step 3: Initial Embedding
  ├─ Convert sparse adj → DGL graph
  ├─ Apply GCN to original features
  └─ Result: e (node embeddings at level height)

Step 4: Hierarchy Loop (for each level from height-1 down to 1)
  For each level i:
    ├─ Apply ASS layer:
    │  ├─ Compute soft assignments S
    │  ├─ Coarsen structural adjacency: S^T @ A @ S
    │  ├─ Compute KNN on coarsened embeddings
    │  └─ Mix adjacencies at cluster level
    │
    ├─ Store results:
    │  ├─ tree_node_embed_dic[level] = cluster embeddings
    │  ├─ s_dic[level] = soft assignments
    │  └─ g_dic[level] = DGL graph at level
    │
    └─ Prepare for next level: update graph and adjacency

Step 5: Top Level
  s_dic[1] = all ones (entire graph is one supercluster)

Step 6: Convert Soft to Hard Assignments
  hard_dic[k] = argmax(accumulated_soft_assignments) for each level
  Result: Each original node assigned to exactly one final cluster

Output:
  ├─ s_dic: Soft assignments at each level
  ├─ tree_node_embed_dic: Embeddings at each level
  └─ g_dic: DGL graphs at each level
```

---

## 🧮 Hard Assignment Computation

**Goal:** Convert hierarchical soft assignments to final cluster IDs

**Algorithm:**
```
For each original node n:
  1. Start at top (level 1) with S[1]
  2. Accumulate: accumulated_S = S[1] @ S[2] @ ... @ S[height]
  3. accumulated_S[n] gives probability over final clusters
  4. final_cluster[n] = argmax(accumulated_S[n])
```

**Interpretation:**
- Each node flows down the hierarchy
- At each level, probabilistically assigned to a supernode
- Accumulation gives final cluster

---

## 📊 Hyperparameter Guide

| Parameter | Range | Default | Sensitivity | Effect |
|-----------|-------|---------|-------------|--------|
| `height` | 1-5 | 2 | **HIGH** | Number of hierarchy levels |
| `embed_dim` | 8-128 | 16 | MEDIUM | Embedding expressiveness |
| `num_clusters_layer` | Variable | [50,10] | **HIGH** | Clusters per level |
| `k` | 2-20 | 2 | MEDIUM | KNN neighbors |
| `se_lamda` | 0.001-1 | 0.01 | **HIGH** | SE loss weight |
| `lp_lamda` | 0.1-2 | 1 | **HIGH** | LP loss weight |
| `beta_f` | 0-1 | 0.2 | MEDIUM | Topology vs feature mix |
| `dropout` | 0-0.5 | 0.1 | LOW | Regularization |
| `activation` | relu/elu/sigmoid | relu | LOW | Non-linearity |

---

## 🎓 Usage Example

```python
# Initialize model
model = DeSE(args, features, device='cuda:0')

# Forward pass
s_dict, embeddings, graphs = model(adjacency, features, degrees)

# Compute losses
se_loss = model.calculate_se_loss1()
lp_loss = model.calculate_lp_loss(graphs[height], neg_edges, embeddings[height])
total_loss = args.se_lamda * se_loss + args.lp_lamda * lp_loss

# Backpropagation
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

# Extract final clusters
final_clusters = model.hard_dic[1]  # Hard assignments at level 1
cluster_ids = torch.argmax(final_clusters, dim=1)  # One ID per node
```

---

## 🔍 Key Design Decisions

### 1. Why Soft Assignments?
- **Differentiable:** Enable gradient-based optimization
- **Smooth:** Gradual transitions between clusters
- **Probabilistic:** Uncertainty representation
- **Graph coarsening:** Preserves edge weights through S^T @ A @ S

### 2. Why KNN Feature Graph?
- **Regularization:** Feature similarity guides topology-based clustering
- **Bridging:** Connects nodes that should cluster together but are distant
- **Adaptive:** Dynamic KNN adjusts to node importance
- **Scale-invariant:** Works across different network scales

### 3. Why Dual Losses?
- **SE loss:** Optimizes cluster quality (structure)
- **LP loss:** Ensures embeddings capture relationships (representation)
- **Complementary:** Together learn both structure and embeddings
- **Balanced:** Trade-off controlled by weights

### 4. Why Hierarchical?
- **Interpretable:** Clustering at multiple granularities
- **Stable:** Less sensitive to initialization
- **Efficient:** Coarsening reduces computation
- **Flexible:** Extract clusters at any level

---

## ⚙️ Advanced Topics

### Graph Coarsening Deep Dive

**Operation:** `new_adj = S^T @ adj @ S`

**What happens:**
```
Original adjacency A (100×100 nodes):
  A[i,j] = weight of edge between nodes i and j

Soft assignments S (100×5 clusters):
  S[i,k] = probability node i belongs to cluster k
  Rows sum to 1

Result adjacency A' (5×5):
  A'[k1,k2] = Σ_{i,j} A[i,j] * S[i,k1] * S[j,k2]
  Represents probabilistic inter-cluster edges

Intuition: Edges are "softly" moved to cluster level
based on assignment probabilities
```

### Attention Mechanism

**Purpose:** Learn which neighbors are most important for each node

**Implementation:**
```
For each edge (u→v):
  1. Compute concatenated feature: [h_u, h_v]
  2. Pass through attention layer: a = MLP([h_u, h_v])
  3. Apply leaky ReLU: a = leaky_relu(a)
  4. Normalize via softmax across incoming edges
  5. Weight message: h_new = Σ attention * h_neighbor
```

**Result:** Model learns to focus on important neighbors

---

## 📈 Typical Performance

On Cora dataset (2,700 nodes, 7 classes):
- **NMI:** 0.50-0.70 (information shared with ground truth)
- **ARI:** 0.40-0.65 (agreement adjusted for chance)
- **ACC:** 0.75-0.85 (node accuracy with optimal label matching)
- **F1:** 0.75-0.85 (balanced precision/recall)

Varies based on hyperparameter choices and random seed.

---

## 🐛 Debugging Tips

| Issue | Likely Cause | Solution |
|-------|-------------|----------|
| Loss not decreasing | Poor hyperparams | Lower lr, adjust se_lamda/lp_lamda |
| Many tiny clusters | Too many clusters | Decrease num_clusters_layer |
| One giant cluster | Too few clusters | Increase num_clusters_layer |
| GPU out of memory | Large embed_dim or graph | Reduce embed_dim, use smaller height |
| Metric fluctuating | High variance | Use multiple random seeds |

