# Dataset Loading and Preparation - Comprehensive Documentation

## 📋 Module Overview

**File:** `dataset.py`  
**Purpose:** Unified interface for loading graph datasets with automatic preprocessing  
**Key Features:** Multi-backend support, sparse tensor operations, negative sampling

---

## 📚 Supported Datasets (7 Benchmarks)

### Dataset Catalog

#### Citation Networks
**Domain:** Academic paper citations, homophilic networks

| Dataset | Nodes | Edges | Classes | Features |
|---------|-------|-------|---------|----------|
| **Cora** | 2,708 | 5,429 | 7 | 1,433 |
| **Citeseer** | 3,327 | 4,732 | 6 | 3,703 |
| **Pubmed** | 19,717 | 44,338 | 3 | 500 |

**Characteristics:**
- Sparse: ~0.1-1% edges of potential edges
- Homophilic: Connected nodes tend to have similar labels
- Small feature dimension (typically ~1500)

#### E-commerce Networks  
**Domain:** Amazon product co-purchasing, heterophilic

| Dataset | Nodes | Edges | Categories | Features |
|---------|-------|-------|-------------|----------|
| **Computers** | 13,752 | 491,722 | 10 | 767 |
| **Photo** | 7,650 | 238,162 | 8 | 745 |

**Characteristics:**
- Larger than citation networks
- Products frequently co-purchased
- Consistent feature dimension

#### Co-authorship Networks
**Domain:** Collaboration graphs, author networks

| Dataset | Nodes | Edges | Topics | Features |
|---------|-------|-------|--------|----------|
| **CS** | 18,333 | 81,351 | 15 | 6,805 |
| **Physics** | 34,493 | 247,962 | 5 | 8,415 |

**Characteristics:**
- Large node count
- Long-tail topic distribution
- High-dimensional features

---

## 🔄 Data Loading Pipeline

### Step 1: Dataset Download & Caching

```python
dataset = Planetoid(
    root='./datasets/Cora',  # Local cache directory
    name='Cora',              # Dataset name
    split='public'            # Train/val/test split
)
```

**What happens:**
- Checks if dataset exists locally
- If not, downloads from PyTorch Geometric
- Caches for future use
- ~1-5 MB per dataset

---

### Step 2: Component Extraction

```
Raw Dataset Object
  ├─ graph[0].x → Node features (num_nodes, features_dim)
  ├─ graph[0].edge_index → Edge connectivity (2, num_edges)
  ├─ graph[0].y → Node labels (num_nodes,)
  └─ graph.num_features → Feature dimension
       ↓
Extracted Components:
  ├─ num_nodes: Total nodes
  ├─ features: (num_nodes, num_features) tensor
  ├─ edge_indices: (2, num_edges) tensor pairs
  ├─ degrees: Sum neighbors per node
  └─ labels: Ground truth labels
```

---

### Step 3: Graph Structure Preparation

```python
# Create sparse adjacency matrix
indices = torch.stack([
    torch.cat([edge_index[0], edge_index[1]]),  # Bidirectional
    torch.cat([edge_index[1], edge_index[0]])
])
adj_sparse = torch.sparse_coo_tensor(
    indices, values, size=(num_nodes, num_nodes)
)

# Create DGL graph (for message passing)
g = dgl.graph((edge_index[0], edge_index[1]))
g = dgl.add_self_loops(g)  # Add self-connections
```

**Why these structures?**
- **Sparse tensor:** Memory efficient, supports matrix operations
- **DGL graph:** Optimized message passing, GPU acceleration
- **Self-loops:** Ensure each node can attend to itself

---

### Step 4: Negative Sampling

```python
# Generate non-existent edges
neg_edge_index = negative_sampling(
    edge_index,
    num_nodes=num_nodes,
    num_neg_samples=edge_index.shape[1]  # Equal to positive edges
)

# Result: Non-existent edges sampled uniformly
# Used for link prediction loss training
```

**Purpose:**
- Balance training signal (positive vs negative)
- Teach model to distinguish connected vs non-connected pairs
- Critical for link prediction loss

---

## 📊 Data Attributes Detailed

### Attribute: `num_nodes`

```python
dataset.num_nodes  # e.g., 2708 for Cora

# Interpretation:
# - Total count of nodes (papers, products, authors)
# - Dimension of node-indexed tensors
# - Size of square adjacency matrix
```

**Range:** 2,700 (Cora) to 34,400 (Physics)

---

### Attribute: `feature`

```python
dataset.feature  # Shape: (num_nodes, num_features)
                 # e.g., (2708, 1433) for Cora

# Each row: feature vector for one node
# Typical values: Binary indicators of keywords, word counts, etc.
# On device: GPU or CPU as specified
```

**Characteristics:**
- Sparse or dense depending on dataset
- Normalized or raw values
- Domain-specific interpretation varies

---

### Attribute: `degrees`

```python
dataset.degrees  # Shape: (num_nodes,)
                 # Each value: number of neighbors for that node

# Example: degrees = [3, 5, 2, 8, ...]
# Interpretation:
#   Node 0 has 3 neighbors
#   Node 1 has 5 neighbors
#   etc.

# Usage: GCN normalization, importance weighting
```

**Distribution characteristics:**
- Most nodes low degree (1-10)
- Few hub nodes high degree (50-200)
- Follows power-law in many networks

---

### Attribute: `adj` (Adjacency Matrix)

```python
dataset.adj  # torch.sparse_coo_tensor
             # Shape: (num_nodes, num_nodes)

# Sparse format: Only stores non-zero edges
# Memory efficient: ~0.1-1% of dense matrix size
# For Cora: ~1.5 MB (vs ~290 MB if dense)

# Usage:
# - Graph operations: adj @ features
# - Coarsening: S.T @ adj @ S
# - Conversions: to DGL, NetworkX, etc.
```

**Sparsity levels:**
- Cora: 0.074% non-zero
- Physics: 0.021% non-zero

---

### Attribute: `graph` (DGL Graph)

```python
dataset.graph  # dgl.DGLGraph with self-loops

# Properties:
graph.num_nodes()  # Total nodes
graph.num_edges()  # Total edges (including self-loops)
graph.edges()      # Return (src, dst) tensors

# Operations:
# - Message passing: graph.update_all()
# - Neighborhood aggregation
# - GPU-accelerated computations
```

**Why DGL format?**
- Optimized for GNNs
- GPU-friendly
- Built-in aggregation functions

---

### Attribute: `labels`

```python
dataset.labels  # Ground truth cluster/class labels
                # Shape: (num_nodes,)
                # Values: 0 to (num_classes-1)

# Example: labels = [0, 1, 0, 2, 1, ...]
# Interpretation:
#   Node 0 belongs to class 0 (paper topic, product category, etc.)
#   Node 1 belongs to class 1
#   etc.

# IMPORTANT: Used ONLY for evaluation, NOT for training
# (Unsupervised clustering)
```

**Label distribution:**
- Balanced or imbalanced depending on dataset
- Not used during model training
- Critical for computing metrics (NMI, ARI, ACC, F1)

---

### Attribute: `neg_edge_index`

```python
dataset.neg_edge_index  # Shape: (2, num_negative_edges)
                        # Usually: num_negative_edges ≈ num_edges

# Example:
# [[100, 250, 500, ...],  # Source nodes
#  [150, 340, 600, ...]]   # Destination nodes
# These pairs do NOT exist in the graph

# Usage: Link prediction loss training
# Binary classification: edge exists (1) or not (0)
```

**Sampling strategy:**
- Uniform random from all non-edges
- Matches positive edge count for balanced training
- Regenerated per training run

---

## 🛠️ Preprocessing Rationale

### Why Sparse Tensor Format?

```
Comparison: Cora dataset (2,708 nodes)

Dense adjacency matrix:
  Size: 2,708 × 2,708 = 7.3 million entries
  Memory: 7.3M × 4 bytes = ~29 MB per copy
  Issue: 99.93% zeros!

Sparse COO tensor:
  Size: 2 × 10,858 entries (bidirectional edges)
  Memory: 10,858 × 8 bytes = ~87 KB
  Savings: 99.7% memory reduction!

Sparse operations:
  Matrix mult: O(edges) not O(nodes²)
  Storage: Linear in edges, not quadratic in nodes
```

**Practical benefits:**
- Fits large graphs in GPU memory
- Faster operations on sparse data
- Standard in GNN libraries

---

### Why Self-Loops?

```
GCN without self-loops:
  h_new[i] = W @ mean(h[neighbors of i])
  Issue: Node i's own features ignored

GCN with self-loops:
  h_new[i] = W @ mean(h[i] + h[neighbors of i])
  Benefit: Node retains identity information
```

**Architecture standard:**
- Original GCN paper recommends self-loops
- Improves stability
- Each node attends to self + neighbors

---

### Why Negative Sampling?

```
Link Prediction Loss requires:
  Positive examples: (u, v) where edge exists
  Negative examples: (u, v) where edge doesn't exist

Without negative sampling:
  Model only learns edges are important
  Doesn't learn to distinguish from non-edges
  Embeddings lack discriminative power

With negative sampling:
  Model learns both:
    "These nodes should be close (connected)"
    "These nodes should be far (not connected)"
  Binary classification problem (structured)
  Gradients more informative
```

**Quantity:** Usually 1:1 ratio (negative:positive)

---

## 🔌 Class Variants

### Data Class (PRIMARY)

```python
dataset = Data(
    dataset_name='Cora',
    device='cuda:0',  # GPU placement
    seed=42
)
```

**Features:**
- ✅ Modern PyTorch Geometric backend
- ✅ Supports all 7 datasets
- ✅ Automatic download and caching
- ✅ Well-maintained upstream
- ✅ Recommended for new projects

**Methods:**
- `print_statistic()`: Display dataset properties
- `load_negative_edges()`: Re-sample negatives

---

### Graph Class (ALTERNATIVE)

```python
graph = Graph(
    dataset_name='Cora',
    device='cuda:0'
)
```

**Features:**
- Uses DGL dataset loaders directly
- Legacy support for older code
- Similar functionality to Data class
- Less commonly used in practice

**When to use:**
- Existing DGL-based pipeline
- Prefer DGL operations
- Specific DGL dataset requirements

---

## 📝 Typical Usage Patterns

### Basic Usage

```python
# 1. Load dataset
dataset = Data('Cora', device='cuda:0')

# 2. Access components
num_nodes = dataset.num_nodes
features = dataset.feature        # (2708, 1433)
adj = dataset.adj                 # Sparse tensor
degrees = dataset.degrees         # (2708,)
labels = dataset.labels           # For eval only
neg_edges = dataset.neg_edge_index  # For LP loss

# 3. In training loop
embeddings = model(features, adj, degrees)
se_loss = model.calculate_se_loss1()
lp_loss = model.calculate_lp_loss(g, neg_edges, embeddings)
```

---

### Multi-Dataset Iteration

```python
datasets = ['Cora', 'Citeseer', 'Photo']

for dataset_name in datasets:
    dataset = Data(dataset_name, device='cuda:0')
    
    # Run experiment on this dataset
    model = train(dataset)
    results = evaluate(model, dataset)
    print(f"{dataset_name}: {results}")
```

---

### GPU vs CPU Switching

```python
# Load on CPU for preprocessing
dataset = Data('Cora', device='cpu')
process_data(dataset)

# Move to GPU for training
device = 'cuda:0'
features = dataset.feature.to(device)
adj = dataset.adj.to(device)
model = model.to(device)
```

---

## 🎯 Data Flow in Training

```
Dataset Loading
    ↓
┌─────────────────────────┐
│ Data Class              │
├─────────────────────────┤
│ • num_nodes: 2708       │
│ • feature: (2708, 1433) │
│ • adj: sparse(2708x2708)│
│ • degrees: (2708,)      │
│ • labels: (2708,)       │
│ • neg_edges: (2, 10858) │
└────────────┬────────────┘
             ↓
      Training Loop
    ↙         ↓         ↖
Forward   Compute    LP Loss
  Pass     SE Loss     from
  through  from        neg_edges
 features  struct.
    ↓         ↓         ↓
   Combined Loss
       ↓
  Backprop & Update
```

---

## 📊 Dataset Statistics

### Memory Requirements

| Dataset | Nodes | Edges | Mem (MB) |
|---------|-------|-------|----------|
| Cora | 2,708 | 5,429 | ~5 |
| Citeseer | 3,327 | 4,732 | ~8 |
| Photo | 7,650 | 238,162 | ~50 |
| Computers | 13,752 | 491,722 | ~100 |
| CS | 18,333 | 81,351 | ~30 |
| Physics | 34,493 | 247,962 | ~80 |
| Pubmed | 19,717 | 44,338 | ~20 |

*Note: Sparse adjacency only, excluding feature matrices*

---

### Feature Dimension Variability

```python
# Features per dataset:
Cora:      1,433 (binary BoW word indicators)
Citeseer:  3,703 (more sparse features)
Pubmed:      500 (dense features)
Photo:       745 (product attributes)
Computers:   767 (product attributes)
CS:        6,805 (author features)
Physics:   8,415 (author features)
```

**Implication:** Models must handle variable input dimensions

---

## 🔍 Common Operations

### Converting Sparse to Dense (for small graphs)

```python
# NOT recommended for large graphs - memory explodes!
adj_dense = dataset.adj.to_dense()  # 2708x2708 matrix

# Only use for visualization, debugging small graphs
```

---

### Extracting Subgraphs

```python
# Get neighbors of node 100
node_id = 100
neighbors = dataset.adj[node_id].nonzero().squeeze()

# Extract features of neighbors
neighbor_features = dataset.feature[neighbors]
```

---

### Computing Graph Statistics

```python
# Degree statistics
degrees = dataset.degrees
print(f"Min degree: {degrees.min()}")
print(f"Max degree: {degrees.max()}")
print(f"Mean degree: {degrees.mean():.2f}")
print(f"Median degree: {degrees.median():.2f}")

# Sparsity
num_edges = dataset.adj.nnz()  # Non-zero count
sparsity = 1 - (num_edges / (num_nodes ** 2))
print(f"Sparsity: {sparsity:.4f}")
```

---

## ⚙️ Advanced Configuration

### Custom Data Splits

```python
# Some datasets support train/val/test splits
from torch_geometric.datasets import Planetoid

dataset = Planetoid(
    root='./datasets',
    name='Cora',
    split='full'  # Use all data (no train/val/test)
)
```

---

### Loading Custom Datasets

```python
# To add a custom dataset:
# 1. Prepare edge list file
# 2. Prepare features
# 3. Create custom loader class
# 4. Integrate with Data class

# For now, use provided 7 benchmarks
```

---

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Dataset too large for GPU | High-dim features | Reduce feature dim, use CPU |
| Negative edges = positive edges | Poor sampling | Re-run sampling |
| Memory error | Dense adjacency | Keep sparse format |
| NaN in features | Data corruption | Re-download dataset |

---

## 📋 Checklist: Before Training

- [ ] Dataset loads without errors
- [ ] `num_nodes` reasonable (100s to 100Ks)
- [ ] Features shape makes sense
- [ ] Adjacency sparse and small
- [ ] Labels present and diverse
- [ ] Negative edges generated
- [ ] Device set correctly (GPU/CPU)

