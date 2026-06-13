# Utilities & Evaluation Metrics - Comprehensive Documentation

## 📋 Module Overview

**File:** `util.py`  
**Purpose:** Graph format conversions, clustering evaluation metrics, activation functions  
**Key Features:** Multi-format graph support, comprehensive metric computation, optimal label alignment

---

## 🗂️ Four Utility Categories

### 1. GRAPH FORMAT CONVERSIONS
Converting between PyTorch Geometric, DGL, and sparse tensor representations

### 2. EDGE LIST OPERATIONS
Transforming between edge indices and adjacency matrices

### 3. ACTIVATION FUNCTIONS
Mapping function names to PyTorch implementations

### 4. CLUSTERING METRICS
Computing multi-perspective clustering quality measures

---

## 🔄 Graph Format Conversions

### The Three Graph Representations

```
┌─────────────────────┬──────────────────────┬──────────────────┐
│   EDGE INDEX        │  SPARSE TENSOR       │   DGL GRAPH      │
├─────────────────────┼──────────────────────┼──────────────────┤
│ (2, num_edges)      │ Sparse COO format    │ DGL graph object │
│ [[src],             │ indices + values     │ Optimized for    │
│  [dst]]             │ (num_nodes, ...)     │ message passing  │
│                     │                      │                  │
│ Compact             │ Matrix operations    │ GPU-friendly     │
│ Lightweight         │ PyTorch standard     │ Built-in ops     │
└─────────────────────┴──────────────────────┴──────────────────┘
       ↓       ↓       ↓       ↓       ↓       ↓
  CONVERSIONS (reversible in all directions)
```

---

### Function: `g_from_torchsparse()`

**Purpose:** Convert PyTorch sparse tensor to DGL graph

```python
# Input: Sparse adjacency tensor
adj_sparse = torch.sparse_coo_tensor(...)  # (num_nodes, num_nodes)

# Output: DGL graph
g = g_from_torchsparse(adj_sparse)

# DGL graph properties:
g.num_nodes()       # Returns num_nodes
g.num_edges()       # Returns total directed edges
g.edges()           # Returns (src, dst) tensors
```

**Workflow:**
```
Sparse Tensor (indices + values)
    ↓
Extract edge indices and values
    ↓
Create DGL graph from edges
    ↓
DGL Graph (optimized for message passing)
```

**When to use:** After computing mixed adjacency (structural + feature), need DGL graph for GCN

---

### Function: `index2adjacency()`

**Purpose:** Convert edge index to adjacency matrix

```python
# Input: Edge indices
edge_index = torch.tensor([
    [0, 1, 1, 2],       # Source nodes
    [1, 0, 2, 1]        # Destination nodes
])

# Output: Adjacency matrix
adj = index2adjacency(num_nodes=3, edge_index=edge_index)

# Result (sparse or dense):
# [[0, 1, 0],
#  [1, 0, 1],
#  [0, 1, 0]]
```

**Options:**
- Dense output for small graphs
- Sparse output for large graphs
- Optional weights

---

### Function: `adjacency2index()`

**Purpose:** Reverse of index2adjacency - extract edges from matrix

```python
# Input: Adjacency matrix
adj = torch.sparse_coo_tensor(...)

# Output: Edge indices
edge_index, weights = adjacency2index(adj, weight=True)

# Can extract weights if adjacency is weighted
```

**Use case:** Extracting edge list from coarsened adjacency matrices

---

## ⚙️ Activation Function Selection

### Function: `select_activation()`

```python
activation = select_activation('relu')    # Returns F.relu
activation = select_activation('elu')     # Returns F.elu
activation = select_activation('sigmoid') # Returns torch.sigmoid
activation = select_activation(None)      # Returns None
```

**Supported Functions:**

| Name | Formula | Use Case |
|------|---------|----------|
| ReLU | max(0, x) | Standard GNNs, stable |
| ELU | x if x>0, e^x-1 if x≤0 | Smoother, potentially better |
| Sigmoid | 1/(1+e^(-x)) | Probability output |
| None | Identity | Skip activation |

**Design:** Simple mapping avoids hardcoding function names throughout code

---

## 📊 Assignment Operations

### Function: `decoding_from_assignment()`

**Purpose:** Convert soft cluster assignments to hard cluster IDs

```
Soft Assignment Matrix (probabilities)
    ↓
Soft (4, 3) - 4 nodes, 3 clusters
  [[0.8, 0.15, 0.05],    Node 0: 80% cluster 0
   [0.1, 0.9,  0.0],     Node 1: 90% cluster 1
   [0.2, 0.3,  0.5],     Node 2: 50% cluster 2
   [0.7, 0.0,  0.3]]     Node 3: 70% cluster 0
    ↓
Hard (4,) - 4 nodes assigned
  [0, 1, 2, 0]           Each node assigned to top cluster
```

**Algorithm:** Argmax over cluster dimension

**Why needed:**
- Soft assignments: probabilistic, differentiable, used during training
- Hard assignments: discrete, interpretable, needed for evaluation/final clusters

---

## 🏆 Clustering Evaluation Metrics

### The Hungarian Algorithm (Optimal Label Matching)

**Problem:** Predicted clusters have arbitrary labels

```
Ground Truth:
  Nodes in class 0: [n1, n5, n9, ...]      (50 nodes)
  Nodes in class 1: [n2, n7, n11, ...]     (40 nodes)
  Nodes in class 2: [n3, n8, n12, ...]     (30 nodes)

Predicted Clusters (arbitrary labels):
  Cluster 0: [n1, n7, n15, ...]            (60 nodes)
  Cluster 1: [n5, n9, n13, ...]            (50 nodes)
  Cluster 2: [n2, n8, n14, ...]            (10 nodes)

Problem: Which predicted cluster maps to which true class?
  Option A: 0→0, 1→1, 2→2  (30 correct)
  Option B: 0→2, 1→0, 2→1  (65 correct) ← BETTER!
  Option C: 0→1, 1→2, 2→0  (55 correct)

Solution: Hungarian algorithm finds Option B (optimal matching)
Result: 65/120 = 54.2% accuracy
```

**Why this matters:**
- Without alignment: Accuracy would be ~33% (chance)
- With alignment: Correctly measures actual clustering quality
- Cost: O(n³) but only run once per evaluation

---

### Metric 1: NMI (Normalized Mutual Information)

**Formula:**
```
NMI = MI(true, pred) / √(H(true) × H(pred))

Where:
  MI = Mutual Information (shared information)
  H = Entropy (uncertainty)
  Range: [0, 1]
```

**Interpretation:**
```
NMI = 0.0  → No agreement (independent)
NMI = 0.3  → Low agreement
NMI = 0.5  → Moderate agreement ← Typical for clustering
NMI = 0.7  → Good agreement
NMI = 1.0  → Perfect agreement
```

**Key property:** Label-alignment independent
- Doesn't require matching predicted→true labels
- Measures pure information sharing
- Robust to label permutations

**Code location:** `cluster_metrics.evaluateFromLabel()`

---

### Metric 2: ARI (Adjusted Rand Index)

**Formula:**
```
ARI = (RI - E[RI]) / (max(RI) - E[RI])

Where:
  RI = Rand Index (fraction of same/different pairs agree)
  E[RI] = Expected RI under random clustering
  Range: [-1, 1]
```

**Interpretation:**
```
ARI = -1.0  → Perfect disagreement
ARI =  0.0  → Random clustering
ARI =  0.5  → Good clustering
ARI =  1.0  → Perfect clustering
```

**Key property:** Adjusted for chance
- Base Rand Index ranges [0, 1]
- Adjustment makes baseline 0 = random chance
- More interpretable for comparison

---

### Metric 3: ACC (Accuracy with Optimal Label Matching)

**Formula:**
```
ACC = (# correctly assigned nodes) / (total nodes)
    = 100 × (correct predictions after optimal matching)
    Range: [0, 1]
```

**Algorithm:**
```
1. Create confusion matrix (num_pred_clusters × num_true_classes)
2. Use Hungarian algorithm to find optimal 1:1 matching
3. Count nodes assigned to matched clusters
4. Compute percentage
```

**Interpretation:**
```
ACC = 0.0   → Completely wrong
ACC = 0.3   → Poor
ACC = 0.5   → Average (for multiclass, better than random)
ACC = 0.8   → Good
ACC = 1.0   → Perfect
```

**Computational cost:** O(clusters³) Hungarian algorithm

**Why important:** Most intuitive metric, direct success measure

---

### Metric 4: F1-Score (Macro-Averaged)

**Formula:**
```
Precision_c = TP_c / (TP_c + FP_c)   (per-class)
Recall_c    = TP_c / (TP_c + FN_c)   (per-class)
F1_c        = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)

Macro F1 = average(F1_c for all classes)
Range: [0, 1]
```

**Interpretation:**
```
F1 = 0.0   → Poor balance of precision/recall
F1 = 0.5   → Moderate performance
F1 = 0.8   → Good balance
F1 = 1.0   → Perfect precision and recall
```

**When to use:** Imbalanced datasets where both FP and FN matter

---

## 📋 Four Metrics Comparison

| Metric | Label-Free | Requires Match | Range | Use Case |
|--------|-----------|----------------|-------|----------|
| **NMI** | ✓ Yes | No | [0,1] | Pure information sharing |
| **ARI** | ✓ Yes | No | [-1,1] | Robust comparison |
| **ACC** | ✗ No | ✓ Yes | [0,1] | Node-level accuracy |
| **F1** | ✗ No | ✓ Yes | [0,1] | Precision-recall balance |

**Recommendation:** Report all four for comprehensive assessment

---

## 🔍 Typical Evaluation Workflow

### Complete Evaluation Pipeline

```python
# Step 1: Get predictions
predictions = model.predict(graph)  # (num_nodes,)

# Step 2: Decode if needed
if soft_assignments:
    predictions = decoding_from_assignment(predictions)

# Step 3: Create evaluator
evaluator = cluster_metrics(ground_truth_labels, predictions)

# Step 4: Compute all metrics
accuracy, nmi, f1_score, ari, aligned_pred = \
    evaluator.evaluateFromLabel(use_acc=True)

# Step 5: Interpret
print(f"NMI: {nmi:.4f}")  # Information: 0.5 good
print(f"ARI: {ari:.4f}")  # Agreement: 0.5 good
print(f"ACC: {accuracy:.4f}")  # Accuracy: 0.75 good
print(f"F1:  {f1_score:.4f}")  # Balance: 0.75 good

# Diagnostic:
if nmi > 0.5 and accuracy < 0.5:
    print("Good information sharing but label mismatch")
if all_metrics > 0.7:
    print("Excellent clustering!")
```

---

## 🛠️ Class: `cluster_metrics`

### Constructor

```python
metrics = cluster_metrics(true_labels, pred_labels)

# Inputs:
#   true_labels: Ground truth (num_nodes,)
#   pred_labels: Predictions (num_nodes,)
```

### Method: `clusterAcc()`

**Purpose:** Compute accuracy with optimal label matching

```python
accuracy, matching = metrics.clusterAcc()

# Returns:
#   accuracy: Single number [0, 1]
#   matching: Dict {pred_cluster_id: true_cluster_id}
```

**Algorithm:**
```
1. Build confusion matrix
2. Hungarian matching
3. Extract diagonal
4. Sum / num_nodes
```

---

### Method: `evaluateFromLabel()`

**Purpose:** Compute all four metrics at once

```python
accuracy, nmi, f1_score, ari, aligned_predictions = \
    metrics.evaluateFromLabel(use_acc=True)

# Returns:
#   accuracy: ACC value
#   nmi: NMI value
#   f1_score: Macro F1 value
#   ari: ARI value
#   aligned_predictions: Hard assignments after matching
```

---

## 📊 Practical Example: Evaluation

```python
# Real scenario: Cora dataset
true_labels = torch.tensor([0, 0, 1, 1, 2, 2, ...])  # 7 classes
predicted = torch.tensor([1, 1, 2, 2, 0, 0, ...])    # arbitrary labels

# Create evaluator
eval_metrics = cluster_metrics(true_labels, predicted)

# Get metrics
acc, nmi, f1, ari, aligned = eval_metrics.evaluateFromLabel(use_acc=True)

# Results (typical for good clustering):
# acc = 0.82 (82% of nodes correctly assigned)
# nmi = 0.65 (good information sharing)
# f1  = 0.81 (balanced precision/recall)
# ari = 0.58 (good agreement)

# aligned[i] now contains optimal label mapping
print(f"Cluster 1 → True class {aligned[1]}")
```

---

## 🎯 When to Use Each Metric

### Use NMI When:
- ✓ Assessing information preservation
- ✓ Don't care about label alignment
- ✓ Comparing with other information-theoretic methods
- ✗ Need interpretable "accuracy"

### Use ARI When:
- ✓ Comparing different clustering methods
- ✓ Want robustness to chance agreement
- ✓ Imbalanced cluster sizes
- ✗ Need intuitive percentage interpretation

### Use ACC When:
- ✓ Most straightforward result
- ✓ Presenting to non-technical audience
- ✓ Primary metric in paper/presentation
- ✗ Label assignment computationally expensive

### Use F1 When:
- ✓ Imbalanced classes/clusters
- ✓ Precision and recall both matter
- ✓ Need balance metric
- ✗ Simple comparison wanted

---

## 📈 Expected Metric Ranges

### Excellent Clustering (>0.7 all metrics)
```
NMI > 0.7   Information well-preserved
ARI > 0.7   Very robust agreement
ACC > 0.8   >80% nodes correct
F1  > 0.8   Balanced classification
```

### Good Clustering (0.5-0.7 range)
```
NMI > 0.5   Decent information sharing
ARI > 0.5   Solid agreement
ACC > 0.75  Most nodes correct
F1  > 0.75  Reasonably balanced
```

### Poor Clustering (<0.3)
```
NMI < 0.3   Little information captured
ARI < 0.3   Near-random agreement
ACC < 0.5   Less than 50% correct
F1  < 0.5   Poor balance
```

---

## 🐛 Common Issues & Solutions

| Symptom | Cause | Solution |
|---------|-------|----------|
| NMI high, ACC low | Label mismatch | Check Hungarian algorithm |
| All metrics = 0 | Wrong data type | Ensure tensors not lists |
| F1 > ACC | Class imbalance | Check label distribution |
| ARI negative | Worse than random | Review model training |

---

## 💡 Advanced Tips

### Confidence in Metrics

```python
# High variance = unreliable
# Run multiple times:
metrics_list = []
for seed in range(5):
    set_seed(seed)
    pred = model.train_and_predict()
    m = cluster_metrics(true_labels, pred)
    metrics_list.append(m.evaluateFromLabel())

# Average
avg_nmi = np.mean([m[1] for m in metrics_list])
std_nmi = np.std([m[1] for m in metrics_list])
print(f"NMI: {avg_nmi:.3f} ± {std_nmi:.3f}")
```

### Debugging Poor Metrics

```python
# Step 1: Check if predictions make sense
unique_pred = torch.unique(pred_labels)
print(f"Num predicted clusters: {len(unique_pred)}")
print(f"Num true classes: {len(torch.unique(true_labels))}")

# Step 2: Check cluster sizes
for c in unique_pred:
    size = (pred_labels == c).sum()
    print(f"Cluster {c}: {size} nodes")

# Step 3: Check majority baselines
baseline_acc = (true_labels == true_labels.mode()[0]).float().mean()
print(f"Baseline accuracy (majority class): {baseline_acc:.3f}")

# If your ACC < baseline: model not learning!
```

