# Command-Line Arguments & Configuration - Comprehensive Documentation

## 📋 Module Overview

**File:** `parser.py`  
**Purpose:** Centralized command-line interface for reproducible experiments  
**Key Feature:** 42 hyperparameters with sensible defaults and documentation

---

## 🎯 Parser Purpose & Design

### Why Centralized Parser?

```
Without parser:
  - Hyperparams scattered in code
  - Hard to reproduce experiments
  - Difficult to run ablation studies
  - Manual parameter edits = error-prone

With parser:
  - Single source of truth
  - Reproducible via command-line
  - Easy experiment tracking
  - Script automation friendly
```

---

## 🔧 Argument Categories & Structure

### Seven Hyperparameter Groups

```
┌─────────────────────────────────────────────────────────┐
│ 1. DATASET & COMPUTATION (3 args)                       │
│    Dataset selection, hardware configuration            │
├─────────────────────────────────────────────────────────┤
│ 2. TRAINING DYNAMICS (3 args)                           │
│    Epochs, learning rate, verbosity                     │
├─────────────────────────────────────────────────────────┤
│ 3. HIERARCHY & STRUCTURE (3 args)                       │
│    Network depth, cluster counts, decay rates           │
├─────────────────────────────────────────────────────────┤
│ 4. MODEL ARCHITECTURE (4 args)                          │
│    Embedding dimension, KNN neighbors, activation       │
├─────────────────────────────────────────────────────────┤
│ 5. LOSS BALANCING (3 args)                              │
│    Weight SE loss, LP loss, and regularization          │
├─────────────────────────────────────────────────────────┤
│ 6. OUTPUT & EXPORT (4 args)                             │
│    Save models, export data, generate figures           │
├─────────────────────────────────────────────────────────┤
│ 7. HARDWARE & REPRODUCIBILITY (2 args)                  │
│    GPU device, random seed                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📋 Complete Argument Reference

### GROUP 1: Dataset & Computation

```bash
--dataset CORA
  Type: str, Choices: [Cora, Citeseer, Pubmed, Computers, Photo, CS, Physics]
  Default: Cora
  Interpretation: Which benchmark dataset to use
  Use: Mandatory, change for different experiments

--gpu 1
  Type: int, Range: [0, 4]
  Default: 1
  Interpretation: GPU device ID (0, 1, 2, ...)
  Use: Select GPU if multi-GPU system available
  Note: Use --gpu 0 for GPU:0, no GPU for CPU fallback needed

--seed 42
  Type: int, Range: [0, ∞)
  Default: 42
  Interpretation: Random seed for reproducibility
  Use: Set fixed seed for deterministic results
  Note: Different seeds → Different results from randomness
```

---

### GROUP 2: Training Dynamics

```bash
--epochs 500
  Type: int, Range: [10, 5000]
  Default: 500
  Interpretation: Number of training iterations
  Use: More epochs = longer training but potentially better
  Tradeoff:
    - 100 epochs: Fast (1-2 min), may not converge
    - 500 epochs: Balanced (5-10 min), typically converges
    - 1000 epochs: Thorough (15-20 min), marginal improvement

--lr 0.01
  Type: float, Range: [0.0001, 1.0]
  Default: 0.01
  Interpretation: Learning rate for Adam optimizer
  Use: Controls optimization step size
  Guidelines:
    - lr=0.1: Fast but unstable, may overshoot
    - lr=0.01: Balanced, recommended default
    - lr=0.001: Slow, takes more epochs to converge
  Debugging: If loss diverges → decrease lr; if no progress → increase lr

--verbose 10
  Type: int, Range: [1, 100]
  Default: 10
  Interpretation: Evaluate metrics every N epochs
  Use: Print progress frequency
  Note:
    - verbose=1: Print every epoch (verbose output)
    - verbose=10: Print every 10 epochs (clean output)
    - verbose=100: Print every 100 epochs (sparse)
```

---

### GROUP 3: Hierarchy & Structure

```bash
--height 2
  Type: int, Range: [1, 5]
  Default: 2
  Interpretation: Number of hierarchy levels in model
  Use: Controls multi-scale clustering
  Impact: **HIGH** - Critical parameter
  Levels:
    - height=1: Flat clustering (single level)
    - height=2: 2-level hierarchy (typical)
    - height=3: 3-level hierarchy (more fine-grained)
  Tradeoff:
    - Smaller height: Faster, coarser clusters
    - Larger height: Slower, finer clusters

--num_clusters_layer [50, 10]
  Type: list of ints
  Default: Auto-computed
  Interpretation: Number of clusters at each hierarchy level
  Example: [50, 10] means:
    - Level 2: 50 clusters
    - Level 1: 10 clusters (final output)
  Use:
    - If None: Auto-compute based on num_nodes and decay_rate
    - Otherwise: Manual specification for fine-grained control

--decay_rate 2
  Type: int, Range: [2, 10]
  Default: 2 (if auto-computing clusters)
  Interpretation: Exponential decay factor for cluster reduction
  Example with 2700 nodes, height=3:
    - Level 3: 2700 / 2^0 = 2700 clusters
    - Level 2: 2700 / 2^1 = 1350 clusters
    - Level 1: 2700 / 2^2 = 675 clusters
  Use: Only affects auto-computed num_clusters_layer
```

---

### GROUP 4: Model Architecture

```bash
--embed_dim 16
  Type: int, Range: [8, 256]
  Default: 16
  Interpretation: Dimension of learned node embeddings
  Impact: MEDIUM sensitivity
  Tradeoff:
    - embed_dim=8: Fast, limited expressiveness
    - embed_dim=16: Balanced (recommended)
    - embed_dim=64: More expressive, 4x slower

--k 2
  Type: int, Range: [1, 20]
  Default: 2
  Interpretation: KNN neighbors for feature-based adjacency
  Use: Connects similar nodes in embedding space
  Effect:
    - k=1: Only nearest neighbor
    - k=2-5: Typical range
    - k=10+: Very connected feature graph

--activation relu
  Type: str, Choices: [relu, elu, sigmoid, None]
  Default: relu
  Interpretation: Non-linearity for neural network layers
  Options:
    - relu: max(0, x), standard for GNNs
    - elu: smooth variant of ReLU
    - sigmoid: probability-like output
  Use: Rarely needs changing from relu

--dropout 0.1
  Type: float, Range: [0, 0.5]
  Default: 0.1
  Interpretation: Dropout rate for regularization
  Use: Prevents overfitting
  Guidelines:
    - 0: No dropout
    - 0.1-0.2: Typical
    - 0.5: Strong regularization
```

---

### GROUP 5: Loss Balancing

```bash
--se_lamda 0.01
  Type: float, Range: [0.0001, 1.0]
  Default: 0.01
  Interpretation: Weight for Structural Entropy loss
  Impact: **HIGH** sensitivity
  Effect:
    - se_lamda=0.001: Less weight on structure
    - se_lamda=0.01: Balanced (recommended)
    - se_lamda=0.1: Heavy emphasis on structure
  Typical: 0.001-0.1, usually 0.01

--lp_lamda 1.0
  Type: float, Range: [0.1, 10.0]
  Default: 1.0
  Interpretation: Weight for Link Prediction loss
  Impact: **HIGH** sensitivity
  Effect:
    - lp_lamda=0.1: Light embedding preservation
    - lp_lamda=1.0: Balanced (recommended)
    - lp_lamda=10.0: Heavy emphasis on LP
  Typical: 0.5-2.0, usually 1.0

--beta_f 0.2
  Type: float, Range: [0, 1]
  Default: 0.2
  Interpretation: Weight mixing structural + feature adjacencies
  Formula: adj = adj_structural + beta_f * adj_feature
  Effect:
    - beta_f=0: Pure graph topology (structure only)
    - beta_f=0.5: Balanced topology and features
    - beta_f=1.0: Pure feature similarity (no structure)
  Typical: 0.1-0.5, usually 0.2
```

---

### GROUP 6: Output & Export

```bash
--save True
  Type: bool
  Default: False
  Interpretation: Save best model checkpoints
  Output:
    - save_model/{dataset}_{metric}.pt
    - One checkpoint per metric (acc, ari, f1, nmi)

--export_ccts True
  Type: bool
  Default: False
  Interpretation: Export data for CCTS explainability
  Output:
    - data/graph/{dataset}.txt (edge list)
    - community_partitions/{method}/{dataset}_partition.pkl

--fig_network True
  Type: bool
  Default: False
  Interpretation: Generate t-SNE visualization
  Output:
    - figure/{dataset}_clusters.png

--ccts_method DeSE
  Type: str
  Default: DeSE
  Interpretation: Name for community partition directory
```

---

### GROUP 7: Hardware & Reproducibility

```bash
--gpu 1
  (Already covered in GROUP 1)

--seed 42
  (Already covered in GROUP 1)
```

---

## 📊 Hyperparameter Sensitivity Analysis

### Critical Parameters (HIGH Impact)

```
SENSITIVITY RANKING:

1. num_clusters_layer: **CRITICAL**
   - Wrong cluster counts = wrong solution space
   - No recovery possible

2. height: **CRITICAL**
   - Changes model architecture
   - Affects all hierarchy levels

3. se_lamda / lp_lamda: **HIGH**
   - Directly control loss balancing
   - Can flip good/poor performance

4. lr: **HIGH**
   - Too high: divergence
   - Too low: no progress
   - Sweet spot narrow
```

### Moderate Parameters (MEDIUM Impact)

```
5. embed_dim:
   - More dims = more expressive
   - But also slower training
   - 16-64 typical range

6. beta_f:
   - Topology vs. feature mix
   - Significant effect but less critical

7. k:
   - KNN neighborhood size
   - Affects feature graph connectivity
```

### Low Sensitivity Parameters

```
8. activation:
   - ReLU works well universally
   - Rarely needs tuning

9. dropout:
   - Helps with overfitting
   - Usually 0.1 fine

10. epochs:
    - More is better (within limits)
    - 500-1000 typical sufficient
```

---

## 🔧 Recommended Configurations

### QUICK TEST (Debug Mode)

```bash
python main.py \
  --dataset Cora \
  --epochs 10 \
  --verbose 5 \
  --height 2 \
  --embed_dim 8
# Runtime: <1 min
# Purpose: Check code correctness
```

### STANDARD TRAINING (Balanced)

```bash
python main.py \
  --dataset Cora \
  --epochs 500 \
  --lr 0.01 \
  --se_lamda 0.01 \
  --lp_lamda 1.0 \
  --beta_f 0.2 \
  --height 2 \
  --embed_dim 16 \
  --k 2
# Runtime: 5-10 min
# Purpose: Typical experiment
```

### DETAILED ANALYSIS (Thorough)

```bash
python main.py \
  --dataset Cora \
  --epochs 1000 \
  --verbose 5 \
  --height 3 \
  --embed_dim 32 \
  --save True \
  --export_ccts True \
  --fig_network True \
  --seed 42
# Runtime: 20-30 min
# Purpose: Publication-quality results
```

### FAST EXPLORATION (Multiple Seeds)

```bash
for seed in {42,123,456,789,999}; do
  python main.py \
    --dataset Cora \
    --epochs 300 \
    --lr 0.01 \
    --seed $seed
done
# Multiple runs with different initialization
# Purpose: Robustness assessment
```

### COMPREHENSIVE HYPERPARAMETER SWEEP

```bash
for se in 0.001 0.01 0.1; do
  for lp in 0.5 1.0 2.0; do
    for embed in 16 32 64; do
      python main.py \
        --dataset Cora \
        --se_lamda $se \
        --lp_lamda $lp \
        --embed_dim $embed
    done
  done
done
# 27 combinations to test
# Purpose: Find best hyperparameters
```

---

## 🎯 Parameter Tuning Guide

### If NMI/ARI Low (Bad Clustering)

```bash
# Check 1: Are clusters meaningful?
# Solution: Increase se_lamda (emphasize structure)
--se_lamda 0.05  # from 0.01

# Check 2: Are embeddings good?
# Solution: Increase lp_lamda (preserve graph)
--lp_lamda 2.0  # from 1.0

# Check 3: Too coarse clustering?
# Solution: Increase final cluster count
--num_clusters_layer [100, 20]  # more granular

# Check 4: Model capacity too small?
# Solution: Increase embedding dimension
--embed_dim 32  # from 16
```

### If Training Diverges (Loss → NaN/Inf)

```bash
# Solution 1: Decrease learning rate
--lr 0.005  # from 0.01

# Solution 2: Add regularization
--dropout 0.2  # from 0.1
--beta_f 0.1   # reduce feature mixing

# Solution 3: Use smaller model
--height 1  # single level
--embed_dim 8  # smaller embeddings
```

### If Too Slow (Slow Training)

```bash
# Solution 1: Reduce hierarchy
--height 1  # from 2

# Solution 2: Smaller embeddings
--embed_dim 8  # from 16

# Solution 3: Fewer epochs needed?
--epochs 300  # from 500

# Solution 4: Reduce KNN
--k 1  # from 2
```

### If Model Overfits (Train metrics good, eval bad)

```bash
# Solution 1: Add dropout
--dropout 0.3  # from 0.1

# Solution 2: Reduce model size
--embed_dim 8  # from 16

# Solution 3: Reduce KNN neighbors (less capacity)
--k 1  # from 2

# Solution 4: Shorter training
--epochs 300  # from 500
```

---

## 📈 Monitoring Metrics During Training

### With `--verbose 10` output:

```
Epoch 0:  Loss=2.341, NMI=0.34, ARI=0.12, ACC=0.45, F1=0.42
Epoch 10: Loss=1.902, NMI=0.42, ARI=0.25, ACC=0.58, F1=0.56
Epoch 20: Loss=1.567, NMI=0.51, ARI=0.38, ACC=0.68, F1=0.67
Epoch 30: Loss=1.234, NMI=0.58, ARI=0.48, ACC=0.75, F1=0.74
...
Epoch 490: Loss=0.892, NMI=0.65, ARI=0.55, ACC=0.82, F1=0.81

Best Results:
  NMI:  0.65 (epoch 485)
  ARI:  0.55 (epoch 480)
  ACC:  0.82 (epoch 490)
  F1:   0.81 (epoch 490)
```

### What to Look For:
- Loss decreasing (generally)
- Metrics increasing
- Stabilizing after ~200-300 epochs
- Not diverging (going to inf/nan)

---

## 🔄 Reproducibility Notes

### For Exact Reproducibility

```python
# Set these IDENTICALLY:
--seed 42          # Same random seed
--dataset Cora     # Same data
--gpu 0            # Same device (if GPU differs, slight differences)
--epochs 500       # Same iterations
--lr 0.01          # Same hyperparams
--se_lamda 0.01
--lp_lamda 1.0
--beta_f 0.2
# Result: Identical results across runs
```

### For Robust Assessment

```bash
# Run multiple seeds:
for seed in 42 123 456 789 999; do
  python main.py --dataset Cora --seed $seed --save True
done

# Aggregate results:
# Average: mean of 5 runs
# StdDev: variability estimate
# Best: maximum performance
```

---

## 🐛 Common Parameter Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| `--se_lamda 1.0` | Over-weighted structure | Decrease to 0.01-0.1 |
| `--lr 1.0` | Loss diverges immediately | Decrease to 0.001-0.01 |
| `--height 5` | Too slow, deep hierarchy | Reduce to 2-3 |
| `--epochs 50` | Underfitting | Increase to 500+ |
| Wrong `--seed` | Not reproducible | Use fixed seed |
| Forgot `--gpu` | CPU slow | Add device ID |

---

## 💾 Output Files

### After running main.py, expect:

```
Results stored in:
✓ save_model/
  - Cora_acc.pt (best accuracy checkpoint)
  - Cora_nmi.pt (best NMI checkpoint)
  - Cora_f1.pt  (best F1 checkpoint)
  - Cora_ari.pt (best ARI checkpoint)

✓ output/
  - Cora.result (metrics JSON/pickle)

✓ figure/ (if --fig_network True)
  - Cora_clusters_tsne.png

✓ data/graph/ (if --export_ccts True)
  - Cora.txt (edge list)

✓ community_partitions/ (if --export_ccts True)
  - DeSE/Cora_partition.pkl
```

