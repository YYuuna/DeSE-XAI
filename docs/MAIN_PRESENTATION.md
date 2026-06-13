# Training & Evaluation Pipeline - Comprehensive Documentation

## 📋 Module Overview

**File:** `main.py`  
**Purpose:** Orchestrates complete training pipeline from data loading to results export  
**Key Responsibilities:** Model training, evaluation, visualization, and explainability export

---

## 🚀 Complete Training Pipeline

### High-Level Execution Flow

```
┌─────────────────────────────────────────────────────────┐
│ 1. INITIALIZATION                                       │
│  • Parse command-line arguments                         │
│  • Set random seeds for reproducibility                 │
│  • Configure GPU/CPU device                             │
│  • Load dataset and initialize data structures          │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 2. MODEL SETUP                                          │
│  • Create DeSE model instance                           │
│  • Initialize optimizer (Adam)                          │
│  • Move model and data to device                        │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 3. TRAINING LOOP (epochs)                               │
│  • Forward pass through model                           │
│  • Compute SE loss + LP loss                            │
│  • Backward propagation                                 │
│  • Optimizer step                                       │
│  • Periodic evaluation (every N epochs)                 │
└──────────────────────┬──────────────────────────────────┘
                       ↓
┌─────────────────────────────────────────────────────────┐
│ 4. POST-TRAINING                                        │
│  • Save best model and results                          │
│  • Export CCTS data for explainability                  │
│  • Generate t-SNE visualizations                        │
│  • Print final metrics and statistics                   │
└─────────────────────────────────────────────────────────┘
```

---

## 🔧 Main Function Architecture

### Function: `train(args)`

**Signature:** `train(args) -> tuple[dict, dict, dict]`

**Returns:**
- `best_cluster_result`: Dict of best metrics for each model
- `best_cluster`: Dict of best metric values
- `best_acc`: Best accuracy achieved

---

### Phase 1: Initialization

```python
# Parse and configure
args = parser.parse_args()
set_seed(args.seed)
device = torch.device('cuda:' + str(args.gpu))

# Load dataset
data = Data(args.dataset, device=device)
num_nodes = data.num_nodes
num_features = data.feature.shape[1]
```

**Dataset Properties Extracted:**
- `data.num_nodes`: Total nodes in graph
- `data.feature`: Node attributes (num_nodes × feature_dim)
- `data.degrees`: Node degrees for normalization
- `data.adj`: Sparse adjacency matrix
- `data.labels`: Ground truth (for evaluation)
- `data.neg_edge_index`: Non-edges for LP loss

---

### Phase 2: Model Creation

```python
# Initialize DeSE model
model = DeSE(args, data.feature, device)
model.to(device)

# Create optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-5
)
```

**Model Configuration:**
- Hyperparameters from `args` (height, embed_dim, num_clusters_layer, etc.)
- Features provided for embedding dimension inference
- Device specified for GPU/CPU execution

---

### Phase 3: Training Loop

```python
for epoch in range(args.epochs):
    # Forward pass
    s_dic, tree_node_embed_dic, g_dic = model(
        data.adj, data.feature, data.degrees
    )
    
    # Compute losses
    se_loss = args.se_lamda * model.calculate_se_loss1()
    lp_loss = args.lp_lamda * model.calculate_lp_loss(
        g_dic[args.height],
        data.neg_edge_index,
        tree_node_embed_dic[args.height]
    )
    loss = se_loss + lp_loss
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Periodic evaluation
    if epoch % args.verbose == 0:
        # Compute metrics
        pred_cluster = get_hard_assignments()
        metrics = evaluate_clustering(data.labels, pred_cluster)
        
        # Track best results
        if metrics['nmi'] > best_nmi:
            best_model_state = model.state_dict()
            best_metrics = metrics
```

**Loss Computation:**
- `SE loss`: Structural Entropy (measured by `calculate_se_loss1()`)
- `LP loss`: Link Prediction (measured by `calculate_lp_loss()`)
- `Total loss`: se_lamda * SE + lp_lamda * LP

**Evaluation Frequency:** Every `args.verbose` epochs

---

### Phase 4: Post-Training Processing

```python
# Save best model
torch.save(best_model_state, f'{args.dataset}_{best_metric}.pt')

# Export CCTS data (for explainability)
if args.export_ccts:
    export_ccts_dataset()
    
# Visualize embeddings
if args.fig_network:
    draw_network()
    
# Print summary statistics
print(f"Best NMI: {best_metrics['nmi']:.4f}")
print(f"Best ARI: {best_metrics['ari']:.4f}")
print(f"Best ACC: {best_metrics['acc']:.4f}")
print(f"Best F1: {best_metrics['f1']:.4f}")
```

---

## 📊 Evaluation Metrics

### Four Clustering Quality Perspectives

| Metric | Range | Use Case | Interpretation |
|--------|-------|----------|-----------------|
| **NMI** | [0, 1] | Information sharing | 0.5 = moderate agreement |
| **ARI** | [-1, 1] | Robust agreement | 0 = random, 1 = perfect |
| **ACC** | [0, 1] | Node accuracy | Requires label alignment |
| **F1** | [0, 1] | Precision-recall balance | Macro-averaged |

### Key Concepts

**Why Multiple Metrics?**
- Single metric may mislead
- Different perspectives reveal different aspects
- Robust assessment requires consensus

**Why Label Alignment?**
- Predicted clusters have arbitrary labels (0, 1, 2, ...)
- True labels have fixed meanings
- Hungarian algorithm finds optimal matching
- Examples:
  - True: [0, 0, 1, 1, 2, 2, ...]
  - Pred: [1, 1, 2, 2, 0, 0, ...] (arbitrary labels)
  - After alignment: [0, 0, 1, 1, 2, 2, ...] (matches true)

---

## 🔄 Data Flow During Training

### Forward Pass

```
Input Graph (sparse COO tensor)
       ↓
Dataset preprocessing
  • Load adjacency matrix
  • Extract node features
  • Compute node degrees
  • Generate negative edges
       ↓
Model forward pass
  • Mix structural + feature adjacencies
  • Process through hierarchy levels
  • Output soft assignments at each level
       ↓
Loss computation
  • SE loss: Graph structure quality
  • LP loss: Embedding preservation
       ↓
Backpropagation
  • Gradients flow back through hierarchy
  • Update model parameters
```

### Backward Pass

```
Loss (scalar)
    ↓
Compute gradients w.r.t. parameters
    ↓
Chain rule through hierarchy
  • SE loss ← assignments ← embeddings ← features
  • LP loss ← embeddings ← features
    ↓
Update: θ_new = θ_old - lr * gradient
```

---

## 💾 Output Functions

### 1. Export CCTS Dataset

**Purpose:** Prepare data for explainability analysis (CCTS module)

**Exports:**
```
data/graph/{dataset}.txt
  • Edge list format: src_id dst_id weight
  • One edge per line
  • Used to understand edge importance

community_partitions/{method}/{dataset}_partition.pkl
  • Pickle file containing:
    - Node-to-cluster assignments
    - Cluster metadata
  • Used for interpretability analysis
```

**Why separate files?**
- CCTS module needs specific format
- Enables post-hoc explainability
- Decouples clustering from interpretation

---

### 2. Save Model and Results

**Checkpoints:**
```
save_model/{dataset}_{metric}_{value}.pt
  • Stores model weights only (not full model)
  • Multiple checkpoints by metric (acc, ari, f1, nmi)
  • Enables comparison and ablation
```

**Results File:**
```
output/{dataset}.result
  • JSON or pickle format
  • Contains:
    - Final metrics (NMI, ARI, ACC, F1)
    - Predictions
    - Training loss history
    - Hyperparameters used
```

---

### 3. Visualization

**t-SNE Network Visualization:**

```python
def draw_network():
    """Generate 2D embedding visualization"""
    # Extract embeddings from model
    embeddings = model.tree_node_embed_dic[height]
    
    # Dimensionality reduction: high-dim → 2D
    embeddings_2d = tsne(embeddings, perplexity=30)
    
    # Visualization
    for node_id in range(num_nodes):
        cluster_id = predictions[node_id]
        color = cluster_colors[cluster_id]
        plot_point(embeddings_2d[node_id], color=color)
    
    save_figure(f'figure/{dataset}_clusters.png')
```

**What it shows:**
- Each dot = one node
- Position = learned embedding (projected to 2D)
- Color = assigned cluster
- Nearby dots = similar nodes
- Color regions = clusters

---

## 🎯 Key Functions

### Function: `export_ccts_dataset()`

**Purpose:** Export clustering results for explainability analysis

**Input:** 
- Model predictions
- Graph structure
- Node features

**Output:**
- Edge list file
- Partition dictionary

**Workflow:**
```
1. Decode hard assignments from model.hard_dic
2. Create partition dict: {node_id: cluster_id}
3. Save graph edges to text file
4. Pickle partition dict
5. Save to community_partitions directory
```

---

### Function: `draw_network()`

**Purpose:** Visualize learned clustering via t-SNE

**Input:**
- Embeddings from `tree_node_embed_dic`
- Cluster assignments
- Dataset name

**Output:**
- PNG figure in `figure/` directory

**Process:**
```
1. Extract node embeddings (high-dimensional)
2. Run t-SNE (reduce to 2D)
3. Assign colors by cluster
4. Plot points with clusters
5. Save figure
```

---

## 📈 Training Metrics & Tracking

### Stored Metrics

```python
best_cluster_result = {
    'dataset': result_dict,  # For each model
    'result': {
        'acc': [...],  # Accuracy history
        'nmi': [...],  # NMI history
        'f1': [...],   # F1-score history
        'ari': [...],  # ARI history
    }
}

best_cluster = {
    'acc': best_acc_value,
    'nmi': best_nmi_value,
    'f1': best_f1_value,
    'ari': best_ari_value,
}
```

### Tracking Logic

```
For each epoch:
  If epoch % verbose == 0:
    Compute 4 metrics (NMI, ARI, ACC, F1)
    For each metric:
      If current > best:
        Update best_cluster[metric]
        Save model checkpoint
        Update best_cluster_result
```

---

## ⚙️ Configuration Parameters

### Critical Parameters

| Param | Default | Range | Impact |
|-------|---------|-------|--------|
| `epochs` | 500 | 100-1000 | Training duration |
| `lr` | 0.01 | 0.001-0.1 | Convergence speed |
| `verbose` | 10 | 1-50 | Evaluation frequency |
| `se_lamda` | 0.01 | 0.001-1 | SE loss weight |
| `lp_lamda` | 1.0 | 0.1-2 | LP loss weight |

### Device & Hardware

```python
device = torch.device('cuda:' + str(args.gpu))
# Enables GPU acceleration for:
# - Matrix operations
# - Neural network computations
# - Loss calculations
```

---

## 🧪 Typical Usage Patterns

### Quick Test
```bash
python main.py --dataset Cora --epochs 10 --verbose 5 --height 2
# Fast run with small epoch count
# Check if code runs without errors
```

### Standard Training
```bash
python main.py --dataset Cora --epochs 500 --lr 0.01 --height 3
# Full training with reasonable hyperparameters
# ~5-10 min runtime on GPU
```

### Detailed Analysis
```bash
python main.py --dataset Cora --epochs 1000 --verbose 5 \
  --se_lamda 0.01 --lp_lamda 1.0 \
  --save --export_ccts --fig_network
# Save model, export for explainability, generate viz
```

---

## 🔍 Results Interpretation

### Expected Performance Ranges

**Citation Networks (Cora, Citeseer, Pubmed):**
- NMI: 0.50-0.70
- ARI: 0.40-0.65
- ACC: 0.75-0.85
- F1: 0.75-0.85

**E-commerce Networks (Computers, Photo):**
- NMI: 0.45-0.65
- ARI: 0.35-0.60
- ACC: 0.70-0.80
- F1: 0.70-0.80

**Co-authorship Networks (CS, Physics):**
- NMI: 0.40-0.60
- ARI: 0.30-0.55
- ACC: 0.65-0.75
- F1: 0.65-0.75

### Metric Diagnostics

| Symptom | Diagnosis | Solution |
|---------|-----------|----------|
| All metrics low | Poor clustering | Tune hyperparameters |
| Loss not decreasing | Learning rate too high/low | Adjust `lr` |
| Converges too fast | May be local minimum | Increase complexity |
| Large variance | High randomness | Multiple runs, average |

---

## 🐛 Debugging & Troubleshooting

### Issue: Loss becomes NaN

**Causes:**
- Numerical instability in loss computation
- Extreme gradients
- Invalid graph structure

**Solutions:**
```python
# Enable anomaly detection
torch.autograd.set_detect_anomaly(True)

# Check gradient norms
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm()}")
```

### Issue: Clustering quality not improving

**Checklist:**
1. Verify data loading: `print(data.adj.shape, data.feature.shape)`
2. Check hyperparameters: Are they reasonable?
3. Verify loss computation: Add print statements
4. Try different random seeds
5. Increase training epochs

### Issue: GPU out of memory

**Solutions:**
- Reduce `embed_dim`
- Use smaller dataset or graph
- Reduce `height` (fewer hierarchy levels)
- Use batch processing

---

## 📊 File Output Summary

After training, expect:

```
save_model/
  ├─ Cora_acc.pt
  ├─ Cora_ari.pt
  ├─ Cora_f1.pt
  └─ Cora_nmi.pt

output/
  └─ Cora.result

figure/
  └─ Cora_clusters_tsne.png

community_partitions/DeSE/
  └─ Cora_partition.pkl

data/graph/
  └─ Cora.txt
```

---

## 🔄 Reproducibility

**Fixed Random Seeds:**
```python
set_seed(args.seed)
# Sets:
# - numpy.random.seed
# - torch.manual_seed
# - torch.cuda.manual_seed_all
# - CUBLAS_WORKSPACE_CONFIG
```

**For full reproducibility:**
1. Use same dataset, device, seed
2. Use same hyperparameters
3. Run on same hardware (slight variations expected)
4. Use `--verbose 1` to track every epoch

