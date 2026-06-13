# CCTS: Community-aware Clustering Transparency System - Comprehensive Documentation

## 📋 Module Overview

**File:** `CCTS.py`  
**Full Name:** Community-aware Clustering Transparency System  
**Purpose:** Interpretability & explainability analysis for clustering results  
**Key Feature:** Identifies explainable regions within predicted communities

---

## 🎯 CCTS Framework Philosophy

### What is CCTS?

CCTS is an **explainability layer** that answers the question:

> **"Why did the model assign this node to this cluster?"**

Instead of treating clusters as black boxes, CCTS:
1. Analyzes each community independently
2. Finds interpretable neighborhoods
3. Visualizes overlapping regions
4. Provides human-readable explanations

---

## 🏗️ CCTS Architecture

### Three-Layer System

```
┌─────────────────────────────────────────────────┐
│ LAYER 1: CLUSTERING RESULTS                     │
│ - Predicted cluster assignments                 │
│ - Graph structure                               │
│ - Node features                                 │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ LAYER 2: COMMUNITY ANALYSIS (CommunityAnalyzer) │
│ - Find best representative center node          │
│ - Determine optimal distance threshold          │
│ - Balance precision vs. error rate              │
└────────────────┬────────────────────────────────┘
                 ↓
┌─────────────────────────────────────────────────┐
│ LAYER 3: VISUALIZATION & EXPORT                 │
│ - Color-coded community visualization           │
│ - Community-aware graph layout                  │
│ - Explainability report                         │
└─────────────────────────────────────────────────┘
```

---

## 🔍 Core Algorithm: Finding Explainable Regions

### Problem Statement

```
Given a predicted cluster C with mixed quality:

True Labels:
  Class 0: [n1, n5, n9, ...]         (pure nodes)
  Class 1: [n2, n7, n11, ...]        (pure nodes)
  
Predicted Cluster:
  [n1, n2, n5, n7, n9, n11, ...]     (mixed!)
       └────┬────┘
      Overlapping region - confusing!

Goal: Find a "center" region that is pure (coherent)
      Explain cluster by its best part
```

### Algorithm: `find_best_center_and_threshold()`

**Step 1: For each community**

```python
community_c = nodes_assigned_to_cluster_c

# Compute precision and error rate
precision = (true_positive_nodes) / (community_nodes)
error_rate = (false_positive_nodes) / (community_nodes)
```

**Step 2: Objective function (precision-error trade-off)**

```
score = α × precision - β × error_rate

Parameters:
  α = weight for precision (how much we value correctness)
  β = weight for error rate (how much we penalize mistakes)
  
Trade-off:
  α=1, β=0: Only precision (maximize correctness)
  α=0.5, β=0.5: Balanced
  α=0, β=1: Only minimize errors
```

**Step 3: For each candidate center node**

```python
for center_node in community_c:
    # Try distance thresholds
    for threshold in [1, 2, 3, ...]:
        # Include neighbors within distance
        neighborhood = neighbors_within_distance(center, threshold)
        
        # Compute precision/error on neighborhood
        neighborhood_precision = compute_precision(neighborhood)
        neighborhood_error = compute_error(neighborhood)
        
        # Score
        score = α × precision - β × error_rate
        
        # Track best
        if score > best_score:
            best_score = score
            best_center = center_node
            best_threshold = threshold
```

**Step 4: Result**

```
Output:
  - best_center: Most representative node
  - best_threshold: Distance for coherent neighborhood
  - community_region: All nodes within threshold of center

Interpretation:
  "Cluster C is best explained by nodes near node {best_center}"
  "These {num_nodes} nodes form the core of the cluster"
```

---

## 📊 Key Concept: Precision vs. Error Rate

### Visualization

```
Center Node C (best representative)
       ↓
   ○ ○ ○
  ○ o o o ○
 ○ o★o o ○
  ○ o o o ○
   ○ ○ ○

★ = Center
o = Nodes within threshold (coherent region)
○ = Nodes outside threshold (noise)

Precision = o_count / (o_count + ○_count)
          High if most included nodes are truly in same class

Error Rate = ○_count / total_count
           Low if few noisy nodes included
```

---

## 🔧 Class: `CommunityAnalyzer`

### Constructor

```python
analyzer = CommunityAnalyzer(
    node_labels,           # Ground truth labels
    predicted_clusters,    # Predicted cluster assignments
    graph,                 # Graph structure
    alpha=1.0,             # Precision weight
    beta=0.0               # Error weight (disabled by default)
)
```

### Main Method: `analyze()`

```python
results = analyzer.analyze()

# Returns: Dict with analysis for each community
# {
#     'community_0': {
#         'center_node': 42,
#         'threshold': 2,
#         'precision': 0.92,
#         'error_rate': 0.08,
#         'coherent_nodes': [n1, n5, n9, ...],
#         'size': 120
#     },
#     'community_1': {...},
#     ...
# }
```

---

### Method: `calculate_precision_and_error_rate()`

**Purpose:** Evaluate cluster purity

```python
precision, error = analyzer.calculate_precision_and_error_rate(
    community_nodes,       # Which nodes in this cluster?
    true_labels            # Ground truth
)

# Precision: % of nodes with majority label
# Error: % of nodes with minority labels
```

**Example:**
```
Community = [n1, n2, n3, n4, n5]
True labels = [0, 0, 0, 1, 2]

Majority label = 0 (3 nodes)
Precision = 3/5 = 0.60

Error = (5-3)/5 = 0.40
```

---

### Method: `find_best_center_and_threshold()`

**Purpose:** Find optimal center node and distance threshold

```python
center, threshold, score = analyzer.find_best_center_and_threshold(
    community_idx          # Which cluster?
)

# Returns:
#   center: Node ID of best center
#   threshold: Optimal distance limit
#   score: Objective function value
```

**Algorithm Flow:**
```
For each node in community:
  For each distance threshold 1...max_distance:
    Compute score = α×precision - β×error
    Save if best

Return: (best_node, best_threshold, best_score)
```

---

### Method: `determine_proportion()`

**Purpose:** Compute what fraction of community to include

```python
proportion = analyzer.determine_proportion(
    precision_value,       # How pure is the region?
    error_rate_value       # How much noise?
)

# Sigmoid-based: high precision → include more nodes
# Returns: Fraction [0, 1] to include in visualization
```

**Formula:**
```
proportion = sigmoid((precision - error) * slope)
           = 1 / (1 + exp(-(precision - error) * slope))

Intuition:
  High precision & low error → proportion ≈ 1.0 (include all)
  Mixed precision/error → proportion ≈ 0.5 (include half)
  Low precision & high error → proportion ≈ 0.0 (include few)
```

---

### Method: `determine_threshold_range()`

**Purpose:** Limit search space for consistency

```python
min_thresh, max_thresh = analyzer.determine_threshold_range(
    num_nodes_in_community
)

# Ensures:
#   - Doesn't search extreme values
#   - Consistent across communities
#   - Computational efficiency
```

---

## 🎨 Visualization: `plot_ccts_results()`

### Color Coding Scheme

```
Node Color Legend:

■ RED (Overlap Nodes)
  Nodes in multiple communities (ambiguous assignment)
  Important: These are clustering mistakes!

■ GREEN (Community-Only)
  Nodes only in this community (pure assignment)
  High confidence nodes

■ BLUE (Threshold-Only)
  Nodes outside threshold but in community
  Lower confidence nodes

■ GRAY (Other)
  Nodes not in this community
  Background of visualization
```

### Visualization Features

```
Input:
  - Graph structure
  - Predicted clusters
  - Explainability analysis

Output:
  - Each community plotted separately
  - Nodes colored by type (overlap/community/threshold/other)
  - Spatial layout using community-aware algorithm
  - Edge drawing showing connections

Layout Strategy:
  1. Global: Place communities in circle
  2. Local: Within community, use spring layout
  3. Result: Inter-community structure clear
             Intra-community structure detailed
```

---

## 🔗 Method: `compute_cluster_layout()`

### Two-Level Layout Algorithm

```
STEP 1: Global Positioning
  ├─ Arrange communities in circle
  ├─ Cluster C placed at angle θ_c = 2π×c/num_clusters
  └─ Each community center positioned

STEP 2: Local Positioning (within each community)
  ├─ Spring layout for nodes in community
  ├─ Attractive forces: connected nodes
  ├─ Repulsive forces: non-connected nodes
  └─ Converges to local minimum (nodes spread nicely)

STEP 3: Final Positions
  └─ Global center + local offset for each node

Result:
  - Communities visually separated
  - Internal structure visible
  - Overlap regions clear
```

---

## 📊 Typical Analysis Workflow

### Complete Pipeline

```python
# 1. Load results
predicted_clusters = model.get_hard_assignments()
graph = load_graph()
true_labels = dataset.labels

# 2. Initialize analyzer
analyzer = CommunityAnalyzer(
    node_labels=true_labels,
    predicted_clusters=predicted_clusters,
    graph=graph,
    alpha=1.0,  # Emphasize precision
    beta=0.1    # Light error rate penalty
)

# 3. Analyze each community
results = analyzer.analyze()

# 4. Print explanations
for community_id, analysis in results.items():
    print(f"\nCommunity {community_id}:")
    print(f"  Best explained by node {analysis['center_node']}")
    print(f"  Within distance {analysis['threshold']}")
    print(f"  Precision: {analysis['precision']:.2%}")
    print(f"  Nodes: {len(analysis['coherent_nodes'])}")

# 5. Visualize
plot_ccts_results(
    graph=graph,
    predicted_clusters=predicted_clusters,
    analysis_results=results,
    output_file='community_analysis.png'
)

# 6. Export for further analysis
export_analysis(results, 'community_analysis.json')
```

---

## 💡 Interpretation Examples

### Example 1: High-Quality Community

```
Community 5 Analysis:
  Center node: 1523
  Threshold: 2 hops
  Precision: 0.94
  Error rate: 0.06
  
Interpretation:
  ✓ "This is a GOOD cluster"
  ✓ "It's well-defined around node 1523"
  ✓ "94% of included nodes match ground truth"
  ✓ "Only 6% noise/overlap"
  
Action: Trust this clustering in this region
```

---

### Example 2: Mixed-Quality Community

```
Community 3 Analysis:
  Center node: 847
  Threshold: 1 hop
  Precision: 0.65
  Error rate: 0.35
  
Interpretation:
  ⚠ "This cluster has quality issues"
  ⚠ "Only 65% of nodes truly belong"
  ⚠ "35% are misclassified or overlapping"
  ⚠ "Best part: immediate neighbors of node 847"
  
Action: Use with caution, investigate misclassifications
```

---

### Example 3: Low-Quality Community

```
Community 7 Analysis:
  Center node: 2101
  Threshold: 0 hops (just center node!)
  Precision: 0.50
  Error rate: 0.50
  
Interpretation:
  ✗ "This cluster is POOR quality"
  ✗ "Even the 'best' node is only 50% pure"
  ✗ "Too much mixing/overlap"
  ✗ "No coherent neighborhood found"
  
Action: This community is unreliable, re-examine model
```

---

## 🔍 Diagnostic Applications

### Finding Problematic Clusters

```python
# Identify low-quality communities
poor_communities = []
for community_id, analysis in results.items():
    if analysis['precision'] < 0.7:
        poor_communities.append({
            'id': community_id,
            'precision': analysis['precision'],
            'center': analysis['center_node']
        })

print(f"Found {len(poor_communities)} problematic communities")
for c in poor_communities:
    print(f"  Community {c['id']}: precision={c['precision']:.2%}")

# Investigate these!
```

---

### Measuring Cluster Stability

```python
# Run analysis on multiple seeds
for seed in [42, 123, 456]:
    model = train(seed=seed)
    clusters = model.get_clusters()
    analyzer = CommunityAnalyzer(...)
    results = analyzer.analyze()
    
    avg_precision = np.mean([r['precision'] for r in results.values()])
    print(f"Seed {seed}: Avg precision = {avg_precision:.3f}")

# Stable model: high precision across seeds
# Unstable model: varying precision
```

---

## 📈 Quantitative Summary

### Aggregate Statistics

```python
# Compute overall metrics
precisions = [r['precision'] for r in results.values()]
errors = [r['error_rate'] for r in results.values()]

print(f"Average precision: {np.mean(precisions):.3f}")
print(f"Std precision: {np.std(precisions):.3f}")
print(f"Min precision: {np.min(precisions):.3f}")
print(f"Max precision: {np.max(precisions):.3f}")

print(f"Average error rate: {np.mean(errors):.3f}")

# Interpretation:
# - High mean & low std: Consistent good quality
# - High mean & high std: Some good, some bad
# - Low mean: Overall poor clustering
```

---

## 🎯 Use Cases

### Use Case 1: Model Validation

**Question:** Is the clustering model trustworthy?

**CCTS Answer:**
- Analyze all communities
- If most have precision > 0.8: ✓ Model is good
- If many have precision < 0.6: ✗ Model needs work

---

### Use Case 2: Identifying Failure Modes

**Question:** Where does the model fail?

**CCTS Answer:**
- Communities with low precision
- Find which node types are confused
- Investigate features of misclassified nodes

---

### Use Case 3: Explaining Individual Predictions

**Question:** Why was this node assigned to this cluster?

**CCTS Answer:**
- Find center node of community
- Check distance from center
- Verify precision/error rate in region
- Report: "Node is in high-precision region" or "borderline region"

---

### Use Case 4: Comparing Models

**Question:** Which model is better?

**CCTS Answer:**
- Analyze both models with CCTS
- Compare average precision across communities
- Model with higher precision: more trustworthy
- Model with more consistent precision: more stable

---

## 🐛 Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| All communities precision < 0.5 | Model failing | Retrain with better hyperparams |
| High variance in precisions | Some clusters good, some bad | Investigate specific communities |
| Center node = 0 (error) | Invalid analysis | Check ground truth labels |
| Visualization all gray | Graph too sparse | Use undirected edges or add KNN |

---

## 📋 Configuration Parameters

### CommunityAnalyzer Parameters

```python
CommunityAnalyzer(
    node_labels,              # Ground truth (required)
    predicted_clusters,       # Predictions (required)
    graph,                    # Graph structure (required)
    
    # Optional parameters:
    alpha=1.0,               # Precision weight (default: 1.0)
    beta=0.0,                # Error weight (default: 0.0)
    max_threshold=10,        # Max distance search (default: 10)
    proportion_threshold=0.5 # Include threshold (default: 0.5)
)
```

### Adjusting for Different Analysis Goals

```python
# Conservative (trust only best regions):
analyzer = CommunityAnalyzer(..., alpha=2.0, beta=0.5)

# Balanced (typical):
analyzer = CommunityAnalyzer(..., alpha=1.0, beta=0.0)

# Permissive (include more nodes):
analyzer = CommunityAnalyzer(..., alpha=0.5, beta=-0.5)
```

---

## 🔄 Integration with Main Pipeline

### Typical Flow

```
main.py training
    ↓
    Model produces clusters
    ↓
main.py exports results
    ├─ Save model
    ├─ Export data/graph/{dataset}.txt
    └─ Export community_partitions/{method}/{dataset}_partition.pkl
         ↓
CCTS Analysis (this module)
    ├─ Load cluster predictions
    ├─ Run CommunityAnalyzer.analyze()
    ├─ Generate visualizations
    └─ Export explainability report
         ↓
Presentation/Paper
    ├─ Use visualizations
    ├─ Report precision/error statistics
    ├─ Explain community structure
    └─ Justify model quality
```

---

## 📄 Output Files

### Generated by CCTS

```
figure/
  └─ {dataset}_ccts_analysis.png      # Main visualization

result/DeSE/
  ├─ {dataset}_analysis.json          # Detailed results
  ├─ {dataset}_precision_summary.txt  # Statistics
  └─ {dataset}_explanations.txt       # Human-readable explanations

community_partitions/DeSE/
  └─ {dataset}_partition.pkl          # Cluster assignments
```

