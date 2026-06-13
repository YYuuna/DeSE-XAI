"""
╔════════════════════════════════════════════════════════════════════════════╗
║         DeSE: Deep Structural Entropy (XAI) - Graph Clustering Model       ║
╚════════════════════════════════════════════════════════════════════════════╝

█ MODULE PURPOSE
  Implements the Deep Structural Entropy (DeSE) model for hierarchical graph clustering
  and community detection with explainability support.

█ KEY INNOVATION
  DeSE combines STRUCTURAL and EMBEDDING information in a hierarchical architecture:
  
  1. STRUCTURAL INFORMATION (Graph Topology)
     - Encoded in adjacency matrix A
     - Captures how nodes are connected
     - Used to compute structural entropy loss
     
  2. EMBEDDING INFORMATION (Feature Similarity)
     - Node embeddings E learned through neural network
     - Capture node feature similarity
     - Used to compute link prediction loss
     
  3. HIERARCHICAL STRUCTURE
     - Multi-level clustering (height parameter controls levels)
     - Level 0: Fine-grained clusters
     - Level h: Coarse-grained clusters (final output)
     - Graph coarsening: S^T @ A @ S (soft assignments compress structure)

█ LOSS FUNCTIONS (Combined Optimization)
  
  STRUCTURAL ENTROPY (SE) LOSS:
  - Measures clustering quality using information theory
  - Formula: SE = -Σ(p_ij * log(p_ij)) where p_ij is soft assignment probability
  - Higher SE indicates better cluster separation
  - Weight: se_lamda (typically 0.01) - balance structural quality
  
  LINK PREDICTION (LP) LOSS:
  - Ensures embeddings preserve graph structure
  - Binary classification: edge exists vs. doesn't exist
  - Uses negative sampling for efficiency
  - Weight: lp_lamda (typically 1.0) - preserve local structure
  
  TOTAL LOSS: L = se_lamda * SE_loss + lp_lamda * LP_loss

█ ARCHITECTURE OVERVIEW
  
  INPUT: Graph(Adjacency A, Features X, Degrees d)
         ↓
  MLP: Transform raw features → embeddings (fixed dimension: embed_dim)
         ↓
  LEVEL 0 ASS:
    - GCN processes node embeddings
    - KNN creates feature-based adjacency
    - Assign_layer computes soft cluster assignments S₀
    - Graph coarsening: A₀' = S₀^T @ A @ S₀
         ↓
  LEVEL 1 ASS:
    - Process coarsened graph
    - Compute soft assignments S₁
    - Further coarsening
         ↓
  ... (repeat for height levels) ...
         ↓
  LEVEL h OUTPUT:
    - Final cluster assignments (num_clusters_layer[h] clusters)
    - Hierarchical representation
    - Embeddings at all levels stored

█ KEY HYPERPARAMETERS
  
  height: Number of hierarchy levels (1=flat, 2=hierarchical with 2 levels, etc.)
  embed_dim: Node embedding dimension (16-128 typical, larger = more expressive)
  num_clusters_layer: Cluster counts per level [c0, c1, ..., c_h]
  k: KNN neighbors for feature-based graph (2-10 typical)
  se_lamda: SE loss weight (0.001-0.1 typical, higher = emphasize structure)
  lp_lamda: LP loss weight (0.1-2.0 typical, higher = preserve edges)
  beta_f: Feature adjacency weight (0-1, 0=pure graph, 1=pure features)

█ TYPICAL USAGE PIPELINE
  
  1. Initialize: model = DeSE(args, features, device)
  2. Forward pass: s_dict, embeddings, graphs = model(adj, features, degrees)
  3. Compute losses:
     - se_loss = model.calculate_se_loss1()
     - lp_loss = model.calculate_lp_loss(adj, neg_edges, embeddings)
  4. Backprop: loss.backward(), optimizer.step()
  5. Extract clusters: hard_clusters = model.hard() or decoding_from_assignment(s_dict[level])

█ DESIGN NOTES FOR PRESENTATION
  
  - Why hierarchical? Captures multi-scale cluster structure, interpretable at each level
  - Why soft assignments? Enable differentiable optimization, smooth gradients
  - Why KNN feature graph? Feature similarity provides regularization signal
  - Why two losses? SE optimizes structure, LP preserves embedding quality
  - Why graph coarsening? Reduces computation at higher levels, learns abstractions

█ COMMON MODIFICATIONS
  
  - Adjust height for finer/coarser clustering
  - Tune se_lamda/lp_lamda trade-off based on dataset
  - Change embed_dim for model capacity
  - Modify k for KNN sensitivity to features
  - Use different activation functions for expressiveness
"""

import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import entropy
from utility.util import select_activation, g_from_torchsparse
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import negative_sampling
from torch_scatter import scatter_sum
from time import time


class MLP(torch.nn.Module):
    """
    Multi-Layer Perceptron (MLP) for feature transformation.
    
    This simple feedforward neural network is used to:
    - Transform node features to embeddings
    - Project embeddings to different dimensional spaces
    - Non-linearly map features through intermediate representations
    
    Architecture: Linear(input_dim) -> Activation -> Dropout -> Linear(output_dim)
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation=None):
        """
        Initialize MLP layers.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of hidden/intermediate layer
            output_dim (int): Dimension of output features
            dropout (float): Dropout rate (default: 0.1)
            activation (str or None): Activation function name ('relu', 'elu', 'sigmoid', None)
        """
        super().__init__()
        # First linear layer: projects input_dim features to hidden_dim
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # Second linear layer: projects hidden_dim features to output_dim
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        # Dropout for regularization to prevent overfitting
        self.dropout = torch.nn.Dropout(dropout)
        # Nonlinear activation function (typically ReLU)
        self.activation = select_activation(activation)
    
    def forward(self, x):
        """
        Forward pass through MLP.
        
        Args:
            x (torch.Tensor): Input features of shape (num_nodes, input_dim)
            
        Returns:
            torch.Tensor: Transformed features of shape (num_nodes, output_dim)
        """
        # Apply first linear transformation
        x = self.fc1(x)
        # Apply activation function if provided
        if self.activation is not None:
            x = self.activation(x)
        # Apply dropout for regularization
        x = self.dropout(x)
        # Apply second linear transformation to get output
        x = self.fc2(x)
        return x

class GCN_layer(torch.nn.Module):
    """
    Graph Convolutional Network (GCN) Layer with optional attention mechanism.
    
    This layer performs message passing over a graph, aggregating information from neighboring nodes.
    When attention is enabled, it learns edge-wise weights (attention scores) to weight the contributions
    of different neighbors differently.
    
    Key operations:
    1. Linear transformation: transforms node features to output dimension
    2. Optional attention: computes importance weights for each edge
    3. Message passing: neighbors send their transformed features
    4. Aggregation: central node aggregates received messages (weighted or unweighted)
    """
    def __init__(self, input_dim, output_dim, activation=None, att=False):
        """
        Initialize GCN layer.
        
        Args:
            input_dim (int): Input feature dimension
            output_dim (int): Output feature dimension
            activation (str or None): Activation function ('relu', 'elu', 'sigmoid', None)
            att (bool): Whether to use attention mechanism (default: False for standard GCN)
        """
        super().__init__()
        # Linear transformation of node features
        self.linear = torch.nn.Linear(input_dim, output_dim)
        # Attention layer: concatenates src and dst feature pairs, outputs single attention score
        # Used to compute edge-wise attention weights when att=True
        self.att_linear = nn.Linear(2*output_dim, 1)
        self.activation = select_activation(activation)
        self.att = att  # Flag to enable/disable attention mechanism

    def edge_attention(self, edges):
        """
        Compute attention scores for edges using source and destination node features.
        
        Args:
            edges: DGL edge batch containing source and destination node features
            
        Returns:
            dict: Edge data containing attention scores after leaky ReLU activation
        """
        # Concatenate source and destination node features: (E, 2*output_dim)
        z2 = torch.cat([edges.src['h'], edges.dst['h']], dim=1)
        # Compute attention score for each edge: (E, 1)
        a = self.att_linear(z2)
        # Apply leaky ReLU activation (allows small negative gradients)
        return {'e': F.leaky_relu(a)}
    
    def message_func(self, edges):
        """
        Prepare messages to send from source nodes to destination nodes.
        
        Returns:
            dict: Edge data containing source node features and attention scores
        """
        return {'h': edges.src['h'], 'e': edges.data['e']}
    
    def reduce_func(self, nodes):
        """
        Aggregate messages received by destination nodes.
        Uses attention-weighted sum: sum(attention_weight * message).
        
        Returns:
            dict: Node data with aggregated embeddings
        """
        # Softmax normalization of attention scores across incoming edges for each node
        alpha = F.softmax(nodes.mailbox['e'], dim=1)  # (num_neighbors, 1)
        # Weighted sum of messages using attention weights
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)  # (node, output_dim)
        return {'h': h}
    
    def forward(self, graph, h):
        """
        Forward pass of GCN layer.
        
        Args:
            graph (dgl.DGLGraph): Input graph with num_nodes nodes
            h (torch.Tensor): Node features of shape (num_nodes, input_dim)
            
        Returns:
            torch.Tensor: Output node embeddings of shape (num_nodes, output_dim)
        """
        with graph.local_scope():
            # Linear transformation: (num_nodes, input_dim) -> (num_nodes, output_dim)
            graph.ndata['h'] = self.linear(h)
            if self.att:
                # Compute attention scores for all edges
                graph.apply_edges(self.edge_attention)
                # Message passing with attention-weighted aggregation
                graph.update_all(self.message_func, self.reduce_func)
            else:
                # Standard GCN: simple mean aggregation (no attention weighting)
                graph.update_all(message_func=fn.copy_u('h', 'm'), 
                               reduce_func=fn.mean(msg='m', out='h'))
            # Extract aggregated features from graph nodes
            h = graph.ndata.pop('h')
            # Apply activation function to introduce non-linearity
            if self.activation is not None:
                h = self.activation(h)
            return h



#fast
class GCN_layer1(torch.nn.Module):
    """
    Optimized/faster GCN layer implementation with improved computational efficiency.
    
    This is an alternative GCN implementation that manually handles message passing
    and aggregation using index operations, which can be faster than DGL's apply_edges
    in some scenarios. The logic is identical to GCN_layer but optimized for speed.
    """
    def __init__(self, input_dim, output_dim, activation=None, att=False):
        """Initialize optimized GCN layer."""
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        self.att_linear = nn.Linear(2*output_dim, 1)
        self.activation = select_activation(activation)
        self.att = att
    
    def edge_attention(self, graph):
        """
        Compute attention scores for all edges efficiently.
        
        Args:
            graph (dgl.DGLGraph): Graph with node features in ndata['h']
            
        Returns:
            torch.Tensor: Attention scores for each edge, shape (num_edges, 1)
        """
        # Extract source and destination node features using edge indices
        src_h = graph.ndata['h'][graph.edges()[0]]  # Features of source nodes
        dst_h = graph.ndata['h'][graph.edges()[1]]  # Features of destination nodes
        # Concatenate source and destination features
        z2 = torch.cat([src_h, dst_h], dim=1)
        # Compute attention score for each edge and apply leaky ReLU
        e = F.leaky_relu(self.att_linear(z2))
        return e
    
    def message_func(self, edges):
        """Prepare messages (same as parent class)."""
        return {'h': edges.src['h'], 'e': edges.data['e']}
    
    def reduce_func(self, nodes):
        """Aggregate messages with attention weighting (same as parent class)."""
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(alpha * nodes.mailbox['h'], dim=1)
        return {'h': h}
    
    def forward(self, graph, h):
        """Forward pass using optimized aggregation."""
        with graph.local_scope():
            # Transform features
            graph.ndata['h'] = self.linear(h)
            if self.att:
                # Compute attention weights efficiently
                e = self.edge_attention(graph)
                # Normalize attention weights using softmax
                e = F.softmax(e, dim=0)
                src, dst = graph.edges()
                # Extract normalized weights (remove singleton dimension)
                edge_weights = e.squeeze(-1)
                
                # Manual message aggregation using index_add_
                # This replaces the DGL message passing with direct tensor operations
                h_new = torch.zeros_like(h)
                # Aggregate weighted messages: h_new[dst] += edge_weights * h[src]
                h_new.index_add_(0, dst, edge_weights[:, None] * h[src])
            else:
                # Standard mean aggregation (DGL version)
                graph.update_all(message_func=fn.copy_u('h', 'm'), 
                               reduce_func=fn.mean(msg='m', out='h'))
                h_new = graph.ndata.pop('h')
            
            # Apply activation
            if self.activation is not None:
                h = self.activation(h_new)
            else:
                h = h_new
            return h


        
class Assign_layer(torch.nn.Module):
    """
    Assignment Layer for computing soft cluster assignments and hierarchical graph coarsening.
    
    This layer takes an embedded graph representation and produces:
    1. Soft cluster assignment matrix (S): probabilistic assignment of nodes to clusters
    2. Coarsened embeddings: cluster-level representations
    3. Coarsened adjacency matrix: graph topology at the cluster level
    
    The key operation is: new_adj = S^T @ adj @ S, which produces a graph where nodes
    represent clusters and edges represent inter-cluster connectivity.
    """
    def __init__(self, embed_dim, num_cluster, activation=None):
        """
        Initialize assignment layer.
        
        Args:
            embed_dim (int): Dimension of node embeddings
            num_cluster (int): Number of clusters to form at this layer
            activation (str or None): Activation function for GCN layers
        """
        super().__init__()
        # GCN layer for embedding refinement: maintains dimensionality
        self.GCN_emb=GCN_layer(embed_dim, embed_dim, activation, att=False)
        # GCN layer for computing assignments: outputs num_cluster values per node
        # Using attention to learn importance weights for assignments
        self.GCN_ass=GCN_layer(embed_dim, num_cluster, activation, att=True)
        self.num_cluster = num_cluster
    
    def update_graph_and_adj(self, adj, s):
        """
        Compute coarsened graph and adjacency matrix at cluster level.
        
        The coarsening operation transforms the graph from node-level to cluster-level:
        - new_adj = S^T @ adj @ S, where S is the soft assignment matrix
        - This aggregates edges between clusters
        
        Args:
            adj (scipy.sparse): Original adjacency matrix (num_nodes, num_nodes)
            s (torch.Tensor): Soft assignment matrix of shape (num_nodes, num_clusters)
            
        Returns:
            tuple: (coarsened_graph (dgl.DGLGraph), coarsened_adj (scipy.sparse))
        """
        # Convert soft assignment matrix to numpy for sparse matrix operations
        s_numpy = s.cpu().numpy()
        # Compute coarsened adjacency: S^T @ A @ S
        # Result shape: (num_clusters, num_clusters)
        adj_s = s_numpy.T @ adj.tocsr() @ s_numpy
        # Convert to lil format for efficient operations
        adj_lil = sp.lil_matrix(adj_s)
        # Remove self-loops (diagonal entries)
        adj_lil.setdiag(0)
        # Convert back to COO format for DGL
        adj_s = adj_lil.tocoo()
        # Create DGL graph from sparse adjacency matrix
        graph = dgl.from_scipy(adj_s)
        # Add self-loops to the graph for GCN processing
        graph = dgl.add_self_loop(graph)
        return graph, adj_s

    def forward(self, graph, adj, x):
        """
        Forward pass: compute soft assignments and coarsen graph.
        
        Args:
            graph (dgl.DGLGraph): Input graph
            adj (scipy.sparse): Adjacency matrix (sparse)
            x (torch.Tensor): Node embeddings, shape (num_nodes, embed_dim)
            
        Returns:
            tuple: (soft_assignment_matrix (S), coarsened_embeddings, coarsened_graph, coarsened_adj)
        """
        # Refine embeddings using GCN
        z = self.GCN_emb(graph, x)
        # Compute soft cluster assignments (logits)
        s = self.GCN_ass(graph, z)
        # Normalize to probability distribution using softmax
        # Each row sums to 1, representing probability of assignment to each cluster
        s = torch.softmax(s, dim=-1)
        # Aggregate embeddings by soft assignment: cluster_emb = S^T @ node_emb
        x = s.t() @ z
        # Coarsen graph and adjacency matrix to cluster level
        graph_higher, adj_higher = self.update_graph_and_adj(adj,s.detach())
        # Return assignment matrix, coarsened embeddings, and coarsened graph
        return s, x, graph_higher, adj_higher  #assignMatrix, father_x, new_graph, new_adj


def KNN(x, k):
    """
    Construct k-Nearest Neighbor (kNN) adjacency matrix for embedding space.
    
    This function creates a feature-based similarity graph where each node connects to
    its k nearest neighbors in the embedding space. This is used to create an additional
    adjacency matrix based on feature similarity (structural adjacency + feature adjacency).
    
    Args:
        x (torch.Tensor): Node embeddings, shape (num_nodes, embed_dim)
        k (int): Number of nearest neighbors for each node
        
    Returns:
        torch.Tensor: Sparse adjacency matrix (symmetric, undirected)
    """
    # Convert tensor to numpy for sklearn processing
    x1 = x.detach().numpy()
    # Fit KNN model on embeddings
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(x1)
    # Find k nearest neighbors for each node
    distances, indices = nbrs.kneighbors(x1)
    # Create edge indices for KNN graph
    # rows: source nodes (repeated k times for each node)
    rows = np.repeat(np.arange(x1.shape[0]), k)
    # cols: neighbor node indices
    cols = indices.flatten()
    N = x1.shape[0]
    # Edge weights (all ones for unweighted graph)
    values = torch.ones(len(rows), dtype=torch.float)
    
    # Ensure rows and cols are numpy arrays (for proper tensor creation)
    rows = np.array(rows) if not isinstance(rows, np.ndarray) else rows
    cols = np.array(cols) if not isinstance(cols, np.ndarray) else cols
    # Create sparse COO tensor with shape (num_nodes, num_nodes)
    indices = torch.tensor(np.stack([rows, cols]), dtype=torch.long)
    adj = torch.sparse_coo_tensor(indices=indices, values=values, size=(N, N), dtype=torch.float)
    
    # Make adjacency matrix symmetric: (adj + adj^T) / 2
    # This ensures the graph is undirected
    adj = (adj + adj.t()) / 2.0
    return adj


def KNN_dynamic(x, degree):
    """
    Construct dynamic k-Nearest Neighbor adjacency matrix where k varies per node.
    
    Instead of using a fixed k for all nodes, this function computes a node-specific k
    based on the node's degree: k = ceil(log2(degree + 1)). Nodes with higher degrees
    can connect to more distant neighbors in embedding space.
    
    Args:
        x (torch.Tensor): Node embeddings, shape (num_nodes, embed_dim)
        degree (torch.Tensor): Degree of each node, shape (num_nodes,)
        
    Returns:
        torch.Tensor: Sparse adjacency matrix with dynamic neighborhoods
    """
    x1 = x.detach().numpy()
    EPS = 1e-6
    # Compute node-specific k values based on logarithm of degree
    # Higher degree nodes get more neighbors
    k_list = np.ceil(np.log2(degree.numpy() + 1) + EPS).astype(int)
    # Find maximum k to determine how many neighbors to query
    max_k = np.max(k_list)
    # Fit KNN model
    nbrs = NearestNeighbors(n_neighbors=max_k, algorithm='auto').fit(x1)
    distances, indices = nbrs.kneighbors(x1)
    
    # Build edge lists with node-specific k
    rows = []
    cols = []
    values = []
    for i, k in enumerate(k_list):
        # Connect node i to its k nearest neighbors
        for j in range(k):
            rows.append(i)
            cols.append(indices[i, j])
            values.append(1)
    
    # Create sparse tensor
    adj = torch.sparse_coo_tensor(
        indices=torch.tensor([rows, cols], dtype=torch.long),
        values=torch.tensor(values, dtype=torch.float32),
        size=(x1.shape[0], x1.shape[0]),
        dtype=torch.float32
    )
    # Make symmetric for undirected graph
    return adj

class ASS(torch.nn.Module):
    """
    Assignment-Sigmoid-Structure (ASS) Layer for hierarchical graph clustering.
    
    This layer combines two types of information at each hierarchy level:
    1. Structural information (graph topology): captured from adjacency matrix
    2. Feature-space information: computed as KNN graph of cluster embeddings
    
    The layer outputs:
    - Soft cluster assignments for current level
    - Coarsened embeddings for next level
    - Updated adjacency matrices (structural and feature-based)
    """
    def __init__(self, embed_dim, num_cluster, k, dropout=0.1, activation=None, flag_feature=True):
        """
        Initialize ASS layer.
        
        Args:
            embed_dim (int): Dimension of node embeddings
            num_cluster (int): Number of clusters at this hierarchy level
            k (int): Number of nearest neighbors for feature-based graph
            dropout (float): Dropout rate for MLP
            activation (str or None): Activation function name
            flag_feature (bool): Whether to include feature-based adjacency (first layer=False)
        """
        super().__init__()
        # GCN for embedding refinement at current level
        self.GCN_emb = GCN_layer(embed_dim, embed_dim, activation, att=False)
        # GCN for computing cluster assignments (produces num_cluster outputs per node)
        self.GCN_ass = GCN_layer(embed_dim, num_cluster, activation, att=True)
        self.num_cluster = num_cluster
        self.k = k  # Number of neighbors for KNN graph
        # MLP to transform embeddings before computing KNN graph
        self.mlp = MLP(embed_dim, embed_dim, embed_dim, dropout, activation)
        # Whether to compute feature-based adjacency matrix
        self.flag = flag_feature
    
    def forward(self, graph, x, adj_g):
        """
        Forward pass: compute assignments and prepare for next hierarchy level.
        
        Args:
            graph (dgl.DGLGraph): Current level graph
            x (torch.Tensor): Node embeddings, shape (num_nodes, embed_dim)
            adj_g (scipy.sparse): Current level adjacency matrix (structural)
            
        Returns:
            tuple: (embeddings_current_level, cluster_embeddings, soft_assignments,
                   (coarsened_adj_structural, coarsened_adj_feature))
        """
        # Refine embeddings using GCN message passing
        h = self.GCN_emb(graph, x)
        # Compute soft cluster assignments: softmax over num_cluster dimensions
        # Shape: (num_nodes, num_cluster), rows sum to 1
        s = torch.softmax(self.GCN_ass(graph, x), dim=-1)
        # Aggregate node embeddings by soft assignment to get cluster embeddings
        # e = S^T @ h, shape: (num_cluster, embed_dim)
        e = s.t() @ h
        # Coarsen structural adjacency matrix: S^T @ A @ S
        # This produces cluster-level graph connectivity
        adj_g1 = s.t() @ adj_g @ s
        
        if self.flag:
            # Compute feature-based adjacency using KNN on transformed cluster embeddings
            # MLP transforms embeddings to a different space for similarity computation
            adj_f1 = KNN(self.mlp(e), self.k)
            # Alternative: could use dynamic KNN based on cluster degrees
            # adj_f1 = KNN_dynamic(self.mlp(e), adj_g1.sum(dim=1))
        else:
            # First layer typically doesn't use feature-based adjacency
            adj_f1 = None
        
        # Return: node-level embeddings, cluster embeddings, assignments, both adjacency matrices
        return h, e, s, (adj_g1, adj_f1)



class DeSE(torch.nn.Module):
    """
    DeSE: Deep Structural Entropy Community Detection Model.
    
    This is the main model implementing hierarchical graph clustering with two complementary
    information sources:
    
    1. STRUCTURAL INFORMATION: Graph topology (adjacency matrix)
       - Captures connections between nodes in the original graph
       - Preserves community structure through message passing
    
    2. EMBEDDING INFORMATION: Feature-space similarity (KNN graph of embeddings)
       - Captures similarity between node features/embeddings
       - Helps connect nodes that are similar but not directly connected
    
    The model builds a multi-level hierarchy:
    - Level 'height': Original graph with all nodes
    - Level 'height-1': First clustering, fewer clusters
    - ... progressive coarsening ...
    - Level 1: Finest clusters/supernodes
    
    Training: Uses two losses:
    - se_loss: Structural Entropy - ensures high-quality clustering by minimizing cut edges
    - lp_loss: Link Prediction - ensures embeddings capture graph structure
    
    The final clustering is obtained from hard assignments (one-hot encoding) at level 1.
    """
    def __init__(self, args, feature, device):
        """
        Initialize DeSE model.
        
        Args:
            args (argparse.Namespace): Configuration arguments containing:
                - height: Number of hierarchy levels
                - embed_dim: Dimension of node embeddings
                - activation: Activation function name
                - num_clusters_layer: List of cluster counts at each level
                - decay_rate: Exponential decay factor for cluster numbers (if auto-computing)
                - k: Number of nearest neighbors for feature graph
                - dropout: Dropout rate
                - beta_f: Weight balancing structural and feature adjacency
            feature (torch.Tensor): Node features, shape (num_nodes, feature_dim)
            device (str): Device to place tensors on ('cpu' or 'cuda:X')
        """
        super().__init__()
        # Total number of nodes in the graph
        self.num_node = feature.shape[0]
        # Input feature dimension
        self.input_dim = feature.shape[-1]
        # Number of hierarchy levels (tree height)
        self.height = args.height
        # Dimension of learned node embeddings
        self.embed_dim = args.embed_dim
        # Activation function for neural network layers
        self.activation = args.activation
        
        # Determine cluster counts for each hierarchy level
        if args.num_clusters_layer is None:
            # Auto-compute: exponential decay from num_nodes to 1
            decay_rate = int(np.exp(np.log(self.num_node) / self.height)) \
                        if args.decay_rate is None else args.decay_rate
            num_clusters_layer = [int(self.num_node / (decay_rate ** i)) 
                                 for i in range(1, self.height)]
        else:
            # Use provided cluster counts
            num_clusters_layer = args.num_clusters_layer
        
        # MLP for transforming raw features to KNN-compatible embeddings
        self.mlp = MLP(self.input_dim, self.embed_dim, self.embed_dim)
        # Initial GCN layer for computing node embeddings from features
        self.gnn = GCN_layer(self.input_dim, self.embed_dim, self.activation, att=False)
        
        # Build hierarchy of assignment layers
        self.assignlayers = nn.ModuleList([])
        for i in range(self.height - 1):
            if i == 0:
                # First layer: no feature adjacency (only structure)
                self.assignlayers.append(
                    ASS(self.embed_dim, num_clusters_layer[i], args.k, 
                        args.dropout, self.activation, flag_feature=False)
                )
            else:
                # Upper layers: use both structural and feature adjacency
                self.assignlayers.append(
                    ASS(self.embed_dim, num_clusters_layer[i], args.k, 
                        args.dropout, self.activation, flag_feature=True)
                )
        
        # Device for tensor placement
        self.device = device
        # Weight for feature-based adjacency: adj = adj_struct + beta_f * adj_feature
        self.beta_f = args.beta_f
        # Number of nearest neighbors for KNN graph
        self.k = args.k

    def hard(self, s_dic):
        """
        Convert soft assignments (probabilities) to hard assignments (one-hot).
        
        This function traverses the hierarchy from top to bottom, accumulating
        assignment matrices. Each level's assignment is computed as the product
        of assignments from all levels above.
        
        For level h: hard_assignment[h] = assign[h+1] @ assign[h+2] @ ... @ assign[height]
        
        This gives the final cluster ID for each original node.
        
        Args:
            s_dic (dict): Soft assignment matrices at each level
                - s_dic[k] is shape (num_nodes_level_k, num_clusters_level_k)
        """
        # Initialize assignment matrix at the bottom: each node is its own cluster
        assign_mat_dict = {self.height: torch.eye(self.num_node).to(self.device)}
        
        # Accumulate assignments from top to bottom
        for k in range(self.height - 1, 0, -1):
            # Multiply assignment from level k+1 with soft assignments at level k
            # assign_mat_dict[k] maps original nodes to clusters at level k
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ s_dic[k + 1]
        
        # Store hard (one-hot) assignments for each level
        self.hard_dic = {}
        for h, assign in assign_mat_dict.items():
            # For each original node, find the cluster with maximum probability
            idx = assign.max(dim=1)[1]
            # Create one-hot encoding
            t = torch.zeros_like(assign)
            t[torch.arange(t.shape[0]), idx] = 1
            self.hard_dic[h] = t

    def forward(self, adj_g, feature, degree):
        """
        Forward pass: hierarchical clustering through multiple levels.
        
        Args:
            adj_g (torch.sparse_coo_tensor): Original graph adjacency matrix (sparse)
            feature (torch.Tensor): Node features, shape (num_nodes, input_dim)
            degree (torch.Tensor): Node degrees, shape (num_nodes,)
            
        Returns:
            tuple:
                - s_dic: Soft assignment matrices at each level
                - tree_node_embed_dic: Node embeddings at each level
                - g_dic: DGL graphs at each level
        """
        # Compute feature-based KNN graph from raw features
        adj_f = KNN(self.mlp(feature), self.k)
        # Alternative: could use dynamic KNN based on node degrees
        # adj_f = KNN_dynamic(self.mlp(feature), degree)
        
        # Combine structural and feature adjacencies with learned weight
        # adj = A_struct + beta_f * A_feature
        adj = adj_g + self.beta_f * adj_f
        
        # Convert sparse adjacency to DGL graph
        g = g_from_torchsparse(adj)
        
        # Initial node embedding using GCN on original features
        e = self.gnn(g, feature)
        
        # Initialize dictionaries for storing hierarchy information
        s_dic = {}  # Soft assignment matrices: s_dic[k] at level k
        tree_node_embed_dic = {self.height: e.to(self.device)}  # Embeddings at each level
        g_dic = {self.height: g}  # DGL graphs at each level
        
        # Progressively cluster through hierarchy levels
        for i, layer in enumerate(self.assignlayers):
            # Apply ASS layer to compute assignments and coarsen
            h, e, s, (adj_g, adj_f) = layer(g, e, adj_g)
            
            # Store coarsened embeddings at next level up
            tree_node_embed_dic[self.height - i - 1] = e.to(self.device)
            # Store soft assignments at current level
            s_dic[self.height - i] = s.to(self.device)
            
            # Check if reached top of hierarchy
            if i == self.height - 2:
                break
            
            # Prepare for next hierarchy level: combine adjacencies
            adj = adj_g + self.beta_f * adj_f
            # Convert to DGL graph for next level
            g = g_from_torchsparse(adj.to_sparse())
            g_dic[self.height - i - 1] = g.to(self.device)
        
        # Top level assignment (single supernode, all nodes belong to it)
        s_dic[1] = torch.ones(s.shape[-1], 1).to(self.device)
        
        # Convert soft assignments to hard assignments
        self.hard(s_dic)
        
        # Store for later use in loss computation
        self.s_dic = s_dic
        self.g_dic = g_dic
        
        return s_dic, tree_node_embed_dic, g_dic
    
    def calculate_se_loss(self, s_dic, g):
        """
        Calculate Structural Entropy (SE) Loss for evaluating clustering quality.
        
        SE measures how well the clustering respects graph structure. The loss is based on
        the Volume/Weight of edges crossing cluster boundaries.
        
        For each cluster, we compute: delta_vol = vol(cluster) - weight_within_cluster
        The loss penalizes large cuts proportionally to their volume relative to the partition.
        
        Formula: SE = -1/vol(G) * sum_levels sum_clusters delta_vol_c * log2(delta_vol_c / vol_parent_c)
        
        Args:
            s_dic (dict): Soft assignment matrices at each level
            g (dgl.DGLGraph): Graph at the level where computing loss
            
        Returns:
            torch.Tensor: Structural entropy loss (scalar)
        """
        # Extract edge information and weights from graph
        edge_index = torch.stack(g.edges())
        weight = g.edata['weight']
        
        # Compute node degrees (weighted in-degree)
        degrees = scatter_sum(weight, edge_index[0])
        # Total graph volume (sum of all degrees = 2 * num_edges for undirected)
        vol_G = torch.sum(degrees).to(self.device)
        EPS = 1e-6
        
        # Initialize: at bottom level, each node is its own "cluster"
        assign_mat_dict = {self.height: torch.eye(self.num_node).to(self.device)}
        # Volume of each bottom-level "cluster" (just node degrees)
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        
        # Propagate assignments and volumes through hierarchy
        for k in range(self.height - 1, 0, -1):
            # Accumulate assignment: maps nodes to clusters at level k
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ s_dic[k + 1]
            # Cluster volume: sum of node degrees in each cluster
            vol_dict[k] = torch.einsum('ij, i->j', assign_mat_dict[k], degrees)
        
        se_loss = 0
        
        # Compute SE loss across all hierarchy levels
        for k in range(1, self.height + 1):
            # Parent volume for each cluster at level k
            vol_parent = torch.einsum('ij, j->i', s_dic[k], vol_dict[k - 1])
            # Log ratio of volumes: measures imbalance
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            
            # Soft assignment probabilities of edge endpoints to clusters
            ass_i = assign_mat_dict[k][edge_index[0]]  # (num_edges, num_clusters_k)
            ass_j = assign_mat_dict[k][edge_index[1]]  # (num_edges, num_clusters_k)
            
            # Probability that both endpoints of an edge are in the same cluster
            # ass_i * ass_j has high values only for edges within clusters
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)
            
            # Volume of edges crossing cluster boundaries
            delta_vol = vol_dict[k] - weight_sum
            
            # Accumulate entropy: high when boundaries have large volume
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        
        # Normalize by total volume
        se_loss = -1 / vol_G * se_loss
        
        return se_loss
    
    def calculate_se_loss1(self):
        """
        Alternative SE loss computation using stored graph and assignments.
        
        This version uses self.g_dic and self.s_dic (stored during forward pass)
        instead of taking them as parameters. More efficient as it reuses computations.
        Same algorithm as calculate_se_loss but optimized.
        
        Returns:
            torch.Tensor: Structural entropy loss (scalar)
        """
        # Get graph from stored dictionary
        g = self.g_dic[self.height]
        
        # Extract edge information
        edge_index = torch.stack(g.edges())
        weight = g.edata['weight']
        
        # Compute degrees and total volume
        degrees = scatter_sum(weight, edge_index[0])
        vol_G = degrees.sum().to(self.device)
        EPS = 1e-6
        
        # Initialize assignment and volume dictionaries
        assign_mat_dict = {self.height: torch.eye(self.num_node, device=self.device)}
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        
        # Propagate through hierarchy using stored soft assignments
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ self.s_dic[k + 1]
            # Optimized: use matrix multiplication instead of einsum
            vol_dict[k] = torch.matmul(assign_mat_dict[k].t(), degrees)
        
        se_loss = 0
        
        # Compute SE loss across levels
        for k in range(1, self.height + 1):
            # Parent volume: optimized matrix multiplication
            vol_parent = torch.matmul(self.s_dic[k], vol_dict[k - 1])
            # Log volume ratio
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))
            
            # Edge endpoint assignments
            ass_i = assign_mat_dict[k][edge_index[0]]
            ass_j = assign_mat_dict[k][edge_index[1]]
            
            # Weight of edges within clusters: optimized with matrix-vector product
            weight_sum = torch.mv((ass_i * ass_j).t(), weight)
            
            # Volume of cut edges
            delta_vol = vol_dict[k] - weight_sum
            
            # Accumulate entropy
            se_loss += torch.dot(delta_vol, log_vol_ratio_k)
        
        # Normalize by total volume (note: negative for minimization)
        se_loss = -se_loss / vol_G
        
        return se_loss

    def calculate_dist(self, x, y):
        """
        Calculate L2 distance between embedding vectors.
        
        Args:
            x (torch.Tensor): Embedding matrix of shape (n, embed_dim)
            y (torch.Tensor): Embedding matrix of shape (m, embed_dim)
            
        Returns:
            torch.Tensor: L2 distances of shape (num_edges,) or (n,) depending on broadcasting
        """
        return torch.norm(x - y, p=2, dim=-1)

    def calculate_lp_loss(self, g, neg_edge_index, embedding):
        """
        Calculate Link Prediction Loss for preserving graph structure in embeddings.
        
        This loss uses binary classification: positive edges (existing edges) should have
        high similarity scores, negative edges (non-existing) should have low scores.
        
        The similarity is computed as: sim = sigmoid((2 - L2_distance) / 1)
        This inverts the distance (smaller distance = higher similarity) and applies sigmoid.
        
        Args:
            g (dgl.DGLGraph): Graph at current level
            neg_edge_index (torch.Tensor): Negative edge indices (non-existent edges)
            embedding (torch.Tensor): Node embeddings at current level
            
        Returns:
            torch.Tensor: Binary cross-entropy loss (scalar)
        """
        # Get positive edges from graph
        edge_index = torch.stack(g.edges())
        
        # Concatenate positive and negative edges for joint prediction
        edge = torch.cat([edge_index, neg_edge_index], dim=-1)
        
        # Compute L2 distances for all edges
        prob = self.calculate_dist(embedding[edge[0]], embedding[edge[1]])
        
        # Transform distance to similarity: larger distance -> lower similarity
        # sigmoid((2 - dist) / 1) ensures: dist=0 -> sim≈0.88, dist=2 -> sim≈0.5, dist>2 -> sim<0.5
        prob = torch.sigmoid((2. - prob) / 1.)
        
        # Create labels: 1 for positive edges, 0 for negative edges
        label = torch.cat([
            torch.ones(edge_index.shape[-1]),
            torch.zeros(neg_edge_index.shape[-1])
        ]).to(self.device)
        
        # Binary cross-entropy loss
        lp_loss = F.binary_cross_entropy(prob, label)
        
        return lp_loss


        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(self.device)
        lp_loss = F.binary_cross_entropy(prob, label)
        return lp_loss


class DSE1(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, height, num_nodes, num_clusters_layer, decay_rate, device, activation=None):
        super().__init__()
        if num_clusters_layer is None:
            decay_rate = int(np.exp(np.log(num_nodes) / height)) if decay_rate is None else decay_rate
            num_clusters_layer = [int(num_nodes / (decay_rate ** i)) for i in range(1, height)]
        self.GCN_f1 = GCN_layer(input_dim, embed_dim, activation, att=False) #feature->embedding
        #self.GCN_f2 = GCN_layer(16, embed_dim, activation, att=False)
        self.assignlayers = nn.ModuleList([])
        for i in range(height - 1):
            self.assignlayers.append(Assign_layer(embed_dim, num_clusters_layer[i], activation))  #embedding->assignment
        self.height = height
        self.num_nodes = num_nodes
        self.device = device


    def hard(self, assignmatrix, tree_node_embed):
        self.embedding = {}
        for h, x in tree_node_embed.items():
            self.embedding[h] = x.detach()
        assign_mat_dict = {self.height: torch.eye(self.num_nodes).to(self.device)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ assignmatrix[k + 1]
        assignment = {}
        for h, assign in assign_mat_dict.items():
            idx = assign.max(dim=1)[1]
            t = torch.zeros_like(assign)
            t[torch.arange(t.shape[0]), idx] = 1
            assignment[h] = t
        return assignment

    def forward(self, graph, adj, feature):
        x=self.GCN_f1(graph, feature)  #feature->embedding
        #x=self.GCN_f2(graph, x)
        x=F.normalize(x, p=2, dim=-1) #normalize
        assignmatrix = {}  #store the assignment matrix of each layer: self.height->1
        tree_node_embed = {self.height: x.to(self.device)}  #store the embedding of each layer: self.height->0
        for i, layer in enumerate(self.assignlayers):  #embedding->assignment
            assign, x, graph, adj = layer(graph, adj, x)
            tree_node_embed[self.height-i-1] = x.to(self.device)
            assignmatrix[self.height-i] = assign.to(self.device)
        tree_node_embed[0] = torch.mean(x).to(self.device)
        assignmatrix[1] = torch.ones(assign.shape[-1], 1).to(self.device)
        self.hard(assignmatrix, tree_node_embed)
        return assignmatrix, tree_node_embed
    
    def calculate_se_loss(self, assignmatrix, degrees, edge_index, weight):
        vol_G = torch.sum(degrees).to(self.device)
        EPS = 1e-6
        assign_mat_dict = {self.height: torch.eye(self.num_nodes).to(self.device)} #each node at the bottom layer forms a cluster
        vol_dict = {self.height: degrees, 0: vol_G.unsqueeze(0)}
        for k in range(self.height - 1, 0, -1):
            assign_mat_dict[k] = assign_mat_dict[k + 1] @ assignmatrix[k + 1]  #assign_mat_dict[k] represent node assigned to which cluster at layer k: self.height->1
            vol_dict[k] = torch.einsum('ij, i->j', assign_mat_dict[k], degrees)  #vol_dict[k] represent vol of clusters at layer k: self.height->0
        se_loss = 0
        for k in range(1, self.height + 1):
            vol_parent = torch.einsum('ij, j->i', assignmatrix[k], vol_dict[k - 1])  # (num_clusters_k, num_clusters_k-1) (num_clusters_k-1, ) -> (num_clusters_k, )
            log_vol_ratio_k = torch.log2((vol_dict[k] + EPS) / (vol_parent + EPS))  # (num_clusters_k, ) / (num_clusters_k, ) -> (num_clusters_k, )
            ass_i = assign_mat_dict[k][edge_index[0]]  # (E, num_clusters_k)
            ass_j = assign_mat_dict[k][edge_index[1]]  # Assignment of nodes at both ends of the edge to the cluster
            weight_sum = torch.einsum('en, e->n', ass_i * ass_j, weight)  # ass_i * ass_j represent the propobalty that node_i node_j assigned to the same cluster: (E, num_clusters_k) (E, ) ->(num_clusters_k, ) total weight within the cluster
            delta_vol = vol_dict[k] - weight_sum    # (num_clusters_k, ) - (num_clusters_k, ) -> (num_clusters_k, )  total weight of cutting edges
            se_loss += torch.sum(delta_vol * log_vol_ratio_k)
        se_loss = -1 / vol_G * se_loss
        return se_loss

    def calculate_entropy(self, p):
        entropy = -torch.sum(p * torch.log2(p))
        return entropy
    
    def calculate_onehot_loss(self, assignmatrix):
        onehot_loss = 0
        for k in range(2, self.height+1):
            entropy_values = [self.calculate_entropy(row) for row in assignmatrix[k]]
            if any(torch.isnan(e).item() for e in entropy_values):
                print(assignmatrix[k])
                raise ValueError("NaN found in entropy calculation")
            onehot_loss += torch.mean(torch.stack(entropy_values))
        return onehot_loss

    def calculate_regularizer_loss(self, tree_node_embed):
        regularizer = 0
        for k in range(1, self.height+1):
            embed = tree_node_embed[k]
            regularizer += embed.norm(2).pow(2)
        return regularizer

    def calculate_dist(self, x, y):
        return torch.norm(x-y, p=2, dim=-1)

    def calculate_lp_loss(self, edge_index, neg_edge_index, embedding):
        edge = torch.cat([edge_index, neg_edge_index], dim=-1)
        prob = self.calculate_dist(embedding[edge[0]], embedding[edge[1]])
        prob = torch.sigmoid((2. - prob) / 1.)
        label = torch.cat([torch.ones(edge_index.shape[-1]), torch.zeros(neg_edge_index.shape[-1])]).to(self.device)
        lp_loss = F.binary_cross_entropy(prob, label)
        return lp_loss
