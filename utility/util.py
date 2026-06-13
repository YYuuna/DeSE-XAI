"""
Utility Functions for DeSE Clustering Module

This module provides essential utilities for clustering evaluation and graph operations:
1. Graph conversions between different formats (sparse tensors, DGL graphs)
2. Edge index/adjacency matrix conversions
3. Activation function selection
4. Clustering assignment decoding
5. Comprehensive clustering quality metrics (NMI, ARI, Accuracy, F1-score)

The cluster_metrics class computes standard clustering evaluation metrics by optimally
matching predicted clusters to ground truth labels using the Hungarian algorithm.
"""

import torch
import torch.nn.functional as F
from sklearn import metrics
from munkres import Munkres
import numpy as np
import dgl
from collections import Counter


def g_from_torchsparse(adj):
    """
    Convert sparse PyTorch COO tensor to DGL graph.
    
    This function converts a sparse adjacency matrix in COO format to a DGL graph
    object that can be used for message passing and graph neural network operations.
    
    Args:
        adj (torch.sparse_coo_tensor): Sparse adjacency matrix (num_nodes, num_nodes)
            in PyTorch COO (coordinate) format
    
    Returns:
        dgl.DGLGraph: DGL graph with edge weights stored in edata['weight']
    """
    # Get number of nodes from adjacency matrix dimensions
    N = adj.shape[0]
    # Convert sparse tensor to COO format and extract edge indices
    edges = adj.coalesce().indices() 
    # Extract source and destination node IDs
    src, dst = edges[0].tolist(), edges[1].tolist()
    # Extract edge weights
    weights = adj.coalesce().values()
    # Create DGL graph from edge lists
    g = dgl.graph((src, dst), num_nodes=adj.shape[0])
    # Store edge weights in graph edge data
    g.edata['weight'] = weights
    return g


def index2adjacency(N, edge_index, weight=None, is_sparse=True):
    """
    Convert edge index representation to adjacency matrix.
    
    Converts a compact edge list representation to an adjacency matrix, either as
    a sparse tensor or a dense matrix depending on the is_sparse parameter.
    
    Args:
        N (int): Number of nodes in graph
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
            where edge_index[0] contains source node IDs
            and edge_index[1] contains destination node IDs
        weight (torch.Tensor, optional): Edge weights. If None, all edges have weight 1.0
        is_sparse (bool): If True, return sparse COO tensor. If False, return dense tensor.
    
    Returns:
        torch.Tensor: Adjacency matrix
            - If is_sparse=True: sparse COO tensor of shape (N, N)
            - If is_sparse=False: dense tensor of shape (N, N)
    """
    if is_sparse:
        # Number of edges
        m = edge_index.shape[1]
        # Use unit weights if not provided
        weight = weight if weight is not None else torch.ones(m).to(edge_index.device)
        # Create sparse COO tensor with edges and weights
        adjacency = torch.sparse_coo_tensor(indices=edge_index, values=weight, size=(N, N))
    else:
        # Create dense adjacency matrix on same device as edge_index
        adjacency = torch.zeros(N, N).to(edge_index.device)
        if weight is None:
            # Set all edges to 1.0 for unweighted graph
            adjacency[edge_index[0], edge_index[1]] = 1
        else:
            # Set edge values according to provided weights
            adjacency[edge_index[0], edge_index[1]] = weight.reshape(-1)
    return adjacency


def adjacency2index(adjacency, weight=False):
    """
    Convert adjacency matrix to edge index representation.
    
    Converts an adjacency matrix back to the compact edge list format, optionally
    returning edge weights.
    
    Args:
        adjacency (torch.Tensor): Adjacency matrix of shape (N, N)
        weight (bool): If True, also return edge weights. If False, only return edge_index.
    
    Returns:
        edge_index (torch.Tensor): Edge index tensor of shape (2, num_edges)
            Contains all non-zero positions in adjacency matrix
        edge_weight (torch.Tensor, optional): Edge weights if weight=True,
            otherwise not returned
    """
    # Get adjacency matrix
    adj = adjacency
    # Find all non-zero positions (edges) and convert to edge index format
    edge_index = torch.nonzero(adj).t().contiguous()
    if weight:
        # Extract weights from adjacency matrix at edge positions
        weight = adjacency[edge_index[0], edge_index[1]].reshape(-1)
        return edge_index, weight
    else:
        return edge_index

    
def select_activation(activation):
    """
    Select and return activation function based on name.
    
    Maps activation function names to PyTorch functional implementations.
    Supported activations: ELU, ReLU, Sigmoid, and None (no activation).
    
    Args:
        activation (str or None): Name of activation function
            - 'elu': Exponential Linear Unit
            - 'relu': Rectified Linear Unit
            - 'sigmoid': Sigmoid function
            - None: No activation (identity function)
    
    Returns:
        function: Activation function or None
    
    Raises:
        NotImplementedError: If activation name is not recognized
    """
    if activation == 'elu':
        return F.elu
    elif activation == 'relu':
        return F.relu
    elif activation == 'sigmoid':
        return F.sigmoid
    elif activation is None:
        return None
    else:
        raise NotImplementedError(f'Activation function "{activation}" is not implemented')

    
def decoding_from_assignment(assignmatrix):
    """
    Convert soft cluster assignments to hard cluster IDs.
    
    Converts a soft assignment matrix (posterior probabilities over clusters) to
    hard cluster assignments by taking the argmax across cluster dimension.
    
    Args:
        assignmatrix (torch.Tensor): Soft assignment matrix of shape (num_nodes, num_clusters)
            where assignmatrix[i, k] is the probability that node i belongs to cluster k
    
    Returns:
        torch.Tensor: Hard assignments of shape (num_nodes,)
            where pred[i] is the cluster ID (0 to num_clusters-1) for node i
    """
    # Take argmax to get cluster with highest assignment probability
    pred = assignmatrix.argmax(dim=1)
    return pred


class cluster_metrics:
    """
    Clustering Quality Evaluation Metrics.
    
    This class computes multiple clustering evaluation metrics by comparing predicted
    cluster assignments to ground truth labels. Metrics include:
    - Normalized Mutual Information (NMI): Measures information shared between true and pred
    - Adjusted Rand Index (ARI): Measures agreement between true and pred assignments
    - Accuracy (ACC): After optimal label mapping via Hungarian algorithm
    - F1-score: Harmonic mean of precision and recall after label mapping
    
    The class handles potential label misalignment by using the Munkres (Hungarian)
    algorithm to find optimal matching between true labels and predicted labels.
    """
    
    def __init__(self, trues, predicts):
        """
        Initialize metrics calculator.
        
        Args:
            trues (list or np.ndarray): Ground truth cluster labels for all nodes
            predicts (torch.Tensor or np.ndarray): Predicted cluster labels for all nodes
        """
        # Ground truth labels: list of true cluster IDs
        self.true_label = trues
        # Predicted labels: convert to numpy array and move to CPU if needed
        self.pred_label = predicts.cpu().numpy()

    def clusterAcc(self):
        """
        Compute clustering accuracy after optimal label matching.
        
        Uses the Munkres (Hungarian) algorithm to find the optimal one-to-one mapping
        between predicted cluster labels and ground truth labels that maximizes accuracy.
        This handles cases where predicted clusters use different label IDs than ground truth.
        
        Returns:
            acc (float): Accuracy after label alignment (0 to 1)
            f1_macro (float): Macro-averaged F1-score
            precision_macro (float): Macro-averaged precision
            recall_macro (float): Macro-averaged recall
            f1_micro (float): Micro-averaged F1-score
            precision_micro (float): Micro-averaged precision
            recall_micro (float): Micro-averaged recall
            new_predict (np.ndarray): Relabeled predictions aligned with ground truth
        """
        # Get unique labels in ground truth
        l1 = list(set(self.true_label))
        numclass1 = len(l1)
        # Count occurrences of each ground truth label
        count_true = Counter(self.true_label)

        # Get unique labels in predictions
        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        # Count occurrences of each predicted label
        count_pred = Counter(self.pred_label)
        
        # Check if number of classes matches
        if numclass1 != numclass2:
            print(f"Class Not equal, class_true({numclass1})={count_true}, class_pred({numclass2})={count_pred}")
            return 0, 0, 0, 0, 0, 0, 0, self.pred_label

        # Build cost matrix: cost[i][j] = # nodes in class i with predicted label j
        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            # Find all nodes with true label c1
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                # Count how many of these nodes are predicted as c2
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]
                cost[i][j] = len(mps_d)

        # Use Munkres (Hungarian) algorithm to find optimal label matching
        # Minimize total cost by maximizing agreement
        m = Munkres()
        # Negate cost to convert min problem to max problem
        cost = cost.__neg__().tolist()
        # Compute optimal assignment: indexes[i] = (true_label_i, pred_label_i)
        indexes = m.compute(cost)

        # Create relabeled predictions using optimal matching
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # Get the predicted label to match with true label l1[i]
            c2 = l2[indexes[i][1]]
            # Find all nodes predicted as c2
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            # Relabel them as c (ground truth label)
            new_predict[ai] = c

        # Store relabeled predictions
        self.new_predicts = new_predict
        
        # Compute evaluation metrics using relabeled predictions
        acc = metrics.accuracy_score(self.true_label, new_predict)
        # Macro-averaged metrics: average across all classes
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        # Micro-averaged metrics: compute metrics globally by counting total true positives
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, new_predict

    def evaluateFromLabel(self, use_acc=False):
        """
        Evaluate clustering quality using multiple metrics.
        
        Computes clustering metrics without requiring label alignment (NMI, ARI) and
        optionally computes metrics with optimal label alignment (Accuracy, F1-score).
        
        Args:
            use_acc (bool): If True, also compute accuracy and F1-score with label alignment.
                           If False, only compute NMI and ARI.
        
        Returns:
            acc (float or 0): Accuracy after label alignment (only if use_acc=True)
            nmi (float): Normalized Mutual Information (0 to 1, higher is better)
            f1_macro (float or 0): Macro-averaged F1-score (only if use_acc=True)
            ari (float): Adjusted Rand Index (-1 to 1, higher is better)
            new_predict (np.ndarray): Relabeled predictions (only if use_acc=True)
        """
        # Compute NMI: Measures mutual information between true and predicted labels
        # normalized by entropy. Range: 0 to 1 (1 = perfect match)
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        # Compute ARI: Measures similarity of cluster assignments adjusted for chance
        # Range: -1 to 1 (-1 = no agreement, 1 = perfect agreement)
        ari = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        
        if use_acc:
            # Compute accuracy after optimal label matching
            acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, new_predict = self.clusterAcc()
            return acc, nmi, f1_macro, ari, new_predict
        else:
            # Return only NMI and ARI without label alignment
            return 0, nmi, 0, ari, self.pred_label