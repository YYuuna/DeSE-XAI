"""
Command-line Argument Parser for DeSE Model

This module defines all command-line arguments for configuring the DeSE clustering model
and training process. It provides a centralized way to pass hyperparameters from the command
line to the training script.
"""

import argparse


def parse_args():
    """
    Parse command-line arguments for DeSE training.
    
    Returns:
        args (argparse.Namespace): Parsed arguments with all hyperparameter settings
    """
    parser = argparse.ArgumentParser(description="DeSE - Deep Structural Entropy Graph Clustering")
    
    # ==================== Dataset Configuration ====================
    parser.add_argument('--dataset', nargs='?', default='Computers', 
                        help='Choose a dataset from {Cora, Citeseer, Pubmed, Computers, Photo, CS, Physics}. '
                             'Citation networks: Cora, Citeseer, Pubmed; '
                             'E-commerce: Computers, Photo; '
                             'Co-authorship: CS, Physics')
    
    # ==================== Training Configuration ====================
    parser.add_argument('--epochs', type=int, default=1000, 
                        help='Total number of training epochs (default: 1000). '
                             'One epoch processes all nodes and edges once.')
    
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='Learning rate for Adam optimizer (default: 0.01). '
                             'Controls step size in gradient descent updates.')
    
    parser.add_argument('--verbose', type=int, default=20,
                        help='Evaluate clustering metrics every N epochs (default: 20). '
                             'Set lower for more frequent evaluation, higher to skip evaluations.')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42). '
                             'Ensures same results across runs.')
    
    # ==================== Hierarchy Configuration ====================
    parser.add_argument('--height', type=int, default=2,
                        help='Number of hierarchy levels in SE tree (default: 2). '
                             'height=1: single-level clustering. '
                             'height=2: 2-level hierarchical clustering. '
                             'height=3: 3-level hierarchical clustering, etc.')
    
    parser.add_argument('--num_clusters_layer', type=list, default=[10],
                        help='Number of clusters in each hierarchy level (default: [10]). '
                             'List length should equal height. '
                             'Example: [50, 10] for 2-level hierarchy with 50 clusters at level 1, 10 at level 2.')
    
    parser.add_argument('--layer_str', type=str, default='[10]',
                        help='String representation of cluster numbers (default: "[10]"). '
                             'Used for parsing and display. Example: "[50, 10]".')
    
    parser.add_argument('--decay_rate', type=int, default=None,
                        help='Decay rate for cluster reduction across hierarchy levels (default: None). '
                             'If set, automatically computes num_clusters for each level. '
                             'Example: decay_rate=5 with height=3 gives [500, 100, 20].')
    
    # ==================== Model Architecture ====================
    parser.add_argument('--embed_dim', type=int, default=16,
                        help='Dimension of node embeddings (default: 16). '
                             'Lower values: faster training, less capacity. '
                             'Higher values: more expressive, slower training.')
    
    parser.add_argument('--k', type=int, default=2,
                        help='Number of nearest neighbors for KNN adjacency matrix (default: 2). '
                             'Used for feature similarity graph in coarsening step.')
    
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate for regularization (default: 0.1). '
                             'Probability of dropping neurons during training. '
                             'Range: 0 (no dropout) to 1 (drop everything).')
    
    parser.add_argument('--activation', type=str, default='relu',
                        help='Activation function for hidden layers (default: "relu"). '
                             'Options: "relu" (ReLU), "elu" (ELU), "sigmoid" (Sigmoid), "None" (no activation).')
    
    # ==================== Loss Function Weights ====================
    parser.add_argument('--se_lamda', type=float, default=0.01,
                        help='Weight coefficient for Structural Entropy (SE) loss (default: 0.01). '
                             'Total loss = se_lamda * se_loss + lp_lamda * lp_loss. '
                             'Higher values emphasize structural clustering quality.')
    
    parser.add_argument('--lp_lamda', type=float, default=1,
                        help='Weight coefficient for Link Prediction (LP) loss (default: 1.0). '
                             'Higher values emphasize embedding fidelity and preserving local structure.')
    
    parser.add_argument('--beta_f', type=float, default=0.2,
                        help='Weight coefficient for feature-based adjacency matrix (default: 0.2). '
                             'In graph coarsening: merged_adj = (1-beta_f) * graph_adj + beta_f * feature_adj.')
    
    # ==================== Output and Visualization ====================
    parser.add_argument('--save', type=bool, default=False,
                        help='Save model checkpoints (default: False). '
                             'Saves best models based on NMI, ARI, ACC, F1 metrics to save_model/ directory.')
    
    parser.add_argument('--fig_network', type=bool, default=False,
                        help='Generate network visualization (default: False). '
                             'Creates t-SNE plots of node embeddings.')
    
    parser.add_argument('--export_ccts', action='store_true',
                        help='Export dataset and partition for CCTS explainer (default: False). '
                             'Exports graph edge list and predicted clustering for explainability analysis.')
    
    parser.add_argument('--ccts_method', type=str, default='DeSE',
                        help='Method name for CCTS exporter (default: "DeSE"). '
                             'Creates output folder: community_partitions/{ccts_method}/')
    
    # ==================== Hardware Configuration ====================
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID to use (default: 0 = first GPU). '
                             'Set to -1 to use CPU instead of GPU.')

    return parser.parse_args()