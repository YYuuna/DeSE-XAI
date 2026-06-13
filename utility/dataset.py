"""
Dataset Loading and Preparation Module

This module provides utilities for loading various graph datasets and preparing them
for use in the DeSE clustering model. It supports multiple benchmark datasets:
- Citation networks: Cora, Citeseer, Pubmed
- E-commerce networks: Computers, Photo
- Co-authorship networks: CS, Physics

Datasets are automatically downloaded and processed through PyTorch Geometric (torch_geometric)
or DGL. The Data class creates necessary graph structures (adjacency matrices, features, degrees)
needed for training the DeSE model.
"""

import sys
sys.path.append('/workspace/DSE')
import torch
import numpy as np
from torch_scatter import scatter_sum
import torch_geometric.datasets
from torch_geometric.datasets import Planetoid, Amazon, Coauthor
from torch_geometric.utils import negative_sampling
from utility.util import index2adjacency
import dgl.data
import scipy.sparse as sp
import time
from collections import Counter


class Data:
    """
    Data container for graph datasets compatible with DeSE model.
    
    This class loads graph datasets from PyTorch Geometric and prepares all necessary
    data structures for model input:
    - Node features (feat)
    - Edge indices and weights
    - Node degrees
    - Ground truth labels (for evaluation)
    - Negative samples for link prediction loss
    """
    
    def __init__(self, dataset_name, device):
        """
        Initialize and load a graph dataset.
        
        Args:
            dataset_name (str): Name of dataset to load
                - 'Cora', 'Citeseer', 'Pubmed': Citation networks
                - 'Computers', 'Photo': Amazon e-commerce networks
                - 'CS', 'Physics': Co-authorship networks
            device (str): Device to place tensors on ('cpu' or 'cuda:X')
        """
        self.name = dataset_name
        
        # Load appropriate dataset based on name
        if dataset_name in ['Cora', 'Citeseer', 'Pubmed']:
            # Planetoid: citation network datasets
            dataset = Planetoid(root='./datasets', name=dataset_name)
        elif dataset_name in ['Computers', 'Photo']:
            # Amazon: e-commerce network datasets
            dataset = Amazon(root='./datasets', name=dataset_name)
        elif dataset_name in ["CS", "Physics"]:
            # Coauthor: co-authorship network datasets
            dataset = Coauthor(root='./datasets', name=dataset_name)
        
        # Extract data object from dataset
        data = dataset.data

        # Graph statistics
        self.num_nodes = data.x.shape[0]  # Number of nodes in graph
        self.feature = data.x.to(device)  # Node features: (num_nodes, feature_dim)
        self.num_features = data.x.shape[1]  # Dimension of node features
        self.num_edges = int(data.edge_index.shape[1] / 2)  # Number of edges (undirected)
        self.edge_index = data.edge_index  # Edge index tensor: (2, num_edges * 2)
        
        # Edge weights (all ones for unweighted graph)
        self.weight = torch.ones(self.edge_index.shape[1])
        
        # Node degrees: sum of edge weights for each node
        self.degrees = scatter_sum(self.weight, self.edge_index[0]).to(device)
        
        # Ground truth cluster labels for evaluation
        self.labels = data.y.tolist()
        # Number of ground truth clusters/classes
        self.num_classes = len(np.unique(self.labels))
        
        # Adjacency matrix as sparse COO tensor
        self.adj = torch.sparse_coo_tensor(
            indices=self.edge_index,
            values=self.weight,
            size=(self.num_nodes, self.num_nodes)
        )
        
        # Create DGL graph for message passing
        graph = dgl.graph((self.edge_index[0], self.edge_index[1]), num_nodes=self.num_nodes).to(device)
        # Move weights and edge_index to device
        self.weight = self.weight.to(device)
        self.edge_index = self.edge_index.to(device)
        # Add self-loops for GCN processing
        self.graph = dgl.add_self_loop(graph)
        
        # Generate negative samples for link prediction loss
        # These are non-existent edges used as negative examples
        self.neg_edge_index = negative_sampling(self.edge_index)

    def print_statistic(self):
        """Print dataset statistics for verification."""
        print(f"Dataset Name: {self.name}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Feature: {self.feature.shape}")
        print(f"edge_index: {self.edge_index.shape}")
        print(f"Graph: {self.graph}")


class Graph:
    """
    Alternative graph data container using DGL for direct graph loading.
    
    This class loads datasets directly using DGL's dataset utilities and
    creates graph structures compatible with message passing algorithms.
    """
    
    def __init__(self, dataset_name):
        """
        Initialize and load a graph dataset using DGL.
        
        Args:
            dataset_name (str): Name of dataset to load
                - 'Cora', 'Citeseer', 'Pubmed': Citation networks
                - 'Computers', 'Photo': Amazon e-commerce networks
                - 'CS', 'Physics': Co-authorship networks
        """
        self.name = dataset_name
        
        # Load appropriate dataset using DGL
        if dataset_name == 'Cora':
            data = dgl.data.CoraGraphDataset()
        elif dataset_name == 'Citeseer':
            data = dgl.data.CiteseerGraphDataset()
        elif dataset_name == 'Pubmed':
            data = dgl.data.PubmedGraphDataset()
        elif dataset_name == 'Computers':
            data = dgl.data.AmazonCoBuyComputerDataset()
        elif dataset_name == 'Photo':
            data = dgl.data.AmazonCoBuyPhotoDataset()
        elif dataset_name == 'CS':
            data = dgl.data.CoauthorPhysicsDataset()
        elif dataset_name == 'Physics':
            data = dgl.data.CoauthorCSDataset()
        
        # Extract first graph from dataset (usually the only one)
        graph = data[0]

        # Graph statistics
        self.num_nodes = graph.number_of_nodes()
        # Node features stored in 'feat' attribute
        self.feature = graph.ndata['feat']
        self.num_features = graph.ndata['feat'].shape[1]
        # Number of edges (undirected, so divide by 2)
        self.num_edges = int(graph.number_of_edges() / 2)
        # Edge index tensor
        self.edge_index = torch.stack(graph.edges(order='eid'))
        
        # Edge weights (all ones for unweighted graph)
        self.weight = torch.ones(self.edge_index.shape[1])
        # Node in-degrees
        self.degrees = graph.in_degrees()
        
        # Ground truth labels
        self.labels = graph.ndata['label'].tolist()
        self.num_classes = len(np.unique(self.labels))
        
        # Adjacency matrix as scipy sparse matrix
        self.adj = sp.coo_matrix(
            (self.weight, (self.edge_index[0], self.edge_index[1])),
            shape=(self.num_nodes, self.num_nodes)
        )
        # Create DGL graph with self-loops for GCN
        self.graph = dgl.add_self_loop(graph)

    def print_statistic(self):
        """Print dataset statistics for verification."""
        print(f"Dataset Name: {self.name}")
        print(f"Number of nodes: {self.num_nodes}")
        print(f"Number of edges: {self.num_edges}")
        print(f"Number of features: {self.num_features}")
        print(f"Number of classes: {self.num_classes}")
        print(f"Feature: {self.feature.shape}")
        print(f"edge_index: {self.edge_index.shape}")
        print(f"Graph: {self.graph}")


        

    


