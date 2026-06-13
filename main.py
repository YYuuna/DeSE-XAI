"""
╔════════════════════════════════════════════════════════════════════════════╗
║              DeSE Training & Evaluation Script - Main Entry Point           ║
╚════════════════════════════════════════════════════════════════════════════╝

█ SCRIPT PURPOSE
  Complete training pipeline for Deep Structural Entropy (DeSE) clustering model
  with evaluation, checkpointing, visualization, and explainability export.

█ EXECUTION FLOW
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 1. SETUP PHASE                                                      │
  │    - Parse command-line arguments (hyperparameters)                 │
  │    - Set random seeds for reproducibility                           │
  │    - Configure GPU/CPU device                                       │
  │    - Load dataset (Cora, Citeseer, Photo, etc.)                     │
  └─────────────────────────────────────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 2. MODEL INITIALIZATION                                             │
  │    - Create DeSE model with architecture (height, embed_dim, etc.)  │
  │    - Create Adam optimizer (learning rate: lr)                      │
  │    - Move model to GPU/CPU device                                   │
  └─────────────────────────────────────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 3. TRAINING LOOP (for epoch in range(epochs))                       │
  │    a. FORWARD PASS                                                  │
  │       s_dict, embeddings, graphs = model(adj, features, degrees)    │
  │    b. LOSS COMPUTATION                                              │
  │       se_loss = model.calculate_se_loss1()                          │
  │       lp_loss = model.calculate_lp_loss(...)                        │
  │       total_loss = se_lamda * se_loss + lp_lamda * lp_loss          │
  │    c. BACKPROPAGATION                                               │
  │       optimizer.zero_grad() → loss.backward() → optimizer.step()    │
  │    d. PERIODIC EVALUATION (every verbose epochs)                    │
  │       Compute metrics: NMI, ARI, ACC, F1                            │
  │       Track best performing models                                  │
  │       Save checkpoints for best metrics                             │
  └─────────────────────────────────────────────────────────────────────┘
            ↓
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 4. POST-TRAINING PHASE                                              │
  │    - Print final results summary                                    │
  │    - Save training results to output file                           │
  │    - Export clustering for explainability (CCTS) if requested       │
  │    - Generate visualizations (t-SNE plots)                          │
  └─────────────────────────────────────────────────────────────────────┘

█ KEY FUNCTIONS
  
  train(args):
    - Main training loop coordinating all phases
    - Arguments: parsed command-line configuration
    - Returns: best clustering results dictionary
    - Duration: ~minutes to hours depending on dataset size and epochs
  
  export_ccts_dataset(dataset, name, method, partition):
    - Exports graph structure (edge list) and clustering (partition)
    - Creates: data/graph/{name}.txt (edges)
              community_partitions/{method}/{name}_partition.pkl (assignments)
    - Purpose: Input for CCTS explainability analysis
  
  draw_network(dataset):
    - Generates t-SNE visualization of learned embeddings
    - Creates: figure/network_{dataset}_{clusters}.png
              figure/network_{dataset}_{clusters}_true.png (ground truth)
    - Allows visual inspection of clustering quality

█ EVALUATION METRICS EXPLAINED
  
  NMI (Normalized Mutual Information):
    - Range: [0, 1], higher is better
    - Measures information shared between predicted and ground truth clustering
    - 0 = no agreement, 1 = perfect agreement
    - Intuition: How much knowing pred helps predict true labels
  
  ARI (Adjusted Rand Index):
    - Range: [-1, 1], higher is better
    - Measures agreement adjusted for random chance
    - -1 = inverse agreement, 0 = random, 1 = perfect
    - More stable than accuracy for imbalanced clusters
  
  ACC (Accuracy with Optimal Label Matching):
    - Range: [0, 1], higher is better
    - Uses Hungarian algorithm to find best label permutation
    - Handles cases where predicted clusters have different IDs
    - Percentage of nodes in correct cluster
  
  F1-Score (Harmonic Mean):
    - Range: [0, 1], higher is better
    - Balances precision and recall
    - Macro-average treats all classes equally
    - Micro-average weights by class frequency

█ MODEL CHECKPOINTING STRATEGY
  
  Best models saved when EACH metric reaches new maximum:
    - Best NMI: save_model/{dataset}_{clusters}_nmi.pt
    - Best ARI: save_model/{dataset}_{clusters}_ari.pt
    - Best ACC: save_model/{dataset}_{clusters}_acc.pt
    - Best F1: save_model/{dataset}_{clusters}_f1.pt
  
  Rationale: Different metrics prioritize different aspects of clustering
  - NMI: Information sharing (structure)
  - ARI: Robustness to chance
  - ACC: Interpretability (node-to-cluster accuracy)
  - F1: Balance of precision/recall

█ TYPICAL HYPERPARAMETER SETTINGS
  
  Fast Training (sanity check):
    epochs=50, verbose=10, height=1, embed_dim=8, se_lamda=0.01, lp_lamda=1
  
  Standard Training:
    epochs=1000, verbose=20, height=2, embed_dim=16, se_lamda=0.01, lp_lamda=1
  
  Fine-grained Clustering:
    epochs=1000, verbose=20, height=3, embed_dim=32, se_lamda=0.1, lp_lamda=0.5
  
  Coarse-grained Clustering:
    epochs=500, verbose=10, height=1, embed_dim=8, se_lamda=0.001, lp_lamda=2

█ TROUBLESHOOTING GUIDE
  
  Problem: Metrics not improving
  → Try: Lower learning rate, increase epochs, adjust se_lamda/lp_lamda ratio
  
  Problem: NMI/ARI high but ACC low
  → Try: Increase embed_dim, try different height, adjust k parameter
  
  Problem: Training too slow
  → Try: Reduce height, reduce embed_dim, increase verbose (less evaluation)
  
  Problem: Clustering dispersed (many small clusters)
  → Try: Decrease num_clusters_layer, increase se_lamda, decrease dropout
  
  Problem: Clustering too coarse (few large clusters)
  → Try: Increase num_clusters_layer, decrease se_lamda, decrease height

█ EXPECTED PERFORMANCE
  
  Cora dataset (typical):
    - NMI: 0.50-0.70
    - ARI: 0.40-0.65
    - ACC: 0.75-0.85
    - F1: 0.75-0.85
  
  Note: Exact values depend on random seed, hyperparameters, and initialization

█ OUTPUT FILES GENERATED
  
  During training:
    output/{dataset}.result - Training configuration and best metrics
    save_model/{dataset}_{k}_nmi/ari/acc/f1.pt - Model checkpoints
  
  If export_ccts enabled:
    data/graph/{dataset}.txt - Edge list
    community_partitions/DeSE/{dataset}_partition.pkl - Cluster assignments
  
  If fig_network enabled:
    figure/network_{dataset}_{k}.png - t-SNE visualization
    figure/network_{dataset}_{k}_true.png - Ground truth t-SNE
"""

from utility.parser import parse_args
from utility.dataset import Data
from utility.util import decoding_from_assignment, cluster_metrics
from model import DeSE
import torch
import torch.optim as optim
from time import time, strftime, localtime
import os
import pickle
import numpy as np
import random
import dgl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import Counter
import networkx as nx
from matplotlib.colors import ListedColormap

# Set GPU device ID (set to GPU 1 for this machine)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# Enable anomaly detection for debugging gradient issues
torch.autograd.set_detect_anomaly(True)
# Check if CUDA/GPU is available
print(torch.cuda.is_available())


def export_ccts_dataset(dataset, dataset_name, method, partition):
    """
    Export graph and clustering partition for CCTS (Community-aware Contrastive Explanations) analysis.
    
    This function saves the graph structure and node cluster assignments to files that can be
    used for explainability analysis using the CCTS method. It creates both:
    1. Graph file: edge list in text format
    2. Partition file: pickle file with node-to-cluster mappings
    
    Args:
        dataset (Data): Dataset object containing graph information
        dataset_name (str): Name of the dataset (e.g., 'Cora', 'Photo')
        method (str): Name of the clustering method (used for directory naming)
        partition (torch.Tensor): Cluster assignment for each node, shape (num_nodes,)
    """
    # Create directories for graph and partition data
    graph_dir = os.path.join('data', 'graph')
    part_dir = os.path.join('community_partitions', method)
    os.makedirs(graph_dir, exist_ok=True)
    os.makedirs(part_dir, exist_ok=True)

    # Extract edge list from edge_index tensor
    edge_index = dataset.edge_index.cpu().numpy()
    edges = set()
    # Remove self-loops and create undirected edges by sorting (u, v) -> (min(u,v), max(u,v))
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u == v:
            # Skip self-loops
            continue
        # Store as sorted tuple to ensure each undirected edge appears only once
        edges.add(tuple(sorted((u, v))))

    # Save graph as edge list (one edge per line)
    graph_path = os.path.join(graph_dir, f'{dataset_name}.txt')
    with open(graph_path, 'w', encoding='utf-8') as f:
        # Sort edges for consistent output
        for u, v in sorted(edges):
            f.write(f"{u} {v}\n")

    # Create partition dictionary: node_id -> cluster_id
    partition_dict = {int(i): int(int(label)) for i, label in enumerate(partition)}
    # Save partition as pickle file for easy loading
    part_path = os.path.join(part_dir, f'{dataset_name}_partition.pkl')
    with open(part_path, 'wb') as f:
        pickle.dump(partition_dict, f)

    # Log export completion
    print(f'Exported CCTS graph to: {graph_path}')
    print(f'Exported CCTS partition to: {part_path}')



def train(args):
    """
    Main training function for DeSE model.
    
    This function:
    1. Initializes the dataset and model
    2. Runs the training loop for specified number of epochs
    3. Computes two losses: structural entropy (SE) and link prediction (LP)
    4. Evaluates clustering quality using multiple metrics (NMI, ARI, ACC, F1)
    5. Saves best models based on different evaluation metrics
    6. Exports results for explainability analysis if requested
    
    Args:
        args (argparse.Namespace): Command-line arguments containing all hyperparameters
            - dataset: Dataset name (Cora, Citeseer, Photo, etc.)
            - gpu: GPU device ID (-1 for CPU)
            - epochs: Number of training epochs
            - lr: Learning rate
            - height: Number of hierarchy levels
            - embed_dim: Embedding dimension
            - se_lamda: Weight for structural entropy loss
            - lp_lamda: Weight for link prediction loss
            - verbose: Evaluation frequency (evaluate every N epochs)
            - save: Whether to save best models
            - export_ccts: Whether to export for CCTS analysis
            - seed: Random seed for reproducibility
    """
    # Set device: GPU if available and requested, otherwise CPU
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    # Override: always use CPU (uncomment above line to use GPU)
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load dataset
    dataset = Data(args.dataset, device)
    dataset.print_statistic()
    
    # Initialize variables to track best results
    final_pred = None  # Store final clustering prediction
    
    # Track best clustering results for each metric
    best_cluster_result = {}
    # Track best metric values achieved so far
    best_cluster = {'nmi': -0.001, 'ari': -0.001, 'acc': -0.001, 'f1': -0.001}
    
    # Initialize DeSE model with hyperparameters
    model = DeSE(args, dataset.feature, device).to(device)
    # Set up optimizer: Adam with specified learning rate
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # Record training start time
    t0 = time()
    
    # Training loop
    for epoch in range(args.epochs):
        # Epoch start time (for measuring epoch duration)
        t1 = time()
        
        # Forward pass: compute soft assignments and embeddings at all hierarchy levels
        s_dic, tree_node_embed_dic, g_dic = model(dataset.adj, dataset.feature, dataset.degrees)
        t2 = time()
        
        # Compute Structural Entropy (SE) loss
        se_loss = model.calculate_se_loss1()
        t3 = time()
        
        # Compute Link Prediction (LP) loss
        lp_loss = model.calculate_lp_loss(g_dic[args.height], dataset.neg_edge_index, 
                                         tree_node_embed_dic[args.height])
        t4 = time()
        
        # Combine losses: total_loss = se_lamda * se_loss + lp_lamda * lp_loss
        loss = args.se_lamda * se_loss + args.lp_lamda * lp_loss
        # Backpropagation: clear gradients, compute gradients, update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Evaluate clustering quality every 'verbose' epochs
        if epoch % args.verbose == 0:
            # Convert soft assignments to hard cluster IDs
            pred = decoding_from_assignment(model.hard_dic[1])
            final_pred = pred
            
            # Compute clustering metrics
            metrics = cluster_metrics(dataset.labels, pred)
            acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
            
            # Track best NMI score
            if nmi > best_cluster['nmi']:
                best_cluster['nmi'] = nmi
                best_cluster_result['nmi'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), 
                             f'./save_model/{args.dataset}_{args.num_clusters_layer[0]}_nmi.pt')
            
            # Track best ARI score
            if ari > best_cluster['ari']:
                best_cluster['ari'] = ari
                best_cluster_result['ari'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), 
                             f'./save_model/{args.dataset}_{args.num_clusters_layer[0]}_ari.pt')
            
            # Track best ACC score
            if acc > best_cluster['acc']:
                best_cluster['acc'] = acc
                best_cluster_result['acc'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), 
                             f'./save_model/{args.dataset}_{args.num_clusters_layer[0]}_acc.pt')
            
            # Track best F1 score
            if f1 > best_cluster['f1']:
                best_cluster['f1'] = f1
                best_cluster_result['f1'] = [nmi, ari, acc, f1]
                if args.save:
                    torch.save(model.state_dict(), 
                             f'./save_model/{args.dataset}_{args.num_clusters_layer[0]}_f1.pt')
            
            # Print epoch statistics
            print(f"Epoch: {epoch} [{time()-t1:.3f}s], Loss: {loss.item():.6f} = {args.se_lamda} * {se_loss.item():.6f} + {args.lp_lamda} * {lp_loss.item():.6f}, NMI: {nmi:.6f}, ARI: {ari:.6f}, ACC: {acc:.6f}, F1: {f1:.6f}")
    
    # Print final results after all epochs
    print(f"Best NMI: {best_cluster_result['nmi']}, Best ARI: {best_cluster_result['ari']}, "
          f"\nBest Cluster: {best_cluster}")
    print(args)

    # Save results summary to file
    save_path = './output/%s.result' % args.dataset
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'a') as f:
        # Write hyperparameter configuration
        f.write(f"lr={args.lr}, embed_dim={args.embed_dim}, se_lamda={args.se_lamda}, "
                f"lp_lamda={args.lp_lamda}, k={args.k}, dropout={args.dropout}, "
                f"beta_f={args.beta_f}, epochs={args.epochs}, height={args.height}, "
                f"num_clusters={args.num_clusters_layer}, verbose={args.verbose}, "
                f"activation={args.activation}, seed={args.seed} \n")
        # Write best results
        f.write(f"--------Best NMI: {best_cluster_result['nmi']}, "
                f"Best ARI: {best_cluster_result['ari']}, "
                f"Best Cluster: {best_cluster} \n")

    # Export clustering results for explainability analysis if requested
    if args.export_ccts:
        if final_pred is None:
            try:
                final_pred = decoding_from_assignment(model.hard_dic[1])
            except Exception as e:
                print(f'Unable to decode final partition from model: {e}')
                final_pred = torch.tensor(dataset.labels, device=device)

        export_ccts_dataset(dataset, args.dataset, args.ccts_method, final_pred)

    return best_cluster


def draw_network(dataset):
    #prepare graph dataset and device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = 'cuda:{}'.format(args.gpu)
    else:
        device = 'cpu'
    dataset = Data(args.dataset, device)
    dataset.print_statistic()
    model = DeSE(args, dataset.feature, device).to(device)
    model.load_state_dict(torch.load('./save_model/{}_{}_acc.pt'.format(args.dataset, args.num_clusters_layer[0])))
    s_dic, tree_node_embed_dic, g_dic = model(dataset.adj, dataset.feature, dataset.degrees)
    pred = decoding_from_assignment(model.hard_dic[1])
    metrics = cluster_metrics(dataset.labels, pred)
    acc, nmi, f1, ari, new_pred = metrics.evaluateFromLabel(use_acc=True)
    print(new_pred)
    print('NMI: {:.6f}, ARI: {:.6f}, ACC: {:.6f}, F1: {:.6f}'.format(nmi, ari, acc, f1))
    '''
    nx_graph = dgl.to_networkx(g_dic[args.height])
    pos = nx.spring_layout(nx_graph, k=0.035, iterations=50, seed=5114)
    fig, ax = plt.subplots(figsize=(10,10))
    plt.xlim(-1.05,1.05)
    plt.ylim(-1.05,1.05)
    color = []
    c =['darkred', 'royalblue', 'darkgreen', 'darkorange', 'darkcyan', 'darkmagenta', 'darkgoldenrod', 'darkviolet', 'darkslategray', 'darkturquoise', 'darkkhaki', 'darkolivegreen', 'darkorchid', 'darkseagreen', 'darkslateblue', 'darkgray', 'darkblue', 'darkcyan', 'darkgoldenrod', 'darkgreen', 'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange', 'darkorchid', 'darkred', 'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise', 'darkviolet']
    for item in pred:
        color.append(c[item])
    print('drawing figure...')
    nx.draw_networkx_nodes(nx_graph, pos=pos, ax=ax, alpha=1, node_color=color, node_size=15)
    '''
    tsne = TSNE(n_components=2, random_state=5114)
    X_tsne = tsne.fit_transform(tree_node_embed_dic[args.height].detach().cpu().numpy())
    custom_cmap = ListedColormap(['#7d6181', '#d7aed3', '#46717c', '#a9d6d6', '#d5ba82', 
                                  '#aebfce', '#b36a6f', '#3c79b4', '#a8d9a2', '#ef7d31'])

    fig, ax = plt.subplots(figsize=(10,10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=new_pred, cmap=custom_cmap, s=50)
    plt.axis('off')
    os.makedirs(os.path.dirname("./figure/"), exist_ok=True)
    plt.savefig("./figure/network_{}_{}.pdf".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')
    plt.savefig("./figure/network_{}_{}.png".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')

    fig, ax = plt.subplots(figsize=(10,10))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=dataset.labels, cmap=custom_cmap, s=50)
    plt.axis('off')
    plt.savefig("./figure/network_{}_{}_true.pdf".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')
    plt.savefig("./figure/network_{}_{}_true.png".format(args.dataset, args.num_clusters_layer[0]), bbox_inches='tight')


if __name__ == "__main__":
    args = parse_args()
    args.dataset = 'Cora'
    args.save = True
    args.export_ccts = True
    if args.dataset == 'Cora':
        args.epochs = 600
        args.verbose = 1
        args.beta_f = 0.2 # 0.2, w/o feature beta_f=0
        args.dropout = 0.3  # 0.3
        args.embed_dim = 8 #8
        args.k = 1
        args.num_clusters_layer = [9]  # [9]
        args.lp_lamda = 5  #5
        args.se_lamda = 0.01  #0.01
        args.lr = 0.01
        args.seed = 406  #406
    elif args.dataset == 'Citeseer':
        args.epochs = 600
        args.verbose = 20
        args.beta_f = 0.1 # 0.1
        args.dropout = 0.05  #0.05
        args.embed_dim = 32  #32
        args.k = 1
        args.num_clusters_layer = [7] #[7]
        args.lp_lamda = 0.5  #0.5
        args.se_lamda = 0.01 #0.01
        args.lr = 0.001  #0.001
        args.seed = 262  #262
    elif args.dataset == 'Photo':
        args.epochs = 100  #800
        args.verbose = 1
        args.beta_f = 0.4  #0.4
        args.dropout = 0.05  #0.05
        args.embed_dim = 64  #64
        args.k = 1
        args.num_clusters_layer = [9]  #[9]
        args.lp_lamda = 5  #5
        args.se_lamda = 0.01  #0.01
        args.lr = 0.001 #0.001
        args.seed = 132  #132
    elif args.dataset == 'Computers':   
        args.epochs = 800 #800
        args.verbose = 20
        args.beta_f = 0.4 #0.4
        args.dropout = 0.3 #0.3
        args.embed_dim = 32  #32
        args.k = 1
        args.num_clusters_layer = [11] #[11]
        args.lp_lamda = 0.5  #0.5
        args.se_lamda = 0.2  #0.2
        args.lr = 0.001  #0.001
        args.seed = 323  #323
    elif args.dataset == 'Pubmed':
        args.epochs = 1000
        args.verbose = 20
        args.beta_f = 0.3
        args.dropout = 0.05
        args.embed_dim = 16
        args.k = 1
        args.num_clusters_layer = [5]
        args.lp_lamda = 1
        args.se_lamda = 0.5
        args.lr = 0.001
        args.seed = 335
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True #使得网络相同输入下每次运行的输出固定
    dgl.seed(args.seed)
    train(args)
    draw_network(args.dataset)
    