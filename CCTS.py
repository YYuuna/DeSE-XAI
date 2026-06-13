"""
CCTS: Community-aware Clustering Transparency System

This module provides explainability analysis for graph clustering results. It implements
the CCTS framework which:

1. Analyzes each predicted community to find representative center nodes
2. Computes explainable regions around each center using distance-based thresholds
3. Evaluates precision and error rate of explainable regions
4. Visualizes communities with cluster-aware layouts and color coding
5. Generates analysis reports showing how well communities are explained

Key Components:
- CommunityAnalyzer: Finds best center nodes and distance thresholds for each community
- plot_ccts_results: Visualizes communities with center nodes and explainable regions
- compute_cluster_layout: Generates community-aware graph layouts using spring layout + small rings
"""

import networkx as nx
import pickle
import os
import numpy as np
import random
import time
import math
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon, Circle

# Supported clustering methods that can be analyzed
method=['Louvain','GN','LPA','FN','DeSE']
# Number of methods being analyzed
num = 4
# Dataset names to analyze
names = ['Cora']
# Current dataset index
index = 0

tag=[False,True]
i=1
# Whether to write results to a file
if_write = True
# if_write = False
# Whether to use all nodes in the community
# if_part = True
if_part = tag[i]
# Whether to limit the threshold range
# if_limit_threshold = True
if_limit_threshold = tag[i]
# Whether to use threshold pruning
# if_threshold_pruning = True
if_threshold_pruning = tag[i]
# Whether to use BFS
# if_BFS = True
if_BFS = tag[i]
# Fixed random seed
random.seed(42)
np.random.seed(42)




class CommunityAnalyzer:
    """
    Analyze communities to find explainable center nodes and distance thresholds.
    
    For each community, this class:
    1. Finds candidate center nodes (high-degree nodes or important hubs)
    2. For each candidate, computes shortest path distances to all community members
    3. Searches for optimal distance threshold that maximizes:
       - Precision: % of threshold-selected nodes that are in the community
       - Minimizes error rate: % of threshold-selected nodes that are NOT in the community
    4. Uses an objective function combining alpha*precision - beta*error_rate
    5. Employs optimization strategies: BFS memoization, threshold pruning, threshold limiting
    """
    
    def __init__(self, G, community_nodes, alpha=1, beta=1, seta=1, if_part=True, 
                 if_limit_threshold=True, if_threshold_pruning=True, if_BFS=True):
        """
        Initialize community analyzer for a single community.
        
        Args:
            G (networkx.Graph): The full graph
            community_nodes (list): Node IDs belonging to this community
            alpha (float): Weight for precision in objective function (default: 1)
            beta (float): Weight for error rate in objective function (default: 1)
            seta (float): Range constraint for threshold search (default: 1)
            if_part (bool): Use only important nodes as candidates (default: True)
            if_limit_threshold (bool): Limit threshold search range between communities (default: True)
            if_threshold_pruning (bool): Early exit if score decreases (default: True)
            if_BFS (bool): Use BFS with memoization for efficiency (default: True)
        """
        self.G = G
        self.community_nodes = community_nodes
        # Precision weight: higher alpha emphasizes including true community members
        self.alpha = alpha
        # Error rate weight: higher beta penalizes including false members
        self.beta = beta
        # Threshold search constraint: limits how far threshold can be from previous community
        self.seta = seta
        # Use only high-degree nodes as candidate centers
        self.if_part = if_part
        # Limit threshold search range for consistency across communities
        self.if_limit_threshold = if_limit_threshold
        # Enable early stopping when objective function score decreases
        self.if_threshold_pruning = if_threshold_pruning
        # Use BFS with dynamic programming for efficiency
        self.if_BFS = if_BFS

    def calculate_precision_and_error_rate(self, community_nodes, threshold_nodes):
        """
        Calculate precision and error rate of a threshold selection.
        
        Precision = (# threshold nodes in community) / (# community nodes)
        Error rate = (# threshold nodes NOT in community) / (# threshold nodes)
        
        Args:
            community_nodes (list): True community member IDs
            threshold_nodes (set): Node IDs selected by distance threshold
        
        Returns:
            precision (float): Fraction of community members correctly identified [0, 1]
            error_rate (float): Fraction of false positives in selection [0, 1]
        """
        # Count nodes that are both in community and selected by threshold
        correct_count = sum(1 for node in community_nodes if node in threshold_nodes)
        # Precision: how many true community members we captured
        precision = correct_count / len(community_nodes) if community_nodes else 0
        # Error rate: how many false positives in our selection
        error_rate = (len(threshold_nodes)-correct_count) / len(threshold_nodes) if threshold_nodes else 0
        return precision, error_rate


    def objective_function(self, precision, error_rate):
        """
        Compute objective score for a threshold configuration.
        
        Objective = alpha * precision - beta * error_rate
        
        Balances two goals:
        - Maximize precision: include as many true community members as possible
        - Minimize error rate: avoid including non-community nodes
        
        Args:
            precision (float): Fraction of true community members captured
            error_rate (float): Fraction of false positives in threshold region
        
        Returns:
            score (float): Objective value (higher is better)
        """
        return self.alpha * precision - self.beta * error_rate

    def find_best_center_and_threshold(self):
        """
        Find optimal center node and distance threshold for this community.
        
        Algorithm:
        1. Select candidate center nodes (high-degree nodes if if_part=True)
        2. For each candidate:
           a. Compute shortest path distances from candidate to all nodes
           b. Find maximum distance to any community member
           c. Search distance thresholds from 1 to max_distance
           d. For each threshold, select all nodes within that distance
           e. Evaluate precision and error_rate
           f. Track best threshold with highest objective score
        3. Return best center node, threshold, and evaluation metrics
        
        Optimizations:
        - if_threshold_pruning: Stop threshold search if score decreases
        - if_BFS: Build distance layers incrementally instead of recomputing each threshold
        - if_limit_threshold: Limit search range based on center_threshold
        
        Returns:
            best_score (float): Objective function value at best threshold
            best_precision (float): Precision at best threshold
            final_error_rate (float): Error rate at best threshold
            best_center_node (int): Selected center node ID
            best_threshold (int): Selected distance threshold (hops)
            proportion (float): Fraction of high-degree nodes used as candidates
            final_threshold_nodes (set): All nodes within best distance threshold
        """
        best_center_node = None
        best_threshold = None
        final_threshold_nodes = None
        best_score = -float('inf')
        center_threshold = -1
        
        # Get candidate nodes: if community is large, use only high-degree nodes
        top_nodes, proportion = self.get_top_nodes(len(self.community_nodes))
        
        # Try each candidate center node
        for candidate_node in top_nodes:
            # Compute shortest path distances from candidate to all nodes
            shortest_path_lengths = nx.single_source_shortest_path_length(G, candidate_node)
            # Find max distance to any community member (upper bound for threshold search)
            max_distance = max(shortest_path_lengths[node] for node in community_nodes if node in shortest_path_lengths)
            # Determine threshold search range
            left, right = self.determine_threshold_range(max_distance, center_threshold, len(self.community_nodes))
            temp_best_score = 0
            
            if self.if_BFS:
                # BFS-based incremental construction of threshold nodes
                threshold_nodes = set([candidate_node])
                # Pre-organize nodes by distance layer for efficient incremental addition
                distance_layered_dict = {}
                flag = True
                for target_node, distance in shortest_path_lengths.items():
                    if distance not in distance_layered_dict:
                        distance_layered_dict[distance] = []
                    distance_layered_dict[distance].append(target_node)

            # Try each distance threshold
            for threshold in range(left, right):
                # Memoized BFS: incrementally add nodes at each distance layer
                if self.if_BFS:
                    if flag:
                        # First iteration: add all nodes up to this threshold
                        new_nodes = []
                        for dist in range(1, threshold + 1):
                            if dist in distance_layered_dict:
                                new_nodes.extend(distance_layered_dict[dist])
                        flag = False
                    else:
                        # Subsequent iterations: only add new layer
                        new_nodes = distance_layered_dict[threshold]
                    threshold_nodes.update(new_nodes)
                else:
                    # Non-BFS: recompute from scratch each time
                    threshold_nodes = [node for node, distance in shortest_path_lengths.items() if distance <= threshold]

                # Evaluate this threshold
                precision, error_rate = self.calculate_precision_and_error_rate(self.community_nodes, threshold_nodes)
                current_score = self.objective_function(precision, error_rate)

                # Update best if this threshold is better
                if current_score > best_score or (current_score == best_score and threshold < best_threshold and best_threshold is None):
                    best_precision = precision
                    final_error_rate = error_rate
                    final_threshold_nodes = threshold_nodes
                    best_score = current_score
                    best_center_node = candidate_node
                    best_threshold = threshold
                    # Update threshold range for next community
                    if if_limit_threshold == True:
                        center_threshold = threshold
                
                # Threshold pruning: stop if score decreases
                if if_threshold_pruning == True:
                    if current_score > temp_best_score:
                        temp_best_score = current_score
                    else:
                        break
                
                # Early stopping if perfect precision achieved
                if precision == 1.0:
                    break

        return best_score, best_precision, final_error_rate, best_center_node, best_threshold, proportion, final_threshold_nodes

    def get_top_nodes(self, num_nodes):
        """
        Get candidate center nodes, filtering by importance if community is large.
        
        For large communities (>100 nodes), uses only high-degree nodes as candidates
        to reduce computation. For small communities, uses all members.
        
        Args:
            num_nodes (int): Number of nodes in community
        
        Returns:
            top_nodes (list): Candidate center node IDs
            proportion (float): Fraction of community used as candidates
        """
        if self.if_part and num_nodes > 100:
            # Use only important (high-degree) nodes for large communities
            return self.find_top_nodes()
        else:
            # Use all community members for small communities
            return self.community_nodes, 1

    # Determine proportion based on community size
    def determine_proportion(self, community_size, max_community_size, min_proportion=0.05):
        """
        Dynamically determine proportion of nodes to use based on community size.
        
        Uses sigmoid function: smaller communities use more candidate nodes (higher proportion),
        larger communities use fewer (lower proportion) for computational efficiency.
        
        Args:
            community_size (int): Size of current community
            max_community_size (int): Size of largest community in dataset
            min_proportion (float): Minimum proportion to always use (default: 0.05)
        
        Returns:
            proportion (float): Fraction of community nodes to use as candidates [min_proportion, 1.0]
        """
        # Sigmoid parameters: control steepness and midpoint
        k = 5 / max_community_size  # Steepness: larger k = sharper transition
        x0 = max_community_size / 2  # Midpoint: shift to half max size
        # Sigmoid: min_proportion + (1 - 1/(1 + exp(-k(x-x0))))
        proportion = min(1.0, min_proportion + (1 - 1 / (1 + math.exp(-k * (community_size - x0)))))
        return proportion


    # Filter eligible nodes by proportion
    def find_top_nodes(self):
        """
        Select high-degree nodes as candidate centers for large communities.
        
        Sorts community members by degree and selects top-k nodes where k is
        determined by the proportion calculation.
        
        Returns:
            top_nodes (list): Selected high-degree node IDs
            proportion (float): Fraction of community used
        """
        # Compute degree for each community node
        degree_dict = {node: self.G.degree(node) for node in community_nodes}
        # Sort by degree descending
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)
        # Calculate the node selection proportion
        proportion = self.determine_proportion(len(community_nodes), max_community_size)
        # Calculate the number of nodes to take
        num_top_nodes = max(100, int(len(sorted_nodes) * proportion))  # Ensure at least 100 nodes
        # Take only the high-degree nodes
        top_nodes = sorted_nodes[:num_top_nodes]
        # print(f'Total: {len(community_nodes)}, selected: {len(top_nodes)}, partition proportion: {proportion * 100:.2f}%')
        return top_nodes, proportion

    def determine_threshold_range(self, max_distance, center_threshold, num_nodes):
        """
        Determine the range of distances to search for optimal threshold.
        
        Can limit search range for consistency:
        - Without limiting: search from 1 to max_distance
        - With limiting: search around previous community's threshold
        
        Args:
            max_distance (int): Maximum distance to any community member
            center_threshold (int): Threshold of previous community (-1 if first community)
            num_nodes (int): Size of community
        
        Returns:
            left (int): Minimum threshold to try
            right (int): Maximum threshold to try (exclusive)
        """
        # Limit threshold range for larger communities if specified
        if self.if_limit_threshold and center_threshold != -1 and num_nodes > 100:
            # Search near previous community's threshold: [center-seta, center+seta]
            left = max(1, center_threshold - self.seta)
            right = min(center_threshold + self.seta, max_distance + 1)
        else:
            # Full range: 1 to max_distance
            left = 1
            right = max_distance + 1
        return left, right


def plot_ccts_results(G, community_results, current_method, dataset_name):
    """
    Visualize clustering results with community-aware layouts.
    
    Creates multiple visualizations:
    1. Community size and threshold bar/line chart
    2. Full graph visualization for top 10 largest communities with:
       - Overlapping nodes (purple): In both community and explainable region
       - Community-only nodes (blue): In community but outside explainable region
       - Threshold-only nodes (red): In explainable region but outside community
       - Other nodes (gray): Not in either region
    
    Args:
        G (networkx.Graph): The full graph
        community_results (list): List of community analysis results with keys:
            - 'community_id': Community identifier
            - 'node_count': Number of nodes in community
            - 'threshold': Distance threshold for explainable region
            - 'original_nodes': Nodes belonging to community
            - 'all_nodes_in_explainable_region': Nodes in threshold region
        current_method (str): Method name (e.g., 'DeSE')
        dataset_name (str): Dataset name (e.g., 'Cora')
    """
    # Create output directory for results
    result_dir = os.path.join('result', current_method)
    os.makedirs(result_dir, exist_ok=True)

    # Extract data from results
    ids = [result['community_id'] for result in community_results]
    sizes = [result['node_count'] for result in community_results]
    thresholds = [result['threshold'] for result in community_results]

    # Create bar + line plot: community sizes and thresholds
    fig, ax1 = plt.subplots(figsize=(10, 5))
    # Bar plot for node counts
    ax1.bar(ids, sizes, color='skyblue', label='Node count')
    ax1.set_xlabel('Community ID')
    ax1.set_ylabel('Node count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    # Line plot for thresholds (dual y-axis)
    ax2 = ax1.twinx()
    ax2.plot(ids, thresholds, color='orange', marker='o', label='Threshold')
    ax2.set_ylabel('Threshold', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle(f'{dataset_name} - Community sizes and thresholds ({current_method})')
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, f'{dataset_name}_community_stats.png'))
    plt.close(fig)

    # Visualize individual communities
    if community_results:
        all_nodes = list(G.nodes())
        # Compute spring layout for consistent visualization
        if len(all_nodes) > 1:
            # Use cluster-aware layout with small radius
            pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(len(all_nodes)), iterations=2000)
        else:
            # Single node: place at origin
            pos = {n: (0, 0) for n in all_nodes}

        # Visualize top 10 largest communities
        max_plots = min(len(community_results), 10)
        visualize_results = sorted(community_results, key=lambda r: r['node_count'], reverse=True)[:max_plots]

        for result in visualize_results:
            # Get node sets from result
            comm_nodes = set(result['original_nodes'])
            threshold_nodes = set(result['all_nodes_in_explainable_region'] or [])
            # Compute different node categories
            overlap_nodes = comm_nodes & threshold_nodes  # Both
            community_only = comm_nodes - threshold_nodes  # Community but not threshold
            threshold_only = threshold_nodes - comm_nodes  # Threshold but not community

            # Color code nodes by category
            colors = []
            sizes = []
            alphas = []
            for node in all_nodes:
                if node in overlap_nodes:
                    # Overlap: both in community and explainable region (good!)
                    colors.append('#9467bd')
                    sizes.append(220)
                    alphas.append(0.95)
                elif node in community_only:
                    # Community only: not explained by threshold region
                    colors.append('#1f77b4')
                    sizes.append(180)
                    alphas.append(0.85)
                elif node in threshold_only:
                    # Threshold only: false positive in explainable region
                    colors.append('#d62728')
                    sizes.append(120)
                    alphas.append(0.75)
                else:
                    # Other: not relevant to this community
                    colors.append('#cccccc')
                    sizes.append(50)
                    alphas.append(0.25)

            # Create large high-quality visualization
            fig = plt.figure(figsize=(28, 28), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            # Draw edges
            nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='#bbbbbb', alpha=0.15, width=0.5)
            # Draw nodes with color coding
            nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, node_size=sizes, 
                                   alpha=0.85, linewidths=0.3, edgecolors='black')

            # Add node labels for small graphs
            if len(all_nodes) <= 200:
                nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=6, font_color='black')

            ax.set_title(f'{dataset_name} Community {result["community_id"]}: overlap with explainable region', fontsize=26)
            ax.set_axis_off()
            plt.tight_layout()
            # Save visualization
            out_path = os.path.join(result_dir, f'{dataset_name}_community_{result["community_id"]}_full_graph.png')
            plt.savefig(out_path, dpi=300)
            plt.close()


def compute_cluster_layout(G, partition, community_results=None):
    """
    Compute cluster-aware layout for graph visualization.
    
    Combines two-stage layout strategy:
    1. **Initial stage**: Places each community at a global position determined by:
       - Community centers arranged in a circle for clear separation
       - Within each community: small local arrangements to keep cohesion
    2. **Refinement stage** (implicit): Spring forces can be applied for smoothing
    
    For small communities (≤30 nodes): Arrange on a small circular ring around community center
    For large communities (>30 nodes): Compute local spring layout within community bounds
    
    This approach balances:
    - Preserving community structure (nodes in same community stay close)
    - Readability (communities clearly separated in global layout)
    - Visualization quality (spring forces create natural-looking layouts)
    
    Args:
        G (networkx.Graph): The full graph
        partition (dict): Node -> community ID mapping
        community_results (list, optional): Community analysis results containing center nodes
    
    Returns:
        pos (dict): Node -> (x, y) position coordinates for visualization
    """
    # Organize nodes by community
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    # Extract representative (center) nodes from community results
    rep_nodes = set()
    if community_results:
        for r in community_results:
            cn = r.get('center_node')
            if cn is not None:
                rep_nodes.add(cn)

    # Get sorted list of community IDs
    comm_list = sorted(communities.keys())
    num_comms = len(comm_list)
    N = max(1, G.number_of_nodes())

    # ============ GLOBAL LAYOUT: Place communities in circle ============
    # Use a small radius so communities don't spread too far apart
    # Radius scales with number of communities
    radius = max(6.0, 6.0 * math.sqrt(max(1, num_comms) / 4.0))

    # Place community centers on a circle
    comm_centers = {}
    for idx, comm in enumerate(comm_list):
        # Evenly distribute communities around circle
        theta = 2 * math.pi * idx / max(1, num_comms)
        comm_centers[comm] = (radius * math.cos(theta), radius * math.sin(theta))

    pos = {}
    
    # ============ LOCAL LAYOUT: Position nodes within each community ============
    for comm in comm_list:
        nodes = communities[comm]
        k = len(nodes)  # Community size
        cx, cy = comm_centers[comm]  # Community center position

        if k == 1:
            # Single node: place exactly at community center
            pos[nodes[0]] = (cx, cy)
            continue

        # ---- For small communities: arrange in circular ring ----
        if k <= 30:
            # Ring radius increases with community size
            ring_r = 1.2 + math.sqrt(k) * 0.6
            # Distribute nodes evenly around a circle centered at (cx, cy)
            for j, n in enumerate(nodes):
                angle = 2 * math.pi * j / k
                pos[n] = (cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle))
            continue

        # ---- For large communities: compute local spring layout ----
        # Extract subgraph containing only this community's nodes
        subG = G.subgraph(nodes)
        try:
            # Compute spring layout within this subgraph
            local_pos = nx.spring_layout(subG, seed=42, k=0.5, iterations=200)
        except Exception:
            # Fallback: random positions if spring layout fails
            local_pos = {n: (random.random(), random.random()) for n in nodes}

        # Compute bounding box of local layout
        xs = [p[0] for p in local_pos.values()]
        ys = [p[1] for p in local_pos.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        
        # Compute dimensions of local layout
        span_x = max(1.0, maxx - minx)
        span_y = max(1.0, maxy - miny)
        
        # Scale factor: larger communities take more space locally
        scale = 1.6 + math.sqrt(k) * 0.5

        # Transform local positions to global coordinates
        # 1. Normalize to [0, 1]
        # 2. Scale by 'scale' factor
        # 3. Center at community center (cx, cy)
        for n, p in local_pos.items():
            nxp = (p[0] - minx) / span_x * scale - scale / 2.0
            nyp = (p[1] - miny) / span_y * scale - scale / 2.0
            pos[n] = (cx + nxp, cy + nyp)

    # Safety for any missing nodes
    for n in G.nodes():
        if n not in pos:
            pos[n] = (random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))

    # Global spring refinement: use pos as initial positions to get a natural graph
    # with moderate inter-community attraction. Do not strictly fix centers so the
    # layout can relax naturally.
    try:
        # choose k relative to graph size: smaller k -> tighter graph
        k_global = max(0.08, 0.6 / math.sqrt(max(1, N)))
        refined = nx.spring_layout(G, pos=pos, iterations=400, seed=42, k=k_global)
        # Post-process positions to enforce more uniform spacing between nodes
        # Compute bounding box and target spacing
        xs = [p[0] for p in refined.values()]
        ys = [p[1] for p in refined.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        width = max(1e-6, maxx - minx)
        height = max(1e-6, maxy - miny)
        target = math.sqrt((width * height) / max(1, N)) * 0.9

        # Simple repulsion iterations: push nodes apart if closer than target
        positions = dict(refined)
        for _iter in range(60):
            disp = {n: [0.0, 0.0] for n in positions}
            moved = False
            for i, a in enumerate(positions):
                ax, ay = positions[a]
                for j, b in enumerate(positions):
                    if a == b:
                        continue
                    bx, by = positions[b]
                    dx = ax - bx
                    dy = ay - by
                    dist = math.hypot(dx, dy) + 1e-9
                    if dist < target:
                        # push proportional to deficit
                        factor = (target - dist) / dist * 0.06
                        disp[a][0] += dx * factor
                        disp[a][1] += dy * factor
                        moved = True
            # apply displacements with a damping factor
            for n in positions:
                positions[n] = (positions[n][0] + disp[n][0], positions[n][1] + disp[n][1])
            if not moved:
                break

        return positions
    except Exception:
        return pos


def plot_cluster_layout(G, partition, current_method, dataset_name, community_results=None):
    result_dir = os.path.join('result', current_method)
    os.makedirs(result_dir, exist_ok=True)

    comm_list = sorted(set(partition.values()))
    num_comms = len(comm_list)

    if num_comms <= 20:
        cmap = plt.get_cmap('tab20')
        colors = [cmap(i % 20) for i in range(num_comms)]
    else:
        cmap = plt.get_cmap('hsv')
        colors = [cmap(i / max(1, num_comms)) for i in range(num_comms)]

    pos = compute_cluster_layout(G, partition, community_results)

    community_nodes = {comm: set() for comm in comm_list}
    for node, comm in partition.items():
        if comm in community_nodes:
            community_nodes[comm].add(node)

    explainable_nodes = set()
    rep_nodes = set()
    if community_results:
        for r in community_results:
            explainable_nodes.update(r.get('all_nodes_in_explainable_region') or [])
            cn = r.get('center_node')
            if cn is not None:
                rep_nodes.add(cn)

    fig = plt.figure(figsize=(28, 28), dpi=300)
    fig.patch.set_facecolor('#f5f5f5')
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor('#f5f5f5')

    for i, comm in enumerate(comm_list):
        points = [pos[n] for n in community_nodes[comm] if n in pos]
        if len(points) >= 3:
            try:
                hull = ConvexHull(points)
                hull_pts = [points[v] for v in hull.vertices]
                patch = Polygon(
                    hull_pts,
                    closed=True,
                    facecolor=colors[i],
                    edgecolor=colors[i],
                    linewidth=1.0,
                    alpha=0.10,
                    zorder=0,
                )
                ax.add_patch(patch)
            except Exception:
                pass

    node_colors = []
    node_sizes = []
    node_edgecolors = []
    node_linewidths = []

    for n in G.nodes():
        comm = partition.get(n, -1)
        if comm in comm_list:
            color = colors[comm_list.index(comm)]
        else:
            color = '#cccccc'

        if n in rep_nodes:
            size = 520
            edgecolor = '#000000'
            lw = 1.4
        elif n in explainable_nodes:
            size = 260
            edgecolor = '#ffbf00'
            lw = 1.2
        else:
            size = 180
            edgecolor = '#333333'
            lw = 0.6

        node_colors.append(color)
        node_sizes.append(size)
        node_edgecolors.append(edgecolor)
        node_linewidths.append(lw)

    # Compute edge membership counts: for each edge, count how many communities it belongs to
    # (i.e., for how many communities both endpoints are members). Use this to style edges.
    edge_membership = {}
    # Precompute community node sets
    comm_node_sets = {comm: set(nodes) for comm, nodes in community_nodes.items()}
    for u, v in G.edges():
        count = 0
        for comm, nodeset in comm_node_sets.items():
            if u in nodeset and v in nodeset:
                count += 1
        edge_membership[(u, v)] = count

    # Map membership counts to visual styles
    # 0 -> faint gray, 1 -> standard, 2 -> highlighted, 3+ -> strong highlight
    def edge_style_for_count(c):
        if c <= 0:
            return {'color': '#dddddd', 'width': 1.2, 'alpha': 0.12}
        if c == 1:
            return {'color': '#bbbbbb', 'width': 1.8, 'alpha': 0.18}
        if c == 2:
            return {'color': '#ff7f0e', 'width': 3.2, 'alpha': 0.32}
        return {'color': '#d62728', 'width': 5.0, 'alpha': 0.42}

    # Draw edges grouped by style for a clean legend and better layering
    styles_to_edges = {}
    for (u, v), c in edge_membership.items():
        key = (edge_style_for_count(c)['color'], edge_style_for_count(c)['width'])
        styles_to_edges.setdefault(key, []).append((u, v, c))

    for (color, width), edges_list in styles_to_edges.items():
        el = [(u, v) for (u, v, _) in edges_list]
        nx.draw_networkx_edges(
            G,
            pos=pos,
            edgelist=el,
            ax=ax,
            edge_color=color,
            width=width,
            alpha=0.22,
            arrows=False,
            style='solid',
            connectionstyle='arc3,rad=0.02',
        )

    # Draw nodes after edges so they sit on top
    nx.draw_networkx_nodes(
        G,
        pos=pos,
        ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
        linewidths=node_linewidths,
        edgecolors=node_edgecolors,
        alpha=0.92,
    )

    if G.number_of_nodes() <= 300:
        labels = {n: str(n) for n in rep_nodes}
        nx.draw_networkx_labels(G, pos=pos, labels=labels, font_size=8, font_color='black')

    ax.set_title(f'{dataset_name} — Cluster layout ({current_method})', fontsize=30, fontweight='bold', pad=20)
    ax.set_axis_off()
    # Add legend for edge membership counts and node types
    from matplotlib.lines import Line2D
    edge_handles = [Line2D([0], [0], color=edge_style_for_count(0)['color'], lw=edge_style_for_count(0)['width'], label='0 communities'),
                    Line2D([0], [0], color=edge_style_for_count(1)['color'], lw=edge_style_for_count(1)['width'], label='1 community'),
                    Line2D([0], [0], color=edge_style_for_count(2)['color'], lw=edge_style_for_count(2)['width'], label='2 communities'),
                    Line2D([0], [0], color=edge_style_for_count(3)['color'], lw=edge_style_for_count(3)['width'], label='3+ communities')]
    node_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#000000', markeredgecolor='#000000', markersize=10, label='Representative node'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ffbf00', markeredgecolor='#333333', markersize=8, label='Explainable node'),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markeredgecolor='#555555', markersize=6, label='Regular node')]
    legend_handles = edge_handles + node_handles
    legend = ax.legend(handles=legend_handles, title='Legend', loc='upper right', bbox_to_anchor=(1.14, 1.0), frameon=True, facecolor='#ffffff', framealpha=0.92, edgecolor='#777777')
    legend.get_title().set_fontsize(12)
    legend.get_title().set_fontweight('bold')
    if num_comms <= 10:
        from matplotlib.patches import Patch
        community_handles = [Patch(facecolor=colors[i], edgecolor='none', alpha=0.4, label=f'Community {comm}') for i, comm in enumerate(comm_list)]
        ax.add_artist(ax.legend(handles=community_handles, title='Community colors', loc='lower right', bbox_to_anchor=(1.14, 0.14), frameon=True, facecolor='#ffffff', framealpha=0.92, edgecolor='#777777'))
    plt.tight_layout()
    out_path = os.path.join(result_dir, f'{dataset_name}_cluster_layout.png')
    plt.savefig(out_path, dpi=300, facecolor=fig.get_facecolor())
    plt.close()


if __name__ == '__main__':
    current_method = method[num]
    for index in range(len(names)):
        print(f'Current community: {names[index]}, partition method: {current_method}')
        dataset_name = f'{names[index]}'
        G = nx.read_edgelist(f'data/graph/{dataset_name}.txt', nodetype=int)
        with open(f'./community_partitions/{current_method}/{dataset_name}_partition.pkl', 'rb') as f:
            partition = pickle.load(f)

        print('Data loaded successfully')

        # Extract nodes for each community and build an adjacency list
        communities = {}
        for node, comm_id in partition.items():
            if comm_id not in communities:
                communities[comm_id] = []
            communities[comm_id].append(node)
        communities = list(communities.values())
        if tag[i]==True:
            start_time = time.time()
        # Find the largest community
        max_community_size = max(len(community) for community in communities)
        # Print the number of nodes in the largest community
        print(f"Number of nodes in the largest community: {max_community_size}")

        if tag[i]==False:
            start_time = time.time()
        # Store center node, threshold, coverage and other info for each community
        community_results = []
        community_thresholds = []
        for index, community_nodes in enumerate(communities):
            analyzer = CommunityAnalyzer(G, community_nodes, if_part=if_part, if_limit_threshold=if_limit_threshold, if_threshold_pruning=if_threshold_pruning, if_BFS=if_BFS)
            part_start_time = time.time()
            score, coverage_rate, error_rate, center_node, threshold_distance, proportion, threshold_nodes = analyzer.find_best_center_and_threshold()
            part_end_time = time.time()
            part_total_cost_time = part_end_time - part_start_time
            print(f'Community {index}, center node {center_node}, threshold {threshold_distance}, time {part_total_cost_time * 1000}ms')
            community_thresholds.append((center_node, threshold_distance))
            community_nodes = list(set(community_nodes))
            community_results.append({
                'community_id': len(community_results),
                'node_count': len(community_nodes),
                'all_nodes_in_explainable_region': threshold_nodes,
                'original_nodes': community_nodes,
                'partition_proportion': proportion,
                'center_node': center_node,
                'threshold': threshold_distance,
                'time_cost': part_total_cost_time,
                'score': score ,
                'coverage_rate': coverage_rate,
                'error_rate': error_rate
            })

        end_time = time.time()
        total_cost_time = end_time - start_time

        # Output results for each community
        for result in community_results:
            print(f"Community {result['community_id']} --> Node count: {result['node_count']}  Score: [{result['score']:.2f}]")
            print(f"  Center node: {result['center_node']}, Threshold: {result['threshold']}")
            print(f"  Original nodes: {result['original_nodes']}")
            print(f"  All nodes in explainable region: {result['all_nodes_in_explainable_region']}")
            print(f"  Partition proportion: {result['partition_proportion'] * 100:.2f}%, Time: {result['time_cost'] * 1000}ms")
            print(f"  Coverage rate: {result['coverage_rate']:.2%}, Error rate: {result['error_rate']:.2%}\n")

        print(f"Total execution time: {total_cost_time * 1000}ms")
        # Calculate total number of communities
        total_communities = len(community_results)

        # Calculate average score, average precision, and average error rate
        average_score = np.mean([result['score'] for result in community_results])
        average_precision = np.mean([result['coverage_rate'] for result in community_results])  # coverage_rate is precision
        average_error_rate = np.mean([result['error_rate'] for result in community_results])

        # Print results
        print(f"Total communities: {total_communities}")
        print(f"Average community score: {average_score:.2f}")
        print(f"Average precision: {average_precision:.2%}")  # Converted to percentage format
        print(f"Average error rate: {average_error_rate:.2%}")  # Converted to percentage format



        if if_write:
            # Check if result folder exists and create it if needed
            result_dir = 'result'
            result_method_dir = os.path.join(result_dir, current_method)
            os.makedirs(result_method_dir, exist_ok=True)
            # Check if dataset_name folder exists and create it if needed
            # dataset_dir = os.path.join(result_dir, dataset_name)
            # if not os.path.exists(dataset_dir):
            #     os.makedirs(dataset_dir)
            param = ''
            if if_part:
                param += '1'
            else:
                param += '0'
            if if_limit_threshold:
                param += '_1'
            else:
                param += '_0'
            if if_threshold_pruning:
                param += '_1'
            else:
                param += '_0'
            # Open a file for writing, create it if it does not exist

            with open(f'result/{current_method}/{dataset_name}_{param}.txt', 'w', encoding='utf-8') as file:
                # Output each community's results
                for result in community_results:
                    file.write(f"Community {result['community_id']} --> Node count: {result['node_count']}  Score: [{result['score']:.2f}]\n")
                    file.write(f"  Center node: {result['center_node']}, Threshold: {result['threshold']}\n")
                    file.write(f"  Partition proportion: {result['partition_proportion'] * 100:.2f}%, Time: {result['time_cost'] * 1000}ms\n")
                    file.write(f"  Coverage rate: {result['coverage_rate']:.2%}, Error rate: {result['error_rate']:.2%}\n\n")

                # Output total execution time of the loop
                file.write(f"Total execution time: {total_cost_time * 1000}ms\n")
                file.write("\n===================\n")
                file.write(f"Total communities: {total_communities}\n")
                file.write(f"Average community score: {average_score:.2f}\n")
                file.write(f"Average precision: {average_precision:.2%}\n")
                file.write(f"Average error rate: {average_error_rate:.2%}\n")        
                # plot_ccts_results(G, community_results, current_method, dataset_name)
                # plot_all_communities(G, partition, current_method, dataset_name)
                plot_cluster_layout(G, partition, current_method, dataset_name, community_results)
                print('#############################################################\n\n')