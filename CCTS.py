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

method=['Louvain','GN','LPA','FN','DeSE']
num = 4
# names = [ 'karate','football', 'personal', 'polblogs', 'polbooks', 'railways','web-spam','road-minnesota','cit-DBLP']
#            0          1          2           3           4           5         6         7              8
# names = [ 'karate','football', 'personal', 'polbooks' ,'Cora' ]
names = ['Cora']
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
    def __init__(self, G, community_nodes, alpha=1, beta=1, seta=1, if_part=True, if_limit_threshold=True, if_threshold_pruning=True, if_BFS=True):
        self.G = G
        self.community_nodes = community_nodes
        self.alpha = alpha
        self.beta = beta
        self.seta = seta
        self.if_part = if_part
        self.if_limit_threshold = if_limit_threshold
        self.if_threshold_pruning = if_threshold_pruning
        self.if_BFS = if_BFS

    def calculate_precision_and_error_rate(self, community_nodes, threshold_nodes):
        """Calculate precision and error rate"""
        correct_count = sum(1 for node in community_nodes if node in threshold_nodes)
        precision = correct_count / len(community_nodes) if community_nodes else 0
        # incorrect_nodes = [node for node in threshold_nodes if node not in community_nodes]
        # incorrect_nodes =
        # error_rate = len(incorrect_nodes) / len(threshold_nodes) if threshold_nodes else 0
        error_rate = (len(threshold_nodes)-correct_count) / len(threshold_nodes) if threshold_nodes else 0
        # return precision, error_rate, incorrect_nodes
        return precision, error_rate


    def objective_function(self, precision, error_rate):
        """Objective function: prefer smaller thresholds"""
        return self.alpha * precision - self.beta * error_rate

    def find_best_center_and_threshold(self):
        best_center_node = None
        best_threshold = None
        final_threshold_nodes = None
        best_score = -float('inf')
        center_threshold = -1
        # Get candidate nodes and partition proportion
        top_nodes, proportion = self.get_top_nodes(len(self.community_nodes))
        for candidate_node in top_nodes:
            shortest_path_lengths = nx.single_source_shortest_path_length(G, candidate_node)
            max_distance = max(shortest_path_lengths[node] for node in community_nodes if node in shortest_path_lengths)
            left, right = self.determine_threshold_range(max_distance, center_threshold, len(self.community_nodes))
            temp_best_score = 0
            if self.if_BFS:
                threshold_nodes = set([candidate_node])
                # Traverse original distance dictionary and store nodes by distance layer
                distance_layered_dict = {}
                flag = True
                for target_node, distance in shortest_path_lengths.items():
                    if distance not in distance_layered_dict:
                        distance_layered_dict[distance] = []
                    distance_layered_dict[distance].append(target_node)

            for threshold in range(left, right):
                # Memoized BFS, dynamic programming
                if self.if_BFS:
                    if flag:
                        new_nodes = []
                        for dist in range(1, threshold + 1):
                            if dist in distance_layered_dict:
                                new_nodes.extend(distance_layered_dict[dist])
                        flag = False
                    else:
                        new_nodes = distance_layered_dict[threshold]
                    threshold_nodes.update(new_nodes)
                else:
                    threshold_nodes = [node for node, distance in shortest_path_lengths.items() if distance <= threshold]

                precision, error_rate = self.calculate_precision_and_error_rate(self.community_nodes, threshold_nodes)
                current_score = self.objective_function(precision, error_rate)

                if current_score > best_score or (current_score == best_score and threshold < best_threshold and best_threshold is None):
                    best_precision = precision
                    final_error_rate = error_rate
                    final_threshold_nodes = threshold_nodes
                    best_score = current_score
                    best_center_node = candidate_node
                    best_threshold = threshold
                    # Use threshold limiting
                    if if_limit_threshold == True:
                        center_threshold = threshold
                # Threshold pruning
                if if_threshold_pruning == True:
                    if current_score > temp_best_score:
                        temp_best_score = current_score
                    else:
                            break
                if precision == 1.0:
                    break

        return best_score, best_precision, final_error_rate, best_center_node, best_threshold, proportion, final_threshold_nodes

    def get_top_nodes(self, num_nodes):
        """Determine the proportion based on community size and filter eligible nodes"""
        if self.if_part and num_nodes > 100:
            # Code to select important nodes in the community can be added here
            return self.find_top_nodes()
        else:
            return self.community_nodes, 1

    # Determine proportion based on community size
    def determine_proportion(self, community_size, max_community_size, min_proportion=0.05):
        # Use a sigmoid function to adjust proportion so smaller communities have higher proportions and larger communities have lower proportions
        # Here we adjust the sigmoid function so it begins to drop quickly when community size approaches half of max_community_size
        k = 5 / max_community_size  # Control the steepness of the curve
        x0 = max_community_size / 2  # Control the horizontal shift of the curve; here set to half the maximum community size
        proportion = min(1.0, min_proportion + (1 - 1 / (1 + math.exp(-k * (community_size - x0)))))

        return proportion


    # Filter eligible nodes by proportion
    def find_top_nodes(self):
        # Sort by node degree
        degree_dict = {node: self.G.degree(node) for node in community_nodes}
        sorted_nodes = sorted(degree_dict, key=degree_dict.get, reverse=True)  # Sort by degree descending
        # Calculate the node selection proportion
        proportion = self.determine_proportion(len(community_nodes), max_community_size)
        # Calculate the number of nodes to take
        num_top_nodes = max(100, int(len(sorted_nodes) * proportion))  # Ensure at least one node
        # Take only the high-degree nodes
        top_nodes = sorted_nodes[:num_top_nodes]
        # print(f'Total: {len(community_nodes)}, selected: {len(top_nodes)}, partition proportion: {proportion * 100:.2f}%')
        return top_nodes, proportion

    def determine_threshold_range(self, max_distance, center_threshold, num_nodes):
        """Determine threshold search range"""
        if self.if_limit_threshold and center_threshold != -1 and num_nodes > 100:
            left = max(1, center_threshold - self.seta)
            right = min(center_threshold + self.seta, max_distance + 1)
        else:
            left = 1
            right = max_distance + 1
        return left, right


def plot_ccts_results(G, community_results, current_method, dataset_name):
    result_dir = os.path.join('result', current_method)
    os.makedirs(result_dir, exist_ok=True)

    ids = [result['community_id'] for result in community_results]
    sizes = [result['node_count'] for result in community_results]
    thresholds = [result['threshold'] for result in community_results]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(ids, sizes, color='skyblue', label='Node count')
    ax1.set_xlabel('Community ID')
    ax1.set_ylabel('Node count', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')

    ax2 = ax1.twinx()
    ax2.plot(ids, thresholds, color='orange', marker='o', label='Threshold')
    ax2.set_ylabel('Threshold', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')

    fig.suptitle(f'{dataset_name} - Community sizes and thresholds ({current_method})')
    fig.tight_layout()
    fig.savefig(os.path.join(result_dir, f'{dataset_name}_community_stats.png'))
    plt.close(fig)

    if community_results:
        all_nodes = list(G.nodes())
        if len(all_nodes) > 1:
            pos = nx.spring_layout(G, seed=42, k=2.0 / math.sqrt(len(all_nodes)), iterations=2000)
        else:
            pos = {n: (0, 0) for n in all_nodes}

        max_plots = min(len(community_results), 10)
        visualize_results = sorted(community_results, key=lambda r: r['node_count'], reverse=True)[:max_plots]

        for result in visualize_results:
            comm_nodes = set(result['original_nodes'])
            threshold_nodes = set(result['all_nodes_in_explainable_region'] or [])
            overlap_nodes = comm_nodes & threshold_nodes
            community_only = comm_nodes - threshold_nodes
            threshold_only = threshold_nodes - comm_nodes

            colors = []
            sizes = []
            alphas = []
            for node in all_nodes:
                if node in overlap_nodes:
                    colors.append('#9467bd')
                    sizes.append(220)
                    alphas.append(0.95)
                elif node in community_only:
                    colors.append('#1f77b4')
                    sizes.append(180)
                    alphas.append(0.85)
                elif node in threshold_only:
                    colors.append('#d62728')
                    sizes.append(120)
                    alphas.append(0.75)
                else:
                    colors.append('#cccccc')
                    sizes.append(50)
                    alphas.append(0.25)

            fig = plt.figure(figsize=(28, 28), dpi=300)
            ax = fig.add_subplot(1, 1, 1)
            nx.draw_networkx_edges(G, pos=pos, ax=ax, edge_color='#bbbbbb', alpha=0.15, width=0.5)
            nx.draw_networkx_nodes(G, pos=pos, ax=ax, node_color=colors, node_size=sizes, alpha=0.85, linewidths=0.3, edgecolors='black')

            if len(all_nodes) <= 200:
                nx.draw_networkx_labels(G, pos=pos, ax=ax, font_size=6, font_color='black')

            ax.set_title(f'{dataset_name} Community {result["community_id"]}: overlap with explainable region', fontsize=26)
            ax.set_axis_off()
            plt.tight_layout()
            out_path = os.path.join(result_dir, f'{dataset_name}_community_{result["community_id"]}_full_graph.png')
            plt.savefig(out_path, dpi=300)
            plt.close()

def compute_cluster_layout(G, partition, community_results=None):
    """Compute a cluster-aware initial layout then refine with a global spring layout.

    The approach places communities near soft centers (small radius) to keep
    related nodes together, then runs a global spring refinement with the
    initial positions to produce a natural-looking graph while preserving
    readability.
    """
    communities = {}
    for node, comm in partition.items():
        communities.setdefault(comm, []).append(node)

    rep_nodes = set()
    if community_results:
        for r in community_results:
            cn = r.get('center_node')
            if cn is not None:
                rep_nodes.add(cn)

    comm_list = sorted(communities.keys())
    num_comms = len(comm_list)
    N = max(1, G.number_of_nodes())

    # Use a small radius so communities are not overly separated
    radius = max(6.0, 6.0 * math.sqrt(max(1, num_comms) / 4.0))

    # Place community centers on a small circle
    comm_centers = {}
    for idx, comm in enumerate(comm_list):
        theta = 2 * math.pi * idx / max(1, num_comms)
        comm_centers[comm] = (radius * math.cos(theta), radius * math.sin(theta))

    pos = {}
    for comm in comm_list:
        nodes = communities[comm]
        k = len(nodes)
        cx, cy = comm_centers[comm]

        if k == 1:
            pos[nodes[0]] = (cx, cy)
            continue

        # For small communities, arrange on a small local ring
        if k <= 30:
            ring_r = 1.2 + math.sqrt(k) * 0.6
            for j, n in enumerate(nodes):
                angle = 2 * math.pi * j / k
                pos[n] = (cx + ring_r * math.cos(angle), cy + ring_r * math.sin(angle))
            continue

        # For larger communities, compute a compact local spring layout
        subG = G.subgraph(nodes)
        try:
            local_pos = nx.spring_layout(subG, seed=42, k=0.5, iterations=200)
        except Exception:
            local_pos = {n: (random.random(), random.random()) for n in nodes}

        xs = [p[0] for p in local_pos.values()]
        ys = [p[1] for p in local_pos.values()]
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        span_x = max(1.0, maxx - minx)
        span_y = max(1.0, maxy - miny)
        scale = 1.6 + math.sqrt(k) * 0.5

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