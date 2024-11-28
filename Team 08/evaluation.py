import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
from community import community_louvain
import os

def calculate_centralities(adj_matrix):
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: shape={adj_matrix.shape}")
    print(f"Processing adjacency matrix of shape: {adj_matrix.shape}")

    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G)

    # Calculate the participation coefficient with the partition
    pc_dict = participation_coefficient(G, partition)

    # Calculate averages of centrality measures
    pr = nx.pagerank(G, alpha=0.9)
    ec = nx.eigenvector_centrality_numpy(G, max_iter=100)
    bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)
    ns = np.array(list(nx.degree_centrality(G).values())) * (len(G.nodes()) - 1)
    acc = nx.average_clustering(G, weight=None)

    # Average participation coefficient
    pc_avg = np.mean(list(pc_dict.values()))

    return {
        'pr': np.mean(list(pr.values())),
        'ec': np.mean(list(ec.values())),
        'bc': np.mean(list(bc.values())),
        'ns': ns,
        'pc': pc_avg,
        'acc': acc
    }

def participation_coefficient(G, partition):
    # Initialize dictionary for participation coefficients
    pc_dict = {}

    # Calculate participation coefficient for each node
    for node in G.nodes():
        node_degree = G.degree(node)
        if node_degree == 0:
            pc_dict[node] = 0.0
        else:
            # Count within-module connections
            within_module_degree = sum(1 for neighbor in G[node] if partition[neighbor] == partition[node])
            # Calculate participation coefficient
            pc_dict[node] = 1 - (within_module_degree / node_degree) ** 2

    return pc_dict


def evaluate_all(true_hr_matrices, predicted_hr_matrices, output_path='ID-randomCV.csv'):
    print(true_hr_matrices.shape)
    print(predicted_hr_matrices.shape)
    
    num_subjects = true_hr_matrices.shape[0]
    results = []

    for i in range(num_subjects):
        true_matrix = true_hr_matrices[i, :, :]
        pred_matrix = predicted_hr_matrices[i, :, :]

        print(f"Evaluating subject {i+1} with matrix shapes: true={true_matrix.shape}, pred={pred_matrix.shape}")

        if true_matrix.shape != pred_matrix.shape or true_matrix.shape[0] != true_matrix.shape[1]:
            print(f"Error: Matrix shape mismatch or not square for subject {i+1}: true={true_matrix.shape}, pred={pred_matrix.shape}")
            continue

        metrics = {
            'ID': i + 1,
            'MAE': mean_absolute_error(true_matrix.flatten(), pred_matrix.flatten()),
            'PCC': pearsonr(true_matrix.flatten(), pred_matrix.flatten())[0],
            'JSD': jensenshannon(true_matrix.flatten(), pred_matrix.flatten()),
        }

        true_metrics = calculate_centralities(true_matrix)
        pred_metrics = calculate_centralities(pred_matrix)

        for key in ['NS', 'PR', 'EC', 'BC', 'PC', 'ACC']:
            metrics[f'MAE in {key}'] = mean_absolute_error([true_metrics[key.lower()]], [pred_metrics[key.lower()]])

        results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        # Check if the file exists to decide whether to write headers
        file_exists = os.path.isfile(output_path)

        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        print(f"Results appended to {output_path}.")
    else:
        print("No data to save.")

