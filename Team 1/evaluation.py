import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
import community as community_louvain
import os

def calculate_centralities(adj_matrix):
    """
    Calculate various centrality measures and clustering coefficients for a graph.

    Parameters:
        adj_matrix (numpy.ndarray): Adjacency matrix of the graph.

    Returns:
        dict: Average centrality measures and clustering coefficients.
    """
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: shape={adj_matrix.shape}")

    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G)

    # Calculate participation coefficient
    pc_dict = participation_coefficient(G, partition)

    # Calculate centrality measures
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
    """
    Calculate the participation coefficient for each node in the graph.

    Parameters:
        G (networkx.Graph): Input graph.
        partition (dict): Community partition of the graph.

    Returns:
        dict: Participation coefficient for each node.
    """
    pc_dict = {}
    for node in G.nodes():
        node_degree = G.degree(node)
        if node_degree == 0:
            pc_dict[node] = 0.0
        else:
            within_module_degree = sum(
                1 for neighbor in G[node] if partition[neighbor] == partition[node]
            )
            pc_dict[node] = 1 - (within_module_degree / node_degree) ** 2

    return pc_dict

def evaluate_all(true_hr_matrices, predicted_hr_matrices, output_path='ID-randomCV.csv'):
    """
    Evaluate prediction performance across multiple subjects.

    Parameters:
        true_hr_matrices (numpy.ndarray): True high-resolution matrices (subjects x nodes x nodes).
        predicted_hr_matrices (numpy.ndarray): Predicted high-resolution matrices (subjects x nodes x nodes).
        output_path (str): File path for saving the evaluation results.
    """
    print(true_hr_matrices.shape)
    print(predicted_hr_matrices.shape)

    num_subjects = true_hr_matrices.shape[0]
    results = []

    for i in range(num_subjects):
        true_matrix = true_hr_matrices[i, :, :]
        pred_matrix = predicted_hr_matrices[i, :, :]

        if i % 25 == 0:
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
            metrics[f'MAE in {key}'] = mean_absolute_error(
                [true_metrics[key.lower()]], [pred_metrics[key.lower()]]
            )

        results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        file_exists = os.path.isfile(output_path)
        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        print(f"Results appended to {output_path}.")
    else:
        print("No data to save.")
