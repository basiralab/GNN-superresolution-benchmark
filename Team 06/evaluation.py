import pandas as pd
import numpy as np
import networkx as nx
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error
import community.community_louvain as community_louvain
import os
from constants import *
from MatrixVectorizer import MatrixVectorizer
import torch
from gan.preprocessing import preprocess_data

def load_random_files(args, return_matrix=False, include_diagonal=False):
    lr_train1_data = np.genfromtxt("data/Cluster-CV/Fold1/lr_clusterA.csv", delimiter=",", skip_header=1)
    hr_train1_data = np.genfromtxt("data/Cluster-CV/Fold1/hr_clusterA.csv", delimiter=",", skip_header=1)
    lr_train2_data = np.genfromtxt("data/Cluster-CV/Fold2/lr_clusterB.csv", delimiter=",", skip_header=1)
    hr_train2_data = np.genfromtxt("data/Cluster-CV/Fold2/hr_clusterB.csv", delimiter=",", skip_header=1)
    lr_train3_data = np.genfromtxt("data/Cluster-CV/Fold3/lr_clusterC.csv", delimiter=",", skip_header=1)
    hr_train3_data = np.genfromtxt("data/Cluster-CV/Fold3/hr_clusterC.csv", delimiter=",", skip_header=1)

    # Pre-processing of the values
    np.nan_to_num(lr_train1_data, copy=False)
    np.nan_to_num(hr_train1_data, copy=False)
    np.nan_to_num(lr_train2_data, copy=False)
    np.nan_to_num(hr_train2_data, copy=False)
    np.nan_to_num(lr_train3_data, copy=False)
    np.nan_to_num(hr_train3_data, copy=False)

    lr_train1_data = np.maximum(lr_train1_data, 0)
    hr_train1_data = np.maximum(hr_train1_data, 0)
    lr_train2_data = np.maximum(lr_train2_data, 0)
    hr_train2_data = np.maximum(hr_train2_data, 0)
    lr_train3_data = np.maximum(lr_train3_data, 0)
    hr_train3_data = np.maximum(hr_train3_data, 0)

    # Split the last 20 samples from each split
    last_20_lr_train2 = lr_train2_data[-20:]
    last_20_hr_train2 = hr_train2_data[-20:]
    last_20_lr_train3 = lr_train3_data[-20:]
    last_20_hr_train3 = hr_train3_data[-20:]

    # Concatenate the last 20 samples to the other splits
    lr_train1_data = np.concatenate((lr_train1_data, last_20_lr_train2, last_20_lr_train3), axis=0)
    hr_train1_data = np.concatenate((hr_train1_data, last_20_hr_train2, last_20_hr_train3), axis=0)
    lr_train2_data = np.concatenate((lr_train2_data, last_20_lr_train1, last_20_lr_train3), axis=0)
    hr_train2_data = np.concatenate((hr_train2_data, last_20_hr_train1, last_20_hr_train3), axis=0)
    lr_train3_data = np.concatenate((lr_train3_data, last_20_lr_train1, last_20_lr_train2), axis=0)
    hr_train3_data = np.concatenate((hr_train3_data, last_20_hr_train1, last_20_hr_train2), axis=0)

    if return_matrix:
        # Apply anti-vectorization for each split
        def create_matrices(lr_data, hr_data):
            lr_matrices = np.empty((lr_data.shape[0], LR_MATRIX_SIZE, LR_MATRIX_SIZE))
            hr_matrices = np.empty((hr_data.shape[0], HR_MATRIX_SIZE, HR_MATRIX_SIZE))
            for i, sample in enumerate(lr_data):
                lr_matrices[i] = MatrixVectorizer.anti_vectorize(sample, LR_MATRIX_SIZE, include_diagonal)
            for i, sample in enumerate(hr_data):
                hr_matrices[i] = MatrixVectorizer.anti_vectorize(sample, HR_MATRIX_SIZE, include_diagonal)
            return lr_matrices, hr_matrices

        lr_train1_matrices, hr_train1_matrices = create_matrices(lr_train1_data, hr_train1_data)
        lr_train2_matrices, hr_train2_matrices = create_matrices(lr_train2_data, hr_train2_data)
        lr_train3_matrices, hr_train3_matrices = create_matrices(lr_train3_data, hr_train3_data)

        A, X = preprocess_data(lr_train1_matrices, args)
        lr_train1_matrices = torch.stack([A, X], dim=1)
        A, X = preprocess_data(lr_train2_matrices, args)
        lr_train2_matrices = torch.stack([A, X], dim=1)
        A, X = preprocess_data(lr_train3_matrices, args)
        lr_train3_matrices = torch.stack([A, X], dim=1)

        return (
            [
                [torch.cat((lr_train2_matrices, lr_train3_matrices), axis=0), np.concatenate((hr_train2_matrices, hr_train3_matrices), axis=0), lr_train1_matrices, hr_train1_matrices],
                [torch.cat((lr_train1_matrices, lr_train3_matrices), axis=0), np.concatenate((hr_train1_matrices, hr_train3_matrices), axis=0), lr_train2_matrices, hr_train2_matrices],
                [torch.cat((lr_train1_matrices, lr_train2_matrices), axis=0), np.concatenate((hr_train1_matrices, hr_train2_matrices), axis=0), lr_train3_matrices, hr_train3_matrices]
            ]
        )

    return (
        [
            [np.concatenate((lr_train2_data, lr_train3_data), axis=0), np.concatenate((hr_train2_data, hr_train3_data), axis=0), lr_train1_data, hr_train1_data],
            [np.concatenate((lr_train1_data, lr_train3_data), axis=0), np.concatenate((hr_train1_data, hr_train3_data), axis=0), lr_train2_data, hr_train2_data],
            [np.concatenate((lr_train1_data, lr_train2_data), axis=0), np.concatenate((hr_train1_data, hr_train2_data), axis=0), lr_train3_data, hr_train3_data]
        ]
    )

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


def evaluate_all(true_hr_matrices, predicted_hr_matrices, output_path='clusterCV.csv'):
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
