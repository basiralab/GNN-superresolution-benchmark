from .MatrixVectorizer import MatrixVectorizer

from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import torch
import networkx as nx
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


def compute_evaluation_measures(pred_matrices, gt_matrices):
    num_test_samples = pred_matrices.shape[0]

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []
    pred_1d_list = []
    gt_1d_list = []

    # Iterate over each test sample
    for i in tqdm(range(num_test_samples)):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i], edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i], edge_attr="weight")

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight")
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight")
        gt_ec = nx.eigenvector_centrality(gt_graph, weight="weight")
        gt_pc = nx.pagerank(gt_graph, weight="weight")

        # Convert centrality dictionaries to lists
        pred_bc_values = list(pred_bc.values())
        pred_ec_values = list(pred_ec.values())
        pred_pc_values = list(pred_pc.values())

        gt_bc_values = list(gt_bc.values())
        gt_ec_values = list(gt_ec.values())
        gt_pc_values = list(gt_pc.values())

        # Compute MAEs
        mae_bc.append(mean_absolute_error(pred_bc_values, gt_bc_values))
        mae_ec.append(mean_absolute_error(pred_ec_values, gt_ec_values))
        mae_pc.append(mean_absolute_error(pred_pc_values, gt_pc_values))

        # Vectorize matrices
        pred_1d_list.append(MatrixVectorizer.vectorize(pred_matrices[i]))
        gt_1d_list.append(MatrixVectorizer.vectorize(gt_matrices[i]))

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    # Concatenate flattened matrices
    pred_1d = np.concatenate(pred_1d_list)
    gt_1d = np.concatenate(gt_1d_list)

    # Compute metrics
    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    print("Average MAE betweenness centrality:", avg_mae_bc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE PageRank centrality:", avg_mae_pc)

    return [mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc]


def plot_evaluation_measures(metrics):
    '''
    Generate and save bar plots of the 3-fold CV evaluation measures.

    4 subplots are generated:
        Fold 1, 2, 3, and avg across folds

    Args:
        metrics (np.array): shape (3, 6) containing evaluation measures for each fold
    '''
    metric_names = ['MAE', 'PCC', 'JSD', 'MAE (PC)', 'MAE (EC)', 'MAE (BC)']
    color_palette = plt.colormaps['tab10'].colors

    # Create figure and subplots
    fig, axs = plt.subplots(2, 4, figsize=(10, 9))

    # Plot evaluation measures for each fold
    for i in range(3):
        row = i // 2
        col = i % 2
        axs[row, col].bar(metric_names[:3], metrics[i, :3], color=color_palette[:3])
        axs[row, col].tick_params(axis='x', rotation=45)
        axs[row, col].set_title(f'Fold {i+1}')

    # Plot average evaluation measures across folds with error bars
    avg_metrics = np.mean(metrics[:, :3], axis=0)
    std_metrics = np.std(metrics[:, :3], axis=0)
    axs[1, 1].bar(metric_names[:3], avg_metrics, yerr=std_metrics, color=color_palette[:3], capsize=5)
    axs[1, 1].tick_params(axis='x', rotation=45)
    axs[1, 1].set_title('Avg. Across Folds')

    # Plot evaluation measures for each fold
    for i in range(3):
        row = i // 2
        col = i % 2 + 2
        axs[row, col].bar(metric_names[3:], metrics[i, 3:], color=color_palette[6:])
        axs[row, col].tick_params(axis='x', rotation=45)
        axs[row, col].set_title(f'Fold {i+1}')

    # Plot average evaluation measures across folds with error bars
    avg_metrics = np.mean(metrics[:, 3:], axis=0)
    std_metrics = np.std(metrics[:, 3:], axis=0)
    axs[1, 3].bar(metric_names[3:], avg_metrics, yerr=std_metrics, color=color_palette[6:], capsize=5)
    axs[1, 3].tick_params(axis='x', rotation=45)
    axs[1, 3].set_title('Avg. Across Folds')

    # Save figure
    plt.tight_layout()
    os.makedirs('figures/', exist_ok=True)
    plt.savefig('figures/evaluation_measures.pdf', bbox_inches='tight')
    plt.savefig('figures/evaluation_measures.png', bbox_inches='tight')
    print(f'Bar plots saved to "figures/evaluation_measures.png"')
    plt.close()


def plot_validation_mae(k_fold_data):
    plt.figure(figsize=(5, 3))
    for fold, data in enumerate(k_fold_data):
        plt.plot(np.arange(1,len(data)+1)*10, data, label=f'Fold {fold+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Validation MAE')
    plt.legend()
    os.makedirs('figures/', exist_ok=True)
    plt.savefig('figures/validation_mae.pdf', bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    measures_list = []
    for i in range(3):
        # the following numbers do not reflect the provided dataset, just for an example
        num_test_samples = 20
        num_roi = 10

        # create a random model output 
        pred_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()

        # post-processing
        pred_matrices[pred_matrices < 0] = 0

        # create random ground-truth data
        gt_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()

        # you do NOT need to that since the ground-truth data we provided you is alread pre-processed.
        gt_matrices[gt_matrices < 0] = 0

        measures = compute_evaluation_measures(pred_matrices, gt_matrices)
        measures_list.append(measures)

    measures_array = np.array(measures_list)
    plot_evaluation_measures(measures_array)
