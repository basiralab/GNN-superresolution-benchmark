from MatrixVectorizer import MatrixVectorizer
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon

def calculate_measures(num_test_samples, num_roi, pred_matrices, gt_matrices):
    """
    Calculate and evaluate various evaluation meatures.

    Args:
        num_test_samples (int): Number of samples in the test set.
        num_roi (int): Number of regions of interest in the adjacency matrices.
        pred_matrices (numpy.ndarray): Array containing predicted adjacency matrices for each test sample.
        gt_matrices (numpy.ndarray): Array containing ground truth adjacency matrices for each test sample.

    Returns:
    - measures (list): List containing computed performance measures, including MAE, PCC, Jensen-Shannon Distance,
                      and average MAE for betweenness centrality, eigenvector centrality, and PageRank centrality.
    """

    # post-processing
    pred_matrices[pred_matrices < 0] = 0
    gt_matrices[gt_matrices < 0] = 0

    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d_list = []
    gt_1d_list = []

    # Iterate over each test sample
    for i in range(num_test_samples):
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
    print("Average MAE PageRank centrality:", avg_mae_pc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE betweenness centrality:", avg_mae_bc)


    return [mae, pcc, js_dis, avg_mae_pc, avg_mae_ec, avg_mae_bc]


def plot_measures(measures = None, fold_index = None, avg = False, mean_values = None, std_dev_values = None):
    """
    Plot and visualize evaluation measures(MAE', 'PCC', 'JSD', 'MAE(EC)', 'MAE(BC)') across folds or for a specific fold.

    Args:
        measures (list): List of numerical values representing evaluation measures for each category.
        fold_index (int or None): Index of the fold if plotting for a specific fold, None for plotting across all folds.
        avg (bool): Flag indicating whether to plot averages across folds or for a specific fold.
        mean_values (list or None): List of mean values for each category across folds.
        std_dev_values (list or None): List of standard deviation values for each category across folds.

    Returns:
        None: The function generates and saves the plot based on the specified parameters.
    """
    # Initialization
    categories = ['MAE', 'PCC', 'JSD', 'MAE(EC)', 'MAE(BC)']
    colors = ['blue', 'green', 'red', 'purple', 'pink']

    if avg:
        # Plot across all folds
        plt.figure(4)
        mean_values = np.delete(mean_values, 3)
        std_dev_values = np.delete(std_dev_values, 3)
        plt.bar(range(len(mean_values)), mean_values, yerr=std_dev_values, capsize=5, color=colors)
        plt.xticks(range(len(mean_values)), categories, rotation=45)
        plt.title('Avg. Across Folds')
        plt.tight_layout()
         # Save plot
        plt.savefig('Avg_Across_Folds_measures.png', dpi=300)
        plt.close()
    else:
        # Plot for each fold
        plt.figure(fold_index)
        fold_measures = measures[:3] + measures[4:]
        plt.bar(categories, fold_measures, color=colors)
        plt.title(f'Fold {fold_index}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        # Save plot
        plt.savefig(f'Fold_{fold_index}_measures.png', dpi=300)
        plt.close()

def plot_mae_pc(measures, mean_value = None, std_dev_value = None):
    """
    Plot and visualize 'MAE(PC)' across folds and for a specific fold.

    Args:
        measures (list): List of 'MAE(PC)' scores.
        mean_value (float): Mean value for MAE(PC).
        std_dev_value (float): Standard deviation value for 'MAE(PC)'.

    Returns:
        None: The function generates and saves the plot based on the specified parameters.
    """
    categories = ['Fold1', 'Fold2', 'Fold3', 'AcrossFolds']
    bar_positions = np.arange(4)
    plt.figure(5)
    plt.bar(bar_positions[:3], measures , color='orange')
    plt.bar(bar_positions[3], mean_value, yerr=std_dev_value, color='orange', capsize=5)
    plt.title('MAE(PC) Scores')
    plt.xticks(bar_positions, categories, rotation=45)
    plt.tight_layout()
    plt.savefig(f'MAE(PC)_Scores.png', dpi=300)
    plt.close()