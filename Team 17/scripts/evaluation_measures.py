from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

from scripts.MatrixVectorizer import MatrixVectorizer

def calculate_evaluation_measures(pred_matrices, gt_matrices):
    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []

    num_test_samples = len(pred_matrices)

    # Iterate over each test sample
    for i in range(num_test_samples):
        # Convert adjacency matrices to NetworkX graphs
        pred_graph = nx.from_numpy_array(pred_matrices[i].numpy(), edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrices[i].numpy(), edge_attr="weight")

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

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    # vectorize and flatten
    pred_1d = []
    for instance in pred_matrices:
        pred_1d.append(MatrixVectorizer.vectorize(instance))
    pred_1d = np.array(pred_1d).flatten()
        
    gt_1d = []
    for instance in gt_matrices:
        gt_1d.append(MatrixVectorizer.vectorize(instance))
    gt_1d = np.array(gt_1d).flatten()

    mae = mean_absolute_error(pred_1d, gt_1d)
    pcc = pearsonr(pred_1d, gt_1d)[0]
    js_dis = jensenshannon(pred_1d, gt_1d)

    print("MAE: ", mae)
    print("PCC: ", pcc)
    print("Jensen-Shannon Distance: ", js_dis)
    print("Average MAE betweenness centrality:", avg_mae_bc)
    print("Average MAE eigenvector centrality:", avg_mae_ec)
    print("Average MAE PageRank centrality:", avg_mae_pc)

    return mae, pcc, js_dis, avg_mae_bc, avg_mae_ec, avg_mae_pc

def plot_evaluation_measures(eval_measures_list):
    measures = [
        "MAE", "PCC", "JSD", "MAE (PC)", "MAE (EC)", "MAE (BC)"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    for i in range(len(eval_measures_list)):
        axes[i // 2][i % 2].bar(
            measures,
            [measure_value for measure_value in eval_measures_list[i]],
            color = ['red', 'green', 'purple', 'orange', 'cyan', 'lightgreen']
        )
        axes[i // 2][i % 2].set_title(f"Fold {i}")
        axes[i // 2][i % 2].set_xticklabels(measures, rotation=45, ha='right')
        axes[i // 2][i % 2].grid(axis='y', linestyle='--', alpha=0.7)

    avg_eval_measures = np.mean(eval_measures_list, axis=0)
    std_eval_measures = np.std(eval_measures_list, axis=0)
    axes[1][1].bar(
        measures,
        avg_eval_measures,
        yerr=std_eval_measures,
        capsize=5,
        color=['red', 'green', 'purple', 'orange', 'cyan', 'lightgreen']
    )
    axes[1][1].set_title('Avg. Across Folds')
    axes[1][1].set_xticklabels(measures, rotation=45, ha='right')
    axes[1][1].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    example_measures_list = [
        [0.561, 0.8245, 0.2434, 0.834, 0.0349, 0.02347],
        [0.235, 0.282, 0.3246, 0.134, 0.3134, 0.08234],
        [0.5134, 0.57, 0.072, 0.0247, 0.0248, 0.06324]
    ]
    plot_evaluation_measures(example_measures_list)
