from MatrixVectorizer import MatrixVectorizer
import numpy as np
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt


def kfold_evaluation_measure(preds, trues, k=3, file=''):
    """Function to evaluate the model trained using k-fold cross-validation"""
    performance_list = []

    for i in range(k):
        pred_matrices = preds[i]
        gt_matrices = trues[i]

        # Initialize lists to store MAEs for each centrality measure
        mae_bc = []
        mae_ec = []
        mae_pc = []
        pred_1d_list = []
        gt_1d_list = []

        # Iterate over each test sample
        for i in tqdm(range(len(pred_matrices))):
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

        performance_list.append([mae, pcc, js_dis, avg_mae_pc, avg_mae_ec, avg_mae_bc])
    
    # Visualize the performance
    performance = np.array(performance_list)
    performance_metric = ["MAE", "PCC", "JSD", "MAE (PC)", "MAE (EC)", "MAE (BC)"]
    color = ['#FF6666', '#66B266', '#6666FF', '#FFC966', '#66FFFF', '#66FF66']

    plt.figure(figsize=(13, 8))

    plt.subplot(2, 2, 1)
    plt.bar(performance_metric, performance[0], color=color)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Fold 1")

    plt.subplot(2, 2, 2)
    plt.bar(performance_metric, performance[1], color=color)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Fold 2")

    plt.subplot(2, 2, 3)
    plt.bar(performance_metric, performance[2], color=color)
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Fold 3")

    plt.subplot(2, 2, 4)
    plt.bar(performance_metric, np.mean(performance, axis=0), color=color)
    plt.errorbar(performance_metric, np.mean(performance, axis=0), yerr=np.std(performance, axis=0),
                 ls='none', capsize=10, capthick=2, elinewidth=2, ecolor='black')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    plt.title("Avg. Across Folds")

    plt.tight_layout()
    plt.savefig(f"./images/performance{file}.png")
    plt.close()