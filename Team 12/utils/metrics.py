from utils.preprocessing import *
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx


def get_metrics(pred_arrays, gt_arrays):
    # Initialize lists to store MAEs for each centrality measure
    mae_bc = []
    mae_ec = []
    mae_pc = []

    # Post-processing
    pred_arrays[pred_arrays < 0] = 0

    # Iterate over each test sample
    for i in range(len(pred_arrays)):
        # Convert adjacency matrices to NetworkX graphs
        pred_matrix = anti_vectorize(pred_arrays[i], 268, include_diagonal=False)
        gt_matrix = anti_vectorize(gt_arrays[i], 268, include_diagonal=False)
        pred_graph = nx.from_numpy_array(pred_matrix, edge_attr="weight")
        gt_graph = nx.from_numpy_array(gt_matrix, edge_attr="weight")

        # Compute centrality measures
        pred_bc = nx.betweenness_centrality(pred_graph, weight="weight", k=5)
        pred_ec = nx.eigenvector_centrality(pred_graph, weight="weight")
        pred_pc = nx.pagerank(pred_graph, weight="weight")

        gt_bc = nx.betweenness_centrality(gt_graph, weight="weight", k=5)
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

        print("Get metrics:", i+1, '/', len(pred_arrays), end='\r')

    # Compute average MAEs
    avg_mae_bc = sum(mae_bc) / len(mae_bc)
    avg_mae_ec = sum(mae_ec) / len(mae_ec)
    avg_mae_pc = sum(mae_pc) / len(mae_pc)

    pred_1d = pred_arrays.flatten()
    gt_1d = gt_arrays.flatten()

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