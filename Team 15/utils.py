from MatrixVectorizer import MatrixVectorizer
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
import networkx as nx
from typing import Tuple
import matplotlib.pyplot as plt

LR_MAT_SIZE = 160
HR_MAT_SIZE = 268

def load_matrix_data(lr_train_path, hr_train_path, num_graphs=167) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads the high-resolution and low-resolution training matrices from CSV files,
    converts them to tensors, and returns them.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two torch.Tensors,
        the first being the low-resolution training data and the second being
        the high-resolution training data.
    """
    hr_train = pd.read_csv(hr_train_path)
    lr_train = pd.read_csv(lr_train_path)

    mv = MatrixVectorizer()

    lr_tensor = torch.empty(num_graphs, LR_MAT_SIZE, LR_MAT_SIZE)
    hr_tensor = torch.empty(num_graphs, HR_MAT_SIZE, HR_MAT_SIZE)

    for idx in range(len(hr_train)):
        vec_lr = lr_train.iloc[idx].to_numpy()
        vec_hr = hr_train.iloc[idx].to_numpy()

        mat_lr = mv.anti_vectorize(vec_lr, LR_MAT_SIZE)
        mat_hr = mv.anti_vectorize(vec_hr, HR_MAT_SIZE)

        lr_tensor[idx] = torch.tensor(mat_lr, dtype=torch.float32)
        hr_tensor[idx] = torch.tensor(mat_hr, dtype=torch.float32)    
    
    return lr_tensor, hr_tensor

def load_matrix_test(lr_test_path) -> torch.Tensor:
    """
    Loads the low-resolution test matrices from a CSV file, converts them to a tensor,
    and returns it.

    Returns:
        torch.Tensor: A tensor containing the low-resolution test data.
    """
    lr_test = pd.read_csv(lr_test_path)

    mv = MatrixVectorizer()

    lr_tensor = torch.empty(112, LR_MAT_SIZE, LR_MAT_SIZE)

    for idx in range(len(lr_test)):
        vec_lr = lr_test.iloc[0].to_numpy()

        mat_lr = mv.anti_vectorize(vec_lr, LR_MAT_SIZE)

        lr_tensor[idx] = torch.tensor(mat_lr, dtype=torch.float32)
    
    return lr_tensor

def calculate_metrics(gt_matrices, pred_matrices, only_mae = False, verbose = False):
    """
    Calculates various metrics between ground truth and predicted matrices, including
    mean absolute error (MAE), Pearson correlation coefficient (PCC), and Jensen-Shannon distance.
    If 'only_mae' is False, additional metrics related to network centrality measures are also computed.

    Args:
        gt_matrices (np.ndarray): Ground truth matrices.
        pred_matrices (np.ndarray): Predicted matrices.
        only_mae (bool, optional): Flag to calculate only MAE. Defaults to False.

    Returns:
        Tuple[float, ...]: A tuple containing calculated metrics. The number and types of metrics
        depend on the value of 'only_mae'.
    """
    num_test_samples = len(gt_matrices)

    mae_bc = []
    mae_ec = []
    mae_pc = []

    pred_1d_list = []
    gt_1d_list = []

    for i in range(num_test_samples):
        
        if not only_mae:

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

    if not only_mae:
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

    if verbose:
    
        print("MAE: ", mae)
        print("PCC: ", pcc)
        print("Jensen-Shannon Distance: ", js_dis)
    
    if not only_mae:
        print("Average MAE betweenness centrality:", avg_mae_bc)
        print("Average MAE eigenvector centrality:", avg_mae_ec)
        print("Average MAE PageRank centrality:", avg_mae_pc)

    if only_mae:
        return mae, pcc, js_dis

    else:
        return [mae, pcc, js_dis, avg_mae_pc, avg_mae_ec, avg_mae_bc]
    

def create_submision_compatible_csv_save(outputs, filename) -> None:
    """
    Creates and saves a submission-compatible CSV file from a list of outputs.

    This function first vectorizes the outputs using a MatrixVectorizer instance.
    Each output is converted into a vector of fixed length (35778 elements) and
    all these vectors are assembled into a matrix. This matrix is then flattened
    into a single long vector. A pandas DataFrame is created with two columns:
    'ID' and 'Predicted', where 'ID' is a sequential identifier starting from 1, 
    and 'Predicted' contains the elements of the flattened matrix. 
    This DataFrame is then saved to a CSV file with the provided filename, 
    without the index column.

    Args:
        outputs (list): A list of outputs to be vectorized and saved.
        filename (str): The name of the file to save the CSV data.

    Returns:
        None
    """
    mv = MatrixVectorizer()

    vectorised_results = np.empty((112, 35778))

    for i in range(len(outputs)):
        vectorised_results[i] = mv.vectorize(outputs[i])

    vectorised_results_melt = vectorised_results.flatten()

    df = pd.DataFrame({
        'ID': np.arange(1, len(vectorised_results_melt) + 1),
        'Predicted': vectorised_results_melt
        })

    df.to_csv(filename, index=False)

def compute_output_hr(args, test_adj, model):
    """
    Compute the high-resolution (HR) output for a set of low-resolution graphs using a specified model.
    
    The function processes each graph in the provided 'test_adj' list (assumed to be low-resolution adjacency matrices)
    using the 'model'. The model is expected to be in evaluation mode and should return a high-resolution output for 
    each input graph. This output is then cropped to remove padding (assumed to be of width 26 units on all sides) and
    clipped to ensure all values are within the range [0, 1].

    Args:
        args (object): A container for various parameters and settings. The exact contents are dependent on user requirements
                       and model design but are not directly used in this function.
        test_adj (list of numpy arrays): A list containing low-resolution adjacency matrices (graphs) to be fed into the model.
        model (torch.nn.Module): The neural network model to be used for generating high-resolution outputs from low-resolution inputs.

    Returns:
        numpy.ndarray: An array of the high-resolution outputs for each input graph, post-cropping and clipping.
    """
    outputs = []
    model.eval()

    for lr_graph in test_adj:
            output = model(lr_graph)
            #unpad and refactorize this
            idx_0 = output.shape[0]-26
            idx_1 = output.shape[1]-26
            output = output[26:idx_0, 26:idx_1]
            # append clipped outputs clipped between 0 and 1
            outputs.append(np.clip(output.detach().numpy(), 0, 1))

    outputs = np.array(outputs)

    return outputs

def plot_adjacencies(prediction, ground_truth):
    """
    Plots the adjacency matrices for a predicted graph and the ground truth graph side by side,
    along with their difference.

    The function creates a matplotlib figure with three subplots: the first for the prediction,
    the second for the ground truth, and the third for the absolute difference between the 
    prediction and the ground truth. Each subplot uses a viridis colormap and displays a 
    colorbar indicating the scale of values in the adjacency matrix.

    Parameters:
    - prediction (numpy.ndarray): The adjacency matrix of the predicted graph. It should be a 
      square matrix where each element represents the presence (1) or absence (0) of an edge 
      between two nodes, or a weighted value representing the strength of the connection.
    - ground_truth (numpy.ndarray): The adjacency matrix of the true graph. It should have the 
      same shape and format as the prediction matrix.

    Returns:
    None, but displays a matplotlib figure containing the three subplots.
    """

    fig, axs = plt.subplots(1, 3, figsize=(18, 6)) 

    # Prediction
    im = axs[0].imshow(prediction, cmap='viridis', interpolation='nearest')
    axs[0].set_title('Prediction')
    axs[0].grid(False)
    fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.04)

    # Ground Truth
    im = axs[1].imshow(ground_truth, cmap='viridis', interpolation='nearest')
    axs[1].set_title('Ground Truth')
    axs[1].grid(False)
    fig.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04)

    # Difference
    im = axs[2].imshow(np.abs(ground_truth - prediction), cmap='viridis', interpolation='nearest')
    axs[2].set_title('Difference')
    axs[2].grid(False)
    fig.colorbar(im, ax=axs[2], fraction=0.046, pad=0.04) 

    plt.show()

def plot_results(data, filename=None):
    """
    Plots the results of evaluations across different folds and metrics, 
    and optionally saves the plot to a file with updated feature of secondary y-axis for 
    specific metrics and differentiated by color-coding. It also adds a legend to clarify which bars correspond to which y-axis.

    Parameters:
    - data (numpy.ndarray): A 2D numpy array where each row contains the results from a single
      fold of cross-validation, and each column corresponds to a different evaluation metric.
    - filename (str, optional): The name of the file to save the plot to. If not specified, the 
      plot will not be saved to a file.

    Returns:
    None, but displays a matplotlib figure and optionally saves it to a file.
    """
    
    data = np.array(data)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)

    labels = ['MAE', 'PCC', 'ISD', 'MAE(PC)', 'MAE(EC)', 'MAE(BC)']
    primary_metrics = ['MAE', 'PCC', 'ISD']
    secondary_metrics = ['MAE(PC)', 'MAE(EC)', 'MAE(BC)']

    primary_color = ['green', 'green', 'green']  # Colors for the primary metrics
    secondary_color = ['blue', 'blue', 'blue']  # Colors for the secondary metrics

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.flatten()

    for i, ax in enumerate(axs[:-1]):  # Exclude the last subplot for the average
        primary_data = [data[i][labels.index(metric)] for metric in primary_metrics]
        secondary_data = [data[i][labels.index(metric)] for metric in secondary_metrics]

        primary_bars = ax.bar(primary_metrics, primary_data, color=primary_color, alpha=0.5)
        ax2 = ax.twinx()  # Create a second y-axis
        secondary_bars = ax2.bar(secondary_metrics, secondary_data, color=secondary_color, alpha=0.5)
        ax.set_title(f'Fold {i + 1}')
        ax.set_ylim(0, 0.8)  # Adjust primary y-axis limits
        ax2.set_ylim(0, 0.03)  # Adjust secondary y-axis limits
        ax.tick_params(axis='x', rotation=45)

    # Handle the averages and standard deviations in the last plot
    primary_means = [means[labels.index(metric)] for metric in primary_metrics]
    secondary_means = [means[labels.index(metric)] for metric in secondary_metrics]
    primary_stds = [stds[labels.index(metric)] for metric in primary_metrics]
    secondary_stds = [stds[labels.index(metric)] for metric in secondary_metrics]

    ax2 = axs[-1].twinx()  # Create a second y-axis for averages
    axs[-1].set_title('Avg. Across Folds')
    axs[-1].set_ylim(0, 0.8)
    ax2.set_ylim(0, 0.03)
    axs[-1].tick_params(axis='x', rotation=45)

    # Legend for color coding
    legend_primary = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=0.5)
    legend_secondary = plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=0.5)
    axs[0].legend([legend_primary, legend_secondary], ['Left Y-Axis', 'Right Y-Axis'], loc='upper right')

    plt.tight_layout()
    plt.show()

    if filename:
        fig.savefig(filename, dpi=300)