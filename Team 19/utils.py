import torch
import numpy as np

def weight_variable_glorot(output_dim):
    """
    Initialize weights using Xavier initialization.

    Args:
        output_dim (int): The number of output dimensions for the weight matrix.

    Returns:
        initial (numpy.ndarray): Randomly initialized square weight matrix.
    """

    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,
                                (input_dim, output_dim))

    return initial

def convert_adj_to_edge_index(adj):
    """
    Convert adjacency matrix to edge index representation.

    Args:
        adj (torch.Tensor): Adjacency matrix.

    Returns:
        edge_index (torch.Tensor): Edge index representation of the graph.
    """
    # Assuming adj is a square matrix
    # Find non-zero elements in the adjacency matrix and 
    # transpose the result for edge index representation
    edge_index = torch.nonzero(adj, as_tuple=False).t().contiguous()
    return edge_index

def pad_HR_adj(label, split):
    """
    Pad the HR matrix with zero padding.

    Args:
        label (numpy.ndarray): The HR matrix.
        split (int): The amount of padding to add to each edge of the matrix.

    Returns:
        label (numpy.ndarray): The padded HR adjacency matrix.
    """

    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    # Set diagonal elements to 1
    np.fill_diagonal(label, 1)
    return label

def unpad(data, split):
    """
    Unpad the padded matrix.

    Args:
        data (numpy.ndarray): The padded matrix .
        split (int): The amount of padding to be removed from each edge of the array.

    Returns:
        train (numpy.ndarray): The unpadded matrix.
    """

    idx_0 = data.shape[0]-split
    idx_1 = data.shape[1]-split
    train = data[split:idx_0, split:idx_1]
    return train

def normalize_adj_torch(mx):
    """
    Normalize the input adjacency matrix using Kipf normalization.
    
    Args:
        mx (torch.Tensor): The input adjacency matrix.

    Returns:
        mx (torch.Tensor): The normalized adjacency matrix.
    """
    rowsum = mx.sum(1) # calculate the row sums
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0. # replace any infinite values with 0
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)

    # Perform Kipf normalization
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def gaussian_noise_layer(input_layer, args):
    """
    Apply Gaussian noise to the input layer.

    Args:
        input_layer (torch.Tensor): Input tensor to which Gaussian noise will be added.
        args (object): Arguments object containing hyperparameters.

    Returns:
        z (torch.Tensor): Output tensor with Gaussian noise.
    """
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z