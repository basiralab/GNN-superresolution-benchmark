import torch
import numpy as np
import torch.nn.functional as F 


def pad_HR_adj(label, split):
    """
    Pads a high-resolution (HR) adjacency matrix with zeros on all sides.

    Parameters:
    - label (torch.Tensor): The HR adjacency matrix to be padded.
    - split (int): The number of zeros to add to each side of the matrix.

    Returns:
    torch.Tensor: The padded HR adjacency matrix.
    """
    padded_label = F.pad(label, pad=(split, split, split, split), mode="constant", value=0)
    return padded_label


def normalize_adj_torch(mx):
    """
    Normalizes an adjacency matrix using the symmetric normalization method.

    Parameters:
    - mx (torch.Tensor): The adjacency matrix to be normalized.

    Returns:
    torch.Tensor: The normalized adjacency matrix.
    """
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def unpad(data, split):
    """
    Removes padding from a matrix.

    Parameters:
    - data (torch.Tensor): The matrix from which padding will be removed.
    - split (int): The number of rows/columns to remove from each side of the matrix.

    Returns:
    torch.Tensor: The unpadded matrix.
    """
    idx_0 = data.shape[0]-split
    idx_1 = data.shape[1]-split
    data = data[split:idx_0, split:idx_1]
    return data

def create_discrepancy(hr, zero_shift = -0.2):
    """
    Adjusts zero values in a high-resolution (HR) adjacency matrix to a specified value.

    Parameters:
    - hr (torch.Tensor): The HR adjacency matrix.
    - zero_shift (float, optional): The value to replace zero entries with. Default is -0.2.

    Returns:
    torch.Tensor: The HR adjacency matrix with adjusted zero values.
    """
    hr[hr == 0] = zero_shift
    return hr


def weight_variable_glorot(output_dim):
    """
    Initializes weights according to the Glorot uniform distribution.

    Parameters:
    - output_dim (int): The dimension of the output layer.

    Returns:
    numpy.ndarray: An array of weights initialized according to the Glorot uniform distribution.
    """
    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,
                                (input_dim, output_dim))

    return initial