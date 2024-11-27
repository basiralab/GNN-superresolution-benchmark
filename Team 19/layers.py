import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GINConv

from utils import *

"""
Define layers:
    GraphConvolution
    GIN
    GAT
    GCN
    GraphUnpool
    GraphPool
    Dense
"""

class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Network (GCN) layer.
    """

    def __init__(self, in_features, out_features, dropout, act=F.relu):
        """
        Initialize the GraphConvolution layer.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            dropout (float): Dropout probability.
            act (function, optional): Activation function, default is ReLU.
        """
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        """
        Resets the parameters using Xavier uniform initialization.
        """
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        """
        Forward pass of the GraphConvolution layer.

        Args:
            input (torch.Tensor): Input matrix.
            adj (torch.Tensor): Adjacency matrix.

        Returns:
            output (torch.Tensor): Aggregation output.
        """
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output

class GIN(nn.Module):
    """
    Graph Isomorphism Network (GIN) module.
    """

    def __init__(self, in_dim, out_dim):
        """
        Initialize the GIN module.

        Args:
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
        """
        super(GIN, self).__init__()
        self.gin_conv = GINConv(nn.Linear(in_dim, out_dim))
        self.act = nn.ReLU()

    def forward(self, A, X):
        """
        Forward pass of the GIN module.

        Args:
            A (torch.Tensor): The adjacency matrix.
            X (torch.Tensor): The input matrix.

        Returns:
            X (torch.Tensor): The output matrix after GIN convolution.
        """
        edge_index = convert_adj_to_edge_index(A)
        X = self.gin_conv(X, edge_index)
        X = self.act(X)

        return X

class GAT(nn.Module):
    """
    Graph Attention Network (GAT) module.
    """

    def __init__(self, in_dim, out_dim):
        """
        Initialize the GIN module.

        Args:
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
        """
        super(GAT, self).__init__()
        self.gat_conv = GATConv(in_dim, out_dim)
        self.act = nn.ReLU()

    def forward(self, A, X):
        """
        Forward pass of the GIN module.

        Args:
            A (torch.Tensor): The adjacency matrix.
            X (torch.Tensor): The input matrix.

        Returns:
            X (torch.Tensor): The output matrix after GIN convolution.
        """
        edge_index = convert_adj_to_edge_index(A)
        X = self.gat_conv(X, edge_index)
        X = self.act(X)
        # Add dropout
        X = F.dropout(X, p=0.6, training=self.training)

        return X

class GCN(nn.Module):
    """
    Graph Convolution Network (GCN) module.
    """
    def __init__(self, in_dim, out_dim):
        """
        Initialize the GCN module.

        Args:
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
        """
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0.6)
        self.act = nn.ReLU()

    def forward(self, A, X):
        """
        Forward pass of the GCN module.

        Args:
            A (torch.Tensor): The adjacency matrix.
            X (torch.Tensor): The input matrix.

        Returns:
            X (torch.Tensor): The output matrix after GCN convolution.
        """

        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        X = self.act(X) # activation
        return X

class GraphUnpool(nn.Module):
    """
    Graph Unpooling layer module.
    """

    def __init__(self):
        """
        Initialize the GraphUnpool layer.
        """
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        """
        Forward pass of the GraphUnpool layer.

        Args:
            A (torch.Tensor): The adjacency matrix.
            X (torch.Tensor): The node embedding matrix.
            idx (torch.Tensor): Nodes to be unpolled.

        Returns:
            A (torch.Tensor): The unpolled adjacency matrix.
            new_X (torch.Tensor): The unpolled node embedding matrix.
        """
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X

class GraphPool(nn.Module):
    """
    Graph Pooling layer module.
    """

    def __init__(self, k, in_dim):
        """
        Initialize the GraphPool layer.

        Args:
            k (float): The ratio of nodes to keep after pooling.
            in_dim (int): Dimensionality of input features.
        """
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        """
        Forward pass of the GraphPool layer.

        Args:
            A (torch.Tensor): The adjacency matrix.
            X (torch.Tensor): The node embedding matrix.

        Returns:
            A (torch.Tensor): The pooled adjacency matrix.
            new_X (torch.Tensor): The pooled node embedding matrix.
            idx (torch.Tensor): Indices of the nodes selected during pooling.
        """
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx

class Dense(nn.Module):
    """
    Dense layer module.
    """
    def __init__(self, n1, n2, args):
        """
        Initialize the Dense layer module.

        Args:
            n1 (int): Number of input features.
            n2 (int): Number of output features.
            args (object): Arguments object containing hyperparameters.
        """
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        """
        Forward pass of the Dense (fully-connected) layer module.

        Args:
            x (torch.Tensor): Input feature matrix.

        Returns:
            out (torch.Tensor): Output feature matrix after dense linear transformation.
        """
        out = torch.mm(x, self.weights) # linear transformation
        return out