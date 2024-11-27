import torch
import torch.nn as nn
import torch.nn.functional as F

from layers import *
from utils import normalize_adj_torch

"""
Define core blocks: 
    GraphUnet
    GSRLayer
    IWASAGSRNet
    Discriminator
"""

class GraphUnet(nn.Module):
    """
    Graph U-Net module.
    """
    def __init__(self, ks, in_dim, out_dim, dim=320):
        """
        Initialize the GraphUnet module.

        Args:
            ks (list): List of pooling ratios for each down-sampling step.
            in_dim (int): Dimensionality of input features.
            out_dim (int): Dimensionality of output features.
            dim (int): Dimensionality of hidden features (default is 320).
        """
        super(GraphUnet, self).__init__()
        self.ks = ks
        
        # Using GAT
        self.start_gat = GAT(in_dim, dim)
        self.bottom_gat = GAT(dim, dim)
        self.end_gat = GAT(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim))
            self.up_gcns.append(GCN(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        """
        Forward pass of the GraphUnet module.

        Args:
            A (torch.Tensor): The adjacency matrix of the input graph.
            X (torch.Tensor): The input node embedding matrix.

        Returns:
            X (torch.Tensor): The output node embedding matrix.
            start_gat_outs (torch.Tensor): The start GAT layer output.
        """
        adj_ms = [] # list of adjacency matrices at each down-sampling step
        indices_list = [] # list of indices for unpooling
        down_outs = [] # list of node embedding matrices at each down-sampling step

        # Initial GAT convolution
        X = self.start_gat(A, X)
        start_gat_outs = X
        org_X = X # original node embedding matrix

        # Build stacked U-Net
        for _ in range(2):
            for i in range(self.l_n):
                X = self.down_gcns[i](A, X) # down-sampling with GCN
                adj_ms.append(A)
                down_outs.append(X)
                A, X, idx = self.pools[i](A, X) # pooling
                indices_list.append(idx)
            
            X = self.bottom_gat(A, X)

            for i in range(self.l_n):
                up_idx = self.l_n - i - 1 # index for unpooling
                A, idx = adj_ms[up_idx], indices_list[up_idx]
                A, X = self.unpools[i](A, X, idx) # unpooling
                X = self.up_gcns[i](A, X) # up-sampling with GCN
                X = X.add(down_outs[up_idx])

            X = torch.cat([X, org_X], 1)
            # Final GAT convolution
            X = self.end_gat(A, X)

        return X, start_gat_outs

class GSRLayer(nn.Module):
    """
    Graph Super Resolution module.
    """

    def __init__(self, hr_dim):
        """
        Initialize the GSRLayer.

        Args:
            hr_dim (int): Dimension of the high-resolution (HR) matrix.
        """
        super(GSRLayer, self).__init__()

        # Initialize weight matrix
        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        """
        Forward pass of the GSRLayer.

        Args:
            A (torch.Tensor): Adjacency matrix of the low-resolution (LR) graph.
            X (torch.Tensor): The input node embedding matrix.

        Returns:
            adj (torch.Tensor): The super-resolved graph structure.
            torch.abs(X) (torch.Tensor): The super-resolved graph node features.
        """
        with torch.autograd.set_detect_anomaly(True):

            # Extract LR adjacency matrix and dimension
            lr = A
            lr_dim = lr.shape[0]

            # Extract node embedding matrix
            f = X

            # Eigenvector decomposition of the LR adjacency matrix
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            # Generate S_d (the transposed concatenation of 2 identity matrices)
            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
            s_d = torch.cat((eye_mat, eye_mat), 0)
            
            # Super-resolving the graph structure
            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, f)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            # Super-resolving the graph node features
            X = torch.mm(adj, adj.t())
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)

        return adj, torch.abs(X)

class IWASAGSRNet(nn.Module):
    """
    Graph Super-Resolution Network (IWAS-AGSRNet) module.
    """

    def __init__(self, args):
        """
        Initialize the IWAS-AGSRNet module.

        Args:
            args (object): Arguments of input hyperparameters.
        """
        super(IWASAGSRNet, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim

        # Initialize graph layers
        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(args.ks, self.lr_dim, self.hr_dim)
        self.gc1 = GraphConvolution(
            self.hr_dim, self.hidden_dim, args.dropout, act=F.relu)
        self.gc2 = GraphConvolution(
            self.hidden_dim, self.hr_dim, args.dropout, act=F.relu)
        self.gin = GIN(self.hr_dim, self.hr_dim)

    def forward(self, lr, lr_dim):
        """
        Forward pass of the IWASAGSRNet module.

        Args:
            lr (torch.Tensor): LR adjacency matrix.
            lr_dim (int): Dimensionality of the LR input features.

        Returns:
            z (torch.Tensor): Generated HR adjacency matrix.
            unet_outs (torch.Tensor): The output node embedding matrix from the stacked U-Net.
            start_gat_outs (torch.Tensor): The output node embedding matrix from the start GAT layer from the stacked U-Net.
        """
        with torch.autograd.set_detect_anomaly(True):
            # Initialize the node embedding matrix to be identity matrix
            I = torch.eye(lr_dim).type(torch.FloatTensor)
            # Normalize LR adjacency matrix
            A = normalize_adj_torch(lr).type(torch.FloatTensor)

            # Pass LR graph through stacked U-Net
            unet_outs, start_gat_outs = self.net(A, I)
            
            # Generate HR graph using GSRLayer
            outputs, Z = self.layer(A, unet_outs)

            # Perform GCN and GIN to aggregate information
            hidden1 = self.gc1(Z, outputs)
            hidden2 = self.gc2(hidden1, outputs)
            z = self.gin(hidden2, outputs)

            # Symmetrize the output adjacency matrix
            z = (z + z.t())/2
            z = z.fill_diagonal_(1)

            # Post-process the output to be non-negative values
            z = torch.abs(z)

        return z, unet_outs, start_gat_outs

class Discriminator(nn.Module):
    """
    Discriminator module.
    """
    def __init__(self, args):
        """
        Initialize the Discriminator module.

        Args:
            args (object): Arguments object containing hyperparameters.
        """
        super(Discriminator, self).__init__()

        # Initialize dense layers and activation functions
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.LeakyReLU(0.1, inplace=False)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.LeakyReLU(0.1, inplace=False)
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        """
        Forward pass of the Discriminator module.

        Args:
            inputs (torch.Tensor): Input feature matrix (generated HR matrix).

        Returns:
            output (torch.Tensor): Probability scores indicating real or fake samples.
        """
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = self.sigmoid(self.dense_3(dc_den2))
        output = torch.abs(output)
        return output