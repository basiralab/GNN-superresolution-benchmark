import torch
import torch.nn as nn
import numpy as np


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]])
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        # scores = torch.abs(scores)
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


class GAT(nn.Module):

    def __init__(self, in_features, out_features):
        super(GAT, self).__init__()
        # Initialize the weights, bias, and attention parameters as
        # trainable parameters
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = nn.ReLU()
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.phi.data.uniform_(-stdv, stdv)
        

    def forward(self, A, X):

        # for i in range(len(A)):
        masked_adj = A + torch.eye(A.shape[0])
        agg = A @ X @ self.weight + self.bias
        hi = agg.unsqueeze(1).expand(-1,agg.shape[0],-1)
        hj = agg.unsqueeze(0).expand(agg.shape[0],-1,-1)
        H = (torch.cat((hi,hj), dim = -1) @ self.phi).squeeze()
        s_masked = torch.where(masked_adj == 0, -float('inf'), H)
        h = nn.functional.softmax(s_masked, dim=-1) @ agg
        return self.activation(h) if self.activation else h


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks

        self.start_gat = GAT(in_dim, dim)
        self.bottom_gat = GAT(dim, dim)
        self.end_gat = GAT(2*dim, out_dim)
        self.down_gats = []
        self.up_gats = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gats.append(GAT(dim, dim))
            self.up_gats.append(GAT(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gat(A, X)
        start_gat_outs = X
        org_X = X
        for i in range(self.l_n):

            X = self.down_gats[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gat(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gats[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gat(A, X)

        return X, start_gat_outs
