import torch
import torch.nn as nn
import torch.nn.functional as F
from initializations import *

class GSRLayer(nn.Module):
    
    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):
            lr = A.to(X.device)
            lr_dim = lr.shape[0]
            f = X.to(lr.device) 
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor).to(lr.device)
            s_d = torch.cat((eye_mat, eye_mat), 0)

            a = torch.matmul(self.weights, s_d).to(lr.device)
            b = torch.matmul(a, torch.t(U_lr)).to(lr.device)
            f_d = torch.matmul(b, f).to(lr.device)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t()).to(lr.device)
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
 
        if torch.cuda.is_available():
            self.weight = torch.nn.Parameter(
                torch.cuda.FloatTensor(in_features, out_features))
        else:
            self.weight = torch.nn.Parameter(
                torch.FloatTensor(in_features, out_features)) 

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output



class GraphUnpool(nn.Module):

    def __init__(self, device):
        super(GraphUnpool, self).__init__()
        self.device = device

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]], device=self.device)
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):

    def __init__(self, k, in_dim, device):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
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


class GCN(nn.Module):

    def __init__(self, in_dim, out_dim, device):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim).to(device)
        self.drop = nn.Dropout(p=0)

    def forward(self, A, X):

        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, device, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.device = device

        self.start_gcn = GCN(in_dim, dim, device)
        self.bottom_gcn = GCN(dim, dim, device)
        self.end_gcn = GCN(2*dim, out_dim, device)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, device))
            self.up_gcns.append(GCN(dim, dim, device))
            self.pools.append(GraphPool(ks[i], dim, device))
            self.unpools.append(GraphUnpool(device))

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):

            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)

        return X, start_gcn_outs
