"""
Modified code based on AGSR-Net (https://github.com/basiralab/AGSR-Net) by Basira Labs.
Original licensing terms apply.
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from utils import *


class GraphUnpool(nn.Module):

    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]], dtype=X.dtype).to(X.device)
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


class GraphUnet(nn.Module):

    def __init__(self, ks, in_dim, out_dim, dim=320, device='cpu'):
        super(GraphUnet, self).__init__()
        self.ks = ks

        self.start_gcn = GCN(in_dim, dim)
        self.bottom_gcn = GCN(dim, dim)
        self.end_gcn = GCN(2*dim, out_dim)
        self.down_gcns = []
        self.up_gcns = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim).to(device))
            self.up_gcns.append(GCN(dim, dim).to(device))
            self.pools.append(GraphPool(ks[i], dim).to(device))
            self.unpools.append(GraphUnpool().to(device))

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        pool_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):

            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
            pool_outs.append(X)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)

        pool_out = pool_outs[0]
        pool_out = torch.mm(pool_out, pool_out.t())
        pool_out = pool_out.fill_diagonal_(1)
        
        return X, start_gcn_outs, pool_out


class GSRLayer(nn.Module):
    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()
        
        self.weights = torch.nn.Parameter(torch.FloatTensor(hr_dim, hr_dim))
        torch.nn.init.xavier_uniform_(self.weights)

    def forward(self,A,X):
        lr = A
        lr_dim = lr.shape[0]
        f = X
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')
        # U_lr = torch.abs(U_lr)
        eye_mat = torch.eye(lr_dim).type(torch.FloatTensor).to(X.device)
        s_d = torch.cat((eye_mat,eye_mat),0)[:self.weights.shape[0], :]
        
        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a ,torch.t(U_lr))
        f_d = torch.matmul(b ,f)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)
        adj = self.f_d

        X = torch.mm(adj, adj.t())
        X = (X + X.t())/2
        X = X.fill_diagonal_(1)

        return adj, torch.abs(X)


class GCN(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GCN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = torch.nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, adj, input):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(adj, input)
        output = torch.mm(support, self.weight) + self.bias
        output = self.act(output)
        return output


class GraphLoong(nn.Module):

    def __init__(self, args):
        super(GraphLoong, self).__init__()
        
        self.lr_dim = args['lr_dim']
        self.hr_dim = args['hr_dim']
        self.hidden_dim = args['hidden_dim']
        self.padding = args['padding']
        self.layer = GSRLayer(self.hr_dim)
        self.net1 = GraphUnet(args['ks'], self.lr_dim, self.hr_dim, device=args['device'])
        self.net2 = GraphUnet(args['kss'], self.lr_dim, self.lr_dim, device=args['device'])
        self.gc1 = GCN(2*self.hr_dim, self.hidden_dim, args['dropout'], act=F.relu)
        self.gc2 = GCN(self.hidden_dim, self.hr_dim, args['dropout'], act=F.relu)

    def forward(self, lr):

        I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(lr.device)
        A = normalize_adj_torch(lr).type(torch.FloatTensor).to(lr.device)

        self.net_outs, self.start_gcn_outs, _ = self.net1(A, I)
        self.layer_A, self.layer_X = self.layer(A, self.net_outs)

        self.net_2_outs, _, self.pool_out = self.net2(A, lr)
        s = self.padding
        self.tile = torch.zeros([self.hr_dim, self.hr_dim], dtype=torch.float32).to(lr.device)
        self.tile[s:self.hr_dim-s, s:self.hr_dim-s] = self.pool_out.tile((2,2))
        self.tile = self.tile.fill_diagonal_(1)

        self.X = torch.concat([self.layer_X, self.tile], 1)
        
        self.hidden1 = self.gc1(self.layer_A, self.X)
        self.hidden2 = self.gc2(self.layer_A, self.hidden1)

        z = self.hidden2
        z = (z + z.t())/2
        z = z.fill_diagonal_(1)
        
        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.net_2_outs


class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args['mean_dense'], std=args['std_dense'])

    def forward(self, x):
        out = torch.mm(x, self.weights)
        return out

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args['hr_dim']-2*args['padding'], args['hr_dim']-2*args['padding'], args)
        self.relu_1 = nn.ReLU(inplace=False)
        self.dense_2 = Dense(args['hr_dim']-2*args['padding'], args['hr_dim']-2*args['padding'], args)
        self.relu_2 = nn.ReLU(inplace=False)
        self.dense_3 = Dense(args['hr_dim']-2*args['padding'], 1, args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = dc_den2
        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)
        return torch.abs(output)

def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args['mean_gaussian'], std=args['std_gaussian'])
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z
