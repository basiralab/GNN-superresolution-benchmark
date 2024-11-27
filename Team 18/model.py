from utils.reproducibility import device
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from preprocessing import normalize_adj_torch


class GraphUnpool(nn.Module):
    '''
    Graph unpooling layer to upsample the number of nodes.
    '''
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]]).to(device)
        new_X[idx] = X
        return A, new_X


class GraphPool(nn.Module):
    '''
    Graph pooling layer to downsample the number of nodes.
    '''
    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)

    def forward(self, A, X):
        scores = self.proj(X).squeeze()
        scores = F.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        values = torch.unsqueeze(values, -1)
        new_X = X[idx, :]
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class DropGCN(nn.Module):
    '''
    DropGNN layer referenced from https://arxiv.org/abs/2111.06283
    '''
    def __init__(self, in_dim, out_dim, act=F.relu):
        super(DropGCN, self).__init__()
        self.drop_p = 0.3  # node drop probability
        self.drop_r = 5  # number of runs for averaging
        self.act = act
        self.fc = nn.Linear(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, A, X):
        n = A.shape[0]
        A_r = torch.zeros([self.drop_r, n, n]).to(device)
        # drop nodes over multiple runs
        for i in range(self.drop_r):
            drop_indices = np.random.choice(np.arange(n), size=int(self.drop_p*n), replace=False)
            A_tmp = A.clone()
            A_tmp[drop_indices, :] = 0
            A_tmp[:, drop_indices] = 0
            A_r[i] = A_tmp
        out = torch.matmul(A_r, X)
        # aggregate over runs by averaging
        out = out.mean(0)
        out = self.fc(out)
        out = self.act(out)
        return out


class GraphUnet(nn.Module):
    '''
    U-net to learn the node embeddings for low resolution graph.
    '''
    def __init__(self, ks, in_dim, out_dim, h_dim):
        super(GraphUnet, self).__init__()
        self.start_gcn = DropGCN(in_dim, h_dim, act=F.relu)
        self.bottom_gcn = DropGCN(h_dim, h_dim, act=F.relu)
        self.end_gcn = DropGCN(2*h_dim, out_dim, act=F.relu)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(DropGCN(h_dim, h_dim, act=F.relu))
            self.up_gcns.append(DropGCN(h_dim, h_dim, act=F.relu))
            self.pools.append(GraphPool(ks[i], h_dim))
            self.unpools.append(GraphUnpool())

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


class GSRLayer(nn.Module):
    '''
    GSR layer referenced from https://arxiv.org/abs/2009.11080
    '''
    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()
        self.hr_dim = hr_dim
        self.weights = torch.from_numpy(self.weight_variable_glorot(self.hr_dim)).float()
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad=True)

    def weight_variable_glorot(self, output_dim):
        input_dim = output_dim
        init_range = np.sqrt(6.0 / (input_dim + output_dim))
        initial = np.random.uniform(-init_range, init_range, (input_dim, output_dim))
        return initial

    def forward(self, A, X):
        lr = A
        lr_dim = lr.shape[0]
        f = X
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')
        eye_mat = torch.eye(lr_dim).to(device)
        s_d = torch.cat((eye_mat, eye_mat), 0)
        a = torch.matmul(self.weights, s_d)
        b = torch.matmul(a, torch.t(U_lr))
        f_d = torch.matmul(b, f)
        f_d = torch.abs(f_d)
        f_d.fill_diagonal_(1)
        adj = normalize_adj_torch(f_d)
        X = torch.mm(adj, adj.t())
        X = (X + X.t())/2
        idx = torch.eye(self.hr_dim, dtype=bool).to(device)
        X[idx] = 1
        return adj, X


class GIN(nn.Module):
    '''
    GIN layer to learn representative node embeddings in high resolution graph.
    '''
    def __init__(self, in_features, out_features, act=F.relu):
        super(GIN, self).__init__()
        self.act = act
        self.mlp = nn.Sequential(nn.Linear(in_features, out_features),
                                 nn.ReLU(),
                                 nn.Linear(out_features, out_features),
                                 nn.ReLU(),
                                 nn.Linear(out_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, input, adj):
        output = self.mlp(input)
        output = torch.mm(adj, output)
        output = self.act(output)
        return output


class GSRGo(nn.Module):
    '''
    The GSR-Go model that is adapted from GSR-net and combines DropGNN, GIN, and cosine similarity measures.
    '''
    def __init__(self, args):
        super(GSRGo, self).__init__()
        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        hidden_dim = args.hidden_dim
        ks = [0.9, 0.7, 0.6, 0.5]
        self.unet = GraphUnet(ks, self.lr_dim, self.hr_dim, hidden_dim)
        self.gsr = GSRLayer(self.hr_dim)
        self.gc1 = GIN(self.hr_dim, hidden_dim, act=F.relu)
        self.gc2 = GIN(hidden_dim, hidden_dim, act=F.relu)
        self.gc3 = GIN(hidden_dim, hidden_dim, act=F.relu)
        self.gc4 = GIN(hidden_dim, self.hr_dim, act=F.relu)

    def forward(self, lr):
        I = torch.eye(self.lr_dim).to(device)
        A = normalize_adj_torch(lr).to(device)

        # apply U-net to learn dense node features
        unet_outs, start_gcn_outs = self.unet(A, I)

        # graph super resolution
        gsr_adj, gsr_z = self.gsr(A, unet_outs)

        # apply GIN block to HR node features
        z = self.gc1(gsr_z, gsr_adj)
        z = self.gc2(z, gsr_adj)
        z = self.gc3(z, gsr_adj)
        z = self.gc4(z, gsr_adj)

        # calculate cosine similarities of node features to get output adjacency
        z_norm = torch.linalg.norm(z, dim=1, keepdim=True) + 1e-8
        A = (z @ z.t()) / (z_norm @ z_norm.t())

        # set the diagonal of the adjacency matrix to be 1
        idx = torch.eye(self.hr_dim, dtype=bool).to(device)
        A[idx] = 1

        return A, unet_outs, start_gcn_outs, gsr_adj
