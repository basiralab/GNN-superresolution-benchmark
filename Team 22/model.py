import torch
import torch.nn as nn
from layers import *
from ops import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class GraphSAGELayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, act=F.relu, aggregator_type='mean'):
        super(GraphSAGELayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.aggregator_type = aggregator_type
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)
        
    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        
        # Aggregation
        if self.aggregator_type == 'mean':
            neigh_feats = torch.mm(adj, input)  # Aggregate neighbor features
        else:
            raise NotImplementedError("Other aggregation methods not implemented")
        
        output = torch.mm(neigh_feats, self.weight)
        
        if self.act is not None:
            output = self.act(output)
            
        return output

class GINLayer(nn.Module):
    """Graph Isomorphism Network layer with an MLP and residual connections."""
    def __init__(self, in_features, out_features, mlp_layers=2, dropout=0.5):
        super(GINLayer, self).__init__()
        self.mlp = self.create_mlp(in_features, out_features, mlp_layers, dropout)
        self.residual = (in_features == out_features)

    def create_mlp(self, in_features, out_features, mlp_layers, dropout):
        layers = [nn.Linear(in_features, out_features), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Linear(out_features, out_features), nn.ReLU(), nn.Dropout(dropout)])
        return nn.Sequential(*layers)

    def forward(self, input, adj):
        neighbor_sum = torch.mm(adj, input)  # Aggregate neighbor features
        self_feats = input  # Self-features
        total = neighbor_sum + self_feats  # Combine self and neighbor features
        out = self.mlp(total)
        if self.residual:
            out = out + input  # Add the residual connection
        return out

class AGSRNet(nn.Module):
    def __init__(self, ks, args):
        super(AGSRNet, self).__init__()
        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)

        # Use GINLayer instead of GraphSAGELayer
        self.gc1 = GINLayer(
            self.hr_dim, self.hidden_dim, mlp_layers=2, dropout=0)
        self.gc2 = GINLayer(
            self.hidden_dim, self.hr_dim, mlp_layers=2, dropout=0)

    def forward(self, lr, lr_dim, hr_dim):
        with torch.autograd.set_detect_anomaly(True):

            I = torch.eye(self.lr_dim).type(torch.FloatTensor)
            A = normalize_adj_torch(lr).type(torch.FloatTensor)

            self.net_outs, self.start_gcn_outs = self.net(A, I)

            self.outputs, self.Z = self.layer(A, self.net_outs)

            self.hidden1 = self.gc1(self.Z, self.outputs)
            self.hidden2 = self.gc2(self.hidden1, self.outputs)
            z = self.hidden2

            z = (z + z.t())/2
            z = z.fill_diagonal_(1)

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs

class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        np.random.seed(1)
        torch.manual_seed(1)

        out = torch.mm(x, self.weights)
        return out


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.dense_1 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_1 = nn.ReLU(inplace=False)
        self.dense_2 = Dense(args.hr_dim, args.hr_dim, args)
        self.relu_2 = nn.ReLU(inplace=False)
        self.dense_3 = Dense(args.hr_dim, 1, args)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        np.random.seed(1)
        torch.manual_seed(1)
        dc_den1 = self.relu_1(self.dense_1(inputs))
        dc_den2 = self.relu_2(self.dense_2(dc_den1))
        output = dc_den2
        output = self.dense_3(dc_den2)
        output = self.sigmoid(output)
        return torch.abs(output)


def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z
