import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from processing import *


class GSRLayer(nn.Module):

    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):

            lr_dim = A.shape[0]
            _, U_lr = torch.linalg.eigh(A, UPLO='U')

            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor)
            s_d = torch.cat((eye_mat, eye_mat), 0)

            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, X)
            f_d = torch.abs(f_d)
            adj = f_d.fill_diagonal_(1)

            X = torch.mm(adj, adj.t())
            X = (X + X.t())/2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, dropout, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
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

class SuperBLTGraph(nn.Module):

    def __init__(self, args):
        super(SuperBLTGraph, self).__init__()

        self.lr_dim = args.lr_dim
        self.double_convolution = args.double_convolution

        if args.double_convolution == False:
            self.gc0 = GraphConvolution(args.lr_dim, args.hr_dim, args.dropout, act=F.relu)
        else:
            self.gc0_0 = GraphConvolution(args.lr_dim, args.hidden_dim, args.dropout, act=F.relu)
            self.gc0_1 = GraphConvolution(args.hidden_dim, args.hr_dim, args.dropout, act=F.relu)
        self.layer = GSRLayer(args.hr_dim)
        self.gc1 = GraphConvolution(args.hr_dim, args.hidden_dim, args.dropout, act=F.relu)
        self.gc2 = GraphConvolution(args.hidden_dim, args.hr_dim, args.dropout, act=F.tanh)
      

    def forward(self, lr):
        with torch.autograd.set_detect_anomaly(True):
           
            X = torch.eye(self.lr_dim).type(torch.FloatTensor)
            A = normalize_adj_torch(lr).type(torch.FloatTensor)

            if self.double_convolution == False:
                self.gcnew_out = self.gc0(X, A)
            else:
                X = self.gc0_0(X, A)
                self.gcnew_out = self.gc0_1(X, A)

            self.outputs, self.Z = self.layer(A, self.gcnew_out)

            self.hidden1 = self.gc1(self.Z, self.outputs)
            z = self.gc2(self.hidden1, self.outputs)
        
            z = (z + z.t())/2
            z = z.fill_diagonal_(0)
            
        return z