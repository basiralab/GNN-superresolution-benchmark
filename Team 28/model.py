import torch
import torch.nn as nn
from layers import *
from preprocessing import normalize_adj_torch
import torch.nn.functional as F
from torch_geometric.nn.norm import GraphNorm

class EAGSRNet(nn.Module):
    """
    Enhanced Adversarial Graph Super Resolution Network with Non-Edge Loss (EASGSR-Net).

    Adapted from AGSR-Net (https://github.com/basiralab/AGSR-Net/tree/master)
    """

    def __init__(self, ks, args):
        super(EAGSRNet, self).__init__()
        self.device = args['device']
        self.lr_dim = args['lr_dim']
        self.hr_dim = args['hr_dim']
        self.hidden_dim = args['hidden_dim']
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim, self.device).to(self.device)
        self.layer = GSRLayer(self.hr_dim).to(self.device)
        self.gn1 = GraphNorm(self.hr_dim).to(self.device)
        self.gc1 = GraphConvolution(
            self.hr_dim, self.hidden_dim, 0, F.relu).to(self.device)
        self.gn2 = GraphNorm(self.hidden_dim).to(self.device)
        self.gc2 = GraphConvolution(
            self.hidden_dim, self.hr_dim, 0, F.relu).to(self.device)

    def forward(self, lr):
        with torch.autograd.set_detect_anomaly(True):

            I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(self.device)
            A = normalize_adj_torch(lr).type(torch.FloatTensor).to(self.device)

            self.net_outs, self.start_gcn_outs = self.net(A, I)

            self.outputs, self.Z = self.layer(A, self.net_outs)
            self.Z = self.gn1(self.Z)

            self.hidden1 = self.gc1(self.Z, self.outputs)
            self.hidden1 = self.gn2(self.hidden1)
            self.hidden2 = self.gc2(self.hidden1, self.outputs)
            z = self.hidden2

            z = (z + z.t())/2
            z = z.fill_diagonal_(1)

        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs



class GCNDiscriminator(nn.Module):
    def __init__(self, args):
        super(GCNDiscriminator, self).__init__()
        self.conv1 = GraphConvolution(args['hr_dim'], args['hidden_dim'], 0, act=F.relu)
        self.conv2 = GraphConvolution(args['hidden_dim'], args['hidden_dim'], 0, act=F.relu)
        self.conv3 = GraphConvolution(args['hidden_dim'], 1, 0, act=F.relu)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x, adj):
        x1 = self.conv1(x, adj)
        x2 = self.conv2(x1, adj)
        x3 = self.conv3(x2, adj)
        x3 = self.sigmoid(x3)
      
        return x3


def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer).to(args['device'])
    noise = z.normal_(mean=args['mean_gaussian'], std=args['std_gaussian'])
    z = torch.abs(input_layer + noise)

    z = (z + z.t())/2
    z = z.fill_diagonal_(1)
    return z
