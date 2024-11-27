import torch
import torch.nn as nn
import torch_geometric as geo
import torch.nn.functional as F
from architecture.GraphGoldNet import GraphGoldNet
from torch_geometric.nn import GCNConv, BatchNorm, global_mean_pool


class ConversionLayer(torch.nn.Module):
    def __init__(self, nodes, dropout=0.1):
        super(ConversionLayer, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Linear(160, nodes*2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),

            nn.Linear(nodes*2, nodes*2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            
            nn.Linear(nodes*2, nodes*2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),

            nn.Linear(nodes*2, nodes*2),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),

            nn.Linear(nodes*2, nodes),
        )
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Model(torch.nn.Module):
    def __init__(self, feature_layer_args):
        super(Model, self).__init__()
       
        self.feature_layer = GraphGoldNet(feature_layer_args["in_channels"],feature_layer_args["hidden_channels"], feature_layer_args["out_channels"], feature_layer_args["hidden_layer"],
                                       pool_ratios=feature_layer_args["pool_ratios"] )
            
        self.linear_lr_hr = ConversionLayer(268)        
        
        self.last_layer_hr_adj = NodeToAdj(268, feature_layer_args["out_channels"] , threshold=0.005)
        self.last_layer_lr_adj = NodeToAdj(160, feature_layer_args["out_channels"] , threshold=0.005)
        
    def forward(self, x, edge_index, edge_weight, batch):        

        batch_size = int(max(batch.tolist())+1)
        lr_features = self.feature_layer(x, edge_index, edge_weight, batch)
        hr_features = self.linear_lr_hr(torch.cat(lr_features.split(lr_features.shape[0]//batch_size,dim=0),dim=1).T).T
        hr_features = torch.cat(hr_features.split(hr_features.shape[1]//batch_size,dim=1),dim=0)        
        
        pred_hr_adj = self.last_layer_hr_adj(hr_features)
        pred_lr_adj = self.last_layer_lr_adj(lr_features)

        return pred_hr_adj, pred_lr_adj


class NodeToAdj(torch.nn.Module):
    def __init__(self, num_node, in_channel, threshold=0.01):
        super(NodeToAdj, self).__init__()
        self.in_channel = in_channel
        self.num_node = num_node
        self.threshold = threshold
        self.weight = nn.Parameter(torch.randn((1,num_node,num_node, in_channel)))        
    
    def forward(self, x):
        batch_size = x.shape[0]//self.num_node
        pred_adj = (x.reshape(batch_size, -1, 1,  self.in_channel)*x.reshape(batch_size, 1, -1, self.in_channel))*self.weight
        pred_adj = pred_adj.sum(dim=-1)
        pred_adj = F.sigmoid(pred_adj+pred_adj.transpose(1,2))
        mask = torch.stack([torch.eye(self.num_node, device=x.device)]*batch_size,dim=0)
        pred_adj = pred_adj*(1-mask) + mask
        pred_adj = pred_adj*(pred_adj > self.threshold)
        return pred_adj.reshape(batch_size*self.num_node,-1)
