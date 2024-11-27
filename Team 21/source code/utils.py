"""
Modified code based on AGSR-Net (https://github.com/basiralab/AGSR-Net) by Basira Labs.
Original licensing terms apply.
"""

import numpy as np
import torch
import random

def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def normalize_adj_torch(mx):
    # mx = mx.to_dense()
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx

def pad_HR_adj(label, split):
    label=np.pad(label,((split,split),(split,split)),mode="constant")
    np.fill_diagonal(label,1)
    return torch.from_numpy(label).type(torch.FloatTensor)

def unpad(data, split):
    idx_0 = data.shape[0]-split
    idx_1 = data.shape[1]-split
    # print(idx_0,idx_1)
    train = data[split:idx_0, split:idx_1]
    return train
