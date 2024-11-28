import torch
import torch.nn.functional as F

def pad_HR_adj(label, split):
    label = F.pad(label, (split, split, split, split), "constant", 0)
    label = label.fill_diagonal_(1)
    return label.to(dtype=torch.float32)

def unpad(data, split):

  idx_0 = data.shape[0]-split
  idx_1 = data.shape[1]-split
  train = data[split:idx_0, split:idx_1]
  return train
