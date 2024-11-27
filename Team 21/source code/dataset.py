import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from provided_code.MatrixVectorizer import MatrixVectorizer
from utils import *

class BrainDataset(Dataset):
    def __init__(self, subjects_adj, subjects_labels, args):
        self.subjects_adj = subjects_adj # LR
        self.subjects_labels = subjects_labels # HR
        self.padding = args['padding']
        self.node_drop = args['node_drop']
        self.edge_drop = args['edge_drop']

    def __len__(self):
        return len(self.subjects_adj)

    def __getitem__(self, idx):
        adj = self.subjects_adj[idx]
        adj = node_drop(adj, p=np.random.uniform(0., self.node_drop)) # Randomly drop nodes
        adj = edge_drop(adj, p=np.random.uniform(0., self.edge_drop)) # Randomly drop edges

        return adj, self.subjects_labels[idx]

def dataset():
    """Anti-vectorize the data"""
    lr_train_df = pd.read_csv('./data/lr_train.csv')
    hr_train_df = pd.read_csv('./data/hr_train.csv')
    lr_test_df = pd.read_csv('./data/lr_test.csv')

    mv = MatrixVectorizer()
    
    subjects_adj, subjects_labels, test_adj = [], [], []
    for i in range(len(lr_train_df)):
        subjects_adj.append(mv.anti_vectorize(lr_train_df.iloc[i].values, 160))
        subjects_labels.append(mv.anti_vectorize(hr_train_df.iloc[i].values, 268))

    for i in range(len(lr_test_df)):
        test_adj.append(mv.anti_vectorize(lr_test_df.iloc[i].values, 160))

    # Return subjects_adj, subjects_labels, test_adj
    return np.array(subjects_adj), np.array(subjects_labels), np.array(test_adj)

def preprocess_dataset(subjects_adj, subjects_ground_truth, test_adj, train_index):
    """Min-max normalization is applied"""
    min_val = np.min(subjects_adj[train_index])
    max_val = np.max(subjects_adj[train_index])

    subjects_adj = [(i - min_val) / (max_val - min_val) for i in subjects_adj]
    test_adj = [(i - min_val) / (max_val - min_val) for i in test_adj]

    subjects_adj = np.array(subjects_adj)
    subjects_ground_truth = np.array(subjects_ground_truth)

    return subjects_adj, subjects_ground_truth, test_adj

def node_drop(x, p=0.025):
    # Random set a whole row and column to nearly 0
    x_copy = x.copy()
    n = x.shape[0]
    
    mask = np.random.choice([True, False], size=(n, 1), p=[p, 1-p])
    score = np.random.uniform(0, 1e-4, size=(n, 1))
    x_copy = np.where(mask @ mask.T, (score @ score.T) * x_copy, x_copy)

    return x_copy

def edge_drop(x, p=0.1):
    # Random set some elements to nearly 0
    x_copy = x.copy()
    n = x.shape[0]
    
    mask = np.random.choice([True, False], size=(n, n), p=[p, 1-p])
    score = np.random.uniform(0, 1e-4, size=(n, n))
    x_copy = np.where(mask, score * x_copy, x_copy)

    return x_copy
