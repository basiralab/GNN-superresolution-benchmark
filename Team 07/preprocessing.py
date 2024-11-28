import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from datasets import BrainDataset
from MatrixVectorizer import MatrixVectorizer



def get_dset():
    # Prepare low + high resolution training set
    training_1 = pd.read_csv('RandomCV/Fold1/lr_split_1.csv').to_numpy()
    training_2 = pd.read_csv('RandomCV/Fold2/lr_split_2.csv').to_numpy()
    training_3 = pd.read_csv('RandomCV/Fold3/lr_split_3.csv').to_numpy()
    lr_train_vectorized = np.concatenate((training_1, training_2, training_3), axis=0)

    training_truths_1 = pd.read_csv('RandomCV/Fold1/hr_split_1.csv').to_numpy()
    training_truths_2 = pd.read_csv('RandomCV/Fold2/hr_split_2.csv').to_numpy()
    training_truths_3 = pd.read_csv('RandomCV/Fold3/hr_split_3.csv').to_numpy()
    hr_train_vectorized = np.concatenate((training_truths_1, training_truths_2, training_truths_3), axis=0)


    # Prepare training dataset
    vectorizer = MatrixVectorizer()

    n_samples = len(lr_train_vectorized)
    train_dataset = []

    for i in range(n_samples):
        lr_train_matrix = vectorizer.anti_vectorize(lr_train_vectorized[i], NUM_LOW_RES_NODES)
        hr_train_matrix = vectorizer.anti_vectorize(hr_train_vectorized[i], NUM_HIGH_RES_NODES)
        train_dataset.append((lr_train_matrix, hr_train_matrix))

    return train_dataset

def get_unseen_test_dset():
    lr_data = pd.read_csv('data/lr_test.csv')
    vectorizer = MatrixVectorizer()
    n_samples = len(lr_data)
    test_dataset = []
    lr_test_vectorized = np.array(lr_data)

    for i in range(n_samples):
        lr_test_matrix = vectorizer.anti_vectorize(lr_test_vectorized[i], NUM_LOW_RES_NODES)
        test_dataset.append(lr_test_matrix)

    return test_dataset

def get_dataloaders(dset, hps, is_train_or_val=True):
    # temporary
    normalization_func = normalize_adj
    batch_size = hps.batch_size

    if is_train_or_val:
        train_dataset, val_dataset = train_test_split(dset, test_size=0.2, random_state=RANDOM_SEED)
        train_dataset = BrainDataset(train_dataset, normalization_func=normalization_func)
        val_dataset = BrainDataset(val_dataset, normalization_func=normalization_func)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
    else:
        test_dataset = BrainDataset(dset, normalization_func=normalization_func, is_train_or_val=is_train_or_val)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        return test_loader

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

NUM_FEATURES_PER_NODE = 5
RANDOM_SEED = 42
NUM_LOW_RES_NODES = 160
NUM_HIGH_RES_NODES = 268
