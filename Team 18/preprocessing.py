import torch
import numpy as np
import pandas as pd
from utils.MatrixVectorizer import MatrixVectorizer


def pad_HR_adj(label, split):
    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    np.fill_diagonal(label, 1)
    return torch.from_numpy(label).float()


def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    return mx


def unpad(data, split):
    idx_0 = data.shape[0]-split
    idx_1 = data.shape[1]-split
    train = data[split:idx_0, split:idx_1]
    return train


def process_vecs(vecs, mat_size, diag=0):
    anti_vecs = np.zeros((vecs.shape[0], mat_size, mat_size))
    for i, vec in enumerate(vecs):
        anti_vecs[i] = MatrixVectorizer.anti_vectorize(vec, mat_size)
        np.fill_diagonal(anti_vecs[i], diag)
    return anti_vecs


def load_data(split='train'):
    '''
    Load training data/labels or testing data from csv files.
    Data is loaded as numpy arrays and processed to anti-vectorize the adjacency matrices.
    '''
    if split == 'train':
        # File paths
        # file_paths_lr = ['data/lr_split_1.csv', 'data/lr_split_2.csv', 'data/lr_split_3.csv']
        file_paths_lr = ['data/lr_clusterA.csv', 'data/lr_clusterB.csv', 'data/lr_clusterC.csv']


        # Read each file into a DataFrame and store in a list
        dataframes = [pd.read_csv(file_path) for file_path in file_paths_lr]

        # Concatenate all DataFrames
        concatenated_df = pd.concat(dataframes, ignore_index=True)

        # Convert the concatenated DataFrame to a NumPy array
        lr_vecs = concatenated_df.to_numpy()
        #lr_vecs = pd.read_csv('data/lr_train.csv').to_numpy()

        # File paths
        # file_paths_hr = ['data/hr_split_1.csv', 'data/hr_split_2.csv', 'data/hr_split_3.csv']
        file_paths_hr = ['data/hr_clusterA_modified.csv', 'data/hr_clusterB_modified.csv', 'data/hr_clusterC_modified.csv']

        # Read each file into a DataFrame and store in a list
        dataframes = [pd.read_csv(file_path) for file_path in file_paths_hr]

        # Concatenate all DataFrames
        concatenated_df = pd.concat(dataframes, ignore_index=True)
        hr_vecs = concatenated_df.to_numpy()
        #hr_vecs = pd.read_csv('data/hr_train.csv').to_numpy()
        lr_anti_vecs = process_vecs(lr_vecs, 160, diag=1)
        hr_anti_vecs = process_vecs(hr_vecs, 268, diag=1)
        return lr_anti_vecs, hr_anti_vecs
    elif split == 'test':
        lr_vecs = pd.read_csv('data/lr_test.csv').to_numpy()
        lr_anti_vecs = process_vecs(lr_vecs, 160, diag=1)
        return lr_anti_vecs


if __name__ == "__main__":
    lr, hr = load_data(split='train')
    print(lr.shape, hr.shape)
    print(lr[0])
