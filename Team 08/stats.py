import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def collect_data(path):
    fold1 = pd.read_csv(path + 'randomCV_fold1.csv')
    fold2 = pd.read_csv(path + 'randomCV_fold2.csv')
    fold3 = pd.read_csv(path + 'randomCV_fold3.csv')

    data = pd.concat([fold1, fold2, fold3], ignore_index=True)
    # collect the mean and std of every column in the data
    mean = data.mean()
    std = data.std()
    # create a new dataframe with the mean and std of every column
    stats = pd.DataFrame(columns=['mean', 'std'])
    stats['mean'] = mean
    stats['std'] = std
    return stats

def collect_data(path):
    fold1 = pd.read_csv(path + 'clusterCV_fold1.csv')
    fold2 = pd.read_csv(path + 'clusterCV_fold2.csv')
    fold3 = pd.read_csv(path + 'clusterCV_fold3.csv')

    data = pd.concat([fold1, fold2, fold3], ignore_index=True)
    # collect the mean and std of every column in the data
    mean = data.mean()
    std = data.std()
    # create a new dataframe with the mean and std of every column
    stats = pd.DataFrame(columns=['mean', 'std'])
    stats['mean'] = mean
    stats['std'] = std
    return stats

def concat_data(path):
    fold1 = pd.read_csv(path + 'clusterCV_fold1.csv')
    fold2 = pd.read_csv(path + 'clusterCV_fold2.csv')
    fold3 = pd.read_csv(path + 'clusterCV_fold3.csv')

    data = pd.concat([fold1, fold2, fold3], ignore_index=False)
    return data

def main():
    path = 'cluster_res/'
    data = concat_data(path)
    data.to_csv("results/" + 'clusterCV.csv')

if __name__ == '__main__':
    main()