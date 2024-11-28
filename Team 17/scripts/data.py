import pandas as pd
import torch
import numpy as np
from scripts.MatrixVectorizer import MatrixVectorizer
from torch_geometric.data import Dataset, Data

def onevector(matrices, latent_dim=10):
    if matrices.dim() == 2:
        return torch.ones((1,matrices.shape[0], latent_dim))
    elif matrices.dim() == 3:
        return torch.ones((matrices.shape[0],matrices.shape[1], latent_dim))
    else:
        raise Exception("Incorrect Dim")

def randomvector(matrices, latent_dim=10):    
    if matrices.dim() == 2:
        return torch.rand(matrices.shape[0], latent_dim)
    elif matrices.dim() == 3:
        return torch.rand(matrices.shape[0],matrices.shape[1], latent_dim)
    else:
        raise Exception("Incorrect Dim")

class BrainTrain():
    def __init__(self, seed=0, eval_mode="clusterCV"):
        
        # CHANGE self.main_path to the correct path
        self.main_path = "path"
        if eval_mode == "clusterCV":
            self.lr_data_path = ["Cluster-CV/Cluster-CV/Fold1/lr_clusterA.csv", "Cluster-CV/Cluster-CV/Fold2/lr_clusterB.csv", "Cluster-CV/Cluster-CV/Fold3/lr_clusterC.csv"]
            self.hr_data_path = ["Cluster-CV/Cluster-CV/Fold1/hr_clusterA.csv", "Cluster-CV/Cluster-CV/Fold2/hr_clusterB.csv", "Cluster-CV/Cluster-CV/Fold3/hr_clusterC.csv"]
        elif eval_mode=="randomCV":
            self.lr_data_path = ["3_fold_data/Train/Fold1/lr_split_1.csv", "3_fold_data/Train/Fold2/lr_split_2.csv", "3_fold_data/Train/Fold3/lr_split_3.csv"]
            self.hr_data_path = ["3_fold_data/Train/Fold1/hr_split_1.csv", "3_fold_data/Train/Fold2/hr_split_2.csv", "3_fold_data/Train/Fold3/hr_split_3.csv"]           

        '''
        self.fold_indices = {}
        self.kf = KFold(n_splits=3,random_state=seed, shuffle=True)
        for i, (train_index, val_index) in enumerate(self.kf.split(self.lr_data)):
            self.fold_indices[i] = (train_index, val_index)
        '''
           
    def get_fold_split(self, fold, transform, type= "train", ):

        
        if type == "train" or type == "val":
            lr_data_1 = pd.read_csv(self.main_path + self.lr_data_path[fold[0]]).to_numpy()
            hr_data_1 = pd.read_csv(self.main_path + self.hr_data_path[fold[0]]).to_numpy()
            lr_data_2 = pd.read_csv(self.main_path + self.lr_data_path[fold[1]]).to_numpy()
            hr_data_2 = pd.read_csv(self.main_path + self.hr_data_path[fold[1]]).to_numpy()

            if type == "train":

                self.lr_data = np.vstack((lr_data_1[:lr_data_1.shape[0]-20,:],lr_data_2[:lr_data_2.shape[0]-20,:]))
                self.hr_data = np.vstack((hr_data_1[:hr_data_1.shape[0]-20,:],hr_data_2[:hr_data_2.shape[0]-20,:]))
            elif type == "val":

                self.lr_data = np.vstack((lr_data_1[-20:,:],lr_data_2[-20:,:]))
                self.hr_data = np.vstack((hr_data_1[-20:,:],hr_data_2[-20:,:]))
            
        elif type == "test":
            self.lr_data = pd.read_csv(self.main_path + self.lr_data_path[2]).to_numpy()
            self.hr_data = pd.read_csv(self.main_path + self.hr_data_path[2]).to_numpy() 
            
        assert self.lr_data.shape[0] == self.hr_data.shape[0]
        
        mv = MatrixVectorizer()
        out = []
        for idx in range(self.lr_data.shape[0]):
            x_adj_maxtrix = mv.anti_vectorize(self.lr_data[idx],160)
            y_adj_maxtrix = torch.from_numpy(mv.anti_vectorize(self.hr_data[idx],268))

            non_zero_edges = x_adj_maxtrix.nonzero()
            edge_index = torch.from_numpy(np.vstack((non_zero_edges[0], non_zero_edges[1])))
            edge_attr = torch.from_numpy(x_adj_maxtrix[non_zero_edges[0], non_zero_edges[1]])
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y_adj_maxtrix, ori_matrix=torch.from_numpy(x_adj_maxtrix))
            data.num_nodes = 160
            out.append(data)

        return (Dummy(out, transform))
    
    def get_all(self, fold, transform):
        mv = MatrixVectorizer()
        out = []
        
        lr_data_1 = pd.read_csv(self.main_path + self.lr_data_path[fold[0]]).to_numpy()
        hr_data_1 = pd.read_csv(self.main_path + self.hr_data_path[fold[0]]).to_numpy()
        lr_data_2 = pd.read_csv(self.main_path + self.lr_data_path[fold[1]]).to_numpy()
        hr_data_2 = pd.read_csv(self.main_path + self.hr_data_path[fold[1]]).to_numpy()
        self.lr_data = np.vstack((lr_data_1,lr_data_2))
        self.hr_data = np.vstack((hr_data_1,hr_data_2))

        for idx in range(self.lr_data.shape[0]):
            x_adj_maxtrix = mv.anti_vectorize(self.lr_data[idx],160)
            y_adj_maxtrix = torch.from_numpy(mv.anti_vectorize(self.hr_data[idx],268))

            non_zero_edges = x_adj_maxtrix.nonzero()
            edge_index = torch.from_numpy(np.vstack((non_zero_edges[0], non_zero_edges[1])))
            edge_attr = torch.from_numpy(x_adj_maxtrix[non_zero_edges[0], non_zero_edges[1]])
            data = Data(edge_index=edge_index, edge_attr=edge_attr, y=y_adj_maxtrix, ori_matrix=torch.from_numpy(x_adj_maxtrix))
            data.num_nodes = 160
            out.append(data)

        return (Dummy(out, transform))


class Dummy(Dataset):
    def __init__(self, data_list, transform):
        super().__init__(transform=transform)
        self.data_list = data_list
       
    def get(self, idx):
        return self.data_list[idx]

    def len(self):
        return len(self.data_list)

def BrainTest(transform):    
    lr_data = pd.read_csv("3_fold_data/Test/lr_test.csv").to_numpy()
    mv = MatrixVectorizer()
            
    out = []
    for idx in range(lr_data.shape[0]):
        x_adj_maxtrix = mv.anti_vectorize(lr_data[idx],160)

        non_zero_edges = x_adj_maxtrix.nonzero()
        edge_index = torch.from_numpy(np.vstack((non_zero_edges[0], non_zero_edges[1])))
        edge_attr = torch.from_numpy(x_adj_maxtrix[non_zero_edges[0], non_zero_edges[1]])
        data = Data(x=None, edge_index=edge_index, edge_attr=edge_attr)
        data.num_nodes = 160
        
        out.append(data)

    return Dummy(out, transform)
