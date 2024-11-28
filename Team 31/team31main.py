import argparse
import os
import random
import time
import matplotlib.pyplot as plt
import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import psutil
import scipy.io
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F

# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Check for CUDA (GPU support) and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    # Additional settings for ensuring reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

path = 'data'
roi_str = 'ROI_FC.mat'

def pad_HR_adj(label, split):
    label = np.pad(label, ((split, split), (split, split)), mode="constant")
    np.fill_diagonal(label, 1)
    return label

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
    idx_0 = data.shape[0] - split
    idx_1 = data.shape[1] - split
    train = data[split:idx_0, split:idx_1]
    return train

def extract_data(subject, session_str, parcellation_str, subjects_roi):
    folder_path = os.path.join(
        path, str(subject), session_str, parcellation_str)
    roi_data = scipy.io.loadmat(os.path.join(folder_path, roi_str))
    roi = roi_data['r']

    # Replacing NaN values
    col_mean = np.nanmean(roi, axis=0)
    inds = np.where(np.isnan(roi))
    roi[inds] = 1

    # Taking the absolute values of the matrix
    roi = np.absolute(roi, dtype=np.float32)

    if parcellation_str == 'xxx':
        roi = np.reshape(roi, (1, 268, 268))
    else:
        roi = np.reshape(roi, (1, 160, 160))

    if subject == 25629:
        subjects_roi = roi
    else:
        subjects_roi = np.concatenate((subjects_roi, roi), axis=0)

    return subjects_roi

def load_data(start_value, end_value):
    subjects_label = np.zeros((1, 268, 268))
    subjects_adj = np.zeros((1, 160, 160))

    for subject in range(start_value, end_value):
        subject_path = os.path.join(path, str(subject))

        if 'session_1' in os.listdir(subject_path):
            subjects_label = extract_data(
                subject, 'session_1', 'xxx', subjects_label)
            subjects_adj = extract_data(
                subject, 'session_1', 'xxx', subjects_adj)

    return subjects_adj, subjects_label

def data():
    subjects_adj, subjects_labels = load_data(25629, 25830)
    test_adj_1, test_labels_1 = load_data(25831, 25863)
    test_adj_2, test_labels_2 = load_data(30701, 30757)
    test_adj = np.concatenate((test_adj_1, test_adj_2), axis=0)
    test_labels = np.concatenate((test_labels_1, test_labels_2), axis=0)
    return subjects_adj, subjects_labels, test_adj, test_labels

def edge_perturbation(adj_matrix, perturb_rate=0.01):
    n = adj_matrix.shape[0]
    num_perturbations = int(n * n * perturb_rate)

    for _ in range(num_perturbations):
        i, j = np.random.randint(0, n, size=2)
        adj_matrix[i, j] = 1 - adj_matrix[i, j]
        adj_matrix[j, i] = adj_matrix[i, j]

    return adj_matrix

def subgraph_sampling(adj_matrix, sample_size=120):
    n = adj_matrix.shape[0]
    indices = np.random.choice(n, sample_size, replace=False)
    subgraph = adj_matrix[np.ix_(indices, indices)]

    resized_subgraph = cv2.resize(subgraph.astype('float32'), (n, n), interpolation=cv2.INTER_NEAREST)
    return resized_subgraph

def graph_interpolation(adj_matrix1, adj_matrix2, alpha=0.5):
    alpha = np.clip(alpha, 0, 1)
    interpolated_graph = alpha * adj_matrix1 + (1 - alpha) * adj_matrix2
    interpolated_graph = (interpolated_graph > 0.5).astype(int)
    return interpolated_graph



class GraphUnpool(nn.Module):
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        new_X = torch.zeros([A.shape[0], X.shape[1]]).to(A.device)
        new_X[idx] = X
        return A, new_X

class GraphPool(nn.Module):
    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores / 100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k * num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx

class GCN(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=0)

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X

class GCNWithDropGNN(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.5):
        super(GCNWithDropGNN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, A, X):
        X = self.drop(X)
        X = torch.matmul(A, X)
        X = self.proj(X)
        return X

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

class Dense(nn.Module):
    def __init__(self, n1, n2, args):
        super(Dense, self).__init__()
        self.weights = torch.nn.Parameter(
            torch.FloatTensor(n1, n2), requires_grad=True)
        nn.init.normal_(self.weights, mean=args.mean_dense, std=args.std_dense)

    def forward(self, x):
        np.random.seed(1)
        torch.manual_seed(1)
        out = torch.mm(x, self.weights)
        return out

class DropGCN(nn.Module):
    def __init__(self, in_dim, out_dim, drop_rate=0.5):
        super(DropGCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.drop = nn.Dropout(p=drop_rate)

    def forward(self, X):
        X = self.proj(X)
        X = self.drop(X)
        return X

class GraphUnet(nn.Module):
    def __init__(self, ks, in_dim, out_dim, dim=320):
        super(GraphUnet, self).__init__()
        self.ks = ks

        self.start_gcn = GCNWithDropGNN(in_dim, dim)
        self.bottom_gcn = GCNWithDropGNN(dim, dim)
        self.end_gcn = GCNWithDropGNN(2 * dim, out_dim)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCNWithDropGNN(dim, dim))
            self.up_gcns.append(GCNWithDropGNN(dim, dim))
            self.pools.append(GraphPool(ks[i], dim))
            self.unpools.append(GraphUnpool())

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gcn(A, X)
        start_gcn_outs = X
        org_X = X
        for i in range(self.l_n):
            X = self.down_gcns[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        X = self.bottom_gcn(A, X)
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1

            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gcns[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gcn(A, X)
        return X, start_gcn_outs

def weight_variable_glorot(output_dim):
    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,
                                (input_dim, output_dim))

    return initial

class GSRLayer(nn.Module):
    def __init__(self, hr_dim):
        super(GSRLayer, self).__init__()

        self.weights = torch.from_numpy(
            weight_variable_glorot(hr_dim)).type(torch.FloatTensor)
        self.weights = torch.nn.Parameter(
            data=self.weights, requires_grad=True)

    def forward(self, A, X):
        with torch.autograd.set_detect_anomaly(True):
            lr = A
            lr_dim = lr.shape[0]
            f = X
            eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')

            eye_mat = torch.eye(lr_dim).type(torch.FloatTensor).to(A.device)
            s_d = torch.cat((eye_mat, eye_mat), 0)

            a = torch.matmul(self.weights, s_d)
            b = torch.matmul(a, torch.t(U_lr))
            f_d = torch.matmul(b, f)
            f_d = torch.abs(f_d)
            f_d = f_d.fill_diagonal_(1)
            adj = f_d

            X = torch.mm(adj, adj.t())
            X = (X + X.t()) / 2
            X = X.fill_diagonal_(1)
        return adj, torch.abs(X)

class AGSRNet(nn.Module):
    def __init__(self, ks, args):
        super(AGSRNet, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.hr_dim)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim)
        self.gc1 = GraphConvolution(
            self.hr_dim, self.hidden_dim, 0, act=F.relu)
        self.gc2 = GraphConvolution(
            self.hidden_dim, self.hr_dim, 0, act=F.relu)

    def forward(self, lr, lr_dim, hr_dim):
        with torch.autograd.set_detect_anomaly(True):
            I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(lr.device)
            A = normalize_adj_torch(lr).type(torch.FloatTensor).to(lr.device)
            self.net_outs, self.start_gcn_outs = self.net(A, I)
            self.outputs, self.Z = self.layer(A, self.net_outs)

            self.hidden1 = self.gc1(self.Z, self.outputs)
            self.hidden2 = self.gc2(self.hidden1, self.outputs)
            z = self.hidden2

            z = (z + z.t()) / 2
            z = z.fill_diagonal_(1)
        return torch.abs(z), self.net_outs, self.start_gcn_outs, self.outputs

class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.layer1 = DropGCN(args.hr_dim, 256, 0.5)
        self.layer2 = DropGCN(256, 128, 0.5)
        self.layer3 = DropGCN(128, 64, 0.5)
        self.layer4 = DropGCN(64, 1, 0.5)
        self.fc = nn.Sequential(
            nn.Linear(320, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, inputs):
        x = self.layer1(inputs)
        x = F.leaky_relu(x, 0.2)
        x = self.layer2(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer3(x)
        x = F.leaky_relu(x, 0.2)
        x = self.layer4(x)
        x = F.leaky_relu(x, 0.2)
        x = torch.permute(x, (1, 0))
        x = self.fc(x)
        return x

def gaussian_noise_layer(input_layer, args):
    z = torch.empty_like(input_layer)
    noise = z.normal_(mean=args.mean_gaussian, std=args.std_gaussian)
    z = torch.abs(input_layer + noise)

    z = (z + z.t()) / 2
    z = z.fill_diagonal_(1)
    return z

class MatrixVectorizer:
    def __init__(self):
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        matrix_size = matrix.shape[0]
        vector_elements = []
        for col in range(matrix_size):
            for row in range(matrix_size):
                if row != col:
                    if row < col:
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        vector_elements.append(matrix[row, col])
        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        matrix = np.zeros((matrix_size, matrix_size))
        vector_idx = 0
        for col in range(matrix_size):
            for row in range(matrix_size):
                if row != col:
                    if row < col:
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
                    elif include_diagonal and row == col + 1:
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
        return matrix

    

def calculate_mae(preds_list, g_t):
    mae_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        ground_truth_np = ground_truth.cpu().detach().numpy()
        mae = np.mean(np.abs(preds - ground_truth_np))
        mae_list.append(mae)
    return mae_list

def calculate_pcc(preds_list, g_t):
    pcc_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        ground_truth_np = ground_truth.cpu().detach().numpy()
        pcc = np.corrcoef(preds, ground_truth_np)[0, 1]
        pcc_list.append(pcc)
    return pcc_list

# compute JSD
def calculate_jsd(preds_list, g_t):
    jsd_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        ground_truth_np = ground_truth.cpu().detach().numpy()
        jsd = jensenshannon(preds, ground_truth_np)
        jsd_list.append(jsd)
    return jsd_list

def calculate_pc(preds_list, g_t):
    pc_mae_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        preds = np.reshape(preds, (268, 268))
        
        # Ground truth is a tensor, so we need to convert it to numpy
        ground_truth = np.reshape(ground_truth.cpu().detach().numpy(), (268, 268))
        g1 = nx.from_numpy_array(preds, create_using=nx.DiGraph)
        g2 = nx.from_numpy_array(ground_truth, create_using=nx.DiGraph)
        
        # Calculate pagerank
        pagerank1 = nx.pagerank(g1)
        pagerank2 = nx.pagerank(g2)
        pagerank1 = np.array(list(pagerank1.values()))
        pagerank2 = np.array(list(pagerank2.values()))
        pc_mae = np.mean(np.abs(pagerank1 - pagerank2))
        pc_mae_list.append(pc_mae)
        
    return pc_mae_list

def calculate_ec(preds_list, g_t):
    ec_mae_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        preds = np.reshape(preds, (268, 268))
        
        # Ground truth is a tensor, so we need to convert it to numpy
        ground_truth = np.reshape(ground_truth.cpu().detach().numpy(), (268, 268))
        g1 = nx.from_numpy_array(preds)
        g2 = nx.from_numpy_array(ground_truth)
        
        # Calculate eigenvector centrality
        ec1 = np.array(list(nx.eigenvector_centrality(g1, max_iter=1000).values()))
        ec2 = np.array(list(nx.eigenvector_centrality(g2, max_iter=1000).values()))
        ec_mae = np.mean(np.abs(ec1 - ec2))
        ec_mae_list.append(ec_mae)
    return ec_mae_list

def calculate_bc(preds_list, g_t):
    bc_mae_list = []
    for preds, ground_truth in zip(preds_list, g_t):
        preds = np.reshape(preds, (268, 268))
        
        # Ground truth is a tensor, so convert it to numpy
        ground_truth = np.reshape(ground_truth.cpu().detach().numpy(), (268, 268))
        g1 = nx.from_numpy_array(preds)
        g2 = nx.from_numpy_array(ground_truth)
        
        # Calculate betweenness centrality
        bc1 = np.array(list(nx.betweenness_centrality(g1).values()))
        bc2 = np.array(list(nx.betweenness_centrality(g2).values()))
        bc_mae = np.mean(np.abs(bc1 - bc2))
        bc_mae_list.append(bc_mae)
        
    return bc_mae_list



def train(model, netD, train_adj, train_labels, val_adj, val_labels, args,epochs, alpha=0.9):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    clip_value = 0.01
    criterion = nn.MSELoss()
    model = model.to(device)
    netD = netD.to(device)

    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    all_epochs_loss = []
    patience = 10
    patience_counter = 0
    best_val_loss = float('inf')
    print("device in train",device)
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    for epoch in range(epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []

            model.train()
            netD.train()
            for lr, hr in zip(train_adj, train_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                padded_hr = pad_HR_adj(hr, args.padding)
                padded_hr = torch.from_numpy(padded_hr).type(torch.FloatTensor).to(device)
                eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                error = criterion(model_outputs, padded_hr)

                fake_data = model_outputs.detach()
                real_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)
                dc_loss = d_fake - d_real

                dc_loss.backward()
                optimizerD.step()

                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                d_fake = netD(model_outputs)
                gen_loss = d_fake

                generator_loss = alpha * mse_loss + (1 - alpha) * gen_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(mse_loss.item())
                epoch_error.append(error.item())

            print(f"Epoch: {epoch}, Loss: {np.mean(epoch_loss)}, Error: {np.mean(epoch_error) * 100}%")
            all_epochs_loss.append(np.mean(epoch_loss))

            # Validation phase
            model.eval()
            netD.eval()
            val_loss = []
            with torch.no_grad():
                for lr, hr in zip(val_adj, val_labels):
                    lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                    padded_hr = pad_HR_adj(hr, args.padding)
                    padded_hr = torch.from_numpy(padded_hr).type(torch.FloatTensor).to(device)

                    model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                        lr, args.lr_dim, args.hr_dim)

                    mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                        model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)
                    val_loss.append(mse_loss.item())

            mean_val_loss = np.mean(val_loss)
            print(f"Epoch: {epoch}, Validation Loss: {mean_val_loss}")

          

    return all_epochs_loss
def test(model, test_adj, test_labels, args, size_dim=268):
    g_t = []
    test_error = []
    preds_list = []
    preds_vectors = []
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    for lr, hr in zip(test_adj, test_labels):
        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)
        if all_zeros_lr == False and all_zeros_hr == False:
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
            preds, _, _, _ = model(lr, args.lr_dim, args.hr_dim)
            preds[preds < 0] = 0  # set negative values to 0
            
            # Crop the output to the same size as the ground truth
            preds = preds[preds.shape[0] - size_dim:preds.shape[0], preds.shape[1] - size_dim:preds.shape[1]]
            preds = preds.to("cpu")
            preds_vector = MatrixVectorizer.vectorize(preds.detach().numpy())
            preds_vectors.append(preds_vector)
            preds_list.append(preds.flatten().detach().numpy())
            preds = preds.to("cuda")
            error = criterion(preds, hr)
            g_t.append(hr.flatten())
            test_error.append(error.item())

    print("Test error MSE: ", np.mean(test_error))
    test_array = np.vstack(preds_vectors).flatten()
    id_list = list(range(1, test_array.shape[0] + 1))
    output_array = {
        'ID': id_list,
        'Predicted': test_array
    }

    output_df = pd.DataFrame(output_array)

    return output_df, preds_list
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, min_epochs=20):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if epoch < self.min_epochs:
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def train_with_early_stopping(model, netD, train_adj, train_labels, val_adj, val_labels, args, alpha=0.9):
    clip_value = 0.01
    criterion = nn.MSELoss()

    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)

    all_epochs_loss = []
    early_stopping = EarlyStopping(patience=10, min_delta=0, min_epochs=10)

    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []

            model.train()
            netD.train()
            for lr, hr in zip(train_adj, train_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                padded_hr = pad_HR_adj(hr, args.padding)
                padded_hr = torch.from_numpy(padded_hr).type(torch.FloatTensor).to(device)
                eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                error = criterion(model_outputs, padded_hr)

                fake_data = model_outputs.detach()
                real_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)
                dc_loss = d_fake - d_real

                dc_loss.backward()
                optimizerD.step()

                for p in netD.parameters():
                    p.data.clamp_(-clip_value, clip_value)

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                d_fake = netD(model_outputs)
                gen_loss = d_fake

                generator_loss = alpha * mse_loss + (1 - alpha) * gen_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(mse_loss.item())
                epoch_error.append(error.item())

            print(f"Epoch: {epoch}, Loss: {np.mean(epoch_loss)}, Error: {np.mean(epoch_error) * 100}%")
            all_epochs_loss.append(np.mean(epoch_loss))

            # Validation phase
            model.eval()
            netD.eval()
            val_loss = []
            with torch.no_grad():
                for lr, hr in zip(val_adj, val_labels):
                    lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                    padded_hr = pad_HR_adj(hr, args.padding)
                    padded_hr = torch.from_numpy(padded_hr).type(torch.FloatTensor).to(device)

                    model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                        lr, args.lr_dim, args.hr_dim)

                    mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                        model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)
                    val_loss.append(mse_loss.item())

            mean_val_loss = np.mean(val_loss)
            print(f"Epoch: {epoch}, Validation Loss: {mean_val_loss}")

            early_stopping(mean_val_loss, epoch)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break

    return all_epochs_loss, epoch

# Data paths
# file_paths_lr = ['data/lr_clusterA.csv', 'data/lr_clusterB.csv', 'data/lr_clusterC.csv']
file_paths_lr = ['data/lr_split_1.csv', 'data/lr_split_2.csv', 'data/lr_split_3.csv']
dataframes = [pd.read_csv(file_path) for file_path in file_paths_lr]
concatenated_df = pd.concat(dataframes, ignore_index=True)
lr_train = concatenated_df

# file_paths_hr = ['data/hr_clusterA_modified.csv', 'data/hr_clusterB_modified.csv', 'data/hr_clusterC_modified.csv']
file_paths_hr = ['data/hr_split_1.csv', 'data/hr_split_2.csv', 'data/hr_split_3.csv']
dataframes = [pd.read_csv(file_path) for file_path in file_paths_hr]
concatenated_df = pd.concat(dataframes, ignore_index=True)
hr_train = concatenated_df

# Model parameters
parser = argparse.ArgumentParser(description='AGSR-Net')
args = parser.parse_args([])
args.epochs = 150
args.lr = 0.0001
args.lmbda = 0.1
args.lr_dim = 160
args.hr_dim = 320
args.hidden_dim = 320
args.padding = 26
args.mean_dense = 0.
args.std_dense = 0.01
args.mean_gaussian = 0.
args.std_gaussian = 0.1
ks = [0.9, 0.7, 0.6, 0.5]
alpha = 0.9  # relative strength of mse vs. generator loss

# Check data dimensions
num_rows = lr_train.shape[0]
num_cols = lr_train.shape[1]

print("LR Number of rows:", num_rows)
print("LR Number of columns:", num_cols)

num_rows = hr_train.shape[0]
num_cols = hr_train.shape[1]

print("HR Number of rows:", num_rows)
print("HR Number of columns:", num_cols)

lr_train_new = lr_train.apply(MatrixVectorizer.anti_vectorize, args=(160,), axis=1)
hr_train_new = hr_train.apply(MatrixVectorizer.anti_vectorize, args=(268,), axis=1)

print(lr_train_new[0].shape)
print(hr_train_new.shape)

# Keep the original data separate
original_lr = lr_train_new.copy()
original_hr = hr_train_new.copy()
print("len original hr",len(original_hr))

# Set random seed for reproducibility
np.random.seed(42)

# Data augmentations
lr_augmented_data = []
hr_augmented_data = []

for i, (lr_data, hr_data) in enumerate(zip(lr_train_new, hr_train_new)):
    lr_enhanced_data = [
        edge_perturbation(lr_data, perturb_rate=0.05),
        subgraph_sampling(lr_data, sample_size=120),
        graph_interpolation(lr_data, lr_train_new[(i + 1) % len(lr_train_new)], alpha=0.5)
    ]
    lr_augmented_data.append(random.choice(lr_enhanced_data))
    hr_augmented_data.append(hr_data)

lr_augmented_data = pd.Series(lr_augmented_data)
hr_augmented_data = pd.Series(hr_augmented_data)

lr_train_new_augmented = pd.concat([lr_train_new, lr_augmented_data]).reset_index(drop=True)
hr_train_new_augmented = pd.concat([hr_train_new, hr_augmented_data]).reset_index(drop=True)

# Run 3-fold CV
k_folds = 3
kf = KFold(n_splits=k_folds, shuffle=False)

# Train model with cross-validation
subjects_adj = lr_train_new_augmented
subjects_ground_truth = hr_train_new_augmented

def memory_usage_psutil():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / float(2 ** 20)
    return mem

memory_before = memory_usage_psutil()
predicted_lists = []
all_folds_loss = []
start_time = time.time()
i = 0

print("-----stage1---------")
max_epochs_list=[]
for train_index, val_index in kf.split(subjects_adj):
    model = AGSRNet(ks, args).to(device)
    netD = Discriminator(args).to(device)

    train_adj, val_adj = [subjects_adj[i] for i in train_index], [subjects_adj[i] for i in val_index]
    train_ground_truth, val_ground_truth = [subjects_ground_truth[i] for i in train_index], [subjects_ground_truth[i] for i in val_index]

    # Split the training data into two parts
    split_idx = len(train_adj) // 2
    train_adj1, train_adj2 = train_adj[:split_idx], train_adj[split_idx:]
    train_ground_truth1, train_ground_truth2 = train_ground_truth[:split_idx], train_ground_truth[split_idx:]

    # Use the last 20 samples from each part for validation
    val_adj = np.concatenate((train_adj1[-20:], train_adj2[-20:]), axis=0)
    val_ground_truth = np.concatenate((train_ground_truth1[-20:], train_ground_truth2[-20:]), axis=0)
    train_adj = np.concatenate((train_adj1[:-20], train_adj2[:-20]), axis=0)
    train_ground_truth = np.concatenate((train_ground_truth1[:-20], train_ground_truth2[:-20]), axis=0)

    epoch_loss, max_epoch = train_with_early_stopping(model, netD, train_adj, train_ground_truth, val_adj, val_ground_truth, args, alpha=alpha)
    all_folds_loss.append(epoch_loss)
    max_epochs_list.append(max_epoch)

    output_df, preds_list = test(model, val_adj, val_ground_truth, args)
    predicted_lists.append(preds_list)
    output_df.to_csv(f'predictions_fold_{i}.csv', index=False)
    i += 1

print("-----stage2---------")
ne = max(max_epochs_list)
print("ne:",ne)
for train_index, val_index in kf.split(subjects_adj):
    model = AGSRNet(ks, args).to(device)
    netD = Discriminator(args).to(device)

    train_adj, val_adj = [subjects_adj[i] for i in train_index], [subjects_adj[i] for i in val_index]
    train_ground_truth, val_ground_truth = [subjects_ground_truth[i] for i in train_index], [subjects_ground_truth[i] for i in val_index]

    epoch_loss = train(model, netD, train_adj, train_ground_truth,val_adj,val_ground_truth, args,ne, alpha=alpha)
    all_folds_loss.append(epoch_loss)

    output_df, preds_list = test(model, val_adj, val_ground_truth, args)
    predicted_lists.append(preds_list)
    output_df.to_csv(f'predictions_fold_{i}.csv', index=False)
    i += 1

memory_after = memory_usage_psutil()
end_time = time.time()
execution_time = end_time - start_time
memory_difference = memory_after - memory_before

# Plotting the loss curves for each fold
plt.figure(figsize=(10, 6))
for fold, loss in enumerate(all_folds_loss):
    plt.plot(loss, label=f'Fold {fold+1}')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Curves')
plt.legend()
plt.show()

# Ensure correct shape for evaluation
print("len before true",len(original_hr))
true_hr_matrices = np.stack(original_hr.apply(lambda x: np.array(x)).values)
print("true shape",true_hr_matrices.shape)
predicted_hr_matrices_list = []
matrix_size=268
# Combine all predicted lists into a single 3D array
for fold_preds_list in predicted_lists:
    for pred in fold_preds_list:
        predicted_hr_matrices_list.append(pred.reshape(matrix_size, matrix_size))

predicted_hr_matrices = np.array(predicted_hr_matrices_list[:len(original_lr)])

print("true.shape:", true_hr_matrices.shape)
print("pred.shape:", predicted_hr_matrices.shape)
from evaluation import evaluate_all
evaluate_all(true_hr_matrices, predicted_hr_matrices, output_path='ID_random_cv_2.csv')



