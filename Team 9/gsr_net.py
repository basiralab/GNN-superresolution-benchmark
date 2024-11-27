# %% [markdown]
# ### Set Up
# Before running all codes at once, please make sure all directory settings in this code match with your environment. You can ctr+f "directory" to find all directory setting that you need to change

# %%
import pandas as pd
import numpy as np
import random
import psutil
import time 
import torch
import torch.nn as nn

import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, train_test_split
from torch_geometric.data import Data

import networkx as nx
from concurrent.futures import ProcessPoolExecutor, as_completed
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler


class MatrixVectorizer:
    """
    A class for transforming between matrices and vector representations.
    
    This class provides methods to convert a symmetric matrix into a vector (vectorize)
    and to reconstruct the matrix from its vector form (anti_vectorize), focusing on 
    vertical (column-based) traversal and handling of elements.
    """

    def __init__(self):
        """
        Initializes the MatrixVectorizer instance.
        
        The constructor currently does not perform any actions but is included for 
        potential future extensions where initialization parameters might be required.
        """
        pass

    @staticmethod
    def vectorize(matrix, include_diagonal=False):
        """
        Converts a matrix into a vector by vertically extracting elements.
        
        This method traverses the matrix column by column, collecting elements from the
        upper triangle, and optionally includes the diagonal elements immediately below
        the main diagonal based on the include_diagonal flag.
        
        Parameters:
        - matrix (numpy.ndarray): The matrix to be vectorized.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the vectorization.
          Defaults to False.
        
        Returns:
        - numpy.ndarray: The vectorized form of the matrix.
        """
        # Determine the size of the matrix based on its first dimension
        matrix_size = matrix.shape[0]

        # Initialize an empty list to accumulate vector elements
        vector_elements = []

        # Iterate over columns and then rows to collect the relevant elements
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Collect upper triangle elements
                        vector_elements.append(matrix[row, col])
                    elif include_diagonal and row == col + 1:
                        # Optionally include the diagonal elements immediately below the diagonal
                        vector_elements.append(matrix[row, col])

        return np.array(vector_elements)

    @staticmethod
    def anti_vectorize(vector, matrix_size, include_diagonal=False):
        """
        Reconstructs a matrix from its vector form, filling it vertically.
        
        The method fills the matrix by reflecting vector elements into the upper triangle
        and optionally including the diagonal elements based on the include_diagonal flag.
        
        Parameters:
        - vector (numpy.ndarray): The vector to be transformed into a matrix.
        - matrix_size (int): The size of the square matrix to be reconstructed.
        - include_diagonal (bool, optional): Flag to include diagonal elements in the reconstruction.
          Defaults to False.
        
        Returns:
        - numpy.ndarray: The reconstructed square matrix.
        """
        # Initialize a square matrix of zeros with the specified size
        matrix = np.zeros((matrix_size, matrix_size))

        # Index to keep track of the current position in the vector
        vector_idx = 0

        # Fill the matrix by iterating over columns and then rows
        for col in range(matrix_size):
            for row in range(matrix_size):
                # Skip diagonal elements if not including them
                if row != col:  
                    if row < col:
                        # Reflect vector elements into the upper triangle and its mirror in the lower triangle
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1
                    elif include_diagonal and row == col + 1:
                        # Optionally fill the diagonal elements after completing each column
                        matrix[row, col] = vector[vector_idx]
                        matrix[col, row] = vector[vector_idx]
                        vector_idx += 1

        return matrix

# %%
def load_data_as_graphs(df, matrix_size):
    graphs = []
    for index, row in df.iterrows():
        vector = row.values
        adjacency_matrix = MatrixVectorizer.anti_vectorize(vector, matrix_size)
        graph = Data(adjacency_matrix=np.array(adjacency_matrix))
        graphs.append(graph)
    return graphs

def get_matrix_size(num_features):
    return int(np.sqrt(num_features * 2)) + 1

def weight_variable_glorot(output_dim):
    input_dim = output_dim
    init_range = np.sqrt(6.0 / (input_dim + output_dim))
    initial = np.random.uniform(-init_range, init_range,(input_dim, output_dim))
    return initial

# Normalize the adjacency matrix of the graph
def normalize_adj_torch(mx):
    rowsum = mx.sum(1)
    r_inv_sqrt = torch.pow(rowsum, -0.5).flatten()
    r_inv_sqrt[torch.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = torch.diag(r_inv_sqrt)
    mx = torch.matmul(mx, r_mat_inv_sqrt)
    mx = torch.transpose(mx, 0, 1)
    mx = torch.matmul(mx,  r_mat_inv_sqrt)
    return mx

# %%
#GAT
class GAT(nn.Module):
    """ 
    This layer applies an attention mechanism in the graph convolution process,
    allowing the model to focus on different parts of the neighborhood
    of each node.
    """
    def __init__(self, in_features, out_features, activation=F.relu):
        """        
        Parameters:
            in_features (int): The number of features of each input node.
            out_features (int): The number of features for each output node.
            activation (callable, optional): The activation function to use. Default is F.relu.
        """

        super(GAT, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.phi = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        self.activation = activation
        self.reset_parameters()
        self.drop = nn.Dropout(p=0.5)
 
    def reset_parameters(self):
        """
        Initializes or resets the parameters of the layer.
        """

        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
 
        stdv = 1. / np.sqrt(self.phi.size(1))
        self.phi.data.uniform_(-stdv, stdv)
 
    def forward(self,adj,input):
        """
        Forward pass of the GAT layer.
 
        Parameters:
        input (Tensor): The input features of the nodes.
        adj (Tensor): The adjacency matrix of the graph.
 
        Returns:
        Tensor: The output features of the nodes after applying the GAT layer.
        """
        
        input = self.drop(input)
        h = torch.mm(input, self.weight) + self.bias 
 
        N = input.size(0) 
        h_expand = h.unsqueeze(1).expand(N, N, -1)
        h_t_expand = h.unsqueeze(0).expand(N, N, -1)
        
        concat_features = torch.cat([h_expand, h_t_expand], dim=-1)
        
        S = torch.matmul(concat_features, self.phi).squeeze(-1)
 
        mask = (adj.to(device) + torch.eye(adj.size(0),device=device)).bool()
        S_masked = torch.where(mask, S, torch.tensor(-9e15, dtype=S.dtype).to(device))
        attention_weights = F.softmax(S_masked, dim=1)
        h = torch.matmul(attention_weights, h)
        return self.activation(h) if self.activation else h
    
class GraphUnpool(nn.Module):
    """    
    This layer "unpools" a graph to a larger graph based on the provided indices.
    """
    def __init__(self):
        super(GraphUnpool, self).__init__()

    def forward(self, A, X, idx):
        """        
        Parameters:
            A (Tensor): The adjacency matrix of the smaller graph.
            X (Tensor): The node features of the smaller graph.
            idx (Tensor): The indices of nodes in the original graph.
        
        Returns:
            Tensor, Tensor: The adjacency matrix and node features of the unpooled graph.
        """

        new_X = torch.zeros([A.shape[0], X.shape[1]]).to(device)
        new_X[idx] = X
        return A, new_X

    
class GraphPool(nn.Module):
    """    
    This layer pools a graph based on the learned scores for each node, reducing the number of nodes in the graph.
    """

    def __init__(self, k, in_dim):
        super(GraphPool, self).__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A, X):
        scores = self.proj(X)
        scores = torch.squeeze(scores)
        scores = self.sigmoid(scores/100)
        num_nodes = A.shape[0]
        values, idx = torch.topk(scores, int(self.k*num_nodes))
        new_X = X[idx, :]
        values = torch.unsqueeze(values, -1)
        new_X = torch.mul(new_X, values)
        A = A[idx, :]
        A = A[:, idx]
        return A, new_X, idx


class GraphUnet(nn.Module):
    """    
    This model combines the GAT layers with graph pooling and unpooling layers to create an architecture for graphs.
    """

    def __init__(self, ks, in_dim, out_dim, dim=268):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.start_gat = GAT(in_dim, dim).to(device)
        self.bottom_gat = GAT(dim, dim).to(device)
        self.end_gat = GAT(2*dim, out_dim).to(device)
        self.down_gats = []
        self.up_gats = []
        self.pools = []
        self.unpools = []
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gats.append(GAT(dim, dim).to(device))
            self.up_gats.append(GAT(dim, dim).to(device))
            self.pools.append(GraphPool(ks[i], dim).to(device))
            self.unpools.append(GraphUnpool().to(device))

    def forward(self, A, X):
        adj_ms = []
        indices_list = []
        down_outs = []
        X = self.start_gat(A, X)
        start_gat_outs = X
        org_X = X
        for i in range(self.l_n):
            X = self.down_gats[i](A, X)
            adj_ms.append(A)
            down_outs.append(X)
            A, X, idx = self.pools[i](A, X)
            indices_list.append(idx)
        
        X = self.bottom_gat(A, X)
        
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            A, idx = adj_ms[up_idx], indices_list[up_idx]
            A, X = self.unpools[i](A, X, idx)
            X = self.up_gats[i](A, X)
            X = X.add(down_outs[up_idx])
        X = torch.cat([X, org_X], 1)
        X = self.end_gat(A, X)
        
        return X, start_gat_outs


class GSRLayer(nn.Module):
    """    
    This layer aims to learn a high-resolution representation of a graph from its low-resolution counterpart.
    """

    def __init__(self,hr_dim):
        super(GSRLayer, self).__init__()
        
        self.hr_dim = hr_dim
        self.weights = torch.from_numpy(weight_variable_glorot(hr_dim)).type(torch.FloatTensor).to(device)
        self.weights = torch.nn.Parameter(data=self.weights, requires_grad = True).to(device)

    def forward(self,A,X):
        lr = A
        lr_dim = lr.shape[0]
        hr_dim = self.hr_dim
        f = X
        eig_val_lr, U_lr = torch.linalg.eigh(lr, UPLO='U')
        eye_mat = torch.eye(lr_dim).type(torch.FloatTensor).to(device)
        s_d = torch.cat((eye_mat, torch.ones(hr_dim - lr_dim, lr_dim).to(device)), dim=0)
        a = torch.matmul(self.weights,s_d)
        b = torch.matmul(a ,torch.t(U_lr))
        f_d = torch.matmul(b ,f)
        f_d = torch.abs(f_d)
        self.f_d = f_d.fill_diagonal_(1)
        adj = normalize_adj_torch(self.f_d)
        X = torch.mm(adj, adj.t())
        X = (X + X.t())/2
        idx = torch.eye(268, dtype=bool)
        X[idx]=1
        return adj, torch.abs(X)
    

class GraphConvolution(nn.Module):
    """    
    This layer applies a graph convolution operation on the graph nodes.
    """

    def __init__(self, in_features, out_features, dropout=0.5, act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = torch.nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.mm(adj, support)
        output = self.act(output)
        return output
    
    
class GSRNet(nn.Module):
    """
    This network combines graph convolutional layers with a Graph U-Net structure and a super-resolution layer to enhance the resolution of graph data.
    """
    
    def __init__(self,ks,args):
        super(GSRNet, self).__init__()

        self.lr_dim = args.lr_dim
        self.hr_dim = args.hr_dim
        self.hidden_dim = args.hidden_dim
        self.layer = GSRLayer(self.hr_dim).to(device)
        self.net = GraphUnet(ks, self.lr_dim, self.hr_dim).to(device)
        self.gc1 = GraphConvolution(self.hr_dim, self.hidden_dim, 0, act=F.relu).to(device)
        self.gc2 = GraphConvolution(self.hidden_dim, self.hr_dim, 0, act=F.relu).to(device)

    def forward(self,lr):
        I = torch.eye(self.lr_dim).type(torch.FloatTensor).to(device)
        A = normalize_adj_torch(lr).type(torch.FloatTensor).to(device)

        self.net_outs, self.start_gat_outs = self.net(A, I)
        self.outputs, self.Z = self.layer(A, self.net_outs)

        self.hidden1 = self.gc1(self.Z, self.outputs)
        self.hidden2 = self.gc2(self.hidden1, self.outputs)

        z = self.hidden2
        z = (z + z.t())/2
        idx = torch.eye(self.hr_dim, dtype=bool) 
        z[idx]=1

        return torch.abs(z), self.net_outs, self.start_gat_outs, self.outputs

# %% [markdown]
# ### Model Training

# %%
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

def get_memory_usage():
    process = psutil.Process()
    return process.memory_info().rss / (1024 ** 2)  # Convert to MB

# %%
criterion = nn.MSELoss()
criterion.to(device)

def create_batches(data, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def train(model, optimizer, subjects_adj, subjects_labels, val_input, val_output, args, early_stopping=True):
    """
    Trains a graph super-resolution network on a dataset of low-resolution and high-resolution adjacency matrices.
    
    Parameters:
        model: The graph super-resolution network to be trained.
        optimizer: The optimizer used to update the model's weights.
        subjects_adj: A list of low-resolution adjacency matrices (the dataset).
        subjects_labels: A list of high-resolution adjacency matrices (the labels).
        val_input: Validation set input matrices.
        val_output: Validation set output matrices.
        args: A namespace or an object containing training parameters such as epochs and batch_size.
    """
    # Initialize training variables
    i = 0
    all_epochs_loss = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    memory_usage_list = []
    batch_size = args.batch_size  # Assuming batch_size is added to Args class

    # Early stopping parameters
    patience = 5  # Number of epochs to wait for improvement
    min_delta = 0.001  # Minimum change to qualify as an improvement

    # Record start time
    start_time = time.time()

    # Training loop
    for epoch in range(args.epochs):
        epoch_loss = []
        epoch_error = []

        # Create batches
        lr_batches = list(create_batches(subjects_adj, batch_size))
        hr_batches = list(create_batches(subjects_labels, batch_size))
        
        for lr_batch, hr_batch in zip(lr_batches, hr_batches):
            model.train()
            optimizer.zero_grad()   

            # Initialize batch_loss as a zero tensor that requires grad
            batch_loss = torch.tensor(0., device=device, requires_grad=True)
            batch_error = 0

            # Process each (lr, hr) pair in the batch
            for lr, hr in zip(lr_batch, hr_batch):
                lr_tensor = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                hr_tensor = torch.from_numpy(hr).type(torch.FloatTensor).to(device)  
                model_outputs, net_outs, start_gat_outs, layer_outs = model(lr_tensor)
                
                hr_p = hr_tensor.cpu()
                eig_val_hr, U_hr = torch.linalg.eigh(hr_p.to(device), UPLO='U')

                # loss function
                loss = args.lmbda1 * criterion(net_outs, start_gat_outs) + criterion(model.layer.weights,U_hr) + args.lmbda2 * criterion(model_outputs, hr_tensor) 
                error = criterion(model_outputs, hr_tensor)

                if batch_loss.grad_fn is not None:
                    batch_loss = batch_loss + loss
                else:
                    batch_loss = loss.detach().requires_grad_()
 
                batch_error += error.item()
            
            loss.backward()
            optimizer.step()

            epoch_loss.append(batch_loss.item())
            epoch_error.append(batch_error / len(lr_batch))  # error
      
        i += 1
        print(f"Epoch: {i}, Loss: {np.mean(epoch_loss)}, Error: {np.mean(epoch_error) * 100}%")
        all_epochs_loss.append(np.mean(epoch_loss))

        # Validation loss calculation
        if early_stopping:
            model.eval()
            with torch.no_grad():
                val_loss = 0
                for val_lr, val_hr in zip(val_input, val_output):
                    val_lr_tensor = torch.from_numpy(val_lr).type(torch.FloatTensor).to(device)
                    val_hr_tensor = torch.from_numpy(val_hr).type(torch.FloatTensor).to(device)
                    val_outputs, _, _, _ = model(val_lr_tensor)
                    val_loss += criterion(val_outputs, val_hr_tensor).item()
                val_loss /= len(val_input)
                val_losses.append(val_loss)
    
            print(f"Validation Loss: {val_loss}")
    
            # Early stopping check
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), args.model_path)  # Save the best model
            else:
                patience_counter += 1
    
            if patience_counter >= patience:
                print(f"Early stopping at epoch {i}")
                break

        # Calculate memory usage in MB
        memory_usage = get_memory_usage()
        memory_usage_list.append(memory_usage)

    end_time = time.time()
    total_training_time = end_time - start_time

    # Load the best model
    model.load_state_dict(torch.load(args.model_path))

    print(f"Total Training Time: {total_training_time:.2f} seconds")
    print(f"\nAverage Memory Usage: {np.mean(memory_usage_list)} MB")


class Args:
    """
    A class to store the configuration settings for the training process.
    
    Attributes:
        epochs (int): Number of epochs to train the model.
        lr (float): Learning rate for the optimizer.
        splits (int): Number of splits for cross-validation.
        lmbda1 (int): Coefficient for the first part of the composite loss function.
        lmbda2 (int): Coefficient for the second part of the composite loss function.
        lr_dim (int): Dimension of the low-resolution graphs.
        hr_dim (int): Dimension of the high-resolution graphs.
        hidden_dim (int): Dimension of the hidden layers in the graph neural network.
        batch_size (int): Number of samples per batch.
        model_path (str): File path to save the trained model.
    """

    def __init__(self):
        self.epochs = 200
        self.lr = 0.0001
        self.splits = 5
        self.lmbda1 = 16
        self.lmbda2 = 2
        self.lr_dim = 160
        self.hr_dim = 268
        self.hidden_dim = 320
        self.batch_size = 2
        self.model_path = "/notebooks/gsr_model.pt" #change directory if needed

args = Args()

# Define the pooling ratios for the Graph U-Net architecture
ks = [0.9, 0.7, 0.6, 0.5]

model = GSRNet(ks, args) 
optimizer = optim.AdamW(model.parameters(), lr=args.lr)

def test(model, test_adj, test_labels, args):
    """    
    Parameters:
        model: The trained GSRNet model.
        test_adj: Numpy array of adjacency matrices for the test dataset (low-resolution).
        test_labels: Numpy array of ground truth adjacency matrices for the test dataset (high-resolution).
        args: Configuration settings that include model and dataset specifications.
    
    Returns:
        pred_matrices: Predicted high-resolution adjacency matrices for the test dataset.
        gt_matrices: Ground truth high-resolution adjacency matrices for the test dataset.
    """

    test_error = []
    preds_list=[]
    g_t = []
    num_test_samples = test_adj.shape[0]
    num_roi = args.hr_dim
    pred_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()
    gt_matrices = torch.randn(num_test_samples, num_roi, num_roi).numpy()

    i = 0
    
    # Iterate over each pair of low-resolution input and high-resolution ground truth
    for lr, hr in zip(test_adj,test_labels):

        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)

        if all_zeros_lr == False and all_zeros_hr==False: #choose representative subject
            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
            np.fill_diagonal(hr,1)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)
            preds,a,b,c = model(lr)

            preds_list.append(preds.flatten().detach().cpu().clone().numpy())
            error = criterion(preds, hr)
            g_t.append(hr.flatten())
            test_error.append(error.item())

            pred_matrices[i] = preds.detach().cpu().clone().numpy()
            gt_matrices[i]   = hr.detach().cpu().clone().numpy()
            i += 1
            
    print ("Test error MSE: ", np.mean(test_error))
    return pred_matrices, gt_matrices