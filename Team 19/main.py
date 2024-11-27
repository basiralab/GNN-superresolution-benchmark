from sklearn.model_selection import KFold
import time
import random
from memory_profiler import profile
import pandas as pd
import scipy.io
from scipy.io import loadmat

from MatrixVectorizer import *
from utils import *
from model import *
from train import *
from evaluation_measures import *

# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

"""
Main utility files
"""
class Argument():
    """
    Argument class to parse hyperparameters
    Please change the args during hyperparameter tuning
    """
    def __init__(self):
        super(Argument, self).__init__()
        self.epochs = 200
        self.lr_d = 2e-4
        self.lr_g = 4e-4
        self.splits = 3
        self.lmbda = 16
        self.lr_dim = 160
        self.hr_dim = 320
        self.hidden_dim = 100
        self.padding = 26
        self.mean_dense = 0.0
        self.std_dense = 0.01
        self.mean_gaussian = 0.0
        self.std_gaussian = 0.1
        self.dropout = 0
        self.ks = [0.9, 0.7, 0.6, 0.5]
        self.data_path = './data/' # path of input csv files

def convert_csv_to_mat(data_path = './'):
    """
    Convert CSV files to MATLAB (.mat) format and save.

    Args:
        data_path (str): The path to the directory containing CSV files. (Default is current directory)
    """
    # Read the input csv files given the data_path
    hr_train = pd.read_csv(data_path + 'hr_train.csv')
    lr_train = pd.read_csv(data_path + 'lr_train.csv')
    lr_test = pd.read_csv(data_path + 'lr_test.csv')

    # Save the .mat files
    scipy.io.savemat(data_path + 'hr_train.mat', {'data': hr_train})
    scipy.io.savemat(data_path + 'lr_train.mat', {'data': lr_train})
    scipy.io.savemat(data_path + 'lr_test.mat', {'data': lr_test})

def load_data(data_path = './'):
    """
    Load data from CSV files converted to MATLAB (.mat) format.

    Args:
        data_path (str): The path to the directory containing the .mat files. (Default is current directory)

    Returns:
        lr_train_vector (pd.DataFrame): Vector containing data from 'lr_train.mat'.
        hr_train_vector (pd.DataFrame): Vector containing data from 'hr_train.mat'.
        lr_test_vector (pd.DataFrame): Vector containing data from 'lr_test.mat'.
    """
    convert_csv_to_mat(data_path) # load input csv files and store in .mat format

    # Load .mat into pd
    lr_train_data = loadmat(data_path + 'lr_train.mat')
    hr_train_data = loadmat(data_path + 'hr_train.mat')
    lr_test_data = loadmat(data_path + 'lr_test.mat')

    lr_train_vector = lr_train_data['data']
    hr_train_vector = hr_train_data['data']
    lr_test_vector = lr_test_data['data']

    return lr_train_vector, hr_train_vector, lr_test_vector

def construct_dataset(lr_train_vector, hr_train_vector, lr_test_vector):
    """
    Construct numpy dataset from vectorized adjacency matrices.

    Args:
        lr_train_vector (pd.DataFrame): LR adjacency matrices for training.
        hr_train_vector (pd.DataFrame): HR adjacency matrices for training.
        lr_test_vector (pd.DataFrame): LR adjacency matrices for testing.

    Returns:
        lr_train_matrix_all_np (numpy.ndarray): Numpy array of size (n, 160, 160) containing anti-vectorized LR adjacency matrices for training.
        hr_train_matrix_all_np (numpy.ndarray): Numpy array of size (n, 268, 268) containing anti-vectorized HR adjacency matrices for training.
        lr_test_matrix_all_np (numpy.ndarray): Numpy array of size (n, 160, 160) containing anti-vectorized LR adjacency matrices for testing.
    """
    # Lists to store all anti-vectorized graphs
    lr_train_matrix_all = []
    hr_train_matrix_all = []
    lr_test_matrix_all = []

    # Anti-vectorize the adj matrix and append to a list
    for i in range(lr_train_vector.shape[0]):
        lr_train_matrix_all.append(MatrixVectorizer.anti_vectorize(lr_train_vector[i,:], 160, include_diagonal=False))
        hr_train_matrix_all.append(MatrixVectorizer.anti_vectorize(hr_train_vector[i,:], 268, include_diagonal=False))

    for i in range(lr_test_vector.shape[0]):
        lr_test_matrix_all.append(MatrixVectorizer.anti_vectorize(lr_test_vector[i,:], 160, include_diagonal=False))
    
    # Convert list to Numpy
    lr_train_matrix_all_np = np.array(lr_train_matrix_all)
    hr_train_matrix_all_np = np.array(hr_train_matrix_all)
    lr_test_matrix_all_np = np.array(lr_test_matrix_all)

    return lr_train_matrix_all_np, hr_train_matrix_all_np, lr_test_matrix_all_np

def cross_validation_3folds(args, train_adj, train_labels, cv):
    """
    Perform 3-fold cross-validation on the given data.

    Args:
        args (object): Input arguments.
        train_adj (numpy.ndarray): Array of adjacency matrices for training.
        train_labels (numpy.ndarray): Array of labels for training.
        cv (object): Cross-validation generator.
    
    Returns:
        measures_folds (list): Evaluation measures for each fold.
    """
    fold_index = 0
    measures_folds = []

    for train_index, test_index in cv.split(train_adj):
        fold_index += 1
        model = IWASAGSRNet(args)
        subjects_adj, val_adj, subjects_ground_truth, val_ground_truth = train_adj[
            train_index], train_adj[test_index], train_labels[train_index], train_labels[test_index]
        train(model, subjects_adj, subjects_ground_truth, args)
        
        pred_matrices, gt_matrices = test(model, val_adj, val_ground_truth, args)

        # Calculate and plot the measures for each fold (MAE, PCC, JSD, MAE(EC), MAE(BC))
        measures = calculate_measures(val_adj.shape[0], val_ground_truth.shape[1], pred_matrices, gt_matrices)
        plot_measures(measures, fold_index)
        measures_folds.append(measures)

        # Save prediction results for every fold
        fold_outputs = pred_matrices.reshape(-1) 
        save_predictions(fold_outputs, fold_index)
        print("========== Predictions Saved into CSV! ==========")

    return measures_folds

def predict_test(test_adj, args, model):
    """
    Predict the output for test adjacency matrices using the given model.

    Args:
        test_adj (numpy.ndarray): Array of test adjacency matrices.
        args (object): Input arguments.
        model (object): Model for prediction.

    Returns:
        total_outputs (numpy.ndarray): Predicted outputs for test adjacency matrices.
    """
    # Set the model to evaluation mode
    model.eval()
    output_list = []

    # Forward pass for prediction
    with torch.no_grad():
        for lr in test_adj:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            outputs, _, _ = model(lr, args.lr_dim)
            outputs, _, _ = model(lr, args.lr_dim)
            outputs = unpad(outputs, args.padding)
            outputs = MatrixVectorizer.vectorize(outputs)
            meltedDF = outputs.flatten()
            output_list.append(meltedDF)
    
    # Stack output lists and reshape into a 1D array
    total_outputs = np.stack(output_list)
    total_outputs = total_outputs.reshape(-1)
    return total_outputs

def save_predictions(total_outputs, fold_num):
    """
    Save predictions into a CSV file with the required format.

    Args:
        total_outputs (numpy.ndarray): Predicted outputs.
        fold_num (int): Fold number for naming the CSV file.
    """

    # Create a DataFrame with ID and Predicted columns
    df = pd.DataFrame({'ID': range(1, len(total_outputs)+1), 'Predicted': total_outputs})
    # Save DataFrame to a CSV file
    df.to_csv(f'predictions_fold_{fold_num}.csv', index=False)

# @profile
def main():
    args = Argument()

    print("Please Specify the Data Path of the Input CSV Files! (currently using the default path under ./data)")
    print("======== Loading Data... ========")
    lr_train_vector, hr_train_vector, lr_test_vector = load_data(args.data_path)
    train_adj, train_labels, test_adj = construct_dataset(lr_train_vector, hr_train_vector, lr_test_vector)
    print("======== Data Loaded! ========")
    
    print("========== Start 3 Folds Cross Validation ==========")
    cv = KFold(n_splits=args.splits, shuffle=True, random_state = random_seed)

    start_time = time.time() # timer for cv run time
    measures_folds = cross_validation_3folds(args, train_adj, train_labels, cv)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total training time for 3F-CV: {elapsed_time} seconds")

    # Plot measures across all folds
    mean_values = np.mean(measures_folds, axis=0)  
    std_dev_values = np.std(measures_folds, axis=0)
    plot_measures(avg = True, mean_values = mean_values, std_dev_values = std_dev_values)

    # Plot MAE(pc) measure
    mae_pc_folds = [measures[3] for measures in measures_folds]
    plot_mae_pc(mae_pc_folds, mean_value = mean_values[3], std_dev_value = std_dev_values[3])
    
    print("========== Generate Final Model with All Training Samples ==========")
    final_model = IWASAGSRNet(args)
    train(final_model, train_adj, train_labels, args)
    final_output = predict_test(test_adj, args, final_model)
    save_predictions(final_output, 'all')

if __name__ == "__main__":
    main()