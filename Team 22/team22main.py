import torch
from torch.utils.data import Dataset
import numpy as np
import random
import pandas as pd
import argparse
import time
import os

from MatrixVectorizer import MatrixVectorizer
from preprocessing import *
from evaluation import evaluate_all
from train import train, test

def parse_args():
    parser = argparse.ArgumentParser(description="AGSR Model Training and Evaluation")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--lr", type=float, default=0.0001, help="learning rate")
    parser.add_argument("--step_size", type=int, default=50, help="StepLR step size")
    parser.add_argument("--gamma", type=float, default=0.5, help="StepLR gamma")
    args = parser.parse_args()
    return args


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        print("CUDA is available. GPU.")
        return torch.device("cuda")
    else:
        print("CUDA not available. CPU.")
        return torch.device("cpu")


class SymmetricMatrixVectorizer:
    """Handles vectorization and devectorization of symmetric matrices."""

    def __init__(self):
        self.vectorizer = MatrixVectorizer()

    def vectorize(self, matrix):
        """Vectorize a symmetric matrix"""
        matrix_np = matrix.numpy()  # Convert to NumPy array for vectorization
        return self.vectorizer.vectorize(matrix_np)

    def devectorize(self, vector, size):
        """Devectorize into a symmetric matrix"""
        if isinstance(vector, torch.Tensor):
            vector = vector.numpy()  # Ensure the vector is a NumPy array
        matrix_np = self.vectorizer.anti_vectorize(vector, size)
        return torch.tensor(matrix_np, dtype=torch.float)


def load_data(file_path):
    """Load dataset from a CSV file."""
    return pd.read_csv(file_path).values


def tensor_conversion(data, dimension, vectorizer):
    """Convert list of matrices to a PyTorch tensor."""
    tensor_list = [vectorizer.devectorize(x, dimension) for x in data]
    tensor_data = torch.stack(tensor_list)  # Use torch.stack to create a 3D tensor from the list
    return tensor_data


def normalisation(adj):
    # Add self-loops to the adjacency matrix
    adj_with_self_loops = adj + torch.eye(adj.shape[0], device=adj.device)
    # Calculate the degree matrix (with added self-loops), and then compute its inverse square root
    D_hat_inv_sqrt = torch.diag(1.0 / torch.sqrt(torch.sum(adj_with_self_loops, axis=0)))
    # Apply normalization
    adj_norm = D_hat_inv_sqrt @ adj_with_self_loops @ D_hat_inv_sqrt
    return adj_norm


class Data(Dataset):
    def __init__(self, features, labels=None, device="cuda"):
        """
        Initializes the dataset.
        Args:
            features (Tensor): The features of the dataset.
            labels (Tensor, optional): The labels of the dataset. Defaults to None.
            device (str, optional): The device to which the tensors will be transferred. Defaults to 'cpu'.
        """
        self.features = features.clone().detach().to(device).float()
        self.labels = None if labels is None else labels.clone().detach().to(device).float()

    def __len__(self):
        """Returns the number of items in the dataset."""
        return len(self.features)

    def __getitem__(self, idx):
        """
        Returns the feature and label at the specified index.
        Args:
            idx (int): The index of the item.
        Returns:
            tuple: A tuple containing the feature and label at the specified index.
        """
        label = self.labels[idx] if self.labels is not None else None
        return self.features[idx], label


def prepare_datasets(X_train, y_train, X_val, y_val):
    """Prepare training and validation datasets."""
    if X_val is None:
        return {"train": Data(X_train, y_train)}

    return {"train": Data(X_train, y_train), "val": Data(X_val, y_val)}


class AGSR:
    def __init__(self, epochs, lr, step_size, gamma):
        self.epochs = epochs
        self.lr = lr
        self.step_size = step_size
        self.gamma = gamma
        self.lmbda = 0.1
        self.lr_dim = 160
        self.hr_dim = 320
        self.hidden_dim = 320
        self.padding = 26
        self.mean_dense = 0.0
        self.std_dense = 0.01
        self.mean_gaussian = 0.0
        self.std_gaussian = 0.1

        # Model setup
        kernel_sizes = [0.9, 0.7, 0.6, 0.5]
        self.model = AGSRNet(kernel_sizes, self)
        self.model_path = f"data/agsr_model_{epochs}_{lr}_{step_size}_{gamma}.pth"

    def save_model(self):
        """Save the model to the specified path."""
        torch.save(self.model.state_dict(), self.model_path)

    def load_model(self):
        """Load the model from the specified path."""
        self.model.load_state_dict(torch.load(self.model_path))

    def train(self, lr_vectors, hr_vectors, val_lr_vectors, val_hr_vectors):
        """Train the model and save it."""
        start_time = time.time()
        best_epoch = train(self.model, lr_vectors, hr_vectors, val_lr_vectors, val_hr_vectors, self)

        end_time = time.time()
        training_duration = end_time - start_time

        print("Time duration", training_duration)
        self.save_model()

        return best_epoch

    def predict(self, lr_vectors):
        """Load the model and predict high-resolution vectors from low-resolution inputs."""
        self.load_model()
        self.model.eval()
        with torch.no_grad():
            predicted_hr_vectors = test(self.model, lr_vectors, self)
        return predicted_hr_vectors


def data_preprocessing(data_path):
    """Load data and cleanse by replacing negative and NaN values with 0."""
    data = pd.read_csv(data_path)
    data = np.maximum(data, 0)  # Ensures all negative values are set to 0
    data = np.nan_to_num(data)  # Replaces NaNs with 0 and returns a numpy array
    return data


def vectorize_data(data):
    """Vectorize the data using a MatrixVectorizer and process it for prediction."""
    vectorized_data = [MatrixVectorizer.anti_vectorize(row, 160) for row in data]
    return np.array(vectorized_data)


def process_predictions(predictions):
    """Vectorize predictions, flatten the array, and prepare submission DataFrame."""
    vectorized_predictions = np.array([MatrixVectorizer.vectorize(pred) for pred in predictions])
    flattened_predictions = vectorized_predictions.flatten()
    predictions_for_csv = pd.DataFrame(
        {
            "ID": np.arange(1, len(flattened_predictions) + 1),
            "Predicted": flattened_predictions,
        }
    )
    return predictions_for_csv


def save_metrics_to_csv(metrics, filename):
    """Save metrics to a CSV file."""
    df = pd.DataFrame(metrics)
    df.to_csv(filename, index=False)

def load_split_data(folder_path, split_number):
    """Load data from the specified split folder."""
    lr_path = os.path.normpath(f"{folder_path}\\Split{split_number}\\lr_cluster{split_number}.csv")
    hr_path = os.path.normpath(f"{folder_path}\\Split{split_number}\\hr_cluster{split_number}.csv")
    print(f"Loading {os.path.abspath(lr_path)} and {os.path.abspath(hr_path)}")
    lr_data = load_data(lr_path)
    hr_data = load_data(hr_path)
    return lr_data, hr_data

def load_test_data(folder_path):
    """Load test data from the specified folder."""
    lr_path = os.path.normpath(f"{folder_path}\\lr.csv")
    hr_path = os.path.normpath(f"{folder_path}\\hr.csv")
    print(f"Loading {os.path.abspath(lr_path)} and {os.path.abspath(hr_path)}")
    lr_data = load_data(lr_path)
    hr_data = load_data(hr_path)
    return lr_data, hr_data

def main():
    args = parse_args()
    random_seed = 42
    set_seeds(random_seed)

    print("Cell done loading")

    # Initialize vectorizer
    vectorizer = SymmetricMatrixVectorizer()

    # Folder containing the splits
    data_folder = os.path.normpath("data\\Train\\Train")

    # Print current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Initialize AGSR model parameters
    num_splits = 3

    # Store the fold results and the best number of epochs
    fold_results = []
    best_epochs = []

    # Load data from splits
    split_data = []
    for i in range(1, num_splits + 1):
        print(f"Loading data from Split {i}")
        lr_data, hr_data = load_split_data(data_folder, i)
        split_data.append((lr_data, hr_data))

    # Perform cross-validation with Leave-One-Out strategy
    for current_fold in range(num_splits):
        print(f"Fold {current_fold + 1}: ")
        
        # Separate test fold and training folds
        test_lr_data, test_hr_data = split_data[current_fold]
        train_lr_data = np.concatenate([split_data[i][0] for i in range(num_splits) if i != current_fold])
        train_hr_data = np.concatenate([split_data[i][1] for i in range(num_splits) if i != current_fold])
        
        # Prepare validation data (last 20 samples from each training split)
        val_lr_data = np.concatenate([split_data[i][0][-20:] for i in range(num_splits) if i != current_fold])
        val_hr_data = np.concatenate([split_data[i][1][-20:] for i in range(num_splits) if i != current_fold])

        # Vectorize the training and testing data
        train_input_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 160) for x in train_lr_data])
        test_input_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 160) for x in test_lr_data])
        train_output_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 268) for x in train_hr_data])
        test_output_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 268) for x in test_hr_data])
        val_input_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 160) for x in val_lr_data])
        val_output_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 268) for x in val_hr_data])

        # Train the model with early stopping
        fold_model = AGSR(args.epochs, args.lr, args.step_size, args.gamma)
        best_epoch = fold_model.train(train_input_matrices, train_output_matrices, val_input_matrices, val_output_matrices)
        best_epochs.append(best_epoch)

        # Evaluate the model on the test set and log the results
        predicted_test_output_matrices = fold_model.predict(test_input_matrices)
        metrics = evaluate_all(test_output_matrices, predicted_test_output_matrices)
        fold_results.append(metrics)
        # Save metrics to CSV
        save_metrics_to_csv(metrics, f"cross_validation_fold_{current_fold + 1}_results_cluster.csv")

    # Output the fold results
    for fold_idx, result in enumerate(fold_results):
        print(f"Fold {fold_idx + 1} Results: {result}")

    # Stage 2: Final Model Training
    final_epochs = int(np.mean(best_epochs))  # Average best epochs from the cross-validation

    # Combine all the training data from the splits
    combined_train_lr_data = np.concatenate([split_data[i][0] for i in range(num_splits)])
    combined_train_hr_data = np.concatenate([split_data[i][1] for i in range(num_splits)])

    combined_train_input_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 160) for x in combined_train_lr_data])
    combined_train_output_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 268) for x in combined_train_hr_data])

    # Train the final model
    final_model = AGSR(final_epochs, args.lr, args.step_size, args.gamma)
    final_model.train(combined_train_input_matrices, combined_train_output_matrices, None, None)

    # Load the test data
    test_folder = os.path.normpath("data\\Test\\Test")
    print("Loading test data")
    test_lr_data, test_hr_data = load_test_data(test_folder)

    # Vectorize the test data
    test_input_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 160) for x in test_lr_data])
    test_output_matrices = np.array([MatrixVectorizer.anti_vectorize(x, 268) for x in test_hr_data])

    # Evaluate the final model on the test data
    predicted_test_output_matrices = final_model.predict(test_input_matrices)
    final_metrics = evaluate_all(test_output_matrices, predicted_test_output_matrices)

    print(f"Final Test Results: {final_metrics}")
    # Save final test metrics to CSV
    save_metrics_to_csv(final_metrics, "final_test_results_cluster.csv")

if __name__ == "__main__":
    main()
