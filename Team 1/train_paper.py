import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from utils import *
from model import *
from processing import *

criterion = nn.L1Loss()

def train(model, subjects_adj, subjects_labels, args, val_adj = None, val_labels = None):
    """
    Trains the model on the provided dataset and optionally evaluates it on a validation set. Tracks the training and validation loss and error.

    This function performs the training of the model using the Adam optimizer and a specified
    loss criterion. It supports dynamic learning rate adjustments and validation performance 
    monitoring. The function outputs the training loss and error, as well as the validation 
    mean absolute error (MAE), if validation data is provided.

    Parameters:
    - model (torch.nn.Module): The model to be trained.
    - subjects_adj (List[torch.Tensor]): List of adjacency matrices for training subjects.
    - subjects_labels (List[torch.Tensor]): List of high-resolution adjacency matrices (ground truth) for training subjects.
    - args (Namespace): Training hyperparameters (includes 'epochs', 'lr', and 'padding').
    - val_adj (List[torch.Tensor], optional): List of adjacency matrices for validation subjects.
    - val_labels (List[torch.Tensor], optional): List of high-resolution adjacency matrices (ground truth) for validation subjects.

    Returns:
    None, but prints the training and validation performance metrics at each epoch.
    """

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # Early stopping parameters
    best_val_mae = float('inf')
    patience = 10
    patience_scheduler = 5
    counter = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=patience_scheduler, factor=0.5, verbose=True, min_lr=1e-5)

    all_epochs_loss = []
    lr_schedule = []  # Store the learning rate schedule
    best_epoch = 0  # Store the best epoch
    all_epochs_val_loss = []
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            
            for i, (lr, hr) in enumerate(zip(subjects_adj, subjects_labels)):
                optimizer.zero_grad()

                padded_hr = pad_HR_adj(hr, args.padding)
                
                _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs = model(lr)

                mae_loss =  criterion(model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)
                
                error_test = torch.clone(model_outputs)
                error_test[error_test < 0] = 0
                error = criterion(error_test, padded_hr)
                
                mae_loss.backward()
                optimizer.step()

                epoch_loss.append(mae_loss.item())
                epoch_error.append(error.item())

            all_epochs_loss.append(np.mean(epoch_loss))
            
            if val_adj is not None and val_labels is not None:
                epoch_val_loss = []
                model.eval()
                for i, (lr, hr) in enumerate(zip(val_adj, val_labels)):
                    padded_hr = pad_HR_adj(hr, args.padding)
                    val_outputs = model(lr)
                    _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')
                    val_loss = criterion(model.layer.weights, U_hr) + criterion(val_outputs, padded_hr)
                    epoch_val_loss.append(val_loss.item()) 
                
                #val_outputs = compute_output_hr(args, val_adj, model)
                #val_metrics = calculate_metrics(val_labels, val_outputs, args)
                #val_mae = val_metrics[0]
                val_mae = np.mean(epoch_val_loss)
                all_epochs_val_loss.append(val_mae)
            
                
                model.train()
                
                print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                      "Error: ", np.mean(epoch_error)*100, "%", "Validation MAE: ", val_mae)

                # Learning rate scheduling
                prev_lr = optimizer.param_groups[0]['lr']
                scheduler.step(val_mae)
                current_lr = optimizer.param_groups[0]['lr']
                lr_schedule.append(current_lr)  # Store the current learning rate
                if current_lr < prev_lr:
                    print(f"Learning rate reduced at epoch {epoch} from {prev_lr} to {current_lr}")
                    # Load the best model weights after learning rate reduction
                    model.load_state_dict(torch.load('best_model.pth'))
                    print(f"Loaded best model weights from epoch {epoch - patience_scheduler}")
                    # Shift the learning rate schedule back by patience_scheduler epochs
                    lr_schedule = lr_schedule[:-patience_scheduler] + [current_lr] * patience_scheduler
                else:
                    lr_schedule.append(current_lr)  # Store the current learning rate
                 # Early stopping logic
                # Save the model weights if the validation MAE improves
                if val_mae < best_val_mae:
                    best_val_mae = val_mae
                    best_epoch = epoch
                    counter = 0
                    torch.save(model.state_dict(), 'best_model.pth')
                else:
                    counter += 1
                    if counter >= patience and epoch > 100:
                        print(f"Early stopping at epoch {epoch}")
                        # Load the best model weights before returning
                        model.load_state_dict(torch.load('best_model.pth'))
                        return all_epochs_loss, all_epochs_val_loss, best_epoch, lr_schedule
            else:
                print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                    "Error: ", np.mean(epoch_error)*100, "%")
    return all_epochs_loss, all_epochs_val_loss, best_epoch, lr_schedule


def retrain_model(model, subjects_adj, subjects_labels, args, lr_schedule, best_epoch):
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    all_epochs_loss = []

    for epoch in range(best_epoch + 1):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []

            for i, (lr, hr) in enumerate(zip(subjects_adj, subjects_labels)):
                optimizer.zero_grad()

                padded_hr = pad_HR_adj(hr, args.padding)

                _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                model_outputs = model(lr)

                mae_loss = criterion(model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                error_test = torch.clone(model_outputs)
                error_test[error_test < 0] = 0
                error = criterion(error_test, padded_hr)

                mae_loss.backward()
                optimizer.step()

                epoch_loss.append(mae_loss.item())
                epoch_error.append(error.item())

            all_epochs_loss.append(np.mean(epoch_loss))

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error) * 100, "%")

            # Set the learning rate according to the saved schedule
            if epoch < len(lr_schedule):
                prev_lr = optimizer.param_groups[0]['lr']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_schedule[epoch]
                current_lr = optimizer.param_groups[0]['lr']
                if current_lr < prev_lr:
                    print(f"Learning rate reduced at epoch {epoch} from {prev_lr} to {current_lr}")

    return all_epochs_loss
                