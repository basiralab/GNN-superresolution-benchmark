import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def loss_function(pred_hr, hr, pred_lr, lr): 
    """
    Calculates the cross-entropy loss between the predicted and target values.

    Args:
        pred_hr (torch.Tensor): The predicted high-resolution values.
        hr (torch.Tensor): The target high-resolution values.
        pred_lr (torch.Tensor): The predicted low-resolution values.
        lr (torch.Tensor): The target low-resolution values.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The cross-entropy loss between the predicted and target values for high-resolution and low-resolution.
    """
    return F.binary_cross_entropy(pred_hr, hr), F.binary_cross_entropy(pred_lr, lr)


def train(fold, model, optimizer, train_dl, val_dl, device, flags):
    """
    Trains the model for a given fold using the specified optimizer, loss function, and data loaders.

    Args:
        fold (int): The fold number.
        model: The model to be trained.
        optimizer: The optimizer used for training.
        train_dl: The data loader for the training set.
        val_dl: The data loader for the validation set.
        device: The device to run the training on.
        flags: Additional flags and settings.

    Returns:
        The trained model.
    """
    sw = SummaryWriter(f"logs/{fold}_v3")
    counter = 0
    val_mae_coll = []

    best_model = None
    best_val_mae = 1
    val_mae = -1
    best_epoch = -1
    epochs = flags.epochs

    if flags.early_stopping:
        epochs = epochs*4

    
    for epoch in range(epochs):
        epoch_loss = []
        model.train()        
        for data in train_dl:            
            x, edge_index, edge_attr, batch = data.x.to(device).float(), data.edge_index.to(device), data.edge_attr.to(device).float(), data.batch.to(device)
            lr_adj_real, hr_adj_real = (data.ori_matrix + torch.cat([torch.eye(160)]*data.batch_size)).to(device).detach().float(), \
                                        (data.y + torch.cat([torch.eye(268)]*data.batch_size)).to(device).detach().float()

            optimizer.zero_grad()

            hr_adj, lr_adj = model(x, edge_index, edge_attr, batch)
            hr_loss, lr_loss = loss_function(hr_adj, hr_adj_real, lr_adj, lr_adj_real)
            loss = lr_loss + flags.omega*hr_loss
            loss.backward()

            optimizer.step()

            sw.add_scalar("Total Loss", loss, counter)
            sw.add_scalar("LR Loss", lr_loss, counter)
            sw.add_scalar("HR Loss", hr_loss, counter)
            counter += 1  
            epoch_loss.append(loss.item())
        
        if val_dl:
            predictions, ground_truth = evaluate(model, val_dl, device)
            val_mae = F.l1_loss(predictions, ground_truth) 
            val_mae_coll.append(val_mae)            
                    
            sw.add_scalar("HR MAE", val_mae, counter)

            if val_mae < best_val_mae:
                best_epoch = int(epoch + 1)
                best_val_mae = val_mae
                best_model = model.state_dict()
            
            if flags.early_stopping == True and epoch > (flags.patience_epochs-1) and \
            min(val_mae_coll[-flags.patience_epochs:]) > min(val_mae_coll):
                print(f"stopped early at epoch {epoch}")
                break


        print(f"Epoch {epoch} loss: {np.mean(epoch_loss)} | val MAE: {val_mae}")
    if flags.early_stopping:
        model.load_state_dict(best_model)    
    return model, best_epoch


def evaluate(model, val_dl, device):
    """
    Evaluate the performance of a model on the validation dataset.

    Args:
        model (torch.nn.Module): The model to evaluate.
        val_dl (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to perform the evaluation on.

    Returns:
        torch.Tensor: The predicted adjacency matrices.
        torch.Tensor: The ground truth adjacency matrices.
    """
    predictions = []
    ground_truth = []

    model.eval()
    with torch.no_grad():
        for data in val_dl:
            repeat_factor = data.batch_size
            x, edge_index, edge_attr, batch = data.x.to(device).float(), data.edge_index.to(device), data.edge_attr.to(device).float(), data.batch.to(device)
            lr_adj_real, hr_adj_real = (data.ori_matrix + torch.cat([torch.eye(160)]*repeat_factor)).to(device).detach().float(), \
                                        (data.y + torch.cat([torch.eye(268)]*repeat_factor)).to(device).detach().float()
            hr_adj, lr_adj = model(x, edge_index, edge_attr, batch)

            predictions.append(hr_adj.view(-1, 268, 268))
            ground_truth.append(hr_adj_real.view(-1, 268, 268))

        predictions = torch.cat(predictions)
    return predictions, torch.cat(ground_truth)


def predict(model, test_dl, device):
    """
    Predicts the high-resolution adjacency matrix using the given model and test data.

    Args:
        model (torch.nn.Module): The trained model used for prediction.
        test_dl (torch.utils.data.DataLoader): The data loader containing the test data.
        device (torch.device): The device on which the model and data should be loaded.

    Returns:
        torch.Tensor: The predicted high-resolution adjacency matrix.
    """
    predictions = []

    model.eval()
    with torch.no_grad():
        for data in test_dl:
            repeat_factor = data.batch_size
            x, edge_index, edge_attr, batch = data.x.to(device).float(), data.edge_index.to(device), data.edge_attr.to(device).float(), data.batch.to(device)            
            hr_adj, lr_adj = model(x, edge_index, edge_attr, batch)
            predictions.append(hr_adj.view(-1, 268, 268))            

        predictions = torch.cat(predictions)
        # predictions[predictions < 0] = 0
    return predictions
