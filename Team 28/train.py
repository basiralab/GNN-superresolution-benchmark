import torch
import torch.nn as nn
import numpy as np

from preprocessing import *
from model import *
import torch.optim as optim


criterion = nn.MSELoss()

class CustomMSELoss(nn.Module):
    """Custom loss that penalizes predicting edges for true non-edges."""
    def __init__(self, zero_penalty=2.0):
        super(CustomMSELoss, self).__init__()
        self.zero_penalty = zero_penalty

    def forward(self, y_pred, y_true):
        # Calculate squared errors element-wise
        squared_errors = torch.square(y_pred - y_true)

        # Apply the penalty to zero values
        zero_mask = (y_true == 0.0).float()
        loss = squared_errors + zero_mask * self.zero_penalty * squared_errors

        # Compute the mean loss across all entries
        return torch.mean(loss)


def train(model, lr_matrices, hr_matrices, args):
    """
    Train model using the given low-resolution and high-resolution matrices.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - lr_matrices (list of np.ndarray): List of low-resolution matrices.
    - hr_matrices (list of np.ndarray): List of high-resolution matrices.
    - args (dict): Arguments including device, learning rate, etc.
    """
    # Move model to device
    device = args['device']
    model.to(device)

    model.train()

    # Initialize discriminator
    netD = GCNDiscriminator(args).to(device)
    # Optimizer for generator
    optimizerG = optim.Adam(model.parameters(), lr=args['lr'])
    # Optimizer for discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr'])

    # Initialize losses
    bce_loss = nn.BCELoss()
    weighted_mse_loss = CustomMSELoss(zero_penalty=args["zero_penalty"])

    for epoch in range(args['epochs']):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            for lr, hr in zip(lr_matrices, hr_matrices):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                # Pad hr matrices and convert to tensors
                hr = pad_HR_adj(hr, args['padding'])
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                # Eigendecomposition of hr
                eig_val_hr, U_hr= torch.linalg.eigh(padded_hr,UPLO='U')

                # Forward pass
                model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)

                # Calculate loss
                u_net_reconstruction_loss = criterion(net_outs, start_gcn_outs)
                eigen_value_loss = criterion(model.layer.weights, U_hr)
                edge_penalty_loss = weighted_mse_loss(model_outputs, padded_hr)
                mse_loss = args['lmbda'] * u_net_reconstruction_loss + eigen_value_loss + edge_penalty_loss
                error = criterion(model_outputs, padded_hr)

                # Discriminator 
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args).to(device)
                fake_adj = torch.ones_like(layer_outs).to(device)
                d_real = netD(real_data, layer_outs.detach())
                d_fake = netD(fake_data, fake_adj)
                # Discriminator loss
                dc_loss_real = bce_loss(d_real, torch.ones(args['hr_dim'], 1).to(device))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args['hr_dim'], 1).to(device))
                dc_loss = dc_loss_real + dc_loss_fake
                dc_loss.backward()
                optimizerD.step()

                # Generator loss
                d_fake = netD(gaussian_noise_layer(padded_hr, args).to(device), fake_adj)
                gen_loss = bce_loss(d_fake, torch.ones(args['hr_dim'], 1).to(device))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%")


def train_with_early_stopping(model, lr_matrices, hr_matrices,
                              test_lr, gt_matrices, args):
    """
    Train model using the given low-resolution and high-resolution matrices.

    Args:
    - model (torch.nn.Module): The model to be trained.
    - lr_matrices (list of np.ndarray): List of low-resolution matrices.
    - hr_matrices (list of np.ndarray): List of high-resolution matrices.
    - args (dict): Arguments including device, learning rate, etc.
    """
    # Move model to device
    device = args['device']
    model.to(device)

    model.train()

    # Initialize discriminator
    netD = GCNDiscriminator(args).to(device)
    # Optimizer for generator
    optimizerG = optim.Adam(model.parameters(), lr=args['lr'])
    # Optimizer for discriminator
    optimizerD = optim.Adam(netD.parameters(), lr=args['lr'])

    # Initialize losses
    bce_loss = nn.BCELoss()
    weighted_mse_loss = CustomMSELoss(zero_penalty=args["zero_penalty"])

    val_losses = []

    for epoch in range(args['epochs']):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            for lr, hr in zip(lr_matrices, hr_matrices):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                # Pad hr matrices and convert to tensors
                hr = pad_HR_adj(hr, args['padding'])
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor).to(device)

                # Eigendecomposition of hr
                eig_val_hr, U_hr= torch.linalg.eigh(padded_hr,UPLO='U')

                # Forward pass
                model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)

                # Calculate loss
                u_net_reconstruction_loss = criterion(net_outs, start_gcn_outs)
                eigen_value_loss = criterion(model.layer.weights, U_hr)
                edge_penalty_loss = weighted_mse_loss(model_outputs, padded_hr)
                mse_loss = args['lmbda'] * u_net_reconstruction_loss + eigen_value_loss + edge_penalty_loss
                error = criterion(model_outputs, padded_hr)

                # Discriminator
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args).to(device)
                fake_adj = torch.ones_like(layer_outs).to(device)
                d_real = netD(real_data, layer_outs.detach())
                d_fake = netD(fake_data, fake_adj)
                # Discriminator loss
                dc_loss_real = bce_loss(d_real, torch.ones(args['hr_dim'], 1).to(device))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args['hr_dim'], 1).to(device))
                dc_loss = dc_loss_real + dc_loss_fake
                dc_loss.backward()
                optimizerD.step()

                # Generator loss
                d_fake = netD(gaussian_noise_layer(padded_hr, args).to(device), fake_adj)
                gen_loss = bce_loss(d_fake, torch.ones(args['hr_dim'], 1).to(device))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%")

            # Predict on test set
            val_loss = calculate_val_loss(model, test_lr, gt_matrices, netD, args)
            loss = val_loss.item()
            val_losses.append(loss)
            print("Validation Loss: ", loss)

            # Early stopping if epoch greater than 100 and the difference between the
            # last 10 val_losses is less than the early stopping threshold
            if epoch > 100 and np.mean(val_losses[-10:]) - np.mean(val_losses[-11:-1]) < args['early_stopping_threshold']:
                print("Early stopping at epoch: ", epoch)
                break


def predict(model, lr_matrices, args):
    """
    Returns predictions returned from model and low-resolution matrices.

    Args:
    - model (torch.nn.Module): trained model for making predictions.
    - lr_matrices (list of np.ndarray): List of low-resolution matrices.
    - args (dict): Additional arguments including padding information.

    Returns:
        np.ndarray: Array of predictions for each low-resolution matrix.
    """
    preds_matrices = [] # To store predictions

    model.eval()
    with torch.no_grad():
        for lr in lr_matrices:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            preds, _, _, _ = model(lr)
            preds = unpad(preds, args['padding'])
            preds_matrices.append(preds.detach().cpu().numpy())

    return np.array(preds_matrices)


def calculate_val_loss(model, lr_matrices, hr_matrices, netD, args):
    """
    Returns predictions returned from model and low-resolution matrices.

    Args:
    - model (torch.nn.Module): trained model for making predictions.
    - lr_matrices (list of np.ndarray): List of low-resolution matrices.
    - args (dict): Additional arguments including padding information.

    Returns:
        np.ndarray: Array of predictions for each low-resolution matrix.
    """
    preds_matrices = [] # To store predictions
    weighted_mse_loss = CustomMSELoss(zero_penalty=args["zero_penalty"])
    bce_loss = nn.BCELoss()

    model.eval()
    with torch.no_grad():
        for lr, hr in zip(lr_matrices, hr_matrices):
            hr = pad_HR_adj(hr, args['padding'])
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

            # Eigendecomposition of hr
            eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)

            u_net_reconstruction_loss = criterion(net_outs, start_gcn_outs)
            eigen_value_loss = criterion(model.layer.weights, U_hr)
            edge_penalty_loss = weighted_mse_loss(model_outputs, padded_hr)
            mse_loss = args['lmbda'] * u_net_reconstruction_loss + eigen_value_loss + edge_penalty_loss

            # Discriminator
            real_data = model_outputs.detach()
            fake_data = gaussian_noise_layer(padded_hr, args)
            fake_adj = torch.ones_like(layer_outs)
            d_real = netD(real_data, layer_outs.detach())
            d_fake = netD(fake_data, fake_adj)

            # Discriminator loss
            dc_loss_real = bce_loss(d_real,torch.ones(args['hr_dim'], 1))
            dc_loss_fake = bce_loss(d_fake,torch.zeros(args['hr_dim'], 1))
            dc_loss = dc_loss_real + dc_loss_fake

            # Generator loss
            d_fake = netD(gaussian_noise_layer(padded_hr, args),
                          fake_adj)
            gen_loss = bce_loss(d_fake, torch.ones(args['hr_dim'], 1))
            generator_loss = gen_loss + mse_loss

    return generator_loss