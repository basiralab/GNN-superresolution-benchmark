import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import *
from model import *
from layers import *

"""
Define training and testing functions
"""

def train(model, subjects_adj, subjects_labels, args):
    """
    Train the IWAS-AGSRNet model.

    Args:
        model (nn.Module): IWAS-AGSRNet model to be trained.
        subjects_adj (list): Training set.
        subjects_labels (list): Training labels.
        args (object): Arguments object containing hyperparameters.
    """

    # Define criterions to calculate loss
    mse_criterion = nn.MSELoss() # for IWAS-AGSRNet
    bce_criterion = nn.BCELoss() # for Discriminator

    # Initialize Discriminator
    netD = Discriminator(args)

    # Prepare optimizers
    optimizerG = optim.Adam(model.parameters(), lr=args.lr_d, betas=(0.5, 0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr_g, betas=(0.5, 0.999))

    # List for loss in all epochs
    all_epochs_loss = []
    for epoch in range(args.epochs):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            epoch_gen = []
            epoch_mse = []
            epoch_dc = []
            epoch_unet = []

            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerD.zero_grad()
                optimizerG.zero_grad()

                # Process input LR and HR matrices
                hr = pad_HR_adj(hr, args.padding) # pad input HR adj matrix
                # Convert to tensor
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

                # Eigenvector decomposition of HR matrix
                _, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                # Pass through IWAS-AGSRNet to generate outputs
                model_outputs, net_outs, start_gat_outs = model(lr, args.lr_dim)

                # Define MSE loss
                mse_loss = args.lmbda * mse_criterion(net_outs, start_gat_outs) + mse_criterion(
                    model.layer.weights, U_hr) + mse_criterion(model_outputs, padded_hr)

                # Calculate MSE loss between the generated and real HR matrices
                error = mse_criterion(model_outputs, padded_hr)

                # Train Discriminator
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args)

                d_real = netD(real_data)
                d_fake = netD(fake_data)

                dc_loss_real = bce_criterion(d_real, torch.ones(args.hr_dim, 1))
                dc_loss_fake = bce_criterion(d_fake, torch.zeros(args.hr_dim, 1))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                d_fake = netD(gaussian_noise_layer(padded_hr, args))

                gen_loss = bce_criterion(d_fake, torch.ones(args.hr_dim, 1))
                generator_loss = gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

                # Append other losses for monitoring
                unet_loss = args.lmbda * mse_criterion(net_outs, start_gat_outs)
                epoch_gen.append(gen_loss.item())
                epoch_mse.append(mse_loss.item())
                epoch_dc.append(dc_loss.item())
                epoch_unet.append(unet_loss.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%",
                  "UNet Loss: ", np.mean(epoch_unet),
                  "DC Loss: ", np.mean(epoch_dc),
                  "Gen Loss: ", np.mean(epoch_gen),
                  "MSE Loss: ", np.mean(epoch_mse))

            all_epochs_loss.append(np.mean(epoch_loss))

def test(model, test_adj, test_labels, args):
    """
    Make prediction on a test set to generate HR matrix for every test sample.

    Args:
        model (torch.nn.Module): The neural network model to be evaluated.
        test_adj (list): List of adjacency matrices representing the test set.
        test_labels (list): List of ground truth matrices corresponding to the test set.
        args (Argument): An Argument object containing various parameters.

    Returns:
        pred_matrices (numpy.ndarray): Array containing the model predictions for each test sample.
        gt_matrices (numpy.ndarray): Array containing the ground truth matrices for each test sample.
    """
    # set model to evaluation mode
    model.eval()

    # Loss function
    criterion = nn.MSELoss()
    criterion_mae = nn.L1Loss()

    # Initializations
    pred_matrices_list = []
    gt_matrices_list = []
    test_error = []
    test_mae_padded = []
    test_mae_unpadded = []

    with torch.no_grad():
        for lr, hr in zip(test_adj, test_labels):
            # check whether lr and hr are empty matrix
            all_zeros_lr = not np.any(lr)
            all_zeros_hr = not np.any(hr)

            if all_zeros_lr == True or all_zeros_hr == True:
                print(all_zeros_lr, all_zeros_hr)

            if all_zeros_lr == False and all_zeros_hr == False:
                # Pad hr from 268 to 320
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                np.fill_diagonal(hr, 1)
                hr_padded = pad_HR_adj(hr, args.padding)
                hr_padded = torch.from_numpy(hr_padded).type(torch.FloatTensor)

                # Forward pass to make prediction
                preds, _, _ = model(lr, args.lr_dim)

                # Store errors
                error = criterion(preds, hr_padded)
                test_error.append(error.item())

                # Store unpadded predictoins and ground truth for plots
                preds_unpadded = unpad(preds, args.padding)
                hr_unpadded = torch.from_numpy(hr).type(torch.FloatTensor)
                pred_matrices_list.append(preds_unpadded.numpy())
                gt_matrices_list.append(hr_unpadded.numpy())
                
                # Use torch L1 norm to calculate MAE (padded & unpadded)
                test_mae_padded.append(criterion_mae(preds, hr_padded).item())
                test_mae_unpadded.append(criterion_mae(preds_unpadded, hr_unpadded).item())

    pred_matrices = np.stack(pred_matrices_list, axis=0)
    gt_matrices = np.stack(gt_matrices_list, axis=0)

    print("Test error MSE: ", np.mean(test_error))
    print("Test MAE (using L1Loss, padded): ", np.mean(test_mae_padded))
    print("Test MAE (using L1Loss, unpadded): ", np.mean(test_mae_unpadded))
    print("====================================================")

    return pred_matrices, gt_matrices