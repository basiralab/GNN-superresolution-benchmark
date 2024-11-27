import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from model import *
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import psutil

# Define the criterion and memory usage function
criterion = nn.MSELoss()
def get_memory_usage():
    process = psutil.Process()
    mem = process.memory_info().rss / float(2 ** 20)  # Memory usage in MB
    return mem

# Train function with modified early stopping logic
def train(model, subjects_adj, subjects_labels, val_subjects_adj, val_subjects_labels, args):
    criterion = nn.MSELoss()
    bce_loss = nn.BCELoss()
    netD = Discriminator(args)
    optimizerG = optim.AdamW(model.parameters(), lr=args.lr)
    optimizerD = optim.AdamW(netD.parameters(), lr=args.lr)

    schedulerG = StepLR(optimizerG, step_size=args.step_size, gamma=args.gamma)
    schedulerD = StepLR(optimizerD, step_size=args.step_size, gamma=args.gamma)

    best_loss = float('inf')
    best_epoch = 0
    patience = 10  # Number of epochs to wait for improvement before early stopping
    patience_counter = 0  # Counter to track the number of epochs since last improvement

    loss_improvement_threshold = 0.0000005
    epoch_losses = []  # List to store average loss per epoch
    val_epoch_losses = []  # List to store average validation loss per epoch

    # Phase 1: Train for 100 epochs unconditionally
    for epoch in range(100):
        model.train()
        epoch_loss = []
        for lr, hr in zip(subjects_adj, subjects_labels):
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            # Preprocess data
            hr = pad_HR_adj(hr, args.padding)
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

            # Perform the forward pass and compute losses
            model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr, args.lr_dim, args.hr_dim)
            mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights, padded_hr) + criterion(model_outputs, padded_hr)
            generator_loss = mse_loss

            # Discriminator losses
            real_data = model_outputs.detach()
            fake_data = gaussian_noise_layer(padded_hr, args)
            d_real = netD(real_data)
            d_fake = netD(fake_data)
            dc_loss_real = bce_loss(d_real, torch.full((args.hr_dim, 1), 0.9))
            dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
            dc_loss = dc_loss_real + dc_loss_fake
            dc_loss.backward()
            optimizerD.step()

            # Generator update
            d_fake = netD(gaussian_noise_layer(padded_hr, args))
            gen_loss = bce_loss(d_fake, torch.full((args.hr_dim, 1), 0.9))
            generator_loss += gen_loss
            generator_loss.backward()
            optimizerG.step()

            epoch_loss.append(dc_loss.item())

        # Learning rate scheduling
        schedulerG.step()
        schedulerD.step()

        # Calculate average loss for the epoch
        avg_loss = np.mean(epoch_loss)
        epoch_losses.append(avg_loss)  # Store average loss for plotting
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Memory Usage: {get_memory_usage()} MB")

        # Validate the model
        model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for lr, hr in zip(val_subjects_adj, val_subjects_labels):
                hr = pad_HR_adj(hr, args.padding)
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

                model_outputs, _, _, _ = model(lr, args.lr_dim, args.hr_dim)
                val_loss = criterion(model_outputs, padded_hr)
                val_epoch_loss.append(val_loss.item())

        avg_val_loss = np.mean(val_epoch_loss)
        val_epoch_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Avg Validation Loss: {avg_val_loss:.6f}")

    # Phase 2: Train with early stopping
    for epoch in range(100, args.epochs):
        model.train()
        epoch_loss = []
        for lr, hr in zip(subjects_adj, subjects_labels):
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            # Preprocess data
            hr = pad_HR_adj(hr, args.padding)
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

            # Perform the forward pass and compute losses
            model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr, args.lr_dim, args.hr_dim)
            mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(model.layer.weights, padded_hr) + criterion(model_outputs, padded_hr)
            generator_loss = mse_loss

            # Discriminator losses
            real_data = model_outputs.detach()
            fake_data = gaussian_noise_layer(padded_hr, args)
            d_real = netD(real_data)
            d_fake = netD(fake_data)
            dc_loss_real = bce_loss(d_real, torch.full((args.hr_dim, 1), 0.9))
            dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
            dc_loss = dc_loss_real + dc_loss_fake
            dc_loss.backward()
            optimizerD.step()

            # Generator update
            d_fake = netD(gaussian_noise_layer(padded_hr, args))
            gen_loss = bce_loss(d_fake, torch.full((args.hr_dim, 1), 0.9))
            generator_loss += gen_loss
            generator_loss.backward()
            optimizerG.step()

            epoch_loss.append(dc_loss.item())

        # Learning rate scheduling
        schedulerG.step()
        schedulerD.step()

        # Calculate average loss for the epoch
        avg_loss = np.mean(epoch_loss)
        epoch_losses.append(avg_loss)  # Store average loss for plotting
        print(f"Epoch {epoch+1}: Avg Loss: {avg_loss:.4f}, Memory Usage: {get_memory_usage()} MB")

        # Validate the model
        model.eval()
        val_epoch_loss = []
        with torch.no_grad():
            for lr, hr in zip(val_subjects_adj, val_subjects_labels):
                hr = pad_HR_adj(hr, args.padding)
                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)

                model_outputs, _, _, _ = model(lr, args.lr_dim, args.hr_dim)
                val_loss = criterion(model_outputs, padded_hr)
                val_epoch_loss.append(val_loss.item())

        avg_val_loss = np.mean(val_epoch_loss)
        val_epoch_losses.append(avg_val_loss)
        print(f"Epoch {epoch+1}: Avg Validation Loss: {avg_val_loss:.6f}")

        # Check for significant improvement
        loss_improvement = best_loss - avg_val_loss
        if loss_improvement > loss_improvement_threshold:
            best_loss = avg_val_loss
            best_epoch = epoch
            patience_counter = 0  # Reset the patience counter
        else:
            patience_counter += 1  # Increment the patience counter

        if patience_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no significant improvement.")
            break

    # Plotting the losses after training
    plt.figure(figsize=(10, 5))
    plt.plot(epoch_losses, label='Training Loss')
    plt.plot(val_epoch_losses, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_plot.png')
    # plt.show()
    return best_epoch




def test(model_instance, test_data, model_parameters):
    predictions = []

    for data_point in test_data:
        is_all_zeros = not np.any(data_point)
        if not is_all_zeros:
            data_point_tensor = torch.from_numpy(data_point).type(torch.FloatTensor)
            prediction_output, _, _, _ = model_instance(data_point_tensor, model_parameters.lr_dim, model_parameters.hr_dim)
            prediction_output = unpad(prediction_output, model_parameters.padding).detach().numpy()
            predictions.append(prediction_output)

    return np.stack(predictions)