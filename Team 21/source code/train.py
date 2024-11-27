"""
Modified code based on AGSR-Net (https://github.com/basiralab/AGSR-Net) by Basira Labs.
Original licensing terms apply.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from provided_code.MatrixVectorizer import MatrixVectorizer
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.swa_utils import AveragedModel
import psutil
import GPUtil
import time
from dataset import *
from model import *
from utils import *


def train(args):
    """Function to train the model and return the trained models, predictions, and ground truth"""
    memory_start = psutil.virtual_memory().used
    memory_running = 0
    if GPUtil.getGPUs():
        gpu_memory_start = GPUtil.getGPUs()[0].memoryUsed
    gpu_memory_running = 0

    time_start = time.time()

    # Create a 3-fold cross validation
    subjects_adj, subjects_ground_truth, test_adj = dataset()
    kf = KFold(n_splits=3, shuffle=True, random_state=args['random_seed'])

    models = []
    swa_models = []
    preds = []
    trues = []
    preds_swa = []
    trues_swa = []

    for fold, (train_index, val_index) in enumerate(kf.split(subjects_adj)):
        print()
        print(f'Fold: {fold+1}')

        # Set random seed for reproducibility
        seed_everything(args['random_seed'])

        # Preprocess the dataset (min-max scaling)
        subjects_adj, subjects_ground_truth, test_adj = preprocess_dataset(subjects_adj, subjects_ground_truth, test_adj, train_index)

        train_dataset = BrainDataset(subjects_adj[train_index], subjects_ground_truth[train_index], args)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)

        # Create our models
        model = GraphLoong(args).to(args['device'])
        swa_model = AveragedModel(model)
        netD = Discriminator(args).to(args['device'])

        # Create a loss function
        criterion = nn.MSELoss()
        d_criterion = nn.BCELoss()
        # Create optimizers
        optimizerG = torch.optim.Adam(model.parameters(), lr=args['lr'])
        optimizerD = torch.optim.Adam(netD.parameters(), lr=args['lr'])
        schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args['epochs'], eta_min=args['eta_min'])
        schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=args['epochs'], eta_min=args['eta_min'])

        for epoch in range(args['epochs']):
            # Train the model
            model.train()
            swa_model.train()
            for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
                data = data[0].float().fill_diagonal_(1).to(args['device'])
                target = target[0].float().fill_diagonal_(1).to(args['device'])

                # Forward + Backward + Optimize
                output, net_outs, start_gcn_outs, net_2_outs = model(data)
                output = unpad(output, args['padding'])

                # Pad everything to 320x320
                padded_target = pad_HR_adj(target.cpu(), args['padding']).to(args['device'])
                eig_val_hr, U_hr = torch.linalg.eigh(padded_target, UPLO='U')

                # Loss calculation
                unet_loss = criterion(net_outs, start_gcn_outs)
                unet2_loss = criterion(net_2_outs, data)
                layer_loss = criterion(model.layer.weights, U_hr)
                output_loss = criterion(output, target)
                mse_loss = args['lambda'] * (unet_loss + unet2_loss) + output_loss + layer_loss

                # Discriminator
                is_real = torch.ones([args['hr_dim']-2*args['padding'], 1]).to(args['device'])
                is_fake = torch.zeros([args['hr_dim']-2*args['padding'], 1]).to(args['device'])
                real_data = gaussian_noise_layer(target, args)
                fake_data = output.detach()

                netD.train()
                d_real = netD(real_data)
                d_fake = netD(fake_data)

                d_loss = - d_criterion(d_real, is_fake) - d_criterion(d_fake, is_real)
                d_loss.backward()

                if (batch_idx+1) % args['batch_size'] == 0:
                    optimizerD.step()
                    optimizerD.zero_grad()

                # Loss for Generator
                netD.requires_grad_(False)
                d_fake = netD(output)
                g_loss = d_criterion(d_fake, is_real)
                loss = mse_loss + g_loss
                loss.backward()
                netD.requires_grad_(True)

                # Gradient accumulation
                if (batch_idx+1) % args['batch_size'] == 0:
                    optimizerG.step()
                    if epoch > args['epochs'] * 0.8:
                        # Update the SWA model
                        swa_model.update_parameters(model)
                    optimizerG.zero_grad()
            
            optimizerD.step()
            optimizerG.step()
            if epoch > args['epochs'] * 0.8:
                # Update the SWA model
                swa_model.update_parameters(model)
            optimizerD.zero_grad()
            optimizerG.zero_grad()

            schedulerG.step()
            schedulerD.step()

            # Track memory usage
            memory_running += psutil.virtual_memory().used - memory_start
            if GPUtil.getGPUs():
                gpu_memory_running += GPUtil.getGPUs()[0].memoryUsed - gpu_memory_start

        # Save models
        models.append(model.state_dict())
        swa_models.append(swa_model.state_dict())

        train_mse, train_mae, _, _ = test(model, subjects_adj[train_index], subjects_ground_truth[train_index], args) # Train MSE and MAE
        train_mse_swa, train_mae_swa, _, _ = test(swa_model, subjects_adj[train_index], subjects_ground_truth[train_index], args) # Train MSE and MAE using SWA model
        print()
        print('Train MSE: ', train_mse)
        print('Train MAE: ', train_mae)
        print('Train MSE SWA: ', train_mse_swa)
        print('Train MAE SWA: ', train_mae_swa)

        val_mse, val_mae, pred, true = test(model, subjects_adj[val_index], subjects_ground_truth[val_index], args, save=f'./predictions/predictions_fold_{fold+1}.csv') # Validation MSE and MAE
        val_mse_swa, val_mae_swa, pred_swa, true_swa = test(swa_model, subjects_adj[val_index], subjects_ground_truth[val_index], args, save=f'./predictions/predictions_swa_fold_{fold+1}.csv') # Validation MSE and MAE using SWA model
        print()
        print('Val MSE: ', val_mse)
        print('Val MAE: ', val_mae)
        print('Val MSE SWA: ', val_mse_swa)
        print('Val MAE SWA: ', val_mae_swa)

        preds.append(pred)
        trues.append(true)
        preds_swa.append(pred_swa)
        trues_swa.append(true_swa)

    memory_running = memory_running /1024**3 /args['epochs'] /3
    gpu_memory_running = gpu_memory_running /1024 /args['epochs'] /3
    time_total = (time.time() - time_start) /60
    print()
    print('Total memory usage:\t{:.2f} GB'.format(memory_running))
    print('Total GPU memory usage:\t{:.2f} GB'.format(gpu_memory_running))
    print('Total time:\t\t{:.2f} minutes'.format(time_total))
    
    return models, swa_models, preds, trues, preds_swa, trues_swa


def test(model, test_adj, test_label, args, save=''):
    """Function to evaluate the model and return the mean squared error, mean absolute error, predictions, and ground truth"""
    vectorizer = MatrixVectorizer()

    model.eval()
    predictions = []
    preds = []
    trues = []
    test_mse = []
    test_mae = []
    for data, target in zip(test_adj, test_label):
        data = torch.from_numpy(data).float().fill_diagonal_(1).to(args['device'])
        np.fill_diagonal(target, 1)
        output, _, _, _ = model(data)
        output = unpad(output, args['padding'])
        predictions.append(vectorizer.vectorize(output.squeeze(0).squeeze(0).detach().cpu().numpy()))
        preds.append(output.detach().cpu().numpy())
        trues.append(target)
        test_mse.append(np.mean((output.detach().cpu().numpy() - target)**2))
        test_mae.append(np.mean(np.abs(output.detach().cpu().numpy() - target)))

    if save:
        flatten_predictions = np.array(predictions).flatten()
        submission = pd.DataFrame({'ID': np.arange(1, len(flatten_predictions)+1), 'Predicted': flatten_predictions})
        submission.to_csv(save, index=False)

    return np.mean(test_mse), np.mean(test_mae), preds, trues


def evaluate(model, args, file):
    vectorizer = MatrixVectorizer()
    
    # Create a dataset
    subjects_adj, subjects_ground_truth, _ = dataset()

    # Generate predictions for the train data
    model.eval()
    predictions = []
    for idx, (data, target) in enumerate(zip(subjects_adj, subjects_ground_truth)):
        data = torch.from_numpy(data).float().fill_diagonal_(1).to(args['device'])
        np.fill_diagonal(target, 1)
        output, _, _, _ = model(data)
        output = unpad(output, args['padding'])
        vectorized_output = vectorizer.vectorize(output.squeeze(0).squeeze(0).detach().cpu().numpy())
        predictions.append(vectorized_output)

    reference = pd.read_csv('./data/hr_train.csv')
    reference = np.array(reference)
    predictions = np.array(predictions)
    # Calculate the mean squared error
    mse = np.mean((predictions - reference)**2)
    print("Train MSE: ", mse)
    # Calculate the mean absolute error
    mae = np.mean(np.abs(predictions - reference))
    print("Train MAE: ", mae)

    # Plot the heatmaps of predictions and the ground truth
    if file == 'swa_1':
        plt.figure(figsize=(10, 5))
        submission_matrix = vectorizer.anti_vectorize(predictions[0], 268)
        reference_matrix = vectorizer.anti_vectorize(reference[0], 268)
        # Predicted HR Heatmap
        plt.subplot(1, 2, 1)
        ax = sns.heatmap(submission_matrix, vmin=0, vmax=1, square=True, cmap='viridis', xticklabels=50, yticklabels=50)
        ax.set_title('Predicted HR Data Heatmap (268x268)')
        # HR Data Heatmap
        plt.subplot(1, 2, 2)
        ax = sns.heatmap(reference_matrix, vmin=0, vmax=1, square=True, cmap='viridis', xticklabels=50, yticklabels=50)
        ax.set_title('HR Data Heatmap (268x268)')
        plt.tight_layout()
        plt.savefig(f'./images/heatmap_{file}_{0}.png')
        plt.close()


def predict(model, args, file):
    vectorizer = MatrixVectorizer()
    
    # Create a dataset
    _, _, test_adj = dataset()

    # Generate submission for the test data
    model.eval()
    predictions = []
    for data in test_adj:
        data = torch.from_numpy(data).float().to(args['device'])
        output, _, _, _ = model(data)
        output = unpad(output, args['padding'])
        vectorized_output = vectorizer.vectorize(output.squeeze(0).squeeze(0).detach().cpu().numpy())
        predictions.append(vectorized_output)

    flatten_predictions = np.array(predictions).flatten()
    submission = pd.DataFrame({'ID': np.arange(1, len(flatten_predictions)+1), 'Predicted': flatten_predictions})
    submission.to_csv(f'./submissions/submission_{file}.csv', index=False)


def best_train(args):
    "Function to train the model using all the data and return the trained models"
    subjects_adj, subjects_ground_truth, test_adj = dataset()

    # Set random seed for reproducibility
    seed_everything(args['random_seed'])

    # Preprocess the dataset (min-max scaling)
    subjects_adj, subjects_ground_truth, test_adj = preprocess_dataset(subjects_adj, subjects_ground_truth, test_adj, np.arange(len(subjects_adj)))

    train_dataset = BrainDataset(subjects_adj, subjects_ground_truth, args)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, persistent_workers=True)

    # Create a model
    model = GraphLoong(args).to(args['device'])
    swa_model = AveragedModel(model)
    netD = Discriminator(args).to(args['device'])

    # Create a loss function
    criterion = nn.MSELoss()
    d_criterion = nn.BCELoss()
    # Create an optimizer
    optimizerG = torch.optim.Adam(model.parameters(), lr=args['lr'])
    optimizerD = torch.optim.Adam(netD.parameters(), lr=args['lr'])
    schedulerG = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerG, T_max=args['epochs'], eta_min=args['eta_min'])
    schedulerD = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerD, T_max=args['epochs'], eta_min=args['eta_min'])

    for epoch in range(args['epochs']):
        # Train the model
        model.train()
        swa_model.train()
        for batch_idx, (data, target) in enumerate(tqdm(train_dataloader)):
            data = data[0].float().fill_diagonal_(1).to(args['device'])
            target = target[0].float().fill_diagonal_(1).to(args['device'])

            # Forward + Backward + Optimize
            output, net_outs, start_gcn_outs, net_2_outs = model(data)
            output = unpad(output, args['padding'])

            # Pad everything to 320x320
            padded_target = pad_HR_adj(target.cpu(), args['padding']).to(args['device'])
            eig_val_hr, U_hr = torch.linalg.eigh(padded_target, UPLO='U')

            # Loss calculation
            unet_loss = criterion(net_outs, start_gcn_outs)
            unet2_loss = criterion(net_2_outs, data)
            layer_loss = criterion(model.layer.weights, U_hr)
            output_loss = criterion(output, target)
            mse_loss = args['lambda'] * (unet_loss + unet2_loss) + output_loss + layer_loss

            # Discriminator
            is_real = torch.ones([args['hr_dim']-2*args['padding'], 1]).to(args['device'])
            is_fake = torch.zeros([args['hr_dim']-2*args['padding'], 1]).to(args['device'])
            real_data = gaussian_noise_layer(target, args)
            fake_data = output.detach()

            netD.train()
            d_real = netD(real_data)
            d_fake = netD(fake_data)

            d_loss = - d_criterion(d_real, is_fake) - d_criterion(d_fake, is_real)
            d_loss.backward()

            if (batch_idx+1) % args['batch_size'] == 0:
                optimizerD.step()
                optimizerD.zero_grad()

            # Loss for Generator
            netD.requires_grad_(False)
            d_fake = netD(output)
            g_loss = d_criterion(d_fake, is_real)
            loss = mse_loss + g_loss
            loss.backward()
            netD.requires_grad_(True)

            if (batch_idx+1) % args['batch_size'] == 0:
                optimizerG.step()
                if epoch > args['epochs'] * 0.8:
                    swa_model.update_parameters(model)
                optimizerG.zero_grad()
        
        optimizerD.step()
        optimizerG.step()
        if epoch > args['epochs'] * 0.8:
            swa_model.update_parameters(model)
        optimizerD.zero_grad()
        optimizerG.zero_grad()

        schedulerG.step()
        schedulerD.step()

    train_mse, train_mae, _, _ = test(model, subjects_adj, subjects_ground_truth, args)
    train_mse_swa, train_mae_swa, _, _ = test(swa_model, subjects_adj, subjects_ground_truth, args)
    print()
    print('Train MSE: ', train_mse)
    print('Train MAE: ', train_mae)
    print('Train MSE SWA: ', train_mse_swa)
    print('Train MAE SWA: ', train_mae_swa)
    
    return model.state_dict(), swa_model.state_dict()