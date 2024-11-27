from utils.reproducibility import device
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from preprocessing import pad_HR_adj, unpad
from utils.MatrixVectorizer import MatrixVectorizer


mse = nn.MSELoss()
mae = nn.L1Loss()


def train(model, optimizer, X, Y, args, epochs, val_callback=None, val_data = None, early_stopping=None):
    optimizer = optimizer(model.parameters(), lr=args.lr, weight_decay=0.001)
    model.train()
    train_losses = []
    train_mse = []
    train_mae = []
    validation_mse = []
    validation_mae = []
    if val_data is not None:
        X_val, Y_val = val_data

    max_epoch = 0
    for epoch in range(epochs):
        losses = []
        error_mse = []
        error_mae = []

        for lr, hr in zip(X, Y):
            lr = torch.from_numpy(lr).float().to(device)
            hr = torch.from_numpy(hr).float().to(device)

            # add small noises to the input
            lr = lr + 0.01 * torch.randn(lr.shape).to(device)

            # forward pass
            model_outputs, net_outs, start_gcn_outs, layer_outs = model(lr)
            model_outputs = unpad(model_outputs, args.padding)

            # calculate eigenvectors of padded high resolution graph
            padded_hr = pad_HR_adj(hr.cpu(), args.padding).to(device)
            eig_val_hr, U_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            # calculate loss
            loss = args.lmbda * mse(net_outs, start_gcn_outs) + mse(model.gsr.weights, U_hr) + mse(model_outputs, hr)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            error_mse.append(mse(model_outputs, hr).item())
            error_mae.append(mae(model_outputs, hr).item())

        train_losses.append(np.mean(losses))
        train_mse.append(np.mean(error_mse))
        train_mae.append(np.mean(error_mae))

        print(f"Epoch: {epoch+1}, Loss: {train_losses[-1]:.6f}, MSE: {train_mse[-1]:.6f}, MAE: {train_mae[-1]:.6f}")

        if (epoch + 1) % 10 == 0 and val_callback is not None:
            val_mse, val_mae, _, _ = val_callback()
            validation_mse.append(val_mse)
            validation_mae.append(val_mae)
        # Validate
        if val_data is not None:
            val_mse, val_mae, _, _ = validate(model, X_val, Y_val, args)
            validation_mse.append(val_mse)
            validation_mae.append(val_mae)

            if early_stopping:
                early_stopping(val_mae, epoch)
                if early_stopping.early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break


    return {
        'train_losses': train_losses,
        'train_mse': train_mse,
        'train_mae': train_mae,
        'validation_mse': validation_mse,
        'validation_mae': validation_mae,
        'max_epoch':max_epoch
    }


def validate(model, X, Y, args):
    model.eval()
    error_mse = []
    error_mae = []
    preds_list = []
    gt_list = []

    for lr, hr in zip(X, Y):
        lr = torch.from_numpy(lr).float().to(device)
        hr = torch.from_numpy(hr).float().to(device)

        # forward pass
        with torch.no_grad():
            preds, _, _, _ = model(lr)
        preds = unpad(preds, args.padding)

        error_mse.append(mse(preds, hr).item())
        error_mae.append(mae(preds, hr).item())
        preds_list.append(preds.cpu().numpy())
        gt_list.append(hr.cpu().numpy())

    mean_mse = np.mean(error_mse)
    mean_mae = np.mean(error_mae)
    preds_list = np.array(preds_list)
    gt_list = np.array(gt_list)

    print(f"Val MSE: {mean_mse:.6f}, MAE: {mean_mae:.6f}")
    return mean_mse, mean_mae, preds_list, gt_list


def predict_kaggle(model, X, args, filename='submission'):
    '''
    Predict high resolution graphs using the provided model and save to the csv file in the format required by kaggle.
    '''
    preds_list = []
    model.eval()
    for lr in X:
        lr = torch.from_numpy(lr).type(torch.FloatTensor).to(device)
        with torch.no_grad():
            preds, _, _, _ = model(lr)
        preds = unpad(preds, args.padding)
        preds = MatrixVectorizer.vectorize(preds.cpu().numpy())
        preds_list.append(preds)

    preds_list = np.array(preds_list).flatten()
    preds_list = np.clip(preds_list, 0, 1)

    preds_df = pd.DataFrame(preds_list, columns=['Predicted'], index=np.arange(1, len(preds_list) + 1))
    preds_df.index.name = 'ID'

    os.makedirs('output/', exist_ok=True)
    preds_df.to_csv(f'output/{filename}.csv')
    print(f'Saved predictions to "output/{filename}.csv"')
