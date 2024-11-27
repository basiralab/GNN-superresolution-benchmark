import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import numpy as np
import pandas as pd
from utils.metrics import get_metrics


def train_graph_cyclegan(model, train_loader, val_loader=None, lr_G=0.0005, lr_D_A=0.0002, lr_D_B=0.0002, epochs=10,
                         device='cpu', fold_num=0):
    model.to(device)

    losses = []
    es = -1

    # Optimizers
    optimizer_G = optim.Adam(list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()), lr=lr_G,
                             betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(model.D_A.parameters(), lr=lr_D_A, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(model.D_B.parameters(), lr=lr_D_B, betas=(0.5, 0.999))

    # Schedulers
    scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.75)
    scheduler_D_A = StepLR(optimizer_D_A, step_size=5, gamma=0.75)
    scheduler_D_B = StepLR(optimizer_D_B, step_size=5, gamma=0.75)

    # Loss functions
    criterion_GAN = nn.BCELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)

    for epoch in range(epochs):
        # For logging
        loss_log = {"G": [], "D_A": [], "D_B": [], "cycle": []}

        for batch in train_loader:
            real_A, real_B, _, _ = batch
            real_A, real_B = real_A.float().to(device), real_B.float().to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1), requires_grad=False).to(device)

            ### Generators A2B and B2A ###
            optimizer_G.zero_grad()

            # Identity loss (Optional, for improved stability)

            # GAN loss
            fake_B = model.G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(model.D_B(fake_B), valid)
            fake_A = model.G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(model.D_A(fake_A), valid)

            # Cycle loss
            recovered_A = model.G_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
            recovered_B = model.G_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_ABA + loss_cycle_BAB)
            loss_G.backward()
            optimizer_G.step()

            ### Discriminator A ###
            optimizer_D_A.zero_grad()

            loss_real_A = criterion_GAN(model.D_A(real_A), valid)
            loss_fake_A = criterion_GAN(model.D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            ### Discriminator B ###
            optimizer_D_B.zero_grad()

            loss_real_B = criterion_GAN(model.D_B(real_B), valid)
            loss_fake_B = criterion_GAN(model.D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Logging
            loss_log["G"].append(loss_G.item())
            loss_log["D_A"].append(loss_D_A.item())
            loss_log["D_B"].append(loss_D_B.item())
            loss_log["cycle"].append((loss_cycle_ABA.item() + loss_cycle_BAB.item()) / 2)

            # Print loss statistics per batch
            print(
                f"Loss_G: {loss_log['G'][-1]:.4f} | Loss_D_A: {loss_log['D_A'][-1]:.4f} | Loss_D_B: {loss_log['D_B'][-1]:.4f} | Loss_Cycle: {loss_log['cycle'][-1]:.4f}",
                end='\r')

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if val_loader != None:
            # Validation process
            val_loss = 0.0
            for inputs, labels, _, _ in val_loader:
                outputs = model.G_A2B(inputs.float().to(device))
                val_loss += criterion_cycle(outputs.to(device), labels.to(device)).item()
            print(
                f"Epoch: {epoch + 1}/{epochs} | Loss_G: {np.mean(loss_log['G']):.4f} | Loss_D_A: {np.mean(loss_log['D_A']):.4f} | Loss_D_B: {np.mean(loss_log['D_B']):.4f} | Loss_Cycle: {np.mean(loss_log['cycle']):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
        else:
            print(
                f"Epoch: {epoch + 1}/{epochs} | Loss_G: {np.mean(loss_log['G']):.4f} | Loss_D_A: {np.mean(loss_log['D_A']):.4f} | Loss_D_B: {np.mean(loss_log['D_B']):.4f} | Loss_Cycle: {np.mean(loss_log['cycle']):.4f}")

    if val_loader != None:
        # Get metrics in the end of training
        predictions = []
        ground_truth = []
        for inputs, labels, _, _ in val_loader:
            predictions += model.G_A2B(inputs.float().to(device)).tolist()
            ground_truth += labels.tolist()
        # Export predictions
        pred_export = np.clip(np.array(predictions).flatten(), 0, 1)
        id_column = np.arange(1, len(pred_export) + 1)
        export_df = pd.DataFrame({
            'ID': id_column,
            'Predicted': pred_export
        })
        export_df.to_csv(f'predictions/predictions_fold_{fold_num}.csv', index=False)
        return get_metrics(np.array(predictions), np.array(ground_truth))
    else:
        return None


# The following function was added for looking for the early stopping point and make sure of fair comparison
# between different models of the paper. The code is based on the original training function,
# but not part of the original code.
def train_graph_cyclegan_es(model, train_loader, val_loader=None, lr_G=0.0005, lr_D_A=0.0002, lr_D_B=0.0002, epochs=10,
                            device='cpu', fold_num=0):
    model.to(device)

    losses = []
    es = -1
    best_mae = float('inf')
    ctr = 0

    # Optimizers
    optimizer_G = optim.Adam(list(model.G_A2B.parameters()) + list(model.G_B2A.parameters()), lr=lr_G,
                             betas=(0.5, 0.999))
    optimizer_D_A = optim.Adam(model.D_A.parameters(), lr=lr_D_A, betas=(0.5, 0.999))
    optimizer_D_B = optim.Adam(model.D_B.parameters(), lr=lr_D_B, betas=(0.5, 0.999))

    # Schedulers
    scheduler_G = StepLR(optimizer_G, step_size=5, gamma=0.75)
    scheduler_D_A = StepLR(optimizer_D_A, step_size=5, gamma=0.75)
    scheduler_D_B = StepLR(optimizer_D_B, step_size=5, gamma=0.75)

    # Loss functions
    criterion_GAN = nn.BCELoss().to(device)
    criterion_cycle = nn.L1Loss().to(device)

    for epoch in range(epochs):
        # For logging
        loss_log = {"G": [], "D_A": [], "D_B": [], "cycle": []}

        for batch in train_loader:
            real_A, real_B, _, _ = batch
            real_A, real_B = real_A.float().to(device), real_B.float().to(device)

            # Adversarial ground truths
            valid = torch.ones((real_A.size(0), 1), requires_grad=False).to(device)
            fake = torch.zeros((real_A.size(0), 1), requires_grad=False).to(device)

            ### Generators A2B and B2A ###
            optimizer_G.zero_grad()

            # Identity loss (Optional, for improved stability)

            # GAN loss
            fake_B = model.G_A2B(real_A)
            loss_GAN_A2B = criterion_GAN(model.D_B(fake_B), valid)
            fake_A = model.G_B2A(real_B)
            loss_GAN_B2A = criterion_GAN(model.D_A(fake_A), valid)

            # Cycle loss
            recovered_A = model.G_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)
            recovered_B = model.G_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)

            # Total loss
            loss_G = loss_GAN_A2B + loss_GAN_B2A + 10.0 * (loss_cycle_ABA + loss_cycle_BAB)
            loss_G.backward()
            optimizer_G.step()

            ### Discriminator A ###
            optimizer_D_A.zero_grad()

            loss_real_A = criterion_GAN(model.D_A(real_A), valid)
            loss_fake_A = criterion_GAN(model.D_A(fake_A.detach()), fake)
            loss_D_A = (loss_real_A + loss_fake_A) / 2
            loss_D_A.backward()
            optimizer_D_A.step()

            ### Discriminator B ###
            optimizer_D_B.zero_grad()

            loss_real_B = criterion_GAN(model.D_B(real_B), valid)
            loss_fake_B = criterion_GAN(model.D_B(fake_B.detach()), fake)
            loss_D_B = (loss_real_B + loss_fake_B) / 2
            loss_D_B.backward()
            optimizer_D_B.step()

            # Logging
            loss_log["G"].append(loss_G.item())
            loss_log["D_A"].append(loss_D_A.item())
            loss_log["D_B"].append(loss_D_B.item())
            loss_log["cycle"].append((loss_cycle_ABA.item() + loss_cycle_BAB.item()) / 2)
            losses.append(loss_log['cycle'][-1])

            # Print loss statistics per batch
            print(
                f"Loss_G: {loss_log['G'][-1]:.4f} | Loss_D_A: {loss_log['D_A'][-1]:.4f} | Loss_D_B: {loss_log['D_B'][-1]:.4f} | Loss_Cycle: {loss_log['cycle'][-1]:.4f}",
                end='\r')

        scheduler_G.step()
        scheduler_D_A.step()
        scheduler_D_B.step()

        if val_loader != None:
            # Validation process
            val_loss = 0.0
            for inputs, labels, _, _ in val_loader:
                outputs = model.G_A2B(inputs.float().to(device))
                outputs = torch.where(outputs < 0, torch.tensor(0.0).to(device), outputs)
                val_loss += criterion_cycle(outputs.to(device), labels.to(device)).item()
            print(
                f"Epoch: {epoch + 1}/{epochs} | Loss_G: {np.mean(loss_log['G']):.4f} | Loss_D_A: {np.mean(loss_log['D_A']):.4f} | Loss_D_B: {np.mean(loss_log['D_B']):.4f} | Loss_Cycle: {np.mean(loss_log['cycle']):.4f} | Val Loss: {val_loss / len(val_loader):.4f}")
            mae = val_loss / len(val_loader)
            if mae < best_mae:
                if best_mae - mae > 0.0001:
                    best_mae = mae
                    print(f"Best MAE: {best_mae}, PASS")
                    ctr = 0
                else:
                    ctr += 1
                    print(f"Best MAE: {best_mae}, BUT UNDER THRESHOLD")
            else:
                ctr += 1
                if ctr >= 10:
                    if epoch >= 99:
                        print(f"Early stopping at epoch {epoch}!!!!!!!")
                        es = epoch
                        break
                    else:
                        print(f"Worse MAE for {ctr} times, continue training.")
                        continue
        else:
            print(
                f"Epoch: {epoch + 1}/{epochs} | Loss_G: {np.mean(loss_log['G']):.4f} | Loss_D_A: {np.mean(loss_log['D_A']):.4f} | Loss_D_B: {np.mean(loss_log['D_B']):.4f} | Loss_Cycle: {np.mean(loss_log['cycle']):.4f}")

    return model, losses, es
