import torch
import torch.nn as nn
import numpy as np
from .preprocessing import *
from .model import *
import torch.optim as optim

criterion = nn.MSELoss()


def train(model, subjects_adj, subjects_labels, args):

    bce_loss = nn.BCELoss()
    netD = Discriminator(args)
    netGD = GlobalDiscriminator(args)

    print(netD)
    print(netGD)

    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerGD = optim.Adam(netGD.parameters(), lr=args.lr)

    all_epochs_loss = []
    all_epochs_error = []
    final_epoch = 0 
    breaks = False
    epochLosses = 0
    x = 0
    for epoch in range(args.epochs):
        if breaks == False:
            with torch.autograd.set_detect_anomaly(True):
                epoch_loss = []
                epoch_error = []
                for lr, hr in zip(subjects_adj, subjects_labels):
                    optimizerG.zero_grad()

                    hr = pad_HR_adj(hr, args.padding)

                    lr = torch.from_numpy(lr).type(torch.FloatTensor)
                    padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)
                    
                    eig_val_hr, U_hr = torch.linalg.eigh(
                        padded_hr, UPLO='U')

                    model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                        lr, args.lr_dim, args.hr_dim)

                    mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                        model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                    error = criterion(model_outputs, padded_hr)
                    real_data = model_outputs.detach()
                    fake_data = gaussian_noise_layer(padded_hr, args)

                    # Train discriminator
                    optimizerD.zero_grad()

                    d_real = netD(padded_hr)
                    d_fake = netD(real_data)

                    dc_loss_real = bce_loss(d_real, torch.ones(args.hr_dim, 1))
                    dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
                    dc_loss = dc_loss_real + dc_loss_fake

                    dc_loss.backward()
                    optimizerD.step()

                    # # Train global discriminator
                    optimizerGD.zero_grad()

                    gd_real = netD(real_data)
                    gd_fake = netD(fake_data)

                    gdc_loss_real = bce_loss(gd_real, torch.ones(args.hr_dim, 1))
                    gdc_loss_fake = bce_loss(gd_fake, torch.zeros(args.hr_dim, 1))
                    gdc_loss = gdc_loss_real + gdc_loss_fake

                    gdc_loss.backward()
                    optimizerGD.step()

                    # Train generator
                    d_fake = netD(gaussian_noise_layer(padded_hr, args))
                    gd_fake = netGD(gaussian_noise_layer(padded_hr, args))

                    d_gen_loss = bce_loss(d_fake, torch.ones(args.hr_dim, 1))
                    gd_gen_loss = bce_loss(gd_fake, torch.ones(args.hr_dim, 1))
                    generator_loss = d_gen_loss + gd_gen_loss + mse_loss
                    generator_loss.backward()
                    optimizerG.step()

                    epoch_loss.append(generator_loss.item())
                    epoch_error.append(error.item())

                print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                    "Error: ", np.mean(epoch_error)*100, "%")
                all_epochs_loss.append(np.mean(epoch_loss))
                all_epochs_error.append(np.mean(epoch_error))
                final_epoch = epoch + 1
                # 0.0001:
                if abs(epochLosses -  np.mean(epoch_error)) < 0.0001:
                    x += 1
                    print("x: ", x)
                    if x >= 10:
                        print("break")
                        breaks = True
                else:
                    print("Difference: ", abs(epochLosses -  np.mean(epoch_error)))
                epochLosses = np.mean(epoch_error)


    optimizerG = optim.Adam(model.parameters(), lr=args.lr)
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr)
    optimizerGD = optim.Adam(netGD.parameters(), lr=args.lr)

    all_epochs_loss = []
    all_epochs_error = []
    for epoch in range(final_epoch):
        with torch.autograd.set_detect_anomaly(True):
            epoch_loss = []
            epoch_error = []
            for lr, hr in zip(subjects_adj, subjects_labels):
                optimizerG.zero_grad()

                hr = pad_HR_adj(hr, args.padding)

                lr = torch.from_numpy(lr).type(torch.FloatTensor)
                padded_hr = torch.from_numpy(hr).type(torch.FloatTensor)


                eig_val_hr, U_hr = torch.linalg.eigh(
                    padded_hr, UPLO='U')

                model_outputs, net_outs, start_gcn_outs, layer_outs = model(
                    lr, args.lr_dim, args.hr_dim)

                mse_loss = args.lmbda * criterion(net_outs, start_gcn_outs) + criterion(
                    model.layer.weights, U_hr) + criterion(model_outputs, padded_hr)

                error = criterion(model_outputs, padded_hr)
                real_data = model_outputs.detach()
                fake_data = gaussian_noise_layer(padded_hr, args)

                # Train discriminator
                optimizerD.zero_grad()

                d_real = netD(padded_hr)
                d_fake = netD(real_data)

                dc_loss_real = bce_loss(d_real, torch.ones(args.hr_dim, 1))
                dc_loss_fake = bce_loss(d_fake, torch.zeros(args.hr_dim, 1))
                dc_loss = dc_loss_real + dc_loss_fake

                dc_loss.backward()
                optimizerD.step()

                # # Train global discriminator
                optimizerGD.zero_grad()

                gd_real = netD(real_data)
                gd_fake = netD(fake_data)

                gdc_loss_real = bce_loss(gd_real, torch.ones(args.hr_dim, 1))
                gdc_loss_fake = bce_loss(gd_fake, torch.zeros(args.hr_dim, 1))
                gdc_loss = gdc_loss_real + gdc_loss_fake

                gdc_loss.backward()
                optimizerGD.step()

                # Train generator
                d_fake = netD(gaussian_noise_layer(padded_hr, args))
                gd_fake = netGD(gaussian_noise_layer(padded_hr, args))

                d_gen_loss = bce_loss(d_fake, torch.ones(args.hr_dim, 1))
                gd_gen_loss = bce_loss(gd_fake, torch.ones(args.hr_dim, 1))
                generator_loss = d_gen_loss + gd_gen_loss + mse_loss
                generator_loss.backward()
                optimizerG.step()

                epoch_loss.append(generator_loss.item())
                epoch_error.append(error.item())

            print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss),
                  "Error: ", np.mean(epoch_error)*100, "%")
            all_epochs_loss.append(np.mean(epoch_loss))
            all_epochs_error.append(np.mean(epoch_error))
    return all_epochs_loss, all_epochs_error


def test(model, test_adj, test_labels, args):

    g_t = []
    test_error = []
    preds_list = []

    for lr, hr in zip(test_adj, test_labels):
        

        all_zeros_lr = not np.any(lr)
        all_zeros_hr = not np.any(hr)
        if all_zeros_lr == False and all_zeros_hr == False:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            np.fill_diagonal(hr, 1)
            hr = pad_HR_adj(hr, args.padding)
            hr = torch.from_numpy(hr).type(torch.FloatTensor)
            preds, a, b, c = model(lr, args.lr_dim, args.hr_dim)

    

            preds_list.append(preds.flatten().detach().numpy())
            error = criterion(preds, hr)
            g_t.append(hr.flatten())
            print(error.item())
            test_error.append(error.item())

    print("Test error MSE: ", np.mean(test_error))

