import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from globals import TORCH_DEVICE, COMPUTE_METRICS_FLAG, FULL_DATA, FULL_TARGETS
from hyperparams import Hyperparams
from model import GSRNet
from utils import pad_HR_adj, unpad
from metrics import compute_metrics
from MatrixVectorizer import MatrixVectorizer
import networkx as nx
from community import community_louvain
# import community as community_louvain
import os
from scipy.stats import pearsonr
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mean_absolute_error

# Vectorize a list of HR graphs (either predictions or ground truth)
# and concatenates the values from all graphs
def vectorize_adj_list(adj_list):
    vect_adj = []
    vectorizer = MatrixVectorizer()

    for adj in adj_list:
        vect_adj.append(vectorizer.vectorize(adj))
    vect_adj = np.concatenate(vect_adj)

    return vect_adj


def get_mae_from_vect_preds(preds_list, test_labels, fold_num):
    # vectorize both the predictions and the ground truth graphs
    vect_hr = vectorize_adj_list(test_labels)
    vect_preds = vectorize_adj_list(preds_list)
    # compute the MAE as done for the submissions
    mean_mae = np.mean(np.absolute(vect_preds - vect_hr))

    return mean_mae


# Apply post-processing on the model outputs
# We want our predictions to all be in [0, 1] and our model returns only
# positive values, so we only upper bound our predictions to 1
def postprocessing(preds):
    final_preds = torch.where(preds > 1, 1, preds)
    return final_preds


def train(model, optimizer, subjects_adj, subjects_labels, hps, num_fold, mode, test_adj=None, test_ground_truth=None):
    all_epochs_loss = []
    no_epochs = hps.epochs

    min_loss, min_loss_epoch = 1e20, -1
    best_loss = 100
    counter = 0
    for epoch in range(1, no_epochs + 1):
        epoch_loss = []
        epoch_error = []

        for group in optimizer.param_groups:
            group['lr'] *= hps.lr_schedule

        # Generate the permutation to apply to the train dataset
        permutation = torch.randperm(hps.lr_dim)
        perm = torch.zeros((hps.lr_dim, hps.lr_dim))

        for i in range(hps.lr_dim):
            perm[i][permutation[i]] = 1

        perm = perm.to(torch.double)
        perm_t = perm.transpose(0, 1)

        # Get the permuted adjacency matrices
        new_adj = torch.matmul(perm_t, torch.matmul(torch.from_numpy(subjects_adj),
                                                    perm)).numpy()

        for lr, hr in zip(new_adj, subjects_labels):
            model.train()
            optimizer.zero_grad()

            lr = torch.from_numpy(lr).type(torch.FloatTensor).to(TORCH_DEVICE)
            hr = torch.from_numpy(hr).type(torch.FloatTensor).to(TORCH_DEVICE)

            model_outputs, net_outs, start_gcn_outs, _ = model(lr)
            model_outputs = postprocessing(model_outputs)
            # Unpad the 320x320 graph to 268x268
            model_outputs = unpad(model_outputs, hps.padding)

            # Get the eigevectors of the padded 320x320 graph for the GT HR
            padded_hr = pad_HR_adj(hr, hps.padding)
            _, upper_hr = torch.linalg.eigh(padded_hr, UPLO='U')

            # Compute the final loss which represents the sum of:
            # * The reconstruction loss of the decoded graph through the UNet
            # * The eigen loss between our modelled weight matrix
            # and the real eigenvectors of the HR
            # * And the final super resolution loss between our modelled output
            # and the ground truth HR graph
            loss = hps.lmbda * hps.train_criterion(net_outs, start_gcn_outs) + \
                   hps.train_criterion(model.layer.weights, upper_hr) + \
                   hps.train_criterion(model_outputs, hr)

            # Record the super resolution loss for logging purposes
            error = hps.train_criterion(model_outputs, hr)
            loss_val = loss.item()

            # Apply the backpropagation with the previously computed loss
            loss.backward()
            optimizer.step()

            epoch_loss.append(loss_val)
            if loss_val < min_loss:
                min_loss = loss_val
                min_loss_epoch = epoch
            epoch_error.append(error.item())

        print("Epoch: ", epoch, "Loss: ", np.mean(epoch_loss), "Error: ", np.mean(epoch_error) * 100, "%")
        all_epochs_loss.append(np.mean(epoch_loss))

        if test_adj is not None and test_ground_truth is not None:
            mean_mae, mean_loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

            if mean_loss >= best_loss:
                counter += 1
                if counter >= 10 and epoch >= 110:
                    break
            else:
                best_loss = mean_loss
                counter = 0
                print(f'----best model epoch: {epoch}----')

    print("Min loss: ", min_loss, "Min loss epoch: ", min_loss_epoch)


def test(model, test_adj, test_labels, hps, num_fold):
    model.eval()
    test_error = []
    preds_list = []
    g_t = []
    test_loss = []

    with torch.no_grad():
        for lr, hr in zip(test_adj, test_labels):
            all_zeros_lr = np.any(lr)
            all_zeros_hr = np.any(hr)
            if all_zeros_lr and all_zeros_hr:
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(TORCH_DEVICE)
                np.fill_diagonal(hr, 1)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(TORCH_DEVICE)

                model_outputs, net_outs, start_gcn_outs, _ = model(lr)
                model_outputs = postprocessing(model_outputs)
                model_outputs = unpad(model_outputs, hps.padding)

                padded_hr = pad_HR_adj(hr, hps.padding)
                _, upper_hr = torch.linalg.eigh(padded_hr, UPLO='U')

                loss = hps.lmbda * hps.train_criterion(net_outs, start_gcn_outs) + \
                       hps.train_criterion(model.layer.weights, upper_hr) + \
                       hps.train_criterion(model_outputs, hr)

                test_loss.append(loss.item())

                preds_list.append(model_outputs.detach().clone().cpu().numpy())

                error = F.l1_loss(model_outputs, hr)
                g_t.append(hr.flatten())
                test_error.append(error.item())

    mean_mae = get_mae_from_vect_preds(preds_list, test_labels, num_fold)
    mean_loss = np.mean(np.array(test_loss))

    print("Test loss: ", mean_loss, "Test error MAE: ", mean_mae, )

    if COMPUTE_METRICS_FLAG:
        return mean_mae, mean_loss, compute_metrics(preds_list, test_labels, hps)
    else:
        return mean_mae, mean_loss, {}


# Function that runs the 3F CV
def run_experiment_fold1(X, Y, num_fold, train_all, mode):
    print("Fold number:", num_fold)

    hps = Hyperparams()

    if mode == 'random':
        if train_all:
            hps.epochs = 155
            # Get the training and testing data for the current fold
            subjects_adj, test_adj = X[93:], X[: 93]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth, test_ground_truth = Y[93:], Y[: 93]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_random_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[93: 166], X[186: 259]), axis=0)
            val_adj = np.concatenate((X[166: 186], X[259:]), axis=0)
            test_adj = X[: 93]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[93: 166], Y[186: 259]), axis=0)
            val_ground_truth = np.concatenate((Y[166: 186], Y[259:]), axis=0)
            test_ground_truth = Y[: 93]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)

    if mode == 'cluster':
        if train_all:
            hps.epochs = 207
            # Get the training and testing data for the current fold
            subjects_adj, test_adj = X[102:], X[: 102]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth, test_ground_truth = Y[102:], Y[: 102]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_cluster_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[102: 184], X[204: 259]), axis=0)
            val_adj = np.concatenate((X[184: 204], X[259:]), axis=0)
            test_adj = X[: 102]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[102: 184], Y[204: 259]), axis=0)
            val_ground_truth = np.concatenate((Y[184: 204], Y[259:]), axis=0)
            test_ground_truth = Y[: 102]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)


# fold2
def run_experiment_fold2(X, Y, num_fold, train_all, mode):
    print("Fold number:", num_fold)

    hps = Hyperparams()

    if mode == 'random':
        if train_all:
            hps.epochs = 130
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 93], X[186: ]), axis=0)
            test_adj = X[93: 186]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 93], Y[186: ]), axis=0)
            test_ground_truth = Y[93: 186]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_random_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 73], X[186: 259]), axis=0)
            val_adj = np.concatenate((X[73: 93], X[259: ]), axis=0)
            test_adj = X[93: 186]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 73], Y[186: 259]), axis=0)
            val_ground_truth = np.concatenate((Y[73: 93], Y[259: ]), axis=0)
            test_ground_truth = Y[93: 186]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)

    if mode == 'cluster':
        if train_all:
            hps.epochs = 108
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 102], X[204:]), axis=0)
            test_adj = X[102: 204]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 102], Y[204:]), axis=0)
            test_ground_truth = Y[102: 204]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_cluster_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 82], X[204: 259]), axis=0)
            val_adj = np.concatenate((X[82: 102], X[259:]), axis=0)
            test_adj = X[102: 204]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 82], Y[204: 259]), axis=0)
            val_ground_truth = np.concatenate((Y[82: 102], Y[259:]), axis=0)
            test_ground_truth = Y[102: 204]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)


def run_experiment_fold3(X, Y, num_fold, train_all, mode):
    print("Fold number:", num_fold)

    hps = Hyperparams()

    if mode == 'random':
        if train_all:
            hps.epochs = 220
            # Get the training and testing data for the current fold
            subjects_adj, test_adj = X[: 186], X[186: ]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth, test_ground_truth = Y[: 186], Y[186: ]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_random_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 73], X[93: 166]), axis=0)
            val_adj = np.concatenate((X[73: 93], X[166: 186]), axis=0)
            test_adj = X[186: ]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 73], Y[93: 166]), axis=0)
            val_ground_truth = np.concatenate((Y[73: 93], Y[166: 186]), axis=0)
            test_ground_truth = Y[186: ]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)

    if mode == 'cluster':
        if train_all:
            hps.epochs = 127
            # Get the training and testing data for the current fold
            subjects_adj, test_adj = X[: 204], X[204:]
            print(subjects_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth, test_ground_truth = Y[: 204], Y[204:]
            print(subjects_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode)
            torch.save(model.state_dict(), f'models/fold{num_fold}_cluster_all.pt')
            mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        else:
            # Get the training and testing data for the current fold
            subjects_adj = np.concatenate((X[: 82], X[102: 184]), axis=0)
            val_adj = np.concatenate((X[82: 102], X[184: 204]), axis=0)
            test_adj = X[204:]
            print(subjects_adj.shape)
            print(val_adj.shape)
            print(test_adj.shape)
            # Get the truths for the current fold
            subjects_ground_truth = np.concatenate((Y[: 82], Y[102: 184]), axis=0)
            val_ground_truth = np.concatenate((Y[82: 102], Y[184: 204]), axis=0)
            test_ground_truth = Y[204:]
            print(subjects_ground_truth.shape)
            print(val_ground_truth.shape)
            print(test_ground_truth.shape)

            model = GSRNet(hps)
            model.to(TORCH_DEVICE)
            optimizer = optim.AdamW(model.parameters(), lr=hps.lr)
            train(model, optimizer, subjects_adj, subjects_ground_truth, hps, num_fold, mode, val_adj, val_ground_truth)


def calculate_centralities(adj_matrix):
    if adj_matrix.shape[0] != adj_matrix.shape[1]:
        raise ValueError(f"Adjacency matrix is not square: shape={adj_matrix.shape}")
    print(f"Processing adjacency matrix of shape: {adj_matrix.shape}")

    G = nx.from_numpy_array(adj_matrix)
    partition = community_louvain.best_partition(G)

    # Calculate the participation coefficient with the partition
    pc_dict = participation_coefficient(G, partition)

    # Calculate averages of centrality measures
    pr = nx.pagerank(G, alpha=0.9)
    ec = nx.eigenvector_centrality_numpy(G, max_iter=100)
    bc = nx.betweenness_centrality(G, normalized=True, endpoints=False)
    ns = np.array(list(nx.degree_centrality(G).values())) * (len(G.nodes()) - 1)
    acc = nx.average_clustering(G, weight=None)

    # Average participation coefficient
    pc_avg = np.mean(list(pc_dict.values()))

    return {
        'pr': np.mean(list(pr.values())),
        'ec': np.mean(list(ec.values())),
        'bc': np.mean(list(bc.values())),
        'ns': ns,
        'pc': pc_avg,
        'acc': acc
    }


def participation_coefficient(G, partition):
    # Initialize dictionary for participation coefficients
    pc_dict = {}

    # Calculate participation coefficient for each node
    for node in G.nodes():
        node_degree = G.degree(node)
        if node_degree == 0:
            pc_dict[node] = 0.0
        else:
            # Count within-module connections
            within_module_degree = sum(1 for neighbor in G[node] if partition[neighbor] == partition[node])
            # Calculate participation coefficient
            pc_dict[node] = 1 - (within_module_degree / node_degree) ** 2

    return pc_dict


def evaluate_all(true_hr_matrices, predicted_hr_matrices, output_path='ID-randomCV.csv'):
    print(true_hr_matrices.shape)
    print(predicted_hr_matrices.shape)

    num_subjects = true_hr_matrices.shape[0]
    results = []

    for i in range(num_subjects):
        true_matrix = true_hr_matrices[i, :, :]
        pred_matrix = predicted_hr_matrices[i, :, :]

        print(f"Evaluating subject {i + 1} with matrix shapes: true={true_matrix.shape}, pred={pred_matrix.shape}")

        if true_matrix.shape != pred_matrix.shape or true_matrix.shape[0] != true_matrix.shape[1]:
            print(
                f"Error: Matrix shape mismatch or not square for subject {i + 1}: true={true_matrix.shape}, pred={pred_matrix.shape}")
            continue

        metrics = {
            'ID': i + 1,
            'MAE': mean_absolute_error(true_matrix.flatten(), pred_matrix.flatten()),
            'PCC': pearsonr(true_matrix.flatten(), pred_matrix.flatten())[0],
            'JSD': jensenshannon(true_matrix.flatten(), pred_matrix.flatten()),
        }

        true_metrics = calculate_centralities(true_matrix)
        pred_metrics = calculate_centralities(pred_matrix)

        for key in ['NS', 'PR', 'EC', 'BC', 'PC', 'ACC']:
            metrics[f'MAE in {key}'] = mean_absolute_error([true_metrics[key.lower()]], [pred_metrics[key.lower()]])

        results.append(metrics)

    df = pd.DataFrame(results)
    if not df.empty:
        # Check if the file exists to decide whether to write headers
        file_exists = os.path.isfile(output_path)

        df.to_csv(output_path, mode='a', header=not file_exists, index=False)
        print(f"Results appended to {output_path}.")
    else:
        print("No data to save.")


def test1(model, test_adj, test_labels, hps, num_fold):
    model.eval()
    preds_list = []

    with torch.no_grad():
        for lr, hr in zip(test_adj, test_labels):
            all_zeros_lr = np.any(lr)
            all_zeros_hr = np.any(hr)
            if all_zeros_lr and all_zeros_hr:
                lr = torch.from_numpy(lr).type(torch.FloatTensor).to(TORCH_DEVICE)
                np.fill_diagonal(hr, 1)
                hr = torch.from_numpy(hr).type(torch.FloatTensor).to(TORCH_DEVICE)

                model_outputs, net_outs, start_gcn_outs, _ = model(lr)
                model_outputs = postprocessing(model_outputs)
                model_outputs = unpad(model_outputs, hps.padding)

                preds_list.append(model_outputs.detach().clone().cpu().numpy())

    return np.array(preds_list)


# mode = 'cluster'
mode = 'random'
train_all = True

hps = Hyperparams()

# fold1
if train_all:
    if mode == 'random':
        X = FULL_DATA
        Y = FULL_TARGETS
        num_fold = 1

        # Get the training and testing data for the current fold
        subjects_adj, test_adj = X[93:], X[: 93]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth, test_ground_truth = Y[93:], Y[: 93]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

    if mode == 'cluster':
        X = FULL_DATA
        Y = FULL_TARGETS
        num_fold = 1

        # Get the training and testing data for the current fold
        subjects_adj, test_adj = X[102:], X[: 102]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth, test_ground_truth = Y[102:], Y[: 102]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

    model = GSRNet(hps)
    model.to(TORCH_DEVICE)

    state_dict_path = f'models/fold1_{mode}_all.pt'
    model.load_state_dict(torch.load(state_dict_path))

    mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

    preds_list = test1(model, test_adj, test_ground_truth, hps, num_fold)
    evaluate_all(test_ground_truth, preds_list, output_path=f'final/7-{mode}CV.csv')


# fold2
if train_all:
    if mode == 'random':
        X = FULL_DATA
        Y = FULL_TARGETS
        num_fold = 2

        # Get the training and testing data for the current fold
        subjects_adj = np.concatenate((X[: 93], X[186:]), axis=0)
        test_adj = X[93: 186]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth = np.concatenate((Y[: 93], Y[186:]), axis=0)
        test_ground_truth = Y[93: 186]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

        model = GSRNet(hps)
        model.to(TORCH_DEVICE)

        state_dict_path = 'models/fold2_random_all.pt'
        model.load_state_dict(torch.load(state_dict_path))

        mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        preds_list = test1(model, test_adj, test_ground_truth, hps, num_fold)
        evaluate_all(test_ground_truth, preds_list, output_path='final/7-randomCV.csv')

    if mode == 'cluster':
        X = FULL_DATA
        Y = FULL_TARGETS
        num_fold = 2

        # Get the training and testing data for the current fold
        subjects_adj = np.concatenate((X[: 102], X[204:]), axis=0)
        test_adj = X[102: 204]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth = np.concatenate((Y[: 102], Y[204:]), axis=0)
        test_ground_truth = Y[102: 204]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

        model = GSRNet(hps)
        model.to(TORCH_DEVICE)

        state_dict_path = 'models/fold2_cluster_all.pt'
        model.load_state_dict(torch.load(state_dict_path))

        mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)
        print(mae, loss)

        preds_list = test1(model, test_adj, test_ground_truth, hps, num_fold)
        evaluate_all(test_ground_truth, preds_list, output_path='final/7-clusterCV.csv')


# fold3
if train_all:
    if mode == 'random':
        X = FULL_DATA
        Y = FULL_TARGETS
        hps = Hyperparams()
        num_fold = 3

        # Get the training and testing data for the current fold
        subjects_adj, test_adj = X[: 186], X[186:]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth, test_ground_truth = Y[: 186], Y[186:]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

        model = GSRNet(hps)
        model.to(TORCH_DEVICE)

        state_dict_path = 'models/fold3_random_all.pt'
        model.load_state_dict(torch.load(state_dict_path))

        mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        preds_list = test1(model, test_adj, test_ground_truth, hps, num_fold)
        evaluate_all(test_ground_truth, preds_list, output_path='final/7-randomCV.csv')

    if mode == 'cluster':
        X = FULL_DATA
        Y = FULL_TARGETS
        hps = Hyperparams()
        num_fold = 3

        # Get the training and testing data for the current fold
        subjects_adj, test_adj = X[: 204], X[204:]
        print(subjects_adj.shape)
        print(test_adj.shape)
        # Get the truths for the current fold
        subjects_ground_truth, test_ground_truth = Y[: 204], Y[204:]
        print(subjects_ground_truth.shape)
        print(test_ground_truth.shape)

        model = GSRNet(hps)
        model.to(TORCH_DEVICE)

        state_dict_path = 'models/fold3_cluster_all.pt'
        model.load_state_dict(torch.load(state_dict_path))

        mae, loss, _ = test(model, test_adj, test_ground_truth, hps, num_fold)

        preds_list = test1(model, test_adj, test_ground_truth, hps, num_fold)
        evaluate_all(test_ground_truth, preds_list, output_path='final/7-clusterCV.csv')
