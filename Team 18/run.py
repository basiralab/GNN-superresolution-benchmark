from utils.reproducibility import random_seed, device  # import to fix seed
import argparse
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from preprocessing import load_data
from model import GSRGo
from train import train, validate, predict_kaggle
from utils.evaluation_measures import compute_evaluation_measures, plot_evaluation_measures, plot_validation_mae
import matplotlib.pyplot as plt  # Import matplotlib for plotting

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, min_epochs=100):
        self.patience = patience
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss, epoch):
        if epoch < self.min_epochs:
            return

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GSR-Go')
    parser.add_argument('--epochs', type=int, default=200, metavar='no_epochs', help='number of episode to train')#200
    parser.add_argument('--lr', type=float, default=0.0002, metavar='lr', help='learning rate')
    parser.add_argument('--splits', type=int, default=3, metavar='n_splits', help='no of cross validation folds')
    parser.add_argument('--lmbda', type=int, default=16, metavar='L', help='self-reconstruction error hyperparameter')
    parser.add_argument('--lr_dim', type=int, default=160, metavar='N', help='adjacency matrix input dimensions')
    parser.add_argument('--hr_dim', type=int, default=320, metavar='N', help='super-resolved adjacency matrix output dimensions')
    parser.add_argument('--hidden_dim', type=int, default=320, metavar='N', help='hidden GraphConvolutional layer dimensions')
    parser.add_argument('--padding', type=int, default=26, metavar='padding', help='dimensions of padding')
    parser.add_argument('--patience', type=int, default=10, metavar='patience', help='early stopping patience')
    parser.add_argument('--min_epoch', type=int, default=100, metavar='min_epoch', help='min epoch for early stopping patience')
    args = parser.parse_args()

    # Load data
    X, Y = load_data(split='train')
    print("len x",len(X))

    # Initialize model
    model = GSRGo(args).to(device)
    print(model)

    optimizer = optim.AdamW
    predicted_hr_matrices_list = []
    true_hr_matrices_list = []
    #stage 1: use last 
    print("------------stage1--------------")
    kf = KFold(n_splits=args.splits, shuffle=False)
    max_epochs_list = []

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        print(f"Fold {fold + 1}")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Split the training data into two parts
        split_idx = len(X_train) // 2
        X_train1, X_train2 = X_train[:split_idx], X_train[split_idx:]
        Y_train1, Y_train2 = Y_train[:split_idx], Y_train[split_idx:]

        # Use the last 20 samples from each part for validation
        X_val = np.concatenate((X_train1[-20:], X_train2[-20:]), axis=0)
        Y_val = np.concatenate((Y_train1[-20:], Y_train2[-20:]), axis=0)
        X_train = np.concatenate((X_train1[:-20], X_train2[:-20]), axis=0)
        Y_train = np.concatenate((Y_train1[:-20], Y_train2[:-20]), axis=0)

        model = GSRGo(args).to(device)
        # optimizer = optim.Adam(model.parameters, lr=args.lr)
        early_stopping = EarlyStopping(patience=args.patience, min_delta=0, min_epochs=args.min_epoch)

        train_results = train(model, optimizer, X_train, Y_train, args, args.epochs, val_data=(X_val, Y_val), early_stopping=early_stopping)
        max_epoch = len(train_results['train_losses'])

        max_epochs_list.append(max_epoch)

        print(f"Training completed for fold {fold + 1}. Max Epochs: {max_epoch}")
    
    print(f"Max epochs for each fold: {max_epochs_list}")


    ne = max(max_epochs_list)
    print("ne:",ne)
    print("-----------stage 2-----------------")
    #stage 2 train the whole fold data
    if args.splits >= 2:
        cv = KFold(n_splits=args.splits, shuffle=False)
        metrics_list = []
        measures_list = []
        min_mae = float('inf')
        all_train_losses = []  # To store training losses for each fold

        # k-fold cross validation
        for fold_num, (train_index, val_index) in enumerate(cv.split(X)):
            epoch = 0
            print(f"Fold: {fold_num+1} / {args.splits}")
            model_k = GSRGo(args).to(device)
            

            # Train model with early stopping
            val_fn = lambda: validate(model_k, X[val_index], Y[val_index], args)
           
            
            fold_metrics = train(model_k, optimizer, X[train_index], Y[train_index], args, ne, val_fn)
            all_train_losses.append(fold_metrics['train_losses'])  # Append training losses for this fold
            val_loss, val_mae, preds, true = val_fn()
            
            print("preds.shape",type(preds))
            true_hr_matrices_list.append(true)
            
            predicted_hr_matrices_list.append(preds)

            val_loss = val_fn()[1]  # Assuming val_fn returns a tuple with (mse, mae, preds, gt)


            metrics_list.append(fold_metrics)

            # Predict HR samples for validation set
            # predict_kaggle(model_k, X[val_index], args, filename=f'predictions_fold_{fold_num+1}')

            mse, mae, preds, gt = val_fn()
       
          
            # Track the best model across folds
            if mae < min_mae:
                min_mae = mae
                model = model_k
        print(f"Best model MAE: {min_mae:.6f}")

    

        # Plot validation MAEs for each fold
        plot_validation_mae([m['validation_mae'] for m in metrics_list])

        # Plot loss curves for each fold
        for fold_num, train_losses in enumerate(all_train_losses):
            plt.plot(train_losses, label=f'Fold {fold_num+1}')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss per Fold')
        plt.legend()
        plt.savefig('training_loss_per_fold.png')
        plt.show()

   
    else:
        # Train over the entire dataset
        train(model, optimizer, X, Y, args)

   
    # Generate evaluation CSV file
from evaluation import evaluate_all

true_hr_matrices = np.concatenate(true_hr_matrices_list, axis=0)
predicted_hr_matrices = np.concatenate(predicted_hr_matrices_list, axis=0)
print("true.shape:",true_hr_matrices.shape)
print("pred.shape:",predicted_hr_matrices.shape)
evaluate_all(np.array(true_hr_matrices), np.array(predicted_hr_matrices), output_path='ID-randomCV22.csv')


   


