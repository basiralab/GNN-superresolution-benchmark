import random
import argparse
import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import torch.optim as optim
import pandas as pd
# from memory_profiler import profile
from scripts.data import BrainTrain, BrainTest
from architecture.model import Model
from scripts.train import train, evaluate, predict
from scripts.evaluation_measures import calculate_evaluation_measures, plot_evaluation_measures
from scripts.MatrixVectorizer import MatrixVectorizer as MV
from datetime import datetime
from scripts import evaluation 
# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)

# Check for CUDA (GPU support) and set device accordingly
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # For multi-GPU setups
    # Additional settings for ensuring reproducibility on CUDA
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

class MakeGaussianNodeFeatures:
    def __init__(self, num_features, mean = 0.0, std = 1.0):
        self.mean = mean
        self.std = std
        self.num_features = num_features

    def __call__(self, data):
        data.x = torch.normal(mean = self.mean, std = self.std, size = (data.num_nodes, self.num_features))
        return data

# @profile
def main():
    # Parameters
    now = datetime.now()
    model_params = {"in_channels":128, "hidden_channels":128, "out_channels":32, "hidden_layer":3, "pool_ratios":0.5}
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--omega", type=float, default=1)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--early_stopping", default=True, action="store_false")
    parser.add_argument("--patience_epochs", type=int, default=10)
    parser.add_argument("--save_model", default=True, action="store_true")
    parser.add_argument("--write_predictions", default=True, action="store_true")
    parser.add_argument("--full_training", default=False, action="store_true")

    flags = parser.parse_args()

    # Data
    
    print("Loading data...")
    #train_transform = T.Compose([T.Constant(1)]*model_params["in_channels"])
    #val_transform = T.Compose([T.Constant(1)]*model_params["in_channels"])
    
    num_features = model_params["in_channels"]
    train_transform = T.Compose([MakeGaussianNodeFeatures(mean = 1.0, std = 0.1, num_features = num_features)])
    val_transform = T.Compose([MakeGaussianNodeFeatures(mean = 1.0, std = 0.1, num_features = num_features)])
    test_transform = val_transform
    
    dataset = BrainTrain(seed=random_seed, eval_mode="randomCV")
    best_epochs = []

    if not flags.full_training:
        # eval_measures_list = []
        fold_rotation = [[0,1, 2], [1,2, 0], [2,0, 1]]
        for i in range(3): 
            print("Starting experiment no: {}".format(i))
            fold = fold_rotation[i]
            
            train_dataset = dataset.get_fold_split(fold, train_transform, "train")
            train_dl = DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
            val_dataset = dataset.get_fold_split(fold, val_transform, "val")
            val_dl = DataLoader(val_dataset, batch_size=flags.batch_size)
            test_dataset = dataset.get_fold_split(fold, test_transform, "test")
            test_dl = DataLoader(test_dataset, batch_size=flags.batch_size)

            # Train model
            print(f"Training fold {i + 1} of 3...")
            model = Model(model_params).to(device)
            print(model)
            print(sum(p.numel() for p in model.parameters()))        
            optimizer = optim.Adam(model.parameters(), lr=flags.lr)

            _, best_epoch = train(i, model, optimizer, train_dl, val_dl, device, flags)
            # val_predictions, val_ground_truth = evaluate(model, val_dl, device)
            
            '''
            if flags.save_model:
                model_name = "model"
                if torch.cuda.is_available():
                    torch.save(model.state_dict(), f"{model_name}_fold_{i}.pt")
                else:
                    torch.save(model.state_dict(), f"{model_name}_fold_{i}.pt", map_location=torch.device('cpu'))
            '''

            best_epochs.append(best_epoch)
        print(best_epochs)

            # eval_measures_list.append(calculate_evaluation_measures(val_predictions.cpu(), val_ground_truth.cpu()))
        # plot_evaluation_measures(eval_measures_list)
    
    if True:
        fold_rotation = [[0,1, 2], [1,2, 0], [2,0, 1]]
        for i in range(3): 
            fold = fold_rotation[i]
            
            #retrain on full model
            flags.early_stopping = False
            flags.epochs = best_epochs[i]
            
            final_model = Model(model_params).to(device)
            train_dataset = dataset.get_all(fold, train_transform)        
            train_dl = DataLoader(train_dataset, batch_size=flags.batch_size, shuffle=True)
            optimizer = optim.Adam(final_model.parameters(), lr=flags.lr)
            train( "All", final_model, optimizer, train_dl, None, device, flags)

            test_dataset = dataset.get_fold_split(fold, test_transform, "test")
            test_dl = DataLoader(test_dataset, batch_size=flags.batch_size)
            test_predictions, test_ground_truth = evaluate(final_model, test_dl, device)
            main_path = "/content/drive/My Drive/Deepen Your Goals in Life/"
            
            save_test_result_path = main_path + "Deepen-randomCV_" + str(i) + ".csv"
            evaluation.evaluate_all(test_ground_truth.cpu().detach().numpy(), test_predictions.cpu().detach().numpy(), output_path = save_test_result_path)

            if flags.save_model:
                    model_name = main_path + "model_full_randomCV_" + str(i)
                    if torch.cuda.is_available():
                        torch.save(final_model.state_dict(), f"{model_name}.pt")
                    else:
                        torch.save(final_model.state_dict(), f"{model_name}.pt", map_location=torch.device('cpu'))
    

    '''
    if flags.write_predictions:
        if not flags.full_training:
            for i in range(3):
                # Load model
                model = Model(model_params).to(device)
                model.load_state_dict(torch.load(f"model_fold_{i}.pt"))

                # Predict test data
                print("Predicting test data...")
                test_transform = val_transform
                test_dataset = BrainTest(test_transform)
                test_dl = DataLoader(test_dataset, batch_size=flags.batch_size)
                predictions = predict(model, test_dl, device)

                # Format test data
                print("Formatting test data...")
                predictions = predictions.cpu().numpy()
                predictions_1d = []
                for pred in predictions:
                    predictions_1d.append(MV.vectorize(pred))
                predictions_1d = np.stack(predictions_1d, axis=0)
                assert predictions_1d.shape[0] == 112 and predictions_1d.shape[1] == 35778
                predictions_1d = predictions_1d.flatten()
                submission_df = pd.DataFrame({'ID': range(1, len(predictions_1d) + 1), 'Predicted': predictions_1d})
                submission_df.to_csv(f"submission_fold_{i + 1}.csv", index=False)
        else:
            model = Model(model_params).to(device)
            model.load_state_dict(torch.load(f"model_full.pt"))

            # Predict test data
            print("Predicting test data...")
            test_transform = val_transform
            test_dataset = BrainTest(test_transform)
            test_dl = DataLoader(test_dataset, batch_size=flags.batch_size)
            predictions = predict(model, test_dl, device)

            # Format test data
            print("Formatting test data...")
            predictions = predictions.cpu().numpy()
            predictions_1d = []
            for pred in predictions:
                predictions_1d.append(MV.vectorize(pred))
            predictions_1d = np.stack(predictions_1d, axis=0)
            assert predictions_1d.shape[0] == 112 and predictions_1d.shape[1] == 35778
            predictions_1d = predictions_1d.flatten()
            submission_df = pd.DataFrame({'ID': range(1, len(predictions_1d) + 1), 'Predicted': predictions_1d})
            submission_df.to_csv(f"submission_full.csv", index=False)
    '''

    stop = datetime.now()
    print("Time Elapse :", stop-now)

if __name__ == "__main__":
    main()
