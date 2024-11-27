import torch
from train import *
from evaluation import *


if __name__ == '__main__':
    # Check for CUDA (GPU support) and set device accordingly
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # All hyperparameters
    args = {'device': device, 'random_seed': 42, 'epochs': 200, 'batch_size': 4, 'lr': 2.5e-4, 'eta_min': 5e-5,
            'node_drop': 0.06, 'edge_drop': 0.06, 'ks': [0.9, 0.7, 0.6, 0.5], 'kss': [0.8375, 0.7, 0.6, 0.5],
            'lr_dim': 160, 'hr_dim': 320, 'hidden_dim': 320, 'padding': 26, 'lambda': 0.1, 'dropout': 0.1,
            'mean_dense': 0, 'std_dense': 0.01, 'mean_gaussian': 0, 'std_gaussian': 0.1,}
    
    models, swa_models, preds, trues, preds_swa, trues_swa = train(args)
    kfold_evaluation_measure(preds, trues, k=3)
    kfold_evaluation_measure(preds_swa, trues_swa, k=3, file='_swa')

    model_full, swa_model_full = best_train(args)

    # Save the model
    count = 1
    for model in models:
        torch.save(model, f'./trained/model_{count}.pth')
        count += 1
    count = 1
    for model in swa_models:
        torch.save(model, f'./trained/swa_model_{count}.pth')
        count += 1
    torch.save(model_full, f'./trained/model_4.pth')
    torch.save(swa_model_full, f'./trained/swa_model_4.pth')
    
    # Evaluate the model
    print()
    print("Evaluating the model...")
    for i in range(1, 5):
        print()
        print(f"Model {i}:")
        seed_everything(args['random_seed'])
        model = GraphLoong(args).to(args['device'])
        model.load_state_dict(torch.load(f'./trained/model_{i}.pth'))
        evaluate(model, args, f'{i}')
        predict(model, args, f'{i}')
    print()
    print("Evaluating the model with SWA...")
    for i in range(1, 5):
        print()
        print(f"Model {i}:")
        seed_everything(args['random_seed'])
        model = AveragedModel(GraphLoong(args).to(args['device']))
        model.load_state_dict(torch.load(f'./trained/swa_model_{i}.pth'))
        evaluate(model, args, f'swa_{i}')
        predict(model, args, f'swa_{i}')