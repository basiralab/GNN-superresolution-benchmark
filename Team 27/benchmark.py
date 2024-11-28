import argparse
import numpy as np
import torch
import model, train
from dataprocessor import DataProcessor as D


class BenchmarkUtil:
    def __init__(self):
        pass

    @staticmethod
    def get_att_dct_model():
        """
        Produce a final model using the basic arguments.
        """
        args = BenchmarkUtil.baseline_args()
        model = model.AGSRNet(
            [0.9, 0.7, 0.6, 0.5],
            args
        )
        return model, args

    @staticmethod
    def get_att_dct_train_function():
        return train.train


    @staticmethod
    def gen_att_multi_dct():
        """
        Train the model on the full training set, then do prediction.
        """
        model, args = BenchmarkUtil.get_att_dct_model()

        lr_train = D.load_data_matrices("data/lr_train.csv", 160)
        hr_train = D.load_data_matrices("data/hr_train.csv", 268)

        loss, error = train.train(model, lr_train, hr_train, args)

        lr_test = D.load_data_matrices("data/lr_test.csv", 160)

        preds_list = []
        for lr in lr_test:
            lr = torch.from_numpy(lr).type(torch.FloatTensor)
            preds, _, _, _ = model(lr, 160, 320)
            preds_unpad = preds[26:294, 26:294]

            preds_list.append(preds_unpad.detach().numpy())
        preds_array = np.array(preds_list)
        D.save_kaggle_csv(preds_array, "data/test_att_multi_dct.csv")
        return loss, error

    @staticmethod
    def baseline_args():
        parser = argparse.ArgumentParser(description='AGSR-Net')
        parser.add_argument('--epochs', type=int, default=1000, metavar='no_epochs',
                            help='number of episode to train ')
        parser.add_argument('--lr', type=float, default=0.0001, metavar='lr',
                            help='learning rate (default: 0.0001 using Adam Optimizer)')
        parser.add_argument('--lmbda', type=float, default=0.1, metavar='L',
                            help='self-reconstruction error hyperparameter')
        parser.add_argument('--lr_dim', type=int, default=160, metavar='N',
                            help='adjacency matrix input dimensions')
        parser.add_argument('--hr_dim', type=int, default=320, metavar='N',
                            help='super-resolved adjacency matrix output dimensions')
        parser.add_argument('--hidden_dim', type=int, default=320, metavar='N',
                            help='hidden GraphConvolutional layer dimensions')
        parser.add_argument('--padding', type=int, default=26, metavar='padding',
                            help='dimensions of padding')
        parser.add_argument('--mean_dense', type=float, default=0., metavar='mean',
                            help='mean of the normal distribution in Dense Layer')
        parser.add_argument('--std_dense', type=float, default=0.01, metavar='std',
                            help='standard deviation of the normal distribution in Dense Layer')
        parser.add_argument('--mean_gaussian', type=float, default=0., metavar='mean',
                            help='mean of the normal distribution in Gaussian Noise Layer')
        parser.add_argument('--std_gaussian', type=float, default=0.1, metavar='std',
                            help='standard deviation of the normal distribution in Gaussian Noise Layer')

        return parser.parse_args()
