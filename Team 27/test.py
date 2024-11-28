# This file is for quickly running test scripts of the project

from evaluation import EvaluationUtil

import random
import numpy as np
import torch


if __name__ == '__main__':
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

    eu = EvaluationUtil()
    # Evaluating the model is to do the 3F-CV, generating the .csv files
    eu.evaluate_model('Att-Multi')
    # Generating the plot images
    eu.plot_metrics_final()
