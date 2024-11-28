# Imports
import torch
from gan.config import Args
from gan.model import GUS
from gan.preprocessing import degree_normalisation
from utils import three_fold_cross_validation
from set_seed import set_seed
from evaluation import load_random_files


# Configurations
# Set a fixed random seed for reproducibility across multiple libraries
random_seed = 42
set_seed(random_seed)

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available. Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA not available. Using CPU.")

# preprocessing of the data
args = Args()
args.device = device
args.normalisation_function = degree_normalisation

splits = load_random_files(args,return_matrix=True)

# print the args
print(args)

def model_init():
    model = GUS(args.ks, args).to(device)
    return model


# run the 3-fold cross-validation
cv_scores = three_fold_cross_validation(model_init, splits, random_state=random_seed,
                                        verbose=True, prediction_vector=False, label_vector=False)

print(cv_scores)
