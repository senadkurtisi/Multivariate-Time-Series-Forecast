import torch
from torch.nn import L1Loss
from argparse import ArgumentParser

# Device on which we port the model and variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Loss function: L1loss (mean absolute error)
loss_func = L1Loss()

parser = ArgumentParser()
parser.add_argument("--mode", type=str, choices=["train", "eval"],
                    default="eval", help="Are we training or testing?")
parser.add_argument("--pretrained_path", type=str,
                    default="pretrained/pollution_net.pt",
                    help="Location of the pretrained net")
# DATASET PREPARATION ARGUMENTS
parser.add_argument("--dataset_path", type=str, default="data/pollution.csv",
                    help="Location of the dataset file")
parser.add_argument("--lag", type=int, default=2,
                    help="Time lag used for preparing train and test X-Y pairs")
parser.add_argument("--train_ratio", type=float, default=0.5,
                    help="Ratio for extracting the train set")
parser.add_argument("--val_ratio", type=float, default=0.25,
                    help="Ratio for extracting the validation set")
# MODEL ARGUMENTS
parser.add_argument("--hidden_dim", type=int, default=6,
                    help="Dimension of the LSTM hidden state vector")
parser.add_argument("--num_layers", type=int, default=1,
                    help="Number of LSTM layers")
# TRAINING ARGUMENTS
parser.add_argument("--epochs", type=int, default=260,
                    help="Number of training epochs")
parser.add_argument("--bs", type=int, default=512,
                    help="Batch size")
parser.add_argument("--lr", type=float, default=2e-4,
                    help="Learning rate")
parser.add_argument("--wd", type=float, default=3e-10,
                    help="L2 regularization weight decay")

# Used to store training/model/dataset hyperparameters
config = parser.parse_args()
