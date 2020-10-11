from globals import *
from utils.utils import *
from utils.visualisation import show_evaluation

from model import PollutionForecast

if __name__ == "__main__":
    # Load the dataset
    dataset, scaler = load_dataset(config.dataset_path, show_data=True)
    # Prepare the dataset for training/testing
    subsequences = extract_subsequences(dataset['scaled'], 
                                        dataset['column_names'], 
                                        lag=config.lag)
    # Split the dataset into train/validation/test set
    train_loader, val_set, test_set = train_val_test_split(subsequences)

    # # Train the network
    if config.mode == "train":
        # Create new instance of the RNN
        net = PollutionForecast(input_dim=train_loader.dataset[0][0].shape[-1],
                                hidden_dim=config.hidden_dim,
                                num_layers=config.num_layers).to(device)
        # Train the model
        train_loop(net, config.epochs, config.lr, config.wd,
                   train_loader, val_set, debug=True)
    else:
        # Create new instance of the RNN using default values
        net = PollutionForecast(input_dim=train_loader.dataset[0][0].shape[-1],
                                hidden_dim=parser.get_default('hidden_dim'),
                                num_layers=parser.get_default('num_layers')).to(device)
        # Load pretrained weights
        net.load_state_dict(torch.load(
            config.pretrained_path, map_location=device))

    # Display the prediction next to the target output values
    show_evaluation(net, subsequences, dataset['target'], scaler, debug=True)
