from globals import *

import torch
import numpy as np
import matplotlib.pyplot as plt


x_ticks = list()
tick_positions = list()


def show_evaluation(net, dataset, target, scaler, debug=True):
    ''' Evaluates performance of the RNN on the entire
        dataset, and shows the prediction as well as
        target values.

    Arguments:
        net (nn.Module): RNN to evaluate
        dataset (numpy.ndarray): dataset
        target (numpy.ndarray): target values for prediction,
                                original (unscaled)
        scaler (MinMaxScaler): used for denormalization
        debug (bool): should we calculate/display eval.
                      MSE/MAE
    '''
    net.eval()
    TRAIN_SPLIT = int(config.train_ratio*len(target))
    VAL_SPLIT = TRAIN_SPLIT + int(config.val_ratio*len(target))

    COL_NUM = int((dataset.shape[-1] - 1)/config.lag)

    # Adapt the entire dataset for the PyTorch LSTM
    total_set = torch.Tensor(dataset[:,:-1]).view(-1, config.lag, COL_NUM).to(device)
    # Prediction on the entire dataset
    prediction = net(total_set).unsqueeze(-1).cpu().data.numpy()

    scaling_temp = np.concatenate([prediction, dataset[:, -8:-1]], axis=1)
    prediction = scaler.inverse_transform(scaling_temp)[:, 0]

    # Plotting the original sequence vs. predicted
    plt.figure(figsize=(8,5))
    plt.plot(range(0, len(target)), target)
    plt.plot(range(config.lag, len(target)), prediction)
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.axvline(x=TRAIN_SPLIT, c='g', linestyle='-')
    plt.axvline(x=VAL_SPLIT, c='r', linestyle='-')
    plt.title('Multivariate Time-Series Forecast')
    plt.xlabel('Year-Month-Day')
    plt.ylabel("Level of pollution ( pm2.5 )")
    plt.legend(['Ground truth', 'Prediction', 'Train-Val split', 'Test split'])
    plt.show()

    if debug:
        # Calculating total MSE & MAE
        train_mse = (np.square(prediction[:TRAIN_SPLIT] - target[:TRAIN_SPLIT])).mean()
        train_mae = (np.abs(prediction[:TRAIN_SPLIT] - target[:TRAIN_SPLIT])).mean()
        # Calculating train MSE & MAE
        val_mse = (np.square(prediction[TRAIN_SPLIT:VAL_SPLIT] - target[TRAIN_SPLIT:VAL_SPLIT])).mean()
        val_mae = (np.abs(prediction[TRAIN_SPLIT:VAL_SPLIT] - target[TRAIN_SPLIT:VAL_SPLIT])).mean()
        # Calculating test MSE & MAE
        test_mse = (np.square(prediction[VAL_SPLIT:] - target[VAL_SPLIT+config.lag:])).mean()
        test_mae = (np.abs(prediction[VAL_SPLIT:] - target[VAL_SPLIT+config.lag:])).mean()

        print(f"Train MSE:  {train_mse:.4f}  |  Train MAE:  {train_mae:.4f}")
        print(f"Val MSE:  {val_mse:.4f}  |  Val MAE:  {val_mae:.4f}")
        print(f"Test MSE:  {test_mse:.4f}   |  Test MAE:  {test_mae:.4f}")


def show_loss(history):
    ''' Display train and evaluation loss

    Arguments:
        history(dict): Contains train and test loss logs
    '''
    plt.figure(figsize=(8, 5))
    plt.plot(history['train_loss'], label='Train loss')
    plt.plot(history['test_loss'], label='Evaluation loss')

    plt.title('Mean Absolute Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()


def display_dataset(dataset, xlabels):
    ''' Displays the loaded data

    Arguments:
        dataset(numpy.ndarray): loaded data
        xlabels(numpy.ndarray): strings representing
                                 according dates
    '''
    global x_ticks
    global tick_positions

    # Remove information about hours (only for plotting purposes)
    xlabels = [x[:10] for x in xlabels]
    # We can't show every date in the dataset
    # on the x axis because we couldn't see
    # any label clearly. So we extract every
    # n-th label/tick
    segment = int(len(dataset) / 6)

    for i, date in enumerate(xlabels):
        if i > 0 and (i + 1) % segment == 0:
            x_ticks.append(date)
            tick_positions.append(i)
        elif i == 0:
            x_ticks.append(date)
            tick_positions.append(i)

    # Display loaded data
    plt.figure(figsize=(8, 5))
    plt.plot(dataset)
    plt.title('Pollution in Beijing')
    plt.xlabel('Year-Month-Day')
    plt.ylabel("Level of pollution ( pm2.5 )")
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.show()
