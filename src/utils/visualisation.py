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
    total_train_size = int(config.split_ratio*len(target))
    COL_NUM = int((dataset.shape[-1] - 1)/config.lag)
    # Prediction on the entire dataset
    total_set = torch.Tensor(dataset[:,:-1]).view(-1, config.lag, COL_NUM).to(device)
    # Prediction on the test set
    test_predict = net(total_set).unsqueeze(-1).cpu().data.numpy()

    scaling_temp = np.concatenate([test_predict, dataset[:, -8:-1]], axis=1)
    test_predict = scaler.inverse_transform(scaling_temp)[:, 0]

    # Plotting the original sequence vs. predicted
    plt.figure(figsize=(8,5))
    plt.plot(range(0, len(target)), target)
    plt.plot(range(config.lag, len(target)), test_predict)
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.axvline(x=total_train_size, c='r', linestyle='-')
    plt.title('Multivariate Time-Series Forecast')
    plt.xlabel('Year-Month')
    plt.ylabel("Level of pollution")
    plt.legend(['Train-Test split', 'Target', 'Prediction'])
    plt.show()

    if debug:
        # Calculating test MSE & MAE
        test_mse = (np.square(test_predict[total_train_size:] - 
                              target[total_train_size+config.lag:])).mean()
        test_mae = (np.abs(test_predict[total_train_size:] - 
                              target[total_train_size+config.lag:])).mean()
                              
        print(f"Test MSE:   {test_mse:.4f}  |  Test MAE:   {test_mae:.4f}")


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
    plt.xlabel('Year-Month')
    plt.ylabel("pm2.5")
    plt.xticks(tick_positions, x_ticks, size='small')
    plt.show()
