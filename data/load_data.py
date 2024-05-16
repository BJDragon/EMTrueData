import numpy as np


def read_data(x_file_path, y_file_path):
    x = np.load(x_file_path)
    y = np.load(y_file_path)
    return x, y


def data_transform(x, y):
    x = x.reshape(-1, 1, 5, 5) - 0.1
    # y = y * 10
    # x = x.reshape(-1, 1, 25) - 0.1
    # x = np.concatenate([x[:16], x[17:31], x[32:]])
    # y = np.concatenate([y[:16], y[17:31], y[32:]])
    # y = np.abs(y)
    # y = 10/y
    # y = 0.5*y/(np.max(y)-np.min(y))
    # print(y)
    return x, y
