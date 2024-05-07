import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def load_data():
    d30 = np.transpose(np.load('reproducted/valid_30度.npy'))
    d90 = np.transpose(np.load('reproducted/valid_90度.npy'))
    n = d30.shape[0]
    l = d30.shape[1]
    return d30, d90, [n, l]


# 简单移动平均函数
def moving_average(signal, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(signal, window, 'same')


def findDataPeaksIndex(data, height=10, prominence=1):
    peak_indices, peak_values = find_peaks(-data, height=height, prominence=prominence)
    return peak_indices, peak_values


def shiftFrequency(peak_indices_1, peak_indices_2):
    # for peak in list(peak_indices_2):
    frequency_result = []
    shift_result = []
    for peak2 in peak_indices_2:
        p = [[peak2, peak2 - peak1] for peak1 in peak_indices_1 if abs(peak2 - peak1) <= 100]
        # print(p)
        if len(p) == 0:
            continue
        else:
            frequency_result.append(p[0][0])
            shift_result.append(p[0][1])
    print(frequency_result)
    print(shift_result)
    return frequency_result, shift_result


def scale_x(x, len_x):
    if type(x) == list:
        x = np.array(x)
    x = 7 * (x / float(len_x)) + 1
    return x


def write_to_excel(data_file, data):
    # 使用pandas的ExcelWriter，指定文件名和引擎
    with pd.ExcelWriter(data_file, engine='openpyxl') as writer:
        # 逐行处理每个列表
        for idx, row in enumerate(data):
            # 将当前列表转换为DataFrame
            df = pd.DataFrame([row])
            # 将DataFrame写入Excel文件
            # 如果是第一行，写入header；否则，继续追加，并且不包括header
            df.to_excel(writer, startrow=idx, index=False, header=False)
        # 保存Excel文件
        writer._save()