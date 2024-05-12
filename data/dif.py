import csv

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

from utils import *

d30, d90, [n, l] = load_data(d30_path='reproducted/valid_origin_data_16plus50/30_all.npy',
                             d90_path='reproducted/valid_origin_data_16plus50/90_all.npy',)
# freq = np.load('reproducted/freq.npy')[2000-1:]
frequency_result_data = []
shift_result_data = []
# 创建一个带有噪声的正弦波信号
freq_axis = np.linspace(1, 8, l)
for i in range(n):
    data_30 = d30[i, :]
    data_90 = d90[i, :]
    # data_30 = moving_average(data_30, 160)
    # data_90 = moving_average(data_90, 160)

    height = 12
    prominence = 1

    peak_indices_1, _ = findDataPeaksIndex(data_30, height, prominence)
    peak_indices_2, _ = findDataPeaksIndex(data_90, height, prominence)

    frequency_result, shift_result = shiftFrequency(peak_indices_1, peak_indices_2)
    frequency_result_data.append(frequency_result)
    shift_result_data.append(shift_result)
    # 结果绘制
    plt.figure()
    plt.plot(freq_axis, data_30, label='T=30', color='#1f77b4', linewidth=2.0)
    plt.plot(scale_x(peak_indices_1, l), data_30[peak_indices_1], "x")

    plt.plot(freq_axis, data_90, label='T=90', color='#ff7f0e', linewidth=1.0)
    plt.plot(scale_x(peak_indices_2, l), data_90[peak_indices_2], "x")

    # 遍历峰值索引和数据，为每个点添加注解
    for j, peak in enumerate(zip(scale_x(frequency_result, l), data_90[frequency_result])):
        x_pos, y_pos = peak
        plt.annotate(f'{shift_result[j]/2}MHz', xy=(x_pos, y_pos), xytext=(20, -5), textcoords='offset points',
                     arrowprops=dict(facecolor='black', shrink=1, width=0.5, headwidth=5))
    plt.title(f'{i + 1}')
    plt.legend()
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('S11 (dB)')
    plt.savefig(f'pngs/valid_origin_data_16plus50/NO_{i + 1}.png', dpi=600)  # 保存为PNG格式，dpi设置分辨率
    plt.show()

write_to_excel('shift_result/valid_origin_data_16plus50/frequency_result_data.xlsx', frequency_result_data)
write_to_excel('shift_result/valid_origin_data_16plus50/shift_result_data.xlsx', shift_result_data)
df = pd.read_excel('shift_result/valid_origin_data_16plus50/shift_result_data.xlsx', header=None)
# 将DataFrame转换为JSON格式
# df.to_json('valid_origin_data_16plus50/data.json', orient='records')
