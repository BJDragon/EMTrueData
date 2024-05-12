import pandas as pd
import os
import csv
import numpy as np


def read_csv_filename(folder_path):
    """
    读取文件夹中的CSV文件，并返回文件名列表
    """

    # 获取文件夹中所有文件的列表
    file_list = os.listdir(folder_path)

    # 筛选出CSV文件
    csv_files = [file for file in file_list if file.endswith('.csv')]
    file_names = [os.path.splitext(file)[0] for file in file_list if file.endswith('.csv')]

    # 打印CSV文件名
    print('CSV files in folder:\n', folder_path)
    for csv_file in csv_files:
        print(csv_file)
    return csv_files, file_names


def read_csv_data(csv_file, freq_needed=False):
    """
    读取CSV文件，并返回数据列表
    """
    # 打开CSV文件
    with open(csv_file, 'r') as file:
        csv_reader = csv.reader(file)
        target_column_index = 2
        current_row = 0
        target_row = 16
        data_list = []
        freq = []

        # 读取文件中的数据
        for row in csv_reader:
            if current_row < target_row:
                current_row = current_row + 1
            else:
                data_list.append(float(row[target_column_index]))
                if freq_needed != False:
                    freq.append(float(row[target_column_index - 1]))
        # print(row[target_column_index])
        # print(len(data_list))
        # print(len(freq))
        print(csv_file, 'finished.')
        return data_list, freq


def read_data_to_df(folder_path, csv_file_name):
    """
    针对于网分仪导出的csv格式数据，输入指定文件夹，自动读取其中的波形数据与文件名称，并将结果保存进指定的csv文件当中
    """
    csv_files, titles = read_csv_filename(folder_path)
    # 预设将变量放置在已经存储freq的变量当中
    _, freq = read_csv_data(folder_path + csv_files[0], freq_needed=True)
    print('freq has been saved.')
    saved_data_list = np.array(freq).reshape(-1, 1)

    # # 按照文件名的总长除以一半
    # for i in range(len(csv_files)):
    #     pass
    for csv_file in csv_files:
        # 只需要读取波形数据以及将其拼接即可
        data_list, _ = read_csv_data(folder_path + csv_file, freq_needed=False)
        saved_data_list = np.concatenate((saved_data_list, np.array(data_list).reshape(-1, 1)), axis=1)

    saved_data_list = np.array(saved_data_list)

    # 创建数据框
    csv_data_titles = ['freq'] + titles
    df = pd.DataFrame(saved_data_list, columns=csv_data_titles)

    # 保存数据框到CSV文件
    df.to_csv(csv_file_name, index=False)

    print(f"Data saved to '{csv_file_name}'")


def data_split(em_data_path: str, valid_em_data_save_path: str):
    """
    将em数据按照频率大于1的频率进行截取，并保存到新的csv文件中
    """
    em_data = pd.read_csv(em_data_path)

    print("读取到的文件为：\n", em_data)

    # 获取列数
    column_count = em_data.shape[1]
    print("总列数为：", column_count)

    freq = em_data[em_data.columns[0]]
    greater_than_1 = freq[freq > 1].index[0]
    print("频率大于1的频率索引：", greater_than_1)

    # 有效的1~8区间的数据截取列
    valid_em_data = em_data.iloc[greater_than_1:, :]
    print("有效的1~8区间的数据:\n", valid_em_data)

    valid_em_data.to_csv(valid_em_data_save_path, index=False)  # 不包含索引列


if __name__ == '__main__':
    read_data_to_df(r'C:\Users\22965\LocalPC\Coding_Metamaterials\0424实测数据文档汇聚\data\origin_data\0416全部测量完成/',
                    'test.csv')
    # data_split(em_data_path, valid_em_data_save_path)
