import numpy as np
import pandas as pd


def data_split(em_data_names, valid_name, greater_than_1):
    for em_name in em_data_names:
        em_data = pd.read_excel(em_name)
        print("读取到的文件为：\n", em_name)
        valid_em_data = em_data.iloc[greater_than_1:, :]
        print("有效的1~8区间的数据:\n", valid_em_data)
        np.save('reproducted/'+valid_name+em_name.split('.')[0], valid_em_data.values)


if __name__ == '__main__':
    data = pd.read_excel('origin_data_16plus50/手动选择的偏移量.xlsx', header=None, sheet_name='Sheet1')
    np.save('reproducted/valid_origin_data_16plus50/shift_value_manual.npy', data.values)
    print(data)

    # # 文件列表
    # em_data_list = ['origin_data_16plus50/30_all.xlsx', 'origin_data_16plus50/90_all.xlsx']
    # # 文件裁剪
    # data_split(em_data_list, 'valid_', 2000-1)



