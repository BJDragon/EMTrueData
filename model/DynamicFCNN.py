import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicFCNN(nn.Module):
    def __init__(self, input_size, output_size, num_layers, hidden_units, activation):
        super(DynamicFCNN, self).__init__()
        layers = []
        in_features = input_size

        # 创建网络层
        for i in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_units[i]))
            if activation[i] == 'relu':
                layers.append(nn.ReLU())
            elif activation[i] == 'tanh':
                layers.append(nn.Tanh())
            elif activation[i] == 'sigmoid':
                layers.append(nn.Sigmoid())
            in_features = hidden_units[i]

        # 输出层
        layers.append(nn.Linear(hidden_units[-1], output_size))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class FCNN(nn.Module):
    def __init__(self, output_size):
        super(FCNN, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(25, 128),  # 输入层到第一个隐藏层, 128个神经元
            nn.LeakyReLU(0.01),  # 使用LeakyReLU激活函数，小斜率设置为0.01
            nn.Linear(128, 256),  # 第二个隐藏层, 256个神经元
            nn.Tanh(),  # LeakyReLU激活函数
            nn.Linear(256, 512),  # 第三个隐藏层, 512个神经元
            nn.LeakyReLU(0.01),  # LeakyReLU激活函数
            nn.Linear(512, 256),  # 第四个隐藏层, 256个神经元
            nn.Tanh(),  # LeakyReLU激活函数
            nn.Linear(256, 128),  # 第五个隐藏层, 128个神经元
            nn.LeakyReLU(0.01),  # LeakyReLU激活函数
            nn.Linear(128, 1)  # 输出层, 1个输出
        )

    def forward(self, x):
        return self.layers(x)
