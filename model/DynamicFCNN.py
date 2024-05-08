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