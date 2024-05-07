import torch.nn as nn



class DynamicNN(nn.Module):
    def __init__(self, layer_sizes):
        super(DynamicNN, self).__init__()

        # 根据给定的数组构建网络层
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.Tanh())  # 添加激活函数，除了最后一层

        self.model = nn.Sequential(*layers)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        return self.softmax(self.model(x))


class FCNN(nn.Module):
    """全连接网络"""

    def __init__(self, input_size, output_size):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 200)
        self.fc2 = nn.Linear(200, 500)
        self.fc3 = nn.Linear(500, 200)
        self.fc4 = nn.Linear(200, 50)
        self.fc5 = nn.Linear(50, output_size)
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.res3 = nn.Linear(500, 200)
        self.res4 = nn.Linear(200, 50)

    def resblock(self, x, res):
        """残差块"""
        return x + res

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.resblock(self.fc3(x), self.res3(x))
        x = self.tanh(x)
        x = self.resblock(self.fc4(x), self.res4(x))
        x = self.tanh(x)
        x = self.fc5(x)
        return self.softmax(x)




class SimpleCNN(nn.Module):
    def __init__(self, num_classes=1):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv1_relu = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv2_relu = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.conv3_relu = nn.LeakyReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1_relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv2_relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.conv3_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class Fully_Connected(nn.Module):
    """全连接网络"""
    def __init__(self, input_size, output_size):
        super(Fully_Connected, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)

        self.sigmoid = nn.Sigmoid()
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        x = self.fc4(x)
        # x = self.sigmoid(x)
        x = self.fc5(x)
        return x