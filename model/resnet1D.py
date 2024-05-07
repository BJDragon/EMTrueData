from torch import nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(out_channel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                # nn.BatchNorm2d(out_channel)
            )
        self.leakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = self.leakyReLU(out)
        return out


class _resnet(nn.Module):
    def __init__(self, ResidualBlock, block_num, num_classes):
        super(_resnet, self).__init__()
        self.in_channel = 32
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            # nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 32, block_num[0], stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 64, block_num[1], stride=1)
        self.layer3 = self.make_layer(ResidualBlock, 128, block_num[2], stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, block_num[3], stride=2)
        self.avg = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Sequential(nn.Linear(256, num_classes),
        #                         nn.Softmax(dim=1))
        self.fc = nn.Sequential(nn.Linear(256, num_classes))

    def make_layer(self, block, channels, num_blocks, stride):
        # strides=[1, 1] or [2, 1]
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channel, channels, stride))
            self.in_channel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet181D(num_classes=4, block_num=[1, 1, 1, 1]):
    return _resnet(ResidualBlock, num_classes=num_classes, block_num=block_num)
