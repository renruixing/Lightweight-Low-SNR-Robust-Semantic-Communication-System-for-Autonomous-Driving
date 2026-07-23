"""
common/backbone.py
===================
Deep JSCC的 Encoder / Decoder 网络定义。

这份定义**与你现有 model.py 中的 Encoder / Decoder 类完全等价**
（层名、通道数、kernel、stride、padding、激活函数全部一一对应），
唯一的区别是：这里不import Channel/utils/GDN，Experiments目录可以
独立运行，不用把整个Codes代码库都拖过来。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(512)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.bn6(self.conv6(x)))
        return x


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=5, stride=1, padding=2)
        self.conv_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=5, stride=1, padding=2)
        self.conv_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2, padding=2)
        self.conv_trans4 = nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2, padding=1)
        self.conv_trans5 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2, padding=1)
        self.conv_trans6 = nn.ConvTranspose2d(16, 3, kernel_size=6, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv_trans1(x))
        x = F.relu(self.conv_trans2(x))
        x = F.relu(self.conv_trans3(x))
        x = F.relu(self.conv_trans4(x))
        x = F.relu(self.conv_trans5(x))
        x = self.conv_trans6(x)
        return torch.sigmoid(x)
