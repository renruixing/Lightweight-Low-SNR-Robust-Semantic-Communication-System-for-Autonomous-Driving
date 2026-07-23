"""
common/backbone_prunable.py
=============================
支持可配置瓶颈宽度（conv6输出通道数 / conv_trans1输入通道数）的
Encoder / Decoder，用于结构化剪枝流程。

关键架构判断（已用参数量核对过）：原始Encoder里**只有bn6一个
BatchNorm层**（conv1~conv5是纯Conv+ReLU，没有BN），所以论文的
"基于BN scaling factor的结构化剪枝"实际上只剪conv6的512个输出
通道（即语义特征向量z的长度k本身），不涉及conv1~conv5。这跟
manuscript.tex "remove channels...along with corresponding kernels
in the adjacent convolutional layers"的表述一致（"adjacent"指的是
decoder侧紧邻的conv_trans1，不是encoder内部其他层）。

conv1~conv5、conv_trans2~conv_trans6结构不变，跟common/backbone.py
完全一致；变化只在conv6/bn6的输出通道数、conv_trans1的输入通道数。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableEncoder(nn.Module):
    def __init__(self, bottleneck_channels: int = 512):
        super().__init__()
        self.bottleneck_channels = bottleneck_channels
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=5, stride=1, padding=2)
        self.conv6 = nn.Conv2d(256, bottleneck_channels, kernel_size=5, stride=1, padding=2)
        self.bn6 = nn.BatchNorm2d(bottleneck_channels)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return torch.sigmoid(self.bn6(self.conv6(x)))


class PrunableDecoder(nn.Module):
    def __init__(self, bottleneck_channels: int = 512):
        super().__init__()
        self.conv_trans1 = nn.ConvTranspose2d(bottleneck_channels, 256, kernel_size=5, stride=1, padding=2)
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
        return torch.sigmoid(self.conv_trans6(x))


def build_pruned_state_dict(full_encoder: PrunableEncoder, full_decoder: PrunableDecoder,
                             keep_indices: torch.Tensor):
    """
    根据keep_indices（要保留的通道下标，长度为剪枝后的宽度）从满宽度
    模型里切出对应的encoder/decoder权重，用于初始化剪枝后的窄模型。

    需要联动裁剪的地方：
        - encoder.conv6.weight: (out_ch, in_ch, k, k) → 按out_ch维裁剪
        - encoder.conv6.bias:   (out_ch,) → 按out_ch维裁剪
        - encoder.bn6.*:        (out_ch,) → 按out_ch维裁剪
        - decoder.conv_trans1.weight: (in_ch, out_ch, k, k) → 按in_ch维裁剪
          （ConvTranspose2d的weight形状是[in_channels, out_channels, k, k]，
          跟普通Conv2d的[out,in,k,k]是反过来的，容易搞错，这里特别注明）
        - decoder.conv_trans1.bias: 不受影响（bias维度是out_channels=256，不变）
    其余层（conv1~conv5, conv_trans2~conv_trans6）原样复制，不受剪枝影响。
    """
    new_state = {}

    enc_state = full_encoder.state_dict()
    for k, v in enc_state.items():
        if k == "conv6.weight":
            new_state[k] = v[keep_indices].clone()
        elif k in ("conv6.bias", "bn6.weight", "bn6.bias", "bn6.running_mean", "bn6.running_var"):
            new_state[k] = v[keep_indices].clone()
        elif k == "bn6.num_batches_tracked":
            new_state[k] = v.clone()
        else:
            new_state[k] = v.clone()

    dec_state = full_decoder.state_dict()
    for k, v in dec_state.items():
        if k == "conv_trans1.weight":
            new_state["dec_" + k] = v[keep_indices].clone()  # 按in_channels维（dim0）裁剪
        else:
            new_state["dec_" + k] = v.clone()

    enc_new = {k: v for k, v in new_state.items() if not k.startswith("dec_")}
    dec_new = {k[len("dec_"):]: v for k, v in new_state.items() if k.startswith("dec_")}
    return enc_new, dec_new
