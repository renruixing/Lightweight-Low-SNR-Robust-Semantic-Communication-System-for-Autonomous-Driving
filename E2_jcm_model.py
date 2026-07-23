"""
E2_jcm_model.py
================
【实验E2】JCM（Joint Coding-Modulation）风格的可微分量化调制模型。

方法来源（非本人推导，来自原文，公式编号对应原文）：
    Y. Bo, Y. Duan, S. Shao, M. Tao, "Joint Coding-Modulation for Digital
    Semantic Communications via Variational Autoencoder,"
    IEEE Trans. Commun., 72(9):5626-5640, 2024. (arXiv:2310.06690)
    也是审稿人1指的[17]/[18]。

核心机制：
1. 编码器最后一层不再输出单一连续值 z_i ∈ (0,1)，而是对每个信道使用位置
   输出M-QAM星座点的类别分布logits（I/Q两路各√M元分布）。
2. 训练时用Gumbel-Softmax重参数化（PyTorch自带F.gumbel_softmax），
   前向硬采样、反向软化近似，使梯度可以直接反传，不需要"训练-部署分离"。
3. 本任务无分类头，原文式(13)损失退化为纯MSE重建损失。
4. 信道复用 common/channel.py 里统一的复瑞利+AWGN实现，与E0/E1同一套。
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.backbone import Encoder, Decoder
from common.channel import rayleigh_awgn_channel_complex


class JCMEncoderHead(nn.Module):
    """
    替换E0/Encoder最后一层conv6+bn6+sigmoid的可微分量化调制头。

    输入：conv5的输出特征图 (B, 256, 32, 32)（与E0完全一致）
    输出：每个信道使用位置的I/Q两路类别分布logits (B, k, 2, √M)
         其中 k = 512 * 32 * 32 = 524288，与论文k/n=2/3设置一致
    """
    def __init__(self, sqrt_M: int, in_channels: int = 256, num_positions: int = 512):
        super().__init__()
        assert sqrt_M in (2, 4, 8, 16), \
            "sqrt_M ∈ {2,4,8,16} 对应 4/16/64/256 QAM"
        self.sqrt_M = sqrt_M
        self.num_positions = num_positions
        out_channels = num_positions * 2 * sqrt_M
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2)

    def forward(self, feat):
        B = feat.shape[0]
        h = self.proj(feat)  # (B, 512*2*√M, 32, 32)
        h = h.view(B, self.num_positions, 2, self.sqrt_M, feat.shape[2], feat.shape[3])
        # 空间维并入信道使用位置维，得到 k = 512*32*32 个位置
        h = h.permute(0, 1, 4, 5, 2, 3).contiguous()
        h = h.view(B, self.num_positions * feat.shape[2] * feat.shape[3], 2, self.sqrt_M)
        return h


def qam_constellation_levels(sqrt_M: int, device):
    """
    生成归一化后的M-QAM I/Q单路电平（原文命题2）:
        c_r = (2r+1)/(√M - 1),  r = -√M/2, ..., √M/2 - 1
    均值0，与common/qam.py里非可微版本的星座点对应一致。
    """
    r = torch.arange(-sqrt_M // 2, sqrt_M // 2, device=device, dtype=torch.float32)
    return (2 * r + 1) / (sqrt_M - 1)


def jcm_sample_symbols(logits, tau=1.5, hard=True):
    """
    Gumbel-Softmax采样，生成可微分的复数星座符号（对应原文式16-18）。

    参数
    ----
    logits : (B, k, 2, √M)，JCMEncoderHead的输出
    tau    : Gumbel-Softmax温度（原文取1.5）
    hard   : True → 前向硬采样(one-hot)、反向软化梯度(straight-through)

    返回
    ----
    z : (B, k) 复数张量（I + jQ）
    """
    B, k, _, sqrt_M = logits.shape
    levels = qam_constellation_levels(sqrt_M, logits.device)

    logits_I = logits[:, :, 0, :]
    logits_Q = logits[:, :, 1, :]
    onehot_I = F.gumbel_softmax(logits_I, tau=tau, hard=hard, dim=-1)
    onehot_Q = F.gumbel_softmax(logits_Q, tau=tau, hard=hard, dim=-1)

    z_I = onehot_I @ levels
    z_Q = onehot_Q @ levels
    return torch.complex(z_I, z_Q)


class JCMEncoder(nn.Module):
    """conv1~conv5与E0/Encoder完全一致，仅替换最后一层。"""
    def __init__(self, sqrt_M: int, num_positions: int = 512):
        super().__init__()
        base = Encoder()
        # 复用backbone的conv1~conv5参数结构
        self.conv1 = base.conv1
        self.conv2 = base.conv2
        self.conv3 = base.conv3
        self.conv4 = base.conv4
        self.conv5 = base.conv5
        self.jcm_head = JCMEncoderHead(sqrt_M=sqrt_M, in_channels=256, num_positions=num_positions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        return self.jcm_head(x)


class JCMDecoder(nn.Module):
    """
    与E0/Decoder结构完全一致，只有conv_trans1的输入通道数从
    num_positions改为2*num_positions（因为要接收I/Q两路各
    num_positions通道的实数特征图）。num_positions=512对应未剪枝，
    剪枝后（比如γ=0.7对应约154）传入对应的更小的数。
    """
    def __init__(self, num_positions: int = 512):
        super().__init__()
        self.num_positions = num_positions
        self.conv_trans1 = nn.ConvTranspose2d(num_positions * 2, 256, kernel_size=5, stride=1, padding=2)
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


class JCMFullModel(nn.Module):
    def __init__(self, sqrt_M: int, gumbel_tau: float = 1.5, num_positions: int = 512):
        super().__init__()
        self.sqrt_M = sqrt_M
        self.gumbel_tau = gumbel_tau
        self.num_positions = num_positions
        self.encoder = JCMEncoder(sqrt_M=sqrt_M, num_positions=num_positions)
        self.decoder = JCMDecoder(num_positions=num_positions)

    def forward(self, x, snr_db):
        B = x.shape[0]
        logits = self.encoder(x)                                            # (B, k, 2, √M)
        z = jcm_sample_symbols(logits, tau=self.gumbel_tau, hard=True)     # (B, k) complex
        z_hat = rayleigh_awgn_channel_complex(z, snr_db)                    # (B, k) complex

        # (B, k) 复数 → (B, 2*num_positions, 32, 32) 实数：I/Q两路num_positions
        # 通道在C维拼接
        feat_I = z_hat.real.view(B, self.num_positions, 32, 32)
        feat_Q = z_hat.imag.view(B, self.num_positions, 32, 32)
        feat = torch.cat([feat_I, feat_Q], dim=1)
        return self.decoder(feat)


def load_warmstart_from_e0(model: JCMFullModel, e0_checkpoint_path: str):
    """
    从E0训练好的AnalogJSCC checkpoint加载Encoder.conv1~conv5和Decoder中形状
    能对上的层作为热启动权重。JCM模型的jcm_head和conv_trans1是新结构、
    形状对不上，会被跳过并打印出来。
    """
    import os
    if not os.path.isfile(e0_checkpoint_path):
        print(f"⚠️ 未找到E0 checkpoint（{e0_checkpoint_path}），从随机初始化开始。")
        return model
    old_state = torch.load(e0_checkpoint_path, map_location="cpu")
    new_state = model.state_dict()
    loaded, skipped = [], []
    for k, v in old_state.items():
        if k in new_state and new_state[k].shape == v.shape:
            new_state[k] = v
            loaded.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(new_state)
    print(f"热启动加载: {len(loaded)}层成功，{len(skipped)}层跳过（新结构或形状不匹配）")
    if skipped:
        show = skipped[:5]
        print(f"  跳过的层示例: {show}{'...' if len(skipped) > 5 else ''}")
    return model


if __name__ == "__main__":
    # 自测：未剪枝(512)和剪枝后(154, 对应γ=0.7)两种宽度，4/16/64/256QAM全跑一遍
    torch.manual_seed(0)
    for num_positions, width_name in [(512, "未剪枝γ=0"), (154, "剪枝γ=0.7")]:
        print(f"\n--- {width_name} (num_positions={num_positions}) ---")
        for sqrt_M, name in [(2, "4QAM"), (4, "16QAM"), (8, "64QAM"), (16, "256QAM")]:
            model = JCMFullModel(sqrt_M=sqrt_M, num_positions=num_positions)
            x = torch.rand(1, 3, 512, 512)
            y = model(x, snr_db=10.0)
            loss = F.mse_loss(y, x)
            loss.backward()
            grad_ok = model.encoder.conv1.weight.grad.abs().sum().item() > 0
            n_params = sum(p.numel() for p in model.parameters())
            print(f"[{name}] shape={tuple(y.shape)}  loss={loss.item():.4f}  "
                  f"参数量={n_params/1e6:.2f}M  梯度→Encoder.conv1={grad_ok}")
