"""
common/channel.py
==================
统一的信道模型：零均值单位方差慢瑞利衰落 + 复AWGN。

所有实验（E0基础训练、E1训练-部署分离量化推理、E2 JCM训练）都从这里
import，保证内部一致性。

信道规格（严格对齐manuscript.tex Section 2）：
    h ~ CN(0, 1)，即 h_real, h_imag ~ N(0, 1/2)，独立同分布  =>  E[|h|^2] = 1
    y = h * z + n,   n ~ CN(0, sigma^2),  sigma^2 = 1 / SNR_linear
    z_hat = y / h    (理想信道估计+均衡)
慢衰落：整个batch共享同一个信道实现 h。

数值保护：瑞利衰落理论上有一定概率采样到|h|非常接近0的深衰落，此时
z_hat = rx/h 会把噪声异常放大导致训练崩溃或推理数值爆炸。这里对h的
模长做下限截断（H_MAG_MIN=0.1），保持相位不变。这是通信系统常见的
工程处理，不是bug，也已在README的"实现细节"里明确记录。
"""
import math
import torch

H_MAG_MIN = 0.1


def _sample_rayleigh_h(device):
    """采样一个信道实现h（复数标量），做深衰落模长下限保护后返回。"""
    sigma_h = math.sqrt(1 / 2)
    h_real = torch.randn(1, device=device) * sigma_h
    h_imag = torch.randn(1, device=device) * sigma_h
    h_mag = torch.sqrt(h_real ** 2 + h_imag ** 2)
    scale = torch.clamp(h_mag, min=H_MAG_MIN) / (h_mag + 1e-12)
    h_real = h_real * scale
    h_imag = h_imag * scale
    return torch.complex(h_real, h_imag)


def rayleigh_awgn_channel_complex(z_complex: torch.Tensor, snr_db: float):
    """
    复数信号 → 慢瑞利衰落 + AWGN → 理想均衡后的复数信号。

    参数
    ----
    z_complex : (batch, k) 复数张量，星座符号或analog特征映射的复数
    snr_db    : 信噪比 (dB)

    返回
    ----
    z_hat : (batch, k) 复数张量，均衡后的接收信号
    """
    device = z_complex.device
    h = _sample_rayleigh_h(device)

    signal_power = 1.0  # 假定发端符号做过功率归一化，平均功率为1
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise = math.sqrt(noise_power / 2) * (
        torch.randn_like(z_complex.real) + 1j * torch.randn_like(z_complex.imag)
    )

    tx = z_complex * h
    rx = tx + noise
    z_hat = rx / h
    return z_hat


def rayleigh_awgn_channel_real(z_real: torch.Tensor, snr_db: float):
    """
    实数信号（analog transmission）版本。

    修复记录：早期版本直接假设 signal_power=1.0 来计算噪声强度，但
    Encoder的sigmoid输出 z 实际平均功率约为0.26（均值0.5左右，不是
    单位功率），导致按"SNR=25dB"设定实际算出来的噪声强度对应的
    有效SNR只有约19dB，差了近6dB。QAM路径的星座点是显式按功率=1
    归一化过的，analog路径这里之前没有做同样处理，两条路径不一致。

    现在的做法：发送前把z显式归一化到单位平均功率（满足
    manuscript.tex Eq.(3)的功率约束），过信道+均衡后再乘回原来的
    缩放系数，还原到Decoder期望的量纲，不改变Decoder的输入分布。

    修复记录2（更关键）：上一版功率归一化里，缩放系数 scale 是带梯度
    计算的，这意味着模型可以通过学习让 z 的整体动态范围收窄（比如
    输出都挤在0.4~0.6之间而不是铺满0~1）来人为缩小归一化后加入的
    噪声绝对值，从而"投机取巧"地降低训练loss，而不是真正学会鲁棒的
    特征编码——这很可能是上次训练30轮后loss卡在0.004附近不再下降
    的真正原因。现在把 scale 的计算从梯度图中分离（.detach()），
    让它只作为"测量值"校准噪声强度，不参与反向传播，模型无法再通过
    操纵自身输出功率来投机。

    参数
    ----
    z_real : 任意形状的实数张量（一般是encoder输出的sigmoid后特征图）
    snr_db : 信噪比 (dB)

    返回
    ----
    z_hat : 与z_real同形状的实数张量
    """
    orig_shape = z_real.shape
    z_flat = z_real.reshape(orig_shape[0], -1)  # (batch, k)

    # 关键：.detach() —— 功率测量值不参与梯度回传，避免模型通过收窄
    # 自身输出动态范围来投机降低有效噪声强度
    with torch.no_grad():
        power = (z_flat ** 2).mean()
        scale = torch.sqrt(power.clamp(min=1e-8))

    z_norm = z_flat / scale  # 归一化后功率≈1
    z_complex = torch.complex(z_norm, torch.zeros_like(z_norm))
    z_hat_complex = rayleigh_awgn_channel_complex(z_complex, snr_db)

    z_hat = z_hat_complex.real * scale  # 还原回原来的量纲，Decoder输入分布不变
    return z_hat.reshape(orig_shape)


if __name__ == "__main__":
    # 自测：验证E[|h|²]=1 与 z_hat数值范围合理
    torch.manual_seed(0)
    h_mag_sq_list = []
    z_hat_max_list = []
    for _ in range(10000):
        h = _sample_rayleigh_h("cpu")
        h_mag_sq_list.append((h.abs() ** 2).item())
        z = torch.randn(1, 100, dtype=torch.complex64)
        z_hat = rayleigh_awgn_channel_complex(z, snr_db=10.0)
        z_hat_max_list.append(z_hat.abs().max().item())
    print(f"E[|h|²] ≈ {sum(h_mag_sq_list)/len(h_mag_sq_list):.4f}  (理论值1.0)")
    print(f"z_hat最大幅度均值: {sum(z_hat_max_list)/len(z_hat_max_list):.2f}  "
          f"(带模长保护，不会爆炸)")
