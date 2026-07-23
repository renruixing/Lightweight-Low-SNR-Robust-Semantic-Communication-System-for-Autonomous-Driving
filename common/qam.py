"""
common/qam.py
==============
M-QAM 星座生成、均匀量化、调制、解调、反量化。

用途：E1（训练-部署分离量化推理）会用到；E2 JCM训练用的是可微分版本
（在 E2_jcm_model.py 里独立定义），不复用这里的非可微分实现。

星座设计：I/Q 各 √M 个电平，等间隔分布，均值为0；整个星座平均功率
归一化为1。这样满足 manuscript.tex Section 2 Eq.(3) 的功率约束
E[|s|²]/k ≤ P（取 P=1）。
"""
import math
import torch


def qam_constellation(M: int, device):
    """
    生成归一化的M-QAM星座点集合。

    参数
    ----
    M : 星座大小，必须是完全平方数 (4/16/64/256)
    device : torch设备

    返回
    ----
    const : (M,) 复数张量，平均功率为1
    """
    sqrt_M = int(round(math.sqrt(M)))
    assert sqrt_M * sqrt_M == M, f"M={M}必须是完全平方数(4/16/64/256)"

    # I/Q 电平：{-(√M-1), -(√M-3), ..., √M-1}
    levels = torch.arange(sqrt_M, device=device, dtype=torch.float32)
    levels = 2 * levels - (sqrt_M - 1)

    I, Q = torch.meshgrid(levels, levels, indexing="ij")
    const = I.flatten() + 1j * Q.flatten()  # (M,) complex

    # 功率归一化：使 E[|s|²] = 1
    avg_power = (const.abs() ** 2).mean()
    const = const / torch.sqrt(avg_power)
    return const


def uniform_quantize_to_index(z: torch.Tensor, M: int):
    """
    对(0,1)区间的实数z做M-level均匀量化，返回整数索引。
    对应manuscript.tex Eq.(14): ⌊z_i × M⌋

    参数
    ----
    z : 任意形状的实数张量，元素应在[0,1)（encoder的sigmoid输出）
    M : 量化级数

    返回
    ----
    idx : 与z同形状的long张量，取值范围[0, M-1]
    """
    idx = (z * M).long().clamp(0, M - 1)
    return idx


def qam_modulate(idx: torch.Tensor, M: int):
    """索引 → 星座符号。返回复数张量，形状同idx。"""
    const = qam_constellation(M, idx.device)
    return const[idx]


def qam_demodulate_to_index(y: torch.Tensor, M: int, chunk_size: int = 65536):
    """
    接收信号 → 最近邻解调回索引。
    对应manuscript.tex Eq.(18): j* = argmin_j |y - C_j|²

    参数
    ----
    y : 复数张量，任意形状
    M : 星座大小
    chunk_size : 分块大小。M大时(如256)一次算所有symbol到所有星座点的距离
                 会占用大量内存，分块处理避免OOM。

    返回
    ----
    idx : 与y同形状的long张量
    """
    const = qam_constellation(M, y.device)  # (M,)
    y_flat = y.reshape(-1)  # (N,)
    N = y_flat.numel()

    idx_flat = torch.empty(N, dtype=torch.long, device=y.device)
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk = y_flat[start:end]  # (chunk_len,)
        # (chunk_len, M)
        dist = (chunk.unsqueeze(-1) - const.view(1, -1)).abs() ** 2
        idx_flat[start:end] = dist.argmin(dim=-1)
    return idx_flat.reshape(y.shape)


def dequantize_from_index(idx: torch.Tensor, M: int):
    """索引 → [0,1)实数（反量化）。对应manuscript.tex中Q^{-1}(j) = j/M。"""
    return idx.float() / M


if __name__ == "__main__":
    # 自测：round-trip量化调制解调反量化应该无损（在无噪声信道下）
    torch.manual_seed(0)
    for M in [4, 16, 64, 256]:
        const = qam_constellation(M, "cpu")
        assert const.numel() == M
        avg_power = (const.abs() ** 2).mean().item()

        z = torch.rand(1, 100)  # [0,1)的随机数
        idx1 = uniform_quantize_to_index(z, M)
        s = qam_modulate(idx1, M)
        # 无噪声：直接解调应恢复索引
        idx2 = qam_demodulate_to_index(s, M)
        z_hat = dequantize_from_index(idx2, M)
        max_err = (z_hat - dequantize_from_index(idx1, M)).abs().max().item()
        print(f"M={M:3d}: 星座平均功率={avg_power:.4f}, "
              f"无噪round-trip最大误差={max_err:.6f}")
