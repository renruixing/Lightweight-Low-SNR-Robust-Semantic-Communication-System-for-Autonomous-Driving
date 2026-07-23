"""
E5_prune_gamma.py
===================
【实验E5】结构化剪枝，复现 manuscript.tex Algorithm 1：
    1. 从E0（满宽度analog模型）热启动，做 E_sparse 轮"稀疏训练"：
       在正常MSE loss基础上，对 bn6.weight（BN scaling factor η）加
       L1正则，把不重要通道的η推向0。
    2. 按 |η| 从小到大排序，剪掉最不重要的 γ 比例的通道（对应
       conv6输出通道 + bn6 + conv_trans1输入通道，联动裁剪）。
    3. 用裁剪后的窄模型做 E_fine_tune 轮纯MSE微调，恢复重建质量。

产出：results/E5_pruned_gamma{γ}.pth（窄模型checkpoint，可直接用于
E6的JCM剪枝版热启动，或者单独评测剪枝后的analog性能）。

运行方式：
    python E5_prune_gamma.py --gamma 0.7
"""
import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from common.backbone_prunable import PrunableEncoder, PrunableDecoder, build_pruned_state_dict
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, LR, SNR_TRAIN_DB,
    E0_CHECKPOINT, RESULTS_DIR, DEVICE, SEED
)

# ============ 剪枝相关超参（对应manuscript.tex Section 4.1） ============
L1_LAMBDA = 1e-5      # L1正则系数，与论文一致
E_SPARSE = 10          # 稀疏训练轮数，与论文N_sparse=10一致
E_FINE_TUNE = 5        # 剪枝后微调轮数，与论文N_fine-tune=5一致
# ==========================================================================


class SparseAnalogJSCC(nn.Module):
    """满宽度(512通道)模型，训练时对bn6.weight加L1正则（稀疏训练阶段用）。"""
    def __init__(self):
        super().__init__()
        self.encoder = PrunableEncoder(bottleneck_channels=512)
        self.decoder = PrunableDecoder(bottleneck_channels=512)

    def forward(self, x, snr_db):
        z = self.encoder(x)
        z_hat = rayleigh_awgn_channel_real(z, snr_db)
        return self.decoder(z_hat)


class NarrowAnalogJSCC(nn.Module):
    """剪枝后的窄模型（微调阶段、以及后续评测用）。"""
    def __init__(self, bottleneck_channels):
        super().__init__()
        self.encoder = PrunableEncoder(bottleneck_channels=bottleneck_channels)
        self.decoder = PrunableDecoder(bottleneck_channels=bottleneck_channels)

    def forward(self, x, snr_db):
        z = self.encoder(x)
        z_hat = rayleigh_awgn_channel_real(z, snr_db)
        return self.decoder(z_hat)


def load_e0_into_sparse_model(model: SparseAnalogJSCC):
    """把E0（普通AnalogJSCC，用的是common/backbone.py而不是backbone_prunable.py，
    但两者层名、形状完全一致，可以直接跨class加载state_dict）加载进来热启动。"""
    if not os.path.isfile(E0_CHECKPOINT):
        raise FileNotFoundError(f"找不到E0 checkpoint: {E0_CHECKPOINT}，请先跑 E0_pretrain_analog.py")
    state = torch.load(E0_CHECKPOINT, map_location="cpu")
    model.load_state_dict(state)
    print(f"已从E0热启动: {E0_CHECKPOINT}")
    return model


def train_one_epoch(model, loader, optimizer, use_l1, tag):
    model.train()
    t_start = time.time()
    running_loss = 0.0
    n_batches = 0
    for batch_idx, inputs in enumerate(loader):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs, snr_db=SNR_TRAIN_DB)
        mse_loss = nn.functional.mse_loss(outputs, inputs)

        if use_l1:
            l1_term = model.encoder.bn6.weight.abs().sum()
            loss = mse_loss + L1_LAMBDA * l1_term
        else:
            loss = mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += mse_loss.item()  # 记录纯MSE，方便跨阶段比较
        n_batches += 1
        if batch_idx % 20 == 0:
            elapsed = time.time() - t_start
            print(f"[{tag}] batch {batch_idx}/{len(loader)}  "
                  f"MSE={mse_loss.item():.4f}  已用{elapsed/60:.1f}分钟")
    return running_loss / max(n_batches, 1)


def run_pruning(gamma, train_loader):
    """核心剪枝流程：稀疏训练+剪枝+微调，供main()单独调用，也供批量脚本
    （E10_prune_multi_gamma.py）循环调用多个gamma复用。"""
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"设备: {DEVICE}，剪枝目标 γ={gamma}")

    # ---------------------- 阶段1: 稀疏训练（L1正则，满宽度） ----------------------
    sparse_model = SparseAnalogJSCC().to(DEVICE)
    sparse_model = load_e0_into_sparse_model(sparse_model)
    optimizer = optim.Adam(sparse_model.parameters(), lr=LR)

    print(f"\n{'='*60}\n[γ={gamma}] 阶段1: 稀疏训练 (L1正则, {E_SPARSE}轮)\n{'='*60}")
    for epoch in range(E_SPARSE):
        avg_mse = train_one_epoch(sparse_model, train_loader, optimizer,
                                   use_l1=True, tag=f"γ={gamma} 稀疏训练 Epoch{epoch+1}/{E_SPARSE}")
        eta_nonzero = (sparse_model.encoder.bn6.weight.abs() > 1e-3).sum().item()
        print(f">>> [γ={gamma}] Epoch{epoch+1}/{E_SPARSE} 完成，平均MSE={avg_mse:.4f}，"
              f"|η|>1e-3的通道数={eta_nonzero}/512")

    # ---------------------- 阶段2: 按|η|剪枝 ----------------------
    eta = sparse_model.encoder.bn6.weight.detach().abs()
    n_keep = int(round(512 * (1 - gamma)))
    keep_indices = torch.argsort(eta, descending=True)[:n_keep].sort().values.cpu()
    print(f"\n[γ={gamma}] 剪枝: 512 → {n_keep} 通道")

    enc_new_state, dec_new_state = build_pruned_state_dict(
        sparse_model.encoder.cpu(), sparse_model.decoder.cpu(), keep_indices
    )
    sparse_model.to(DEVICE)

    narrow_model = NarrowAnalogJSCC(bottleneck_channels=n_keep).to(DEVICE)
    narrow_model.encoder.load_state_dict(enc_new_state)
    narrow_model.decoder.load_state_dict(dec_new_state)
    print(f"[γ={gamma}] 剪枝后模型权重加载完成")

    # ---------------------- 阶段3: 微调（窄模型，纯MSE） ----------------------
    optimizer2 = optim.Adam(narrow_model.parameters(), lr=LR)
    print(f"\n{'='*60}\n[γ={gamma}] 阶段3: 微调 (纯MSE, {E_FINE_TUNE}轮)\n{'='*60}")
    for epoch in range(E_FINE_TUNE):
        avg_mse = train_one_epoch(narrow_model, train_loader, optimizer2,
                                   use_l1=False, tag=f"γ={gamma} 微调 Epoch{epoch+1}/{E_FINE_TUNE}")
        print(f">>> [γ={gamma}] 微调 Epoch{epoch+1}/{E_FINE_TUNE} 完成，平均MSE={avg_mse:.4f}")

    # ---------------------- 保存 ----------------------
    gamma_str = str(gamma).replace(".", "")
    out_path = os.path.join(RESULTS_DIR, f"E5_pruned_gamma{gamma_str}.pth")
    full_state = {}
    for k, v in narrow_model.encoder.state_dict().items():
        full_state[f"encoder.{k}"] = v
    for k, v in narrow_model.decoder.state_dict().items():
        full_state[f"decoder.{k}"] = v
    full_state["_bottleneck_channels"] = torch.tensor(n_keep)
    torch.save(full_state, out_path)
    print(f"\n✅ [γ={gamma}] 完成，剪枝后模型（{n_keep}通道）保存至 {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, required=True,
                        help="剪枝比例，例如0.7表示剪掉70%通道")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = FlatImageFolderDataset(DATASET_PATH, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"训练集: {len(train_dataset)}张")

    run_pruning(args.gamma, train_loader)


if __name__ == "__main__":
    main()
