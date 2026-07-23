"""
E9_train_bandwidth_ratio.py
==============================
【实验E9第1步】不同带宽压缩比 k/n 下的analog模型训练，回应审稿人3
意见2（"应补充在不同带宽压缩比条件下系统性能的系统性研究"）。

跟E5剪枝不同：这里是**独立从零训练**不同宽度的模型（随机初始化，不
从E0热启动），对应你论文Section 4.4里"直接设计不同k/n的模型"这一
方法论（跟"剪枝出来的窄模型"是两回事，二者的对比正是Figure 6在做
的事）。

k/n=2/3（num_positions=512）就是已经训好的E0，不在这里重复训练；
本脚本依次训练 common/config.py::E9_KN_RATIOS 里另外三个比例
（1/6, 1/3, 1/2），每个200轮，每训完一个立刻保存checkpoint，
互不阻塞、中途中断不丢已完成的部分。

运行方式：
    python E9_train_bandwidth_ratio.py
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, LR, SNR_TRAIN_DB,
    E9_NUM_EPOCHS, E9_KN_RATIOS, RESULTS_DIR, DEVICE, SEED
)


class AnalogJSCC_KN(nn.Module):
    """跟E0的AnalogJSCC结构一致，只是瓶颈宽度可配置。"""
    def __init__(self, num_positions):
        super().__init__()
        self.encoder = PrunableEncoder(bottleneck_channels=num_positions)
        self.decoder = PrunableDecoder(bottleneck_channels=num_positions)

    def forward(self, x, snr_db):
        z = self.encoder(x)
        z_hat = rayleigh_awgn_channel_real(z, snr_db)
        return self.decoder(z_hat)


def train_one_ratio(kn_name, num_positions, train_loader):
    print(f"\n{'='*60}\n[E9] 开始训练 k/n={kn_name} (num_positions={num_positions})\n{'='*60}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = AnalogJSCC_KN(num_positions=num_positions).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M（随机初始化，独立训练，不从E0热启动）")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    kn_str = kn_name.replace("/", "_")
    checkpoint_path = os.path.join(RESULTS_DIR, f"E9_kn_{kn_str}_analog.pth")

    for epoch in range(E9_NUM_EPOCHS):
        model.train()
        t_epoch_start = time.time()
        running_loss, n_batches = 0.0, 0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs, snr_db=SNR_TRAIN_DB)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if batch_idx % 20 == 0:
                elapsed = time.time() - t_epoch_start
                print(f"[k/n={kn_name}][Epoch {epoch+1}/{E9_NUM_EPOCHS}] "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  已用{elapsed/60:.1f}分钟")

        avg_loss = running_loss / max(n_batches, 1)
        print(f">>> [k/n={kn_name}] Epoch {epoch+1}/{E9_NUM_EPOCHS} 完成，"
              f"平均loss={avg_loss:.4f}，耗时{(time.time()-t_epoch_start)/60:.1f}分钟")
        torch.save(model.state_dict(), checkpoint_path)

    print(f"✅ k/n={kn_name} 训练完成，保存至 {checkpoint_path}")


def main():
    print(f"设备: {DEVICE}")
    print(f"本次将依次训练 k/n = {list(E9_KN_RATIOS.keys())}（各{E9_NUM_EPOCHS}轮）")
    print(f"k/n=2/3 已有E0现成checkpoint，不在本脚本训练范围内")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = FlatImageFolderDataset(DATASET_PATH, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"训练集: {len(train_dataset)}张")

    t_all_start = time.time()
    for i, (kn_name, num_positions) in enumerate(E9_KN_RATIOS.items(), start=1):
        print(f"\n########## 总进度 {i}/{len(E9_KN_RATIOS)}：k/n={kn_name} "
              f"（已用总时间 {(time.time()-t_all_start)/60:.1f} 分钟） ##########")
        train_one_ratio(kn_name, num_positions, train_loader)

    print(f"\n✅ E9全部训练完成，总耗时 {(time.time()-t_all_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
