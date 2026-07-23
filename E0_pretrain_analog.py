"""
E0_pretrain_analog.py
======================
【实验E0】训练一个基础的Analog Deep JSCC模型（未剪枝，γ=0）。

这是后续所有实验（E1、E2）的共同起点：
  - E1 直接加载它的权重，在推理阶段套上"训练-部署分离量化"评测；
  - E2 加载它的Encoder/Decoder作为JCM训练的热启动权重。

架构：直接复用你现有 model.py 里的 Encoder / Decoder 类（原样不动），
但训练时的信道模块换成 common/channel.py 里统一实现的
rayleigh_awgn_channel_real（严格对齐 manuscript.tex Section 2 的
"零均值单位方差瑞利" 规格）。这样后续所有实验的信道方差都是一致的。

预计训练时间：单epoch约1分钟，30个epoch约30分钟（参考之前的实测速度）。
每个epoch都会保存最新checkpoint到 results/E0_analog_baseline.pth，
中途中断也不会丢失进度。
"""
import os
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from common.backbone import Encoder, Decoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, LR, SNR_TRAIN_DB,
    E0_NUM_EPOCHS, E0_CHECKPOINT, DEVICE, SEED
)


class AnalogJSCC(nn.Module):
    """
    Analog Deep JSCC模型：Encoder → 统一信道（analog实数版本）→ Decoder。
    结构与你 model.py 的 JSCCmodel_Withoutchannelcodec_city 等价，
    区别仅在于信道模块换成了 common/channel.py 里的统一实现。
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x, snr_db):
        z = self.encoder(x)                                # (B, 512, 32, 32), [0,1]
        z_hat = rayleigh_awgn_channel_real(z, snr_db)     # 同形状，经过统一信道
        x_hat = self.decoder(z_hat)
        return x_hat


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"设备: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = FlatImageFolderDataset(DATASET_PATH, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"训练集: {len(train_dataset)}张，batch数={len(train_loader)}")

    model = AnalogJSCC().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(E0_NUM_EPOCHS):
        model.train()
        t_epoch_start = time.time()
        running_loss = 0.0
        n_batches = 0
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
                print(f"[E0][Epoch {epoch+1}/{E0_NUM_EPOCHS}] "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  已用{elapsed/60:.1f}分钟")

        avg_loss = running_loss / max(n_batches, 1)
        print(f">>> [E0] Epoch {epoch+1}/{E0_NUM_EPOCHS} 完成，"
              f"平均loss={avg_loss:.4f}，耗时{(time.time()-t_epoch_start)/60:.1f}分钟")
        torch.save(model.state_dict(), E0_CHECKPOINT)

    print(f"\n✅ E0训练完成，checkpoint保存至 {E0_CHECKPOINT}")
    print("接下来运行 E1_eval_train_deploy_quantization.py 和 E2_train_jcm.py")


if __name__ == "__main__":
    main()
