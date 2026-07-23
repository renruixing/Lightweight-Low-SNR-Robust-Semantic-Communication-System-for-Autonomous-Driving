"""
E6_train_jcm_gamma07.py
=========================
【实验E6】在γ=0.7剪枝后的骨干网络上训练JCM，可直接对比论文Figure 4
现有的 "Deep JSCC(γ=0.7)+4/16/64/256QAM" 曲线。

热启动来源：E5_prune_gamma.py 跑出的剪枝后checkpoint
（results/E5_pruned_gamma07.pth），提取其中的bottleneck_channels
（剪枝后的宽度，比如γ=0.7对应约154），构建同样宽度的JCM模型，
加载conv1~conv5的权重作为热启动（conv6/bn6/conv_trans1形状对不上，
不加载，JCM的jcm_head本来就是全新结构）。

运行前提：必须先跑完 E5_prune_gamma.py --gamma 0.7

运行方式：
    python E6_train_jcm_gamma07.py
"""
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from E2_jcm_model import JCMFullModel, load_warmstart_from_e0
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, LR, SNR_TRAIN_DB,
    E2_NUM_EPOCHS, E2_GUMBEL_TAU, E2_GRAD_CLIP,
    E2_MOD_LIST, EVAL_SNR_GRID, RESULTS_DIR, DEVICE, SEED
)

GAMMA = 0.7
GAMMA_STR = str(GAMMA).replace(".", "")
E5_CHECKPOINT = os.path.join(RESULTS_DIR, f"E5_pruned_gamma{GAMMA_STR}.pth")
E6_RESULTS_JSON = os.path.join(RESULTS_DIR, f"E6_jcm_gamma{GAMMA_STR}_results.json")


def get_pruned_width():
    if not os.path.isfile(E5_CHECKPOINT):
        raise FileNotFoundError(
            f"找不到E5剪枝checkpoint: {E5_CHECKPOINT}\n"
            f"请先运行: python E5_prune_gamma.py --gamma {GAMMA}"
        )
    state = torch.load(E5_CHECKPOINT, map_location="cpu")
    n_keep = int(state["_bottleneck_channels"].item())
    print(f"从E5 checkpoint读取剪枝后宽度: {n_keep} (γ={GAMMA})")
    return n_keep


def compute_psnr_ssim(x_true, x_pred):
    x_true_np = (x_true.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    x_pred_np = (x_pred.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    psnr_list, ssim_list = [], []
    for i in range(x_true_np.shape[0]):
        psnr_list.append(peak_signal_noise_ratio(x_true_np[i], x_pred_np[i], data_range=255))
        ssim_list.append(structural_similarity(x_true_np[i], x_pred_np[i],
                                                channel_axis=2, data_range=255))
    return np.mean(psnr_list), np.mean(ssim_list)


def save_result(mod_name, sqrt_M, num_positions, n_params, psnr_curve, ssim_curve):
    all_results = {}
    if os.path.isfile(E6_RESULTS_JSON):
        with open(E6_RESULTS_JSON, "r") as f:
            all_results = json.load(f)
    all_results[mod_name] = {
        "sqrt_M": sqrt_M,
        "num_positions": num_positions,
        "gamma": GAMMA,
        "n_params_M": n_params / 1e6,
        "snr_grid": EVAL_SNR_GRID,
        "psnr": psnr_curve,
        "ssim": ssim_curve,
    }
    with open(E6_RESULTS_JSON, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"结果已写入 {E6_RESULTS_JSON}（{mod_name}）")


def train_and_eval_one(sqrt_M, mod_name, num_positions, train_loader, val_loader):
    print(f"\n{'='*60}\n[E6] 开始训练 {mod_name}, γ={GAMMA} (sqrt_M={sqrt_M}, "
          f"num_positions={num_positions})\n{'='*60}")

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    model = JCMFullModel(sqrt_M=sqrt_M, gumbel_tau=E2_GUMBEL_TAU,
                          num_positions=num_positions).to(DEVICE)
    model = load_warmstart_from_e0(model, E5_CHECKPOINT)  # 函数名沿用E2那个，逻辑通用
    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params/1e6:.2f}M")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    checkpoint_path = os.path.join(RESULTS_DIR, f"E6_jcm_gamma{GAMMA_STR}_{mod_name}.pth")

    for epoch in range(E2_NUM_EPOCHS):
        model.train()
        t_epoch_start = time.time()
        running_loss, n_batches = 0.0, 0
        for batch_idx, inputs in enumerate(train_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs, snr_db=SNR_TRAIN_DB)
            loss = criterion(outputs, inputs)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=E2_GRAD_CLIP)
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1
            if batch_idx % 20 == 0:
                elapsed = time.time() - t_epoch_start
                print(f"[{mod_name}][Epoch {epoch+1}/{E2_NUM_EPOCHS}] "
                      f"batch {batch_idx}/{len(train_loader)}  "
                      f"loss={loss.item():.4f}  已用{elapsed/60:.1f}分钟")

        avg_loss = running_loss / max(n_batches, 1)
        print(f">>> [{mod_name}] Epoch {epoch+1}/{E2_NUM_EPOCHS} 完成，"
              f"平均loss={avg_loss:.4f}，耗时{(time.time()-t_epoch_start)/60:.1f}分钟")
        torch.save(model.state_dict(), checkpoint_path)

    model.eval()
    psnr_curve, ssim_curve = [], []
    with torch.no_grad():
        for snr in EVAL_SNR_GRID:
            psnr_list, ssim_list = [], []
            for inputs in val_loader:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs, snr_db=float(snr))
                psnr, ssim = compute_psnr_ssim(inputs, outputs)
                psnr_list.append(psnr)
                ssim_list.append(ssim)
            psnr_avg = float(np.mean(psnr_list))
            ssim_avg = float(np.mean(ssim_list))
            psnr_curve.append(psnr_avg)
            ssim_curve.append(ssim_avg)
            print(f"[{mod_name}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}")

    save_result(mod_name, sqrt_M, num_positions, n_params, psnr_curve, ssim_curve)


def main():
    print(f"设备: {DEVICE}")
    num_positions = get_pruned_width()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = FlatImageFolderDataset(DATASET_PATH, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集: {len(train_dataset)}张，验证集: {len(val_dataset)}张")
    print(f"本次将依次训练: {[name for _, name in E2_MOD_LIST]}")

    t_all_start = time.time()
    for i, (sqrt_M, mod_name) in enumerate(E2_MOD_LIST, start=1):
        print(f"\n########## 总进度 {i}/{len(E2_MOD_LIST)}：{mod_name} "
              f"（已用总时间 {(time.time()-t_all_start)/60:.1f} 分钟） ##########")
        train_and_eval_one(sqrt_M, mod_name, num_positions, train_loader, val_loader)

    print(f"\n✅ E6全部完成，总耗时 {(time.time()-t_all_start)/60:.1f} 分钟")
    print(f"完整结果见 {E6_RESULTS_JSON}")


if __name__ == "__main__":
    main()
