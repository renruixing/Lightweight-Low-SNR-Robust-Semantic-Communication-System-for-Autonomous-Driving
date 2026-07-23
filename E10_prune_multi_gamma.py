"""
E10_prune_multi_gamma.py
===========================
【补齐Figure 3剩余剪枝率】依次跑 γ=0.2, 0.5, 0.9（γ=0是E0，γ=0.7是
已有的E5，这三个是目前缺的），复用E5_prune_gamma.py里重构出来的
run_pruning()函数，逻辑完全一致，只是循环跑多个γ。

每个γ跑完：
    1. 保存剪枝后checkpoint（results/E5_pruned_gamma{X}.pth）
    2. 立刻跑一遍完整0~30dB SNR曲线评测（analog，带N_EVAL_REPEATS
       重复采样修正），存进 results/E10_gamma{X}_analog_results.json

这样五个γ（0/0.2/0.5/0.7/0.9）全部跑完后，Figure 3里"Deep JSCC"
那五条曲线会是同一套复现体系下自洽的结果（BPG+LDPC那几条基线维持
原样不动，见此前已披露的caveat）。

运行方式：
    python E10_prune_multi_gamma.py
"""
import os
import json
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from E5_prune_gamma import run_pruning
from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, EVAL_SNR_GRID, N_EVAL_REPEATS,
    RESULTS_DIR, DEVICE
)

TARGET_GAMMAS = [0.2, 0.5, 0.9]  # γ=0(E0)和γ=0.7(已有E5)不在这里重复跑


def compute_psnr_ssim(x_true, x_pred):
    x_true_np = (x_true.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    x_pred_np = (x_pred.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    psnr_list, ssim_list = [], []
    for i in range(x_true_np.shape[0]):
        psnr_list.append(peak_signal_noise_ratio(x_true_np[i], x_pred_np[i], data_range=255))
        ssim_list.append(structural_similarity(x_true_np[i], x_pred_np[i],
                                                channel_axis=2, data_range=255))
    return np.mean(psnr_list), np.mean(ssim_list)


def eval_one_gamma(gamma, checkpoint_path, val_loader):
    """对刚训完的剪枝checkpoint跑完整SNR曲线（analog），逻辑同
    E7_eval_analog_full_curve.py --source e5，独立复制一份避免
    额外的命令行参数依赖。"""
    state = torch.load(checkpoint_path, map_location=DEVICE)
    n_keep = int(state["_bottleneck_channels"].item())
    encoder = PrunableEncoder(bottleneck_channels=n_keep).to(DEVICE)
    decoder = PrunableDecoder(bottleneck_channels=n_keep).to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval()
    decoder.eval()

    psnr_curve, ssim_curve = [], []
    with torch.no_grad():
        for snr in EVAL_SNR_GRID:
            psnr_list, ssim_list = [], []
            for repeat in range(N_EVAL_REPEATS):
                for inputs in val_loader:
                    inputs = inputs.to(DEVICE)
                    z = encoder(inputs)
                    z_hat = rayleigh_awgn_channel_real(z, snr_db=float(snr))
                    outputs = decoder(z_hat)
                    psnr, ssim = compute_psnr_ssim(inputs, outputs)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
            psnr_avg, ssim_avg = float(np.mean(psnr_list)), float(np.mean(ssim_list))
            psnr_curve.append(psnr_avg)
            ssim_curve.append(ssim_avg)
            print(f"[γ={gamma}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}")

    return {"n_keep": n_keep, "snr_grid": EVAL_SNR_GRID, "psnr": psnr_curve, "ssim": ssim_curve}


def main():
    print(f"设备: {DEVICE}")
    print(f"本次将依次跑齐 γ = {TARGET_GAMMAS}（γ=0已是E0，γ=0.7已是现有E5，不重复）")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    train_dataset = FlatImageFolderDataset(DATASET_PATH, split="train", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"训练集: {len(train_dataset)}张，验证集: {len(val_dataset)}张")

    all_results = {}
    t_all_start = time.time()
    for i, gamma in enumerate(TARGET_GAMMAS, start=1):
        print(f"\n########## 总进度 {i}/{len(TARGET_GAMMAS)}：γ={gamma} "
              f"（已用总时间 {(time.time()-t_all_start)/60:.1f} 分钟） ##########")

        checkpoint_path = run_pruning(gamma, train_loader)

        print(f"\n[γ={gamma}] 剪枝训练完成，开始评测完整SNR曲线...")
        result = eval_one_gamma(gamma, checkpoint_path, val_loader)
        all_results[str(gamma)] = result

        # 每跑完一个就立刻写盘，避免中途中断丢失已完成的部分
        out_path = os.path.join(RESULTS_DIR, "E10_multi_gamma_results.json")
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[γ={gamma}] 结果已写入 {out_path}")

    print(f"\n✅ 全部完成，总耗时 {(time.time()-t_all_start)/60:.1f} 分钟")


if __name__ == "__main__":
    main()
