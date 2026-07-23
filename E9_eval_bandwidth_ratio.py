"""
E9_eval_bandwidth_ratio.py
=============================
【实验E9第1步-评测】对4个k/n比例（1/6, 1/3, 1/2, 2/3）的analog模型
跑完整0~30dB SNR曲线。k/n=2/3直接复用E0 checkpoint，另外三个读
E9_train_bandwidth_ratio.py训出的checkpoint。

运行方式：
    python E9_eval_bandwidth_ratio.py
"""
import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, E0_CHECKPOINT,
    EVAL_SNR_GRID, N_EVAL_REPEATS, E9_KN_RATIOS, RESULTS_DIR, DEVICE
)

# 四个k/n对应的宽度和checkpoint路径（2/3复用E0，其余读E9训练产物）
ALL_KN = dict(E9_KN_RATIOS)
ALL_KN["2/3"] = 512


def get_checkpoint_path(kn_name):
    if kn_name == "2/3":
        return E0_CHECKPOINT
    kn_str = kn_name.replace("/", "_")
    return os.path.join(RESULTS_DIR, f"E9_kn_{kn_str}_analog.pth")


def compute_psnr_ssim(x_true, x_pred):
    x_true_np = (x_true.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    x_pred_np = (x_pred.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    psnr_list, ssim_list = [], []
    for i in range(x_true_np.shape[0]):
        psnr_list.append(peak_signal_noise_ratio(x_true_np[i], x_pred_np[i], data_range=255))
        ssim_list.append(structural_similarity(x_true_np[i], x_pred_np[i],
                                                channel_axis=2, data_range=255))
    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    print(f"设备: {DEVICE}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"验证集: {len(val_dataset)}张")

    all_results = {}
    for kn_name, num_positions in sorted(ALL_KN.items(), key=lambda kv: kv[1]):
        checkpoint_path = get_checkpoint_path(kn_name)
        if not os.path.isfile(checkpoint_path):
            print(f"⚠️ 跳过 k/n={kn_name}：找不到checkpoint {checkpoint_path}")
            continue

        print(f"\n{'='*60}\n评测 k/n={kn_name} (num_positions={num_positions})\n{'='*60}")
        state = torch.load(checkpoint_path, map_location=DEVICE)
        encoder = PrunableEncoder(bottleneck_channels=num_positions).to(DEVICE)
        decoder = PrunableDecoder(bottleneck_channels=num_positions).to(DEVICE)
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
                print(f"[k/n={kn_name}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}")

        all_results[kn_name] = {
            "num_positions": num_positions,
            "snr_grid": EVAL_SNR_GRID,
            "psnr": psnr_curve,
            "ssim": ssim_curve,
        }

    out_path = os.path.join(RESULTS_DIR, "E9_bandwidth_ratio_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()
