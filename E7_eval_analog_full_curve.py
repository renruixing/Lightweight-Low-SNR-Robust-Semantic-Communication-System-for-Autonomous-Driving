"""
E7_eval_analog_full_curve.py
==============================
【消融实验用】给E0（未剪枝）或E5（剪枝）的analog checkpoint跑一条完整的
0~30dB SNR曲线（之前check_e0/e5_analog_performance.py只测了1~2个点，
这里补成完整曲线，用于跟量化方案的曲线放在一起画消融对比图）。

运行方式：
    python E7_eval_analog_full_curve.py --source e0
    python E7_eval_analog_full_curve.py --source e5 --gamma 0.7
"""
import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone import Encoder, Decoder
from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, E0_CHECKPOINT,
    EVAL_SNR_GRID, N_EVAL_REPEATS, RESULTS_DIR, DEVICE
)


def compute_psnr_ssim(x_true, x_pred):
    x_true_np = (x_true.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    x_pred_np = (x_pred.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    psnr_list, ssim_list = [], []
    for i in range(x_true_np.shape[0]):
        psnr_list.append(peak_signal_noise_ratio(x_true_np[i], x_pred_np[i], data_range=255))
        ssim_list.append(structural_similarity(x_true_np[i], x_pred_np[i],
                                                channel_axis=2, data_range=255))
    return np.mean(psnr_list), np.mean(ssim_list)


def load_e0():
    state = torch.load(E0_CHECKPOINT, map_location=DEVICE)
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    return encoder, decoder


def load_e5(gamma):
    gamma_str = str(gamma).replace(".", "")
    checkpoint_path = os.path.join(RESULTS_DIR, f"E5_pruned_gamma{gamma_str}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到E5 checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=DEVICE)
    n_keep = int(state["_bottleneck_channels"].item())
    encoder = PrunableEncoder(bottleneck_channels=n_keep).to(DEVICE)
    decoder = PrunableDecoder(bottleneck_channels=n_keep).to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    return encoder, decoder, n_keep


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["e0", "e5"], required=True)
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    if args.source == "e0":
        encoder, decoder = load_e0()
        tag = "E0_unpruned_analog"
        print("已加载E0（未剪枝）checkpoint")
    else:
        encoder, decoder, n_keep = load_e5(args.gamma)
        tag = f"E5_pruned_gamma{str(args.gamma).replace('.', '')}_analog"
        print(f"已加载E5（剪枝，宽度={n_keep}）checkpoint")

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"验证集: {len(val_dataset)}张")

    psnr_curve, ssim_curve = [], []
    with torch.no_grad():
        for snr in EVAL_SNR_GRID:
            psnr_list, ssim_list = [], []
            for repeat in range(N_EVAL_REPEATS):  # 每次重新采样信道实现，降低慢衰落评测噪声
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
            print(f"SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}  "
                  f"(重复{N_EVAL_REPEATS}次取平均)")

    out = {"snr_grid": EVAL_SNR_GRID, "psnr": psnr_curve, "ssim": ssim_curve}
    out_path = os.path.join(RESULTS_DIR, f"E7_{tag}_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()
