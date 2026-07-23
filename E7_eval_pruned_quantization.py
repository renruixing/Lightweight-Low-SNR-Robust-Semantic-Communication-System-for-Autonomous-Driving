"""
E7_eval_pruned_quantization.py
=================================
【消融实验第4格：剪枝+训练-部署分离量化】

逻辑跟 E1_eval_train_deploy_quantization.py 完全一样（均匀量化+M-QAM+
最近邻解调），唯一区别是这次用的是E5剪枝后的checkpoint（宽度约154，
而不是E0未剪枝的512）。两个脚本刻意保持同样的量化/信道逻辑，保证
四格消融实验里"量化方式"这一个变量是一致的，只有"剪没剪枝"在变。

运行方式：
    python E7_eval_pruned_quantization.py --gamma 0.7
"""
import os
import json
import argparse
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_complex
from common.qam import (
    uniform_quantize_to_index, qam_modulate,
    qam_demodulate_to_index, dequantize_from_index,
)
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE, EVAL_SNR_GRID, N_EVAL_REPEATS, RESULTS_DIR, DEVICE
)


def load_e5(gamma):
    gamma_str = str(gamma).replace(".", "")
    checkpoint_path = os.path.join(RESULTS_DIR, f"E5_pruned_gamma{gamma_str}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"找不到E5 checkpoint: {checkpoint_path}\n"
            f"请先运行: python E5_prune_gamma.py --gamma {gamma}"
        )
    state = torch.load(checkpoint_path, map_location=DEVICE)
    n_keep = int(state["_bottleneck_channels"].item())
    encoder = PrunableEncoder(bottleneck_channels=n_keep).to(DEVICE)
    decoder = PrunableDecoder(bottleneck_channels=n_keep).to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    return encoder, decoder, n_keep


@torch.no_grad()
def train_deploy_forward(encoder, decoder, x, snr_db, M, n_keep):
    z = encoder(x)  # (B, n_keep, 32, 32), [0,1)
    original_shape = z.shape

    idx = uniform_quantize_to_index(z, M)
    s = qam_modulate(idx, M)

    s_flat = s.reshape(original_shape[0], -1)
    y_flat = rayleigh_awgn_channel_complex(s_flat, snr_db)

    idx_hat_flat = qam_demodulate_to_index(y_flat, M)
    z_hat_flat = dequantize_from_index(idx_hat_flat, M)
    z_hat = z_hat_flat.reshape(original_shape)

    return decoder(z_hat)


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.7)
    args = parser.parse_args()

    print(f"设备: {DEVICE}")
    encoder, decoder, n_keep = load_e5(args.gamma)
    encoder.eval()
    decoder.eval()
    print(f"已加载E5（剪枝，宽度={n_keep}，γ={args.gamma}）checkpoint")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"验证集: {len(val_dataset)}张")

    MOD_LIST = [4, 16, 64, 256]
    all_results = {}

    for M in MOD_LIST:
        mod_name = f"{M}QAM"
        print(f"\n{'='*60}\n评测 剪枝+{mod_name} (M={M})\n{'='*60}")

        psnr_curve, ssim_curve = [], []
        for snr in EVAL_SNR_GRID:
            psnr_list, ssim_list = [], []
            for repeat in range(N_EVAL_REPEATS):  # 每次重新采样信道实现，降低慢衰落评测噪声
                for inputs in val_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = train_deploy_forward(encoder, decoder, inputs,
                                                    snr_db=float(snr), M=M, n_keep=n_keep)
                    psnr, ssim = compute_psnr_ssim(inputs, outputs)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
            psnr_avg, ssim_avg = float(np.mean(psnr_list)), float(np.mean(ssim_list))
            psnr_curve.append(psnr_avg)
            ssim_curve.append(ssim_avg)
            print(f"[{mod_name}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}")

        all_results[mod_name] = {
            "M": M, "n_keep": n_keep, "gamma": args.gamma,
            "snr_grid": EVAL_SNR_GRID, "psnr": psnr_curve, "ssim": ssim_curve,
        }

    gamma_str = str(args.gamma).replace(".", "")
    out_path = os.path.join(RESULTS_DIR, f"E7_pruned_gamma{gamma_str}_quantization_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ 完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()
