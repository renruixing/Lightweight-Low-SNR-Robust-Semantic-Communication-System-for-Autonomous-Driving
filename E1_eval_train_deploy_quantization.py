"""
E1_eval_train_deploy_quantization.py
=====================================
【实验E1】训练-部署分离量化推理评测（论文自己的核心方案）。

流程（严格对应 manuscript.tex Section 3.2 描述）：
    Encoder → 均匀量化 → M-QAM映射 → 复瑞利+AWGN信道 → 均衡
    → 最近邻解调 → 反量化 → Decoder

复用E0训练好的Encoder/Decoder，训练阶段本身没有量化（就是E0的
analog训练），只在推理阶段引入量化调制模块——这就是论文所称的
"训练-部署分离"策略。

产出结果：results/E1_train_deploy_results.json，每种调制阶数
（4/16/64/256QAM）在 EVAL_SNR_GRID 上的PSNR/SSIM曲线。
"""
import os
import json
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone import Encoder, Decoder
from common.channel import rayleigh_awgn_channel_complex
from common.qam import (
    uniform_quantize_to_index, qam_modulate,
    qam_demodulate_to_index, dequantize_from_index,
)
from common.dataset import FlatImageFolderDataset
from common.config import (
    DATASET_PATH, IMG_SIZE, BATCH_SIZE,
    E0_CHECKPOINT, EVAL_SNR_GRID, N_EVAL_REPEATS, RESULTS_DIR, DEVICE, SEED
)


def build_model_from_e0_checkpoint():
    """从E0的checkpoint加载Encoder和Decoder"""
    if not os.path.isfile(E0_CHECKPOINT):
        raise FileNotFoundError(
            f"找不到E0 checkpoint: {E0_CHECKPOINT}\n"
            f"请先运行 python E0_pretrain_analog.py 训练基础模型。"
        )
    state = torch.load(E0_CHECKPOINT, map_location=DEVICE)
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
    # E0保存的是完整AnalogJSCC模型的state_dict，key前缀是encoder./decoder.
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval()
    decoder.eval()
    return encoder, decoder


@torch.no_grad()
def train_deploy_forward(encoder, decoder, x, snr_db, M):
    """
    训练-部署分离方案的推理前向：
    Encoder → 量化 → 调制 → 复数信道 → 解调 → 反量化 → Decoder
    """
    z = encoder(x)                                  # (B, 512, 32, 32), [0,1)
    original_shape = z.shape

    idx = uniform_quantize_to_index(z, M)          # long, 同形状
    s = qam_modulate(idx, M)                        # complex, 同形状

    # 展平成 (B, k) 复数序列送入信道
    s_flat = s.reshape(original_shape[0], -1)
    y_flat = rayleigh_awgn_channel_complex(s_flat, snr_db)

    idx_hat_flat = qam_demodulate_to_index(y_flat, M)
    z_hat_flat = dequantize_from_index(idx_hat_flat, M)
    z_hat = z_hat_flat.reshape(original_shape)

    x_hat = decoder(z_hat)
    return x_hat


def compute_psnr_ssim(x_true, x_pred):
    """
    在numpy上计算逐张图的PSNR/SSIM，然后batch内平均。
    输入是tensor (B, 3, H, W)，[0,1]范围。
    """
    x_true_np = (x_true.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    x_pred_np = (x_pred.clamp(0, 1) * 255).permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    psnr_list, ssim_list = [], []
    for i in range(x_true_np.shape[0]):
        psnr_list.append(peak_signal_noise_ratio(x_true_np[i], x_pred_np[i], data_range=255))
        ssim_list.append(structural_similarity(x_true_np[i], x_pred_np[i],
                                                channel_axis=2, data_range=255))
    return np.mean(psnr_list), np.mean(ssim_list)


def main():
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    print(f"设备: {DEVICE}")
    encoder, decoder = build_model_from_e0_checkpoint()
    print(f"已加载E0 checkpoint: {E0_CHECKPOINT}")

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
        print(f"\n{'='*60}\n评测 {mod_name} (M={M})\n{'='*60}")

        psnr_curve, ssim_curve = [], []
        for snr in EVAL_SNR_GRID:
            psnr_list, ssim_list = [], []
            for repeat in range(N_EVAL_REPEATS):  # 每次重新采样信道实现，降低慢衰落评测噪声
                for inputs in val_loader:
                    inputs = inputs.to(DEVICE)
                    outputs = train_deploy_forward(encoder, decoder, inputs,
                                                    snr_db=float(snr), M=M)
                    psnr, ssim = compute_psnr_ssim(inputs, outputs)
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)
            psnr_avg = float(np.mean(psnr_list))
            ssim_avg = float(np.mean(ssim_list))
            psnr_curve.append(psnr_avg)
            ssim_curve.append(ssim_avg)
            print(f"[{mod_name}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}  "
                  f"(重复{N_EVAL_REPEATS}次取平均)")

        all_results[mod_name] = {
            "M": M,
            "snr_grid": EVAL_SNR_GRID,
            "psnr": psnr_curve,
            "ssim": ssim_curve,
        }

    out_path = os.path.join(RESULTS_DIR, "E1_train_deploy_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ E1评测完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()
