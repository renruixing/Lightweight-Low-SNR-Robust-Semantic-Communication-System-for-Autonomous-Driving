"""
check_e5_analog_performance.py
================================
独立诊断脚本：检查 E5（γ=0.7剪枝后模型）在**不接JCM、纯analog传输**
情况下的PSNR/SSIM，跟以下两个参照对比：
    1. 论文 Table 2 里 γ=0.7 那一行：PSNR=30.36dB, SSIM=0.91 (SNR=25dB)
    2. 论文 Plot_Fig4.py 里 "Deep JSCC(γ=0.7)-analog" 那条曲线
       （高SNR封顶约30.4~30.8dB）

这是判断"E6里JCM比训练-部署分离方案高3~4dB"这个结果能不能相信的关键
检查——如果E5自己的analog基线就已经严重偏低（比如20多dB），说明E5
继承了E0的问题，E6的"提升"很可能是相对于一个不健康的基线算出来的，
不能直接采信；如果E5的analog基线接近30dB，则E6里JCM的相对提升是
可信的。

运行方式：
    python check_e5_analog_performance.py --gamma 0.7
"""
import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, RESULTS_DIR, DEVICE
import os

# 论文里对应参照值
TABLE2_REFERENCE = {0.7: (30.36, 0.91)}  # (PSNR, SSIM) at SNR=25dB
FIG4_ANALOG_HIGH_SNR_REFERENCE = {0.7: 30.4}  # Plot_Fig4.py "Deep JSCC(γ=0.7)-analog"高SNR封顶


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
    gamma = args.gamma
    gamma_str = str(gamma).replace(".", "")

    checkpoint_path = os.path.join(RESULTS_DIR, f"E5_pruned_gamma{gamma_str}.pth")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"找不到E5 checkpoint: {checkpoint_path}")

    print(f"设备: {DEVICE}")
    state = torch.load(checkpoint_path, map_location=DEVICE)
    n_keep = int(state["_bottleneck_channels"].item())
    print(f"剪枝后宽度: {n_keep} (γ={gamma})")

    encoder = PrunableEncoder(bottleneck_channels=n_keep).to(DEVICE)
    decoder = PrunableDecoder(bottleneck_channels=n_keep).to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    val_dataset = FlatImageFolderDataset(DATASET_PATH, split="val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"验证集: {len(val_dataset)}张")

    print(f"\n=== 检查1: SNR=25dB下的analog表现 ===")
    with torch.no_grad():
        psnr_list, ssim_list = [], []
        for inputs in val_loader:
            inputs = inputs.to(DEVICE)
            z = encoder(inputs)
            z_hat = rayleigh_awgn_channel_real(z, snr_db=25.0)
            outputs = decoder(z_hat)
            psnr, ssim = compute_psnr_ssim(inputs, outputs)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        psnr_25, ssim_25 = np.mean(psnr_list), np.mean(ssim_list)
    ref_psnr, ref_ssim = TABLE2_REFERENCE.get(gamma, (None, None))
    print(f"E5实测: PSNR={psnr_25:.2f}dB, SSIM={ssim_25:.4f}")
    if ref_psnr is not None:
        print(f"论文Table 2 (γ={gamma}): PSNR={ref_psnr}dB, SSIM={ref_ssim}")
        print(f"差距: {ref_psnr - psnr_25:.2f}dB")

    print(f"\n=== 检查2: 高SNR(30dB)封顶表现，对比Plot_Fig4.py的analog曲线 ===")
    with torch.no_grad():
        psnr_list, ssim_list = [], []
        for inputs in val_loader:
            inputs = inputs.to(DEVICE)
            z = encoder(inputs)
            z_hat = rayleigh_awgn_channel_real(z, snr_db=30.0)
            outputs = decoder(z_hat)
            psnr, ssim = compute_psnr_ssim(inputs, outputs)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        psnr_30, ssim_30 = np.mean(psnr_list), np.mean(ssim_list)
    ref_high = FIG4_ANALOG_HIGH_SNR_REFERENCE.get(gamma)
    print(f"E5实测(SNR=30dB): PSNR={psnr_30:.2f}dB, SSIM={ssim_30:.4f}")
    if ref_high is not None:
        print(f"论文Plot_Fig4.py 'Deep JSCC(γ={gamma})-analog' 高SNR封顶: 约{ref_high}dB")
        gap = ref_high - psnr_30
        print(f"差距: {gap:.2f}dB")
        if abs(gap) < 3:
            print("✅ 差距在3dB以内，E5的analog基线基本健康，"
                  "E6里JCM相对训练-部署分离方案的提升可以采信。")
        else:
            print("⚠️ 差距较大，E5继承了跟E0类似的收敛问题，"
                  "E6里看到的'JCM更好'很可能是相对于一个不健康基线算出来的，"
                  "不能直接采信，需要先解决E0/E5的收敛问题。")


if __name__ == "__main__":
    main()
