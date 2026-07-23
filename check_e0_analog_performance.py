"""
check_e0_analog_performance.py
================================
独立诊断脚本：检查 E0（基础analog模型）本身在 SNR=25dB 下的PSNR/SSIM，
跟论文 Table 2 里 γ=0 那一行（PSNR=31.42dB, SSIM=0.91）对比。

这是排查"E1量化结果偏低"问题的第一步：如果连E0自己（没有任何量化）
都到不了31dB附近，说明瓶颈在E0训练本身（欠拟合/没收敛），而不是
量化调制流程的问题；如果E0本身就接近31dB，那问题出在E1的量化/信道
实现上，需要往那边排查。

运行方式：
    python check_e0_analog_performance.py
"""
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone import Encoder, Decoder
from common.channel import rayleigh_awgn_channel_real
from common.dataset import FlatImageFolderDataset
from common.config import DATASET_PATH, IMG_SIZE, BATCH_SIZE, E0_CHECKPOINT, DEVICE


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
    state = torch.load(E0_CHECKPOINT, map_location=DEVICE)
    encoder = Encoder().to(DEVICE)
    decoder = Decoder().to(DEVICE)
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

    print("\n=== 检查1：SNR=25dB下的analog表现（直接对比论文Table 2的31.42dB / 0.91） ===")
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
        psnr_avg, ssim_avg = np.mean(psnr_list), np.mean(ssim_list)
    print(f"E0实测: PSNR={psnr_avg:.2f}dB, SSIM={ssim_avg:.4f}")
    print(f"论文Table 2 (γ=0): PSNR=31.42dB, SSIM=0.91")
    gap = 31.42 - psnr_avg
    print(f"差距: {gap:.2f}dB")
    if abs(gap) < 2:
        print("✅ 差距在2dB以内，E0训练基本正常，问题大概率在E1的量化/信道实现上。")
    else:
        print("⚠️ 差距较大，E0本身可能没训好（欠拟合），需要先解决E0，而不是排查E1。")

    print("\n=== 检查2：无信道噪声（SNR=100dB近似无噪）下的重建质量 ===")
    print("（这一步排除信道噪声因素，纯粹看encoder/decoder本身重建能力上限）")
    with torch.no_grad():
        psnr_list, ssim_list = [], []
        for inputs in val_loader:
            inputs = inputs.to(DEVICE)
            z = encoder(inputs)
            z_hat = rayleigh_awgn_channel_real(z, snr_db=100.0)  # 近似无噪声
            outputs = decoder(z_hat)
            psnr, ssim = compute_psnr_ssim(inputs, outputs)
            psnr_list.append(psnr)
            ssim_list.append(ssim)
        psnr_avg2, ssim_avg2 = np.mean(psnr_list), np.mean(ssim_list)
    print(f"E0实测（近似无噪）: PSNR={psnr_avg2:.2f}dB, SSIM={ssim_avg2:.4f}")
    if psnr_avg2 < 28:
        print("⚠️ 即使几乎没有信道噪声，PSNR也远低于31dB，说明问题在encoder/decoder"
              "本身没训好（欠拟合），不是信道/量化的问题。建议检查E0训练loss是否"
              "真正收敛，或增加训练轮数。")
    else:
        print("✅ 无噪声下表现正常，说明encoder/decoder训得OK，"
              "25dB时的差距可能来自信道模型细节，需要进一步排查。")


if __name__ == "__main__":
    main()
