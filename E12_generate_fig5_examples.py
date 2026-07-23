"""
E12_generate_fig5_examples.py
================================
生成Figure 5（重建图像示例，LaTeX label=fig6）的重建图像，逐张保存为
独立PNG文件，命名方式匹配manuscript.tex里原有的
`Fig6_row{行号}col{列号}.png` / `Fig6_first_column.png` 体例，可以
直接替换/插入进现有的 \\includegraphics 网格，不需要重新排版整个
figure环境。

列的含义（跟原LaTeX一致，只是在col3和原col4之间插入了JCM这一新列）：
    col1 = Original（所有行共用同一张图，跟原LaTeX里"first_column"
           被4行重复引用的做法一致，只生成一份）
    col2 = γ=0.7-analog
    col3 = γ=0.7+256QAM（所提方案）
    col4 = γ=0.7+256QAM（JCM，新插入的列）
    col5 = BPG+LDPC+QAM —— 不在本脚本生成范围内（LDPC复现已决定不做），
           请用你自己现有的 `BPG + LDPC + BPSK.py` 类脚本单独生成，
           命名成 Fig6_row{1..4}col5.png 放进同一输出目录即可

行 = 4个SNR点（默认 0/9/15/21dB，对应原图1/9/15/22dB的近似替代，
因为我们的评测网格是0:3:30步长3，没有1/22这种精确值）。

同一张测试图像贯穿所有行/列，保证视觉对比公平。

运行方式：
    python E12_generate_fig5_examples.py
"""
import os
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.backbone_prunable import PrunableEncoder, PrunableDecoder
from common.channel import rayleigh_awgn_channel_real, rayleigh_awgn_channel_complex
from common.qam import uniform_quantize_to_index, qam_modulate, qam_demodulate_to_index, dequantize_from_index
from E2_jcm_model import JCMFullModel, load_warmstart_from_e0
from common.config import DATASET_PATH, IMG_SIZE, RESULTS_DIR, DEVICE

# ============ 可调整的配置 ============
TEST_IMAGE_DIR = os.path.join(os.path.dirname(DATASET_PATH), "test_img")  # 跟DATASET_PATH同目录下的test_img文件夹
TEST_IMAGE_FILENAME = None   # 指定具体文件名（比如"0.jpg"）；留None则自动取该文件夹下第一张图
TARGET_SNRS = [0, 9, 15, 21]  # 对应原图1/9/15/22dB的近似替代（评测网格步长3，没有精确值）
E5_CHECKPOINT = os.path.join(RESULTS_DIR, "E5_pruned_gamma07.pth")
E6_JCM_256QAM_CHECKPOINT = os.path.join(RESULTS_DIR, "E6_jcm_gamma07_256QAM.pth")
M = 256  # 调制阶数，跟原图一致用256QAM
# ========================================


def load_e5():
    state = torch.load(E5_CHECKPOINT, map_location=DEVICE)
    n_keep = int(state["_bottleneck_channels"].item())
    encoder = PrunableEncoder(bottleneck_channels=n_keep).to(DEVICE)
    decoder = PrunableDecoder(bottleneck_channels=n_keep).to(DEVICE)
    enc_state = {k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")}
    dec_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    encoder.load_state_dict(enc_state)
    decoder.load_state_dict(dec_state)
    encoder.eval()
    decoder.eval()
    return encoder, decoder, n_keep


def load_jcm():
    state = torch.load(E6_JCM_256QAM_CHECKPOINT, map_location=DEVICE)
    n_keep = int(round(154))  # γ=0.7对应宽度，与E5一致
    sqrt_M = int(round(M ** 0.5))
    model = JCMFullModel(sqrt_M=sqrt_M, num_positions=n_keep).to(DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def reconstruct_analog(encoder, decoder, x, snr_db):
    z = encoder(x)
    z_hat = rayleigh_awgn_channel_real(z, snr_db)
    return decoder(z_hat)


@torch.no_grad()
def reconstruct_proposed_quant(encoder, decoder, x, snr_db, n_keep):
    z = encoder(x)
    idx = uniform_quantize_to_index(z, M)
    s = qam_modulate(idx, M)
    s_flat = s.reshape(1, -1)
    y_flat = rayleigh_awgn_channel_complex(s_flat, snr_db)
    idx_hat_flat = qam_demodulate_to_index(y_flat, M)
    z_hat_flat = dequantize_from_index(idx_hat_flat, M)
    z_hat = z_hat_flat.reshape(z.shape)
    return decoder(z_hat)


@torch.no_grad()
def reconstruct_jcm(model, x, snr_db):
    return model(x, snr_db=snr_db)


def to_numpy_img(tensor):
    arr = tensor.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (arr * 255).astype(np.uint8)


def compute_psnr_ssim(x_true_np, x_pred_np):
    psnr = peak_signal_noise_ratio(x_true_np, x_pred_np, data_range=255)
    ssim = structural_similarity(x_true_np, x_pred_np, channel_axis=2, data_range=255)
    return psnr, ssim


def save_single_image(img_np, out_path):
    """保存单张纯图像（无边框/标题/坐标轴），供LaTeX \\includegraphics直接引用。"""
    Image.fromarray(img_np).save(out_path)


def main():
    print(f"设备: {DEVICE}")
    encoder, decoder, n_keep = load_e5()
    print(f"已加载E5（γ=0.7，宽度={n_keep}）")
    jcm_model = load_jcm()
    print(f"已加载E6 JCM（256QAM）")

    print(f"测试图像目录: {TEST_IMAGE_DIR}")
    if not os.path.isdir(TEST_IMAGE_DIR):
        raise FileNotFoundError(
            f"找不到目录: {TEST_IMAGE_DIR}\n"
            f"请检查脚本顶部 TEST_IMAGE_DIR 的路径设置是否正确"
            f"（默认假设是 DATASET_PATH 同级目录下的 test_img 文件夹，"
            f"如果实际路径不是这样，请直接改成绝对路径）。"
        )
    candidate_files = sorted(
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    if not candidate_files:
        raise FileNotFoundError(f"{TEST_IMAGE_DIR} 下没有找到任何jpg/png图像。")
    if TEST_IMAGE_FILENAME is not None:
        if TEST_IMAGE_FILENAME not in candidate_files:
            raise FileNotFoundError(
                f"指定的文件 {TEST_IMAGE_FILENAME} 不在 {TEST_IMAGE_DIR} 下，"
                f"该目录里实际有: {candidate_files}"
            )
        img_path = os.path.join(TEST_IMAGE_DIR, TEST_IMAGE_FILENAME)
    else:
        img_path = os.path.join(TEST_IMAGE_DIR, candidate_files[0])
        if len(candidate_files) > 1:
            print(f"⚠️ test_img目录下有{len(candidate_files)}张图，"
                  f"未指定TEST_IMAGE_FILENAME，默认取第一张: {candidate_files[0]}")
    print(f"测试图像: {img_path}")

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])
    original_img = Image.open(img_path).convert("RGB")
    x = transform(original_img).unsqueeze(0).to(DEVICE)
    original_np = to_numpy_img(x)

    out_dir = os.path.join(RESULTS_DIR, "fig6_examples")
    os.makedirs(out_dir, exist_ok=True)

    # 第1列(Original)所有行共用同一张图，跟原LaTeX里"Fig6_first_column.png"
    # 被4行重复引用的做法一致，只存一份
    save_single_image(original_np, os.path.join(out_dir, "Fig6_first_column.png"))
    print(f"已保存 Fig6_first_column.png（Original，所有行共用）")

    psnr_ssim_log = []
    # 列号：2=analog, 3=Proposed+256QAM, 4=JCM+256QAM
    # （列1=Original已单独存好；列5=BPG+LDPC+QAM由你自己脚本生成，不在这里）
    for row_idx, snr in enumerate(TARGET_SNRS, start=1):
        x_analog = reconstruct_analog(encoder, decoder, x, float(snr))
        x_proposed = reconstruct_proposed_quant(encoder, decoder, x, float(snr), n_keep)
        x_jcm = reconstruct_jcm(jcm_model, x, float(snr))

        col_data = {2: ("analog", x_analog), 3: ("proposed+256QAM", x_proposed),
                    4: ("JCM+256QAM", x_jcm)}

        for col_idx, (name, tensor) in col_data.items():
            img_np = to_numpy_img(tensor)
            fname = f"Fig6_row{row_idx}col{col_idx}.png"
            save_single_image(img_np, os.path.join(out_dir, fname))
            psnr, ssim = compute_psnr_ssim(original_np, img_np)
            psnr_ssim_log.append((row_idx, snr, col_idx, name, psnr, ssim))
            print(f"已保存 {fname}  [{name}, SNR={snr}dB]  PSNR={psnr:.2f}dB  SSIM={ssim:.4f}")

    print(f"\n✅ 全部完成，图像保存至 {out_dir}/")
    print("命名对应关系：col1=Original(所有行共用) / col2=analog / col3=Proposed+256QAM / col4=JCM+256QAM")
    print("col5(BPG+LDPC+QAM)请用你自己现有脚本单独生成，命名成 Fig6_row{1..4}col5.png 放进同一目录即可直接替换LaTeX里的引用")
    print("\nPSNR/SSIM汇总（可用于图注或核对）：")
    for row_idx, snr, col_idx, name, psnr, ssim in psnr_ssim_log:
        print(f"  row{row_idx}(SNR={snr}dB) col{col_idx}({name}): {psnr:.2f}dB / {ssim:.4f}")


if __name__ == "__main__":
    main()