"""
E13_single_bpg_ldpc_example.py
=================================
复用E11_bpg_ldpc_qam.py的核心pipeline，只生成**一张**BPG+LDPC+QAM
重建示例图（不是完整曲线），用于补上Figure 5（fig6）第5列缺的部分。

用的是跟E12_generate_fig5_examples.py**同一张**测试图片（test_img
文件夹），保证Figure 5五列用的是同一个场景，视觉对比公平。

默认用3/4码率+4QAM（E11里已经验证过能跑通的配置），可以改
TARGET_SNRS加测其他信噪比点，或者改CODE_RATE_KEY换成2/3码率。

运行方式：
    python E13_single_bpg_ldpc_example.py
"""
import os
import tempfile
import numpy as np
import pyldpc
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from E11_bpg_ldpc_qam import (
    BPGENC_CMD, BPGDEC_CMD, BPG_QUALITY, N_CODE, CODE_RATES,
    _check_executables, bpg_compress, bpg_decompress,
    bytes_to_bits, bits_to_bytes, encode_stream, transmit_and_decode,
)
from E12_generate_fig5_examples import TEST_IMAGE_DIR, TEST_IMAGE_FILENAME
from common.config import RESULTS_DIR

# ============ 可调整的配置 ============
TARGET_SNRS = [0, 9, 15, 21]   # 跟E12用的SNR点保持一致，方便直接对应Figure5的4行
CODE_RATE_KEY = "3_4"          # "3_4" 或 "2_3"，对应E11里定义的两种码率
ROW_MAP = {0: 1, 9: 2, 15: 3, 21: 4}  # SNR -> Figure5行号，跟E12的row编号对应
# 注：只跑单次信道实现，不做"多次重复选最好"——analog/Proposed/JCM那几列
# 也都是单次随机信道实现，BPG这列必须用同样的方法论，不能单独给它开
# "多选一"的特权，否则不同列之间的比较就不公平了。
# ========================================


def compute_psnr_ssim(x_true, x_pred):
    if x_true.shape != x_pred.shape:
        h = min(x_true.shape[0], x_pred.shape[0])
        w = min(x_true.shape[1], x_pred.shape[1])
        x_true, x_pred = x_true[:h, :w], x_pred[:h, :w]
    psnr = peak_signal_noise_ratio(x_true, x_pred, data_range=255)
    ssim = structural_similarity(x_true, x_pred, channel_axis=2, data_range=255)
    return psnr, ssim


def main():
    _check_executables()

    # 定位测试图片（跟E12用同一张）
    candidate_files = sorted(
        f for f in os.listdir(TEST_IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    img_filename = TEST_IMAGE_FILENAME or candidate_files[0]
    img_path = os.path.join(TEST_IMAGE_DIR, img_filename)
    print(f"测试图像: {img_path}")

    rate_cfg = CODE_RATES[CODE_RATE_KEY]
    print(f"码率: {rate_cfg['label']}, 调制: 4QAM")
    H, tG = pyldpc.make_ldpc(N_CODE, rate_cfg["d_v"], rate_cfg["d_c"],
                              systematic=True, sparse=True, seed=42)
    k_bits = tG.shape[1]

    out_dir = os.path.join(RESULTS_DIR, "fig6_examples")
    os.makedirs(out_dir, exist_ok=True)

    original = np.array(Image.open(img_path).convert("RGB"))

    with tempfile.TemporaryDirectory() as tmp_dir:
        compressed = bpg_compress(img_path, tmp_dir)
        bits = bytes_to_bits(compressed)
        half = len(bits) // 2
        bits_I, bits_Q = bits[:half], bits[half:]
        cw_I, len_I = encode_stream(bits_I, tG, k_bits)
        cw_Q, len_Q = encode_stream(bits_Q, tG, k_bits)
        print(f"BPG压缩后大小: {len(compressed)}字节, I路{len(cw_I)}个codeword, Q路{len(cw_Q)}个codeword")

        for snr in TARGET_SNRS:
            rng = np.random.default_rng(hash((CODE_RATE_KEY, snr)) % (2**32))
            try:
                dec_I, dec_Q = transmit_and_decode(cw_I, cw_Q, H, tG, float(snr), rng)
                bits_I_hat = dec_I[:len_I]
                bits_Q_hat = dec_Q[:len_Q]
                bits_full = np.concatenate([bits_I_hat, bits_Q_hat])
                recovered_bytes = bits_to_bytes(bits_full)
                decoded_img = bpg_decompress(recovered_bytes, tmp_dir)
                psnr, ssim = compute_psnr_ssim(original, decoded_img)
            except Exception:
                psnr, ssim, decoded_img = 0.0, 0.0, np.zeros_like(original)

            row_idx = ROW_MAP.get(snr, "?")
            out_path = os.path.join(out_dir, f"Fig6_row{row_idx}col5.png")
            Image.fromarray(decoded_img.astype(np.uint8)).save(out_path)
            print(f">>> SNR={snr}dB: PSNR={psnr:.2f}dB SSIM={ssim:.4f}  已保存至 {out_path}")


if __name__ == "__main__":
    main()