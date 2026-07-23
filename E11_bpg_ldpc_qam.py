"""
E11_bpg_ldpc_qam.py
=====================
复现 Figure 3 里的 BPG+LDPC+4QAM 基线（3/4和2/3两种码率），补上此前
一直没做的这最后一块拼图。16QAM按之前商量的暂不做（需要额外写多电平
软解调LLR公式，工作量大，价值相对4QAM更低）。

核心设计：把4QAM(QPSK)当成两路独立的BPSK+LDPC处理（I路一份LDPC
codeword，Q路另一份），复用pyldpc现成的BP译码器，不用自己写QAM的
软解调LLR公式。信道用common/channel.py同一套瑞利衰落模型（慢衰落，
整张图共享同一个信道实现h），跟本项目其余部分的信道口径保持一致。

流程（每张测试图像）：
    1. BPG压缩（固定quality=35，与原论文脚本"BPG + LDPC + BPSK.py"
       里的设置一致）
    2. 压缩字节流转成比特流，对半分成I路/Q路两个子流
    3. 每路各自按LDPC码长切块、编码
    4. 每个信噪比下：I/Q两路符号一起过瑞利信道+AWGN，均衡后分别用
       pyldpc译码
    5. 译码结果拼回原比特流，尝试BPG解码；译码失败/BPG解码报错时，
       按照论文"悬崖效应"的约定记为 PSNR=0, SSIM=0

⚠️ 运行前提：
- 需要本地 bpgenc.exe/bpgdec.exe（下载：https://bellard.org/bpg/），
  跟 E4_capacity_bound.py 用的是同一套工具，配置方式相同
- 需要先 pip install pyldpc（--break-system-packages 如果需要）
- 默认只测N_TEST_IMAGES=30张（不是全部500张），把500张全跑一遍在
  当前译码速度下要几十小时，30张是经过权衡的折中，曲线仍有统计意义

运行方式：
    python E11_bpg_ldpc_qam.py
"""
import os
import json
import time
import subprocess
import tempfile
import numpy as np
import pyldpc
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.config import DATASET_PATH, RESULTS_DIR, EVAL_SNR_GRID

# ============ 需要你根据本地实际情况修改 ============
BPGENC_CMD = r"/home/rrx/hdd0/files/libbpg-0.9.8/bpgenc"
BPGDEC_CMD = r"/home/rrx/hdd0/files/libbpg-0.9.8/bpgdec"
N_TEST_IMAGES = 30           # 折中后的测试集大小，不是全部500张
BPG_QUALITY = 35             # 固定画质参数，与原论文脚本一致
N_CODE = 384                 # LDPC码长（codeword长度）
MAXITER = 50                 # BP译码最大迭代次数
CODE_RATES = {
    "3_4": {"d_v": 3, "d_c": 12, "label": "3/4"},   # 码率 = 1 - 3/12 = 3/4
    "2_3": {"d_v": 4, "d_c": 12, "label": "2/3"},   # 码率 = 1 - 4/12 = 2/3
}
# ======================================================


def _check_executables():
    for name, path in (("BPGENC_CMD", BPGENC_CMD), ("BPGDEC_CMD", BPGDEC_CMD)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到 {name}='{path}'。请从 https://bellard.org/bpg/ 下载并配置完整路径。"
            )


def bpg_compress(image_path, tmp_dir):
    """固定quality=35压缩，返回压缩后的字节内容（bytes对象）。"""
    bin_path = os.path.join(tmp_dir, "tmp_enc.bin")
    subprocess.run([BPGENC_CMD, "-m", "1", "-b", "8", "-q", str(BPG_QUALITY),
                    image_path, "-o", bin_path], check=True, capture_output=True)
    with open(bin_path, "rb") as f:
        return f.read()


def bpg_decompress(data_bytes, tmp_dir):
    """尝试BPG解码，失败则抛异常（调用方按译码失败处理）。"""
    bin_path = os.path.join(tmp_dir, "tmp_dec.bin")
    png_path = os.path.join(tmp_dir, "tmp_dec.png")
    with open(bin_path, "wb") as f:
        f.write(data_bytes)
    subprocess.run([BPGDEC_CMD, "-o", png_path, bin_path],
                    check=True, capture_output=True, timeout=10)
    return np.array(Image.open(png_path).convert("RGB"))


def bytes_to_bits(data_bytes):
    return np.unpackbits(np.frombuffer(data_bytes, dtype=np.uint8))


def bits_to_bytes(bits):
    n_pad = (-len(bits)) % 8
    if n_pad:
        bits = np.concatenate([bits, np.zeros(n_pad, dtype=bits.dtype)])
    return np.packbits(bits).tobytes()


def encode_stream(bits, tG, k_bits):
    """把比特流切块、逐块LDPC编码，返回codeword列表和原始比特长度（用于译码后裁剪）。"""
    n_blocks = int(np.ceil(len(bits) / k_bits))
    padded_len = n_blocks * k_bits
    bits_padded = np.concatenate([bits, np.zeros(padded_len - len(bits), dtype=bits.dtype)])
    codewords = []
    for i in range(n_blocks):
        msg = bits_padded[i * k_bits:(i + 1) * k_bits]
        cw = pyldpc.utils.binaryproduct(tG, msg)
        codewords.append(cw)
    return codewords, len(bits)


def transmit_and_decode(codewords_I, codewords_Q, H, tG, snr_db, rng):
    """I/Q两路codeword一起过同一个瑞利信道实现(慢衰落)，译码后返回拼接好的比特流。"""
    sigma_h = np.sqrt(0.5)
    h = complex(rng.normal(0, sigma_h), rng.normal(0, sigma_h))
    h_mag = abs(h)
    if h_mag < 0.1:  # 深衰落保护，跟common/channel.py一致
        h = h / h_mag * 0.1
    signal_power = 1.0
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    decoded_I_bits, decoded_Q_bits = [], []
    for cw_I, cw_Q in zip(codewords_I, codewords_Q):
        x_I = (-1.0) ** cw_I
        x_Q = (-1.0) ** cw_Q
        s = x_I + 1j * x_Q

        noise = np.sqrt(noise_power / 2) * (rng.standard_normal(len(s)) + 1j * rng.standard_normal(len(s)))
        rx = s * h + noise
        y_eq = rx / h

        y_I = y_eq.real / noise_power
        y_Q = y_eq.imag / noise_power

        x_I_hat = pyldpc.decode(H, y_I, snr=0.0, maxiter=MAXITER)
        x_Q_hat = pyldpc.decode(H, y_Q, snr=0.0, maxiter=MAXITER)
        decoded_I_bits.append(pyldpc.get_message(tG, x_I_hat))
        decoded_Q_bits.append(pyldpc.get_message(tG, x_Q_hat))

    return np.concatenate(decoded_I_bits), np.concatenate(decoded_Q_bits)


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

    val_dir = os.path.join(DATASET_PATH, "val")
    image_files = sorted(
        os.path.join(val_dir, f) for f in os.listdir(val_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )[:N_TEST_IMAGES]
    print(f"测试图像数量: {len(image_files)}（折中版，不是全部500张）")

    all_results = {}
    out_path = os.path.join(RESULTS_DIR, "E11_bpg_ldpc_qam_results.json")
    if os.path.isfile(out_path):
        with open(out_path, "r") as f:
            all_results = json.load(f)
        print(f"检测到已有部分结果: {list(all_results.keys())}，将跳过")

    for rate_key, rate_cfg in CODE_RATES.items():
        result_key = f"{rate_cfg['label']}LDPC+4QAM"
        if result_key in all_results:
            print(f"\n跳过 {result_key}（已有结果）")
            continue

        print(f"\n{'='*60}\n{result_key}\n{'='*60}")
        H, tG = pyldpc.make_ldpc(N_CODE, rate_cfg["d_v"], rate_cfg["d_c"],
                                  systematic=True, sparse=True, seed=42)
        k_bits = tG.shape[1]
        print(f"LDPC参数: n_code={N_CODE}, k_bits={k_bits}, 实际码率={k_bits/N_CODE:.4f}")

        t_start = time.time()
        psnr_curve, ssim_curve = [], []
        with tempfile.TemporaryDirectory() as tmp_dir:
            # 预先对所有测试图像做BPG压缩+LDPC编码（跟SNR无关，只算一次）
            image_data = []
            for img_path in image_files:
                compressed = bpg_compress(img_path, tmp_dir)
                bits = bytes_to_bits(compressed)
                half = len(bits) // 2
                bits_I, bits_Q = bits[:half], bits[half:]
                cw_I, len_I = encode_stream(bits_I, tG, k_bits)
                cw_Q, len_Q = encode_stream(bits_Q, tG, k_bits)
                original = np.array(Image.open(img_path).convert("RGB"))
                image_data.append({
                    "cw_I": cw_I, "cw_Q": cw_Q, "len_I": len_I, "len_Q": len_Q,
                    "original": original,
                })

            for snr in EVAL_SNR_GRID:
                rng = np.random.default_rng(hash((rate_key, snr)) % (2**32))
                psnr_list, ssim_list = [], []
                for idx, data in enumerate(image_data):
                    try:
                        dec_I, dec_Q = transmit_and_decode(
                            data["cw_I"], data["cw_Q"], H, tG, float(snr), rng
                        )
                        bits_I_hat = dec_I[:data["len_I"]]
                        bits_Q_hat = dec_Q[:data["len_Q"]]
                        bits_full = np.concatenate([bits_I_hat, bits_Q_hat])
                        recovered_bytes = bits_to_bytes(bits_full)
                        decoded_img = bpg_decompress(recovered_bytes, tmp_dir)
                        psnr, ssim = compute_psnr_ssim(data["original"], decoded_img)
                    except Exception:
                        # 译码失败/BPG解码报错，按论文"悬崖效应"约定记为0
                        psnr, ssim = 0.0, 0.0
                    psnr_list.append(psnr)
                    ssim_list.append(ssim)

                    if (idx + 1) % 10 == 0:
                        elapsed = time.time() - t_start
                        print(f"[{result_key}][SNR={snr}dB] {idx+1}/{len(image_data)}张  "
                              f"已用{elapsed/60:.1f}分钟")

                psnr_avg, ssim_avg = float(np.mean(psnr_list)), float(np.mean(ssim_list))
                psnr_curve.append(psnr_avg)
                ssim_curve.append(ssim_avg)
                print(f">>> [{result_key}] SNR={snr:>3}dB  PSNR={psnr_avg:.2f}dB  SSIM={ssim_avg:.4f}")

        all_results[result_key] = {
            "code_rate": rate_cfg["label"],
            "n_test_images": len(image_files),
            "snr_grid": EVAL_SNR_GRID,
            "psnr": psnr_curve,
            "ssim": ssim_curve,
        }
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"[{result_key}] 结果已写入 {out_path}，耗时{(time.time()-t_start)/60:.1f}分钟")

    print(f"\n✅ 完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()