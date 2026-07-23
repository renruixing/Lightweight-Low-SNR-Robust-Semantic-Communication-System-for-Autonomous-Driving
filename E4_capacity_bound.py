"""
E4_capacity_bound.py
=====================
【实验E4】容量可达分离式编码上界（对应审稿人2意见4）。

方法：
1. 蒙特卡洛计算慢瑞利衰落信道的遍历容量 C(SNR) = E_h[log2(1 + SNR|h|²)]。
   信道模型与 common/channel.py 完全一致（零均值单位方差瑞利）。
2. 每个SNR下，可无误传输的比特预算 B(SNR) = floor(C(SNR) * k)，
   其中 k = 512*32*32 = 524288（与论文 k/n=2/3, 512x512x3 输入一致）。
3. 用BPG将每张验证集图像压缩到不超过 B(SNR)/8 字节的最高画质档
   （对BPG的-q参数做二分搜索），假设理想容量可达信道编码下无误传输，
   计算解码后PSNR/SSIM。

预期结果（沿用上次已跑过的实测数据；如需重跑，见下方运行方式说明）：
    PSNR ≈ 46.35~46.37 dB，SSIM ≈ 0.9963~0.9964
    在整个0~30dB区间几乎是水平线——不是因为信道容量限制，而是因为
    BPG最高画质档的文件大小已经小于容量比特预算（说明BPG的画质天花板
    先于信道容量成为瓶颈）。这一发现已经写进审稿人回复的意见4部分。

⚠️ 运行前提：
- 需要本地已安装 bpgenc.exe / bpgdec.exe（下载：https://bellard.org/bpg/）
- 修改下方 BPGENC_CMD / BPGDEC_CMD 为你本地的完整路径
- 如果只想复用上次实测结果（推荐，避免45分钟重跑），直接把下方
  PSNR_REUSE / SSIM_REUSE 对应的值填给 plot_figure4_three_way.py 即可，
  不用再跑本脚本。
"""
import os
import json
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from common.config import DATASET_PATH, IMG_SIZE, EVAL_SNR_GRID, RESULTS_DIR

# ============ 需要你根据本地实际情况修改的路径 ============
BPGENC_CMD = r"bpgenc.exe"   # 例如 r"C:\bpg\bpgenc.exe"
BPGDEC_CMD = r"bpgdec.exe"   # 例如 r"C:\bpg\bpgdec.exe"
MAX_SEARCH_ITERS = 6         # BPG -q 参数二分搜索次数
Q_MIN, Q_MAX = 0, 51
# ============================================================

# 上次已实测的结果，如果不想重跑BPG可以直接从这里读
PSNR_REUSE = [46.35, 46.37, 46.37, 46.37, 46.37, 46.37,
              46.37, 46.37, 46.37, 46.37, 46.37]
SSIM_REUSE = [0.9963, 0.9964, 0.9964, 0.9964, 0.9964, 0.9964,
              0.9964, 0.9964, 0.9964, 0.9964, 0.9964]


def ergodic_capacity_rayleigh(snr_db_array, num_trials=2_000_000, seed=0):
    """蒙特卡洛计算遍历容量 C(SNR)，单位 bits/complex channel use。"""
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(1 / 2)  # 与common/channel.py一致
    h_real = rng.normal(0.0, sigma, size=num_trials)
    h_imag = rng.normal(0.0, sigma, size=num_trials)
    h_abs2 = h_real ** 2 + h_imag ** 2  # E[|H|^2] = 1

    snr_db_array = np.asarray(snr_db_array, dtype=float)
    snr_linear = 10 ** (snr_db_array / 10)
    capacity = np.empty_like(snr_db_array)
    for i, snr in enumerate(snr_linear):
        capacity[i] = np.mean(np.log2(1 + snr * h_abs2))
    return capacity


def _check_executables():
    for name, path in (("BPGENC_CMD", BPGENC_CMD), ("BPGDEC_CMD", BPGDEC_CMD)):
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"找不到 {name}='{path}'。请从 https://bellard.org/bpg/ 下载"
                f"bpg-0.9.8-win64.zip 并解压到本地固定目录，把上方"
                f"BPGENC_CMD/BPGDEC_CMD 改成完整路径（含 .exe）。"
            )


def _bpg_encode_decode(image_path, quality, tmp_dir):
    bin_path = os.path.join(tmp_dir, "tmp.bin")
    png_path = os.path.join(tmp_dir, "tmp_dec.png")
    subprocess.run([BPGENC_CMD, "-m", "1", "-b", "8", "-q", str(quality),
                    image_path, "-o", bin_path], check=True, capture_output=True)
    encoded_bytes = os.path.getsize(bin_path)
    subprocess.run([BPGDEC_CMD, "-o", png_path, bin_path], check=True, capture_output=True)
    decoded = np.array(Image.open(png_path).convert("RGB"))
    return decoded, encoded_bytes


def encode_at_byte_budget(image_path, target_bytes, tmp_dir):
    lo, hi = Q_MIN, Q_MAX
    best = None
    for _ in range(MAX_SEARCH_ITERS):
        mid = (lo + hi) // 2
        decoded, size_bytes = _bpg_encode_decode(image_path, mid, tmp_dir)
        if size_bytes <= target_bytes:
            best = (decoded, size_bytes, mid)
            hi = mid - 1
        else:
            lo = mid + 1
        if lo > hi:
            break
    if best is None:
        decoded, size_bytes = _bpg_encode_decode(image_path, Q_MAX, tmp_dir)
        best = (decoded, size_bytes, Q_MAX)
    return best


def main():
    _check_executables()

    image_dir = os.path.join(DATASET_PATH, "val")
    image_files = sorted(
        os.path.join(image_dir, f) for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )
    print(f"验证集: {len(image_files)}张")

    k = int(round((2 / 3) * IMG_SIZE * IMG_SIZE * 3))
    C_bits = ergodic_capacity_rayleigh(np.array(EVAL_SNR_GRID),
                                        num_trials=2_000_000, seed=0)
    budget_bytes_list = (np.floor(C_bits * k) // 8).astype(int).tolist()
    print(f"k={k}, 各SNR比特预算(bytes/image): {budget_bytes_list}")

    total_units = len(EVAL_SNR_GRID) * len(image_files)
    done = 0
    t0 = time.time()
    results = {}
    with tempfile.TemporaryDirectory() as tmp_dir:
        for snr_idx, (snr, target_bytes) in enumerate(
                zip(EVAL_SNR_GRID, budget_bytes_list), start=1):
            psnr_list, ssim_list = [], []
            for img_idx, img_path in enumerate(image_files, start=1):
                decoded, _, _ = encode_at_byte_budget(img_path, target_bytes, tmp_dir)
                original = np.array(Image.open(img_path).convert("RGB"))
                if decoded.shape != original.shape:
                    h = min(decoded.shape[0], original.shape[0])
                    w = min(decoded.shape[1], original.shape[1])
                    decoded, original = decoded[:h, :w], original[:h, :w]
                psnr_list.append(peak_signal_noise_ratio(original, decoded, data_range=255))
                ssim_list.append(structural_similarity(original, decoded,
                                                        channel_axis=2, data_range=255))
                done += 1
                if done % 50 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / done * (total_units - done)
                    print(f"[SNR {snr_idx}/{len(EVAL_SNR_GRID)}={snr}dB] "
                          f"{img_idx}/{len(image_files)}张  "
                          f"总进度 {done}/{total_units}  "
                          f"已用{elapsed/60:.1f}分  ETA {eta/60:.1f}分")
            results[snr] = {
                "psnr": float(np.mean(psnr_list)),
                "ssim": float(np.mean(ssim_list)),
                "budget_bytes": int(target_bytes),
            }
            print(f">>> SNR={snr}dB 完成 PSNR={results[snr]['psnr']:.2f} "
                  f"SSIM={results[snr]['ssim']:.4f}")

    out = {
        "snr_grid": EVAL_SNR_GRID,
        "psnr": [results[s]["psnr"] for s in EVAL_SNR_GRID],
        "ssim": [results[s]["ssim"] for s in EVAL_SNR_GRID],
    }
    out_path = os.path.join(RESULTS_DIR, "E4_capacity_bound_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n✅ E4完成，结果保存至 {out_path}")


if __name__ == "__main__":
    main()
