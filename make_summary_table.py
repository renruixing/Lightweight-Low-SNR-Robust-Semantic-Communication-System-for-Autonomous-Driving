"""
make_summary_table.py
=======================
生成 SNR=25dB 下 E1 vs E2 的对比表格（文字版+markdown版），
方便直接复制进回复信正文引用具体数字。

产出：results/summary_table.md
"""
import os
import json
from common.config import RESULTS_DIR, EVAL_SNR_GRID

TARGET_SNR = 24  # 评测网格是0:3:30(步长3)，不含25，用最接近的24dB代替，
                  # 表格标题会注明这一点，避免误认为是精确的25dB


def load_json_or_none(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_at_snr(curve, snr_grid, target_snr):
    # 找评测网格里离target_snr最近的点（评测网格是0:3:30步长3，不一定含target_snr）
    closest_idx = min(range(len(snr_grid)), key=lambda i: abs(snr_grid[i] - target_snr))
    return curve[closest_idx], snr_grid[closest_idx]


def main():
    e1_data = load_json_or_none(os.path.join(RESULTS_DIR, "E1_train_deploy_results.json"))
    e2_data = load_json_or_none(os.path.join(RESULTS_DIR, "E2_jcm_results.json"))

    if e1_data is None or e2_data is None:
        print("⚠️ E1或E2结果缺失，请确认两个实验都跑完了。")
        return

    lines = [
        f"# Supplementary Table: E1 (train-deploy separation) vs E2 (JCM), near SNR={TARGET_SNR}dB",
        "",
        "*(same γ=0 backbone, same channel model; not for absolute comparison with Table 2, "
        "which reports at exactly SNR=25dB — our eval grid steps by 3dB and doesn't include 25 "
        "exactly, so the closest available point is used and shown in the 'SNR' column)*",
        "",
        "| Modulation | SNR(dB) | E1 PSNR (dB) | E2 PSNR (dB) | ΔPSNR (E2-E1) | E1 SSIM | E2 SSIM | ΔSSIM (E2-E1) |",
        "|---|---|---|---|---|---|---|---|",
    ]

    for mod_name in ["4QAM", "16QAM", "64QAM", "256QAM"]:
        if mod_name not in e1_data or mod_name not in e2_data:
            continue
        snr_grid = e1_data[mod_name]["snr_grid"]
        e1_psnr, actual_snr = get_at_snr(e1_data[mod_name]["psnr"], snr_grid, TARGET_SNR)
        e2_psnr, _ = get_at_snr(e2_data[mod_name]["psnr"], snr_grid, TARGET_SNR)
        e1_ssim, _ = get_at_snr(e1_data[mod_name]["ssim"], snr_grid, TARGET_SNR)
        e2_ssim, _ = get_at_snr(e2_data[mod_name]["ssim"], snr_grid, TARGET_SNR)
        d_psnr = e2_psnr - e1_psnr
        d_ssim = e2_ssim - e1_ssim
        lines.append(
            f"| {mod_name} | {actual_snr} | {e1_psnr:.2f} | {e2_psnr:.2f} | "
            f"{'+' if d_psnr >= 0 else ''}{d_psnr:.2f} | "
            f"{e1_ssim:.4f} | {e2_ssim:.4f} | "
            f"{'+' if d_ssim >= 0 else ''}{d_ssim:.4f} |"
        )

    out_text = "\n".join(lines)
    print(out_text)

    out_path = os.path.join(RESULTS_DIR, "summary_table.md")
    with open(out_path, "w") as f:
        f.write(out_text)
    print(f"\n✅ 已保存至 {out_path}")


if __name__ == "__main__":
    main()
