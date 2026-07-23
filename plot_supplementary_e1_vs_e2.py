"""
plot_supplementary_e1_vs_e2.py
================================
【独立补充图，专用于回应审稿人1】E1（训练-部署分离）vs E2（JCM）对比。

数据来源：
    - E1: results/E1_train_deploy_results.json
    - E2: results/E2_jcm_results.json

生成的图：
    - supp_E1_vs_E2_PSNR_{modname}.png
    - supp_E1_vs_E2_SSIM_{modname}.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import RESULTS_DIR, EVAL_SNR_GRID

# 上次E4实测的容量上界结果（信道模型与E1/E2统一，可以放心作为参考线）
E4_REUSE_PSNR = [46.35, 46.37, 46.37, 46.37, 46.37, 46.37,
                 46.37, 46.37, 46.37, 46.37, 46.37]
E4_REUSE_SSIM = [0.9963, 0.9964, 0.9964, 0.9964, 0.9964, 0.9964,
                 0.9964, 0.9964, 0.9964, 0.9964, 0.9964]


def get_e4_curves():
    path = os.path.join(RESULTS_DIR, "E4_capacity_bound_results.json")
    data = load_json_or_none(path)
    if data is not None:
        return data["psnr"], data["ssim"]
    return E4_REUSE_PSNR, E4_REUSE_SSIM


def load_json_or_none(path):
    if not os.path.isfile(path):
        return None
    with open(path, "r") as f:
        return json.load(f)


def plot_one_modulation(mod_name, e1_data, e2_data, e4_psnr, e4_ssim):
    snr = np.array(EVAL_SNR_GRID)

    for metric, ylabel in [("psnr", "PSNR (dB)"), ("ssim", "SSIM")]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        if e1_data is not None and mod_name in e1_data:
            ax.plot(snr, e1_data[mod_name][metric], marker="o", markersize=5,
                    label="Train-deploy separation (proposed)", linewidth=1.5)

        if e2_data is not None and mod_name in e2_data:
            ax.plot(snr, e2_data[mod_name][metric], marker="s", markersize=5,
                    label="JCM (Gumbel-Softmax, end-to-end)", linewidth=1.5)

        curve = e4_psnr if metric == "psnr" else e4_ssim
        ax.plot(snr, curve, linestyle="--", color="black",
                marker="o", markerfacecolor="none", markersize=5,
                label="Capacity-achieving separation bound (reference)", linewidth=1.2)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Supplementary comparison, γ=0, {mod_name}\n"
                      f"(same E0 backbone; not for absolute comparison with Fig.3/4)",
                      fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
        fig.tight_layout()

        out_path = os.path.join(RESULTS_DIR,
                                 f"supp_E1_vs_E2_{metric.upper()}_{mod_name}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存 {out_path}")


def main():
    e1_data = load_json_or_none(os.path.join(RESULTS_DIR, "E1_train_deploy_results.json"))
    e2_data = load_json_or_none(os.path.join(RESULTS_DIR, "E2_jcm_results.json"))
    e4_psnr, e4_ssim = get_e4_curves()

    if e1_data is None:
        print("⚠️ 未找到E1结果")
    if e2_data is None:
        print("⚠️ 未找到E2结果，脚本会先只画出E1（如果有的话），JCM跑完后重跑本脚本即可")

    for mod_name in ["4QAM", "16QAM", "64QAM", "256QAM"]:
        plot_one_modulation(mod_name, e1_data, e2_data, e4_psnr, e4_ssim)

    print("\n✅ 绘图完成。这些图是独立的补充材料，不替换、不叠加进现有Figure 3/Figure 4。")


if __name__ == "__main__":
    main()
