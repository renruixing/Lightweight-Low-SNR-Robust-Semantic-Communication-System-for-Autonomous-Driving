"""
make_ablation_table.py
========================
把消融实验四格结果拼成对比表（markdown）+ 对比图，用于回应审稿人3
意见3（剪枝vs量化贡献隔离）。

四格数据来源：
    (a) 未剪枝+analog        results/E7_E0_unpruned_analog_results.json
    (b) 未剪枝+量化(训练-部署分离)  results/E1_train_deploy_results.json
    (c) 剪枝+analog          results/E7_E5_pruned_gamma07_analog_results.json
    (d) 剪枝+量化(训练-部署分离)    results/E7_pruned_gamma07_quantization_results.json

(b)和(d)各自包含4/16/64/256QAM，表格/图里为避免过于拥挤，默认取64QAM
（跟论文Table 2里"BPG+LDPC+64QAM"作为参照点的选择一致），如需其他
调制阶数改 TARGET_MOD 常量即可。

产出：
    results/ablation_table.md
    results/ablation_PSNR.png / ablation_SSIM.png
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.config import RESULTS_DIR, EVAL_SNR_GRID

TARGET_MOD = "64QAM"  # (b)/(d)取哪个调制阶数展示，可改成4QAM/16QAM/256QAM


def load_json(path):
    if not os.path.isfile(path):
        print(f"⚠️ 未找到 {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def get_at_snr(curve, snr_grid, target_snr):
    idx = min(range(len(snr_grid)), key=lambda i: abs(snr_grid[i] - target_snr))
    return curve[idx], snr_grid[idx]


def main():
    a = load_json(os.path.join(RESULTS_DIR, "E7_E0_unpruned_analog_results.json"))
    b = load_json(os.path.join(RESULTS_DIR, "E1_train_deploy_results.json"))
    c = load_json(os.path.join(RESULTS_DIR, "E7_E5_pruned_gamma07_analog_results.json"))
    d = load_json(os.path.join(RESULTS_DIR, "E7_pruned_gamma07_quantization_results.json"))

    missing = [name for name, v in
               [("(a)未剪枝+analog", a), ("(b)未剪枝+量化", b),
                ("(c)剪枝+analog", c), ("(d)剪枝+量化", d)] if v is None]
    if missing:
        print(f"缺少: {missing}，请先跑齐对应脚本。")
        return

    b_curve_psnr, b_curve_ssim = b[TARGET_MOD]["psnr"], b[TARGET_MOD]["ssim"]
    d_curve_psnr, d_curve_ssim = d[TARGET_MOD]["psnr"], d[TARGET_MOD]["ssim"]

    # ---------------- 表格：SNR≈24dB这一行的数字对比 ----------------
    target_snr = 24
    lines = [
        f"# Ablation: Pruning vs. Quantization Contribution (near SNR={target_snr}dB, {TARGET_MOD})",
        "",
        "| Condition | PSNR (dB) | SSIM |",
        "|---|---|---|",
    ]
    for name, curve_psnr, curve_ssim, snr_grid in [
        ("(a) Unpruned + analog", a["psnr"], a["ssim"], a["snr_grid"]),
        (f"(b) Unpruned + {TARGET_MOD}", b_curve_psnr, b_curve_ssim, b[TARGET_MOD]["snr_grid"]),
        ("(c) Pruned(γ=0.7) + analog", c["psnr"], c["ssim"], c["snr_grid"]),
        (f"(d) Pruned(γ=0.7) + {TARGET_MOD}", d_curve_psnr, d_curve_ssim, d[TARGET_MOD]["snr_grid"]),
    ]:
        p, actual_snr = get_at_snr(curve_psnr, snr_grid, target_snr)
        s, _ = get_at_snr(curve_ssim, snr_grid, target_snr)
        lines.append(f"| {name} | {p:.2f} | {s:.4f} |")

    lines += [
        "",
        f"**Pruning-only cost** = (a) − (c)  ",
        f"**Quantization-only cost** = (a) − (b)  ",
        f"**Combined cost** = (a) − (d), compared against (pruning-only + quantization-only) "
        f"to check whether the two effects are roughly additive or interact.",
    ]

    a_p, _ = get_at_snr(a["psnr"], a["snr_grid"], target_snr)
    b_p, _ = get_at_snr(b_curve_psnr, b[TARGET_MOD]["snr_grid"], target_snr)
    c_p, _ = get_at_snr(c["psnr"], c["snr_grid"], target_snr)
    d_p, _ = get_at_snr(d_curve_psnr, d[TARGET_MOD]["snr_grid"], target_snr)

    pruning_cost = a_p - c_p
    quant_cost = a_p - b_p
    combined_cost = a_p - d_p
    additive_prediction = pruning_cost + quant_cost

    lines += [
        "",
        f"- Pruning-only cost: {pruning_cost:+.2f} dB",
        f"- Quantization-only cost: {quant_cost:+.2f} dB",
        f"- Combined (measured) cost: {combined_cost:+.2f} dB",
        f"- Additive prediction (pruning + quantization): {additive_prediction:+.2f} dB",
        f"- Interaction term (combined − additive): {combined_cost - additive_prediction:+.2f} dB",
    ]

    out_text = "\n".join(lines)
    print(out_text)
    with open(os.path.join(RESULTS_DIR, "ablation_table.md"), "w") as f:
        f.write(out_text)

    # ---------------- 图 ----------------
    snr = np.array(EVAL_SNR_GRID)
    for metric, ylabel, a_c, b_c, c_c, d_c in [
        ("PSNR", "PSNR (dB)", a["psnr"], b_curve_psnr, c["psnr"], d_curve_psnr),
        ("SSIM", "SSIM", a["ssim"], b_curve_ssim, c["ssim"], d_curve_ssim),
    ]:
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.plot(snr, a_c, marker="o", label="(a) Unpruned + analog")
        ax.plot(snr, b_c, marker="s", label=f"(b) Unpruned + {TARGET_MOD}")
        ax.plot(snr, c_c, marker="^", label="(c) Pruned(γ=0.7) + analog")
        ax.plot(snr, d_c, marker="d", label=f"(d) Pruned(γ=0.7) + {TARGET_MOD}")
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Ablation: pruning vs. quantization contribution ({metric})")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9)
        fig.tight_layout()
        out_path = os.path.join(RESULTS_DIR, f"ablation_{metric}.png")
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"已保存 {out_path}")


if __name__ == "__main__":
    main()
