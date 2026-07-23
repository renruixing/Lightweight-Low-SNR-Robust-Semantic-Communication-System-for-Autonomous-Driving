"""
E8_measure_latency.py
=======================
【实验E8】测量Encoder（车端实际运行的部分）在不同剪枝率下的推理延迟，
回应审稿人3意见5（硬件部署可行性）。

关键点：推理延迟只取决于网络架构（层数、通道数、卷积核大小），不取决
于具体权重数值，所以不需要等对应γ的checkpoint训练/剪枝完成——直接
用未训练的模型实例计时即可，五个剪枝率可以一次性全部测完。

只测Encoder，不测Decoder：车端（source vehicle）只需要跑Encoder把
图像编码成语义特征后发送，Decoder运行在接收端（ego vehicle），通常
不受车载终端算力限制，因此"车端部署可行性"主要看Encoder的延迟。

运行方式：
    python E8_measure_latency.py
"""
import os
import json
import time
import numpy as np
import torch

from common.backbone_prunable import PrunableEncoder
from common.config import RESULTS_DIR, IMG_SIZE, DEVICE

GAMMAS = [0, 0.2, 0.5, 0.7, 0.9]
N_WARMUP = 10
N_TIMED = 100
BATCH_SIZE_DEPLOY = 1  # 部署场景：单张图像实时编码，不是训练时的batch=32


def measure_latency(model, device, batch_size, n_warmup, n_timed):
    model = model.to(device).eval()
    x = torch.rand(batch_size, 3, IMG_SIZE, IMG_SIZE, device=device)

    with torch.no_grad():
        # 预热：排除首次调用的初始化开销（尤其GPU的cuDNN kernel选择等）
        for _ in range(n_warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()

        times = []
        for _ in range(n_timed):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    return float(np.mean(times)), float(np.std(times))


def main():
    devices_to_test = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices_to_test.append(torch.device("cuda"))

    print(f"待测设备: {[str(d) for d in devices_to_test]}")
    print(f"批大小(部署场景): {BATCH_SIZE_DEPLOY}, 预热{N_WARMUP}次, 计时{N_TIMED}次取均值±标准差\n")

    results = {}
    for gamma in GAMMAS:
        n_keep = int(round(512 * (1 - gamma)))
        encoder = PrunableEncoder(bottleneck_channels=n_keep)
        n_params = sum(p.numel() for p in encoder.parameters())

        results[str(gamma)] = {"n_keep": n_keep, "encoder_params_M": n_params / 1e6}
        print(f"γ={gamma} (宽度={n_keep}, Encoder参数量={n_params/1e6:.2f}M):")

        for device in devices_to_test:
            mean_ms, std_ms = measure_latency(encoder, device, BATCH_SIZE_DEPLOY, N_WARMUP, N_TIMED)
            results[str(gamma)][f"latency_{device.type}_ms"] = mean_ms
            results[str(gamma)][f"latency_{device.type}_std_ms"] = std_ms
            fps = 1000.0 / mean_ms
            print(f"  [{device.type:>4}] {mean_ms:.2f} ± {std_ms:.2f} ms/image  (~{fps:.1f} FPS)")
        print()

    out_path = os.path.join(RESULTS_DIR, "E8_latency_results.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ 完成，结果保存至 {out_path}")

    # 额外打印一份markdown表格，方便直接贴进回复信/论文
    lines = ["| γ | Encoder宽度 | Encoder参数量(M) |"]
    dev_names = [d.type for d in devices_to_test]
    header = "| γ | Encoder宽度 | Encoder参数量(M) |" + "".join(f" {d.upper()}延迟(ms) |" for d in dev_names)
    sep = "|---|---|---|" + "---|" * len(dev_names)
    md_lines = [header, sep]
    for gamma in GAMMAS:
        r = results[str(gamma)]
        row = f"| γ={gamma} | {r['n_keep']} | {r['encoder_params_M']:.2f} |"
        for d in dev_names:
            row += f" {r[f'latency_{d}_ms']:.2f} ± {r[f'latency_{d}_std_ms']:.2f} |"
        md_lines.append(row)
    md_text = "\n".join(md_lines)
    print("\n" + md_text)
    with open(os.path.join(RESULTS_DIR, "E8_latency_table.md"), "w") as f:
        f.write(md_text)


if __name__ == "__main__":
    main()
  