import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')
plt.rcParams["font.family"] = ["Times New Roman"]

# ============================================================
# Figure 6（最终对齐版）：剪枝(γ=0.5, 宽度256) vs 直接设计带宽压缩比
# (k/n=1/3, 宽度256)。两者瓶颈宽度完全相同(256)，是严格同参数量对比。
# 数据来源：
#   γ=0.5剪枝  <- E10_multi_gamma_results.json
#   k/n=1/3    <- E9_bandwidth_ratio_results.json
# ============================================================

snr = np.arange(0, 31, 3)

psnr = np.array([
    [23.801525603147912,25.571993528090477,26.79930251944989,27.37897947642175,28.086893090120363,28.27652861568573,28.531542447637378,28.710510868345626,28.73787407240088,28.7679153844903,28.813926634586323],  # Deep JSCC(γ=0.5)-analog（剪枝）
    [24.04126534900474,25.28821735420398,26.112270203169164,26.530795123629,27.353644347599527,27.402176072726796,27.68831222134857,27.776500830736552,27.861262283157203,27.858329012956823,27.87556671940738],      # Deep JSCC-1/3-analog（直接设计带宽压缩比）
])
ssim = np.array([
    [0.6354647433620831,0.7149845769028422,0.76886363546753,0.7924854809987542,0.8240438250226492,0.830657617925848,0.8408611579324884,0.8476127887533998,0.8487507845040383,0.8495181003663219,0.8512317098786856],
    [0.6592412909472569,0.7163171459569249,0.7530570266327281,0.7715244171848973,0.8078332373858521,0.8093672791255928,0.8212337461671801,0.82472020540952,0.8280573555277229,0.8279848365336804,0.8286154395242649],
])

labels = ['Deep JSCC(γ=0.5)-analog', 'Deep JSCC-1/3-analog']

plt.figure(1, figsize=(6, 4.5))
plt.plot(snr, psnr[0], marker='o', markersize=4.5)
plt.plot(snr, psnr[1], marker='v', markersize=4.5)
plt.xlabel('SNR')
plt.ylabel('PSNR(dB)')
plt.legend(labels, loc='lower right')
plt.tight_layout()
plt.savefig('./figures/fig6a.pdf', bbox_inches='tight', dpi=1024)
plt.show()

plt.figure(2, figsize=(6, 4.5))
plt.plot(snr, ssim[0], marker='o', markersize=4.5)
plt.plot(snr, ssim[1], marker='v', markersize=4.5)
plt.xlabel('SNR')
plt.ylabel('SSIM')
plt.legend(labels, loc='lower right')
plt.tight_layout()
plt.savefig('./figures/fig6b.pdf', bbox_inches='tight', dpi=1024)
plt.show()