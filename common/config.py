"""
common/config.py
=================
所有实验共用的配置。改路径/超参只需要改这一个文件。
"""
import os
import numpy as np
import torch

# ============ 你必须确认/修改的路径（只有这一个地方需要改） ============
# Cityscapes扁平数据集路径，里面应直接包含 train/ 和 val/ 两个子文件夹
DATASET_PATH = r"./dataset/cityscapes"

# ============ 一般不需要修改的实验参数 ============
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 结果输出目录（脚本会自动创建）
RESULTS_DIR = "./results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# 数据 & 训练超参
IMG_SIZE = 512
BATCH_SIZE = 32
LR = 1e-4
SNR_TRAIN_DB = 25       # 训练固定SNR，与论文Section 4.1一致

# E0 基础analog训练超参
E0_NUM_EPOCHS = 200     # 30轮实测只能到23.8dB(论文31.42dB)，150轮约1小时，这是目前唯一还没验证过的变量
E0_CHECKPOINT = os.path.join(RESULTS_DIR, "E0_analog_baseline.pth")

# E2 JCM训练超参（4个调制阶数共用一套设置）
E2_NUM_EPOCHS = 50      # JCM量化训练稍慢，需要更多轮次
E2_GUMBEL_TAU = 1.5
E2_GRAD_CLIP = 5.0
E2_MOD_LIST = [(2, "4QAM"), (4, "16QAM"), (8, "64QAM"), (16, "256QAM")]
E2_RESULTS_JSON = os.path.join(RESULTS_DIR, "E2_jcm_results.json")

# 评测SNR网格（与论文Fig.3/4完全一致）
EVAL_SNR_GRID = np.arange(0, 31, 3).tolist()

# 每个SNR点重复评测的次数（每次重新采样信道实现，取平均降低慢衰落评测噪声）。
# 500张验证图/batch_size=32 ≈ 16次独立信道采样，方差较大，尤其在低/中SNR区间
# 容易出现"PSNR不随SNR单调"的评测噪声（不是真实效应）。重复采样次数越多，
# 曲线越平滑，但评测耗时也线性增加。
N_EVAL_REPEATS = 5

# E9 不同带宽压缩比 k/n 实验超参
# k/n = num_positions / 768 （n=512*512*3=786432, 空间维固定32x32，
# k = num_positions*32*32）。k/n=2/3 对应 num_positions=512，就是已经
# 训好的 E0，不用重训；这里只需要新训另外三个。
E9_NUM_EPOCHS = 200
E9_KN_RATIOS = {
    "1/6": 128,
    "1/3": 256,
    "1/2": 384,
    # "2/3": 512,  # 已经是E0，不在这里重复训练
}

# 随机种子（保证可复现）
SEED = 42
