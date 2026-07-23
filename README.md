# Lightweight Low-SNR-Robust Semantic Communication System for Autonomous Driving

This repository contains the supplementary experiment code used to address reviewer comments during the revision of *"Lightweight Low-SNR-Robust Semantic Communication System for Autonomous Driving."* It includes a from-scratch, self-consistent reproduction of the paper's core deep JSCC pipeline (structured pruning + train-deploy separation quantization), a reproduction of the JCM (joint coding-modulation) baseline from [Bo et al., WCSP 2022 / IEEE TCOM 2024], and several supporting studies (bandwidth compression ratio, ablation, hardware latency, capacity bound).

This code is **independent of the original paper's codebase** and does not depend on it. It was built to guarantee internal consistency across all new experiments, after identifying several inconsistencies in the original codebase's channel model implementations.

---

## Contents

- [Overview](#overview)
- [Repository structure](#repository-structure)
- [Setup](#setup)
- [Experiment index](#experiment-index)
- [Reproducing paper figures/tables](#reproducing-paper-figurestables)
- [Known limitations](#known-limitations)
- [Citation](#citation)

---

## Overview

The original paper proposes a lightweight, low-SNR-robust deep JSCC system for V2V image transmission, combining:
1. **Structured pruning** (BN-scaling-factor + L1 regularization, following Network Slimming) to reduce model size for on-board deployment.
2. **Uniform quantization + M-QAM modulation** with a **train-deploy separation** strategy to make the system compatible with digital communication systems.

During revision, we implemented from scratch:
- A unified, internally-consistent slow-Rayleigh-fading channel model (`common/channel.py`), matching the manuscript's stated specification ($h \sim \mathcal{CN}(0,1)$).
- A reproduction of the proposed pruning + quantization pipeline (E0, E1, E5, E6, E7, E9, E10).
- A reproduction of an end-to-end differentiable JCM baseline (E2, E6), trained via Gumbel-Softmax reparameterization, to directly compare against the paper's train-deploy separation strategy.
- A capacity-achieving separation bound (E4).
- An ablation isolating the individual and joint contributions of pruning and quantization (E7 + `make_ablation_table.py`).
- A systematic study of the bandwidth compression ratio $k/n$ (E9).
- Hardware inference-latency measurements (E8).
- A partial reproduction of the classical BPG+LDPC+QAM baseline via `pyldpc` (E11, E13) — used only for a single qualitative example image, not for the main quantitative curves (see [Known limitations](#known-limitations)).

---

## Repository structure

```
.
├── README.md
├── common/
│   ├── config.py              # All paths & hyperparameters — edit this first
│   ├── channel.py              # Unified slow Rayleigh fading + AWGN channel model
│   ├── qam.py                  # M-QAM constellation, uniform quantization, modulation/demodulation
│   ├── backbone.py             # Fixed-width Encoder/Decoder (unpruned)
│   ├── backbone_prunable.py    # Configurable-width Encoder/Decoder (for pruning experiments)
│   └── dataset.py              # Flat-folder Cityscapes image loader
│
├── E0_pretrain_analog.py               # Train base analog Deep JSCC model (γ=0)
├── check_e0_analog_performance.py      # Sanity-check E0 against Table 2's γ=0 row
│
├── E1_eval_train_deploy_quantization.py # Train-deploy separation quantization eval (γ=0, 4/16/64/256-QAM)
│
├── E2_jcm_model.py                      # JCM model definition (Gumbel-Softmax quantization head)
├── E2_train_jcm.py                      # JCM training, γ=0
│
├── E4_capacity_bound.py                 # Capacity-achieving separation bound (Monte Carlo + BPG)
│
├── E5_prune_gamma.py                    # Structured pruning (Algorithm 1: sparse train → prune → fine-tune)
├── check_e5_analog_performance.py       # Sanity-check pruned model against Table 2
│
├── E6_train_jcm_gamma07.py              # JCM training on pruned (γ=0.7) backbone
│
├── E7_eval_analog_full_curve.py         # Full-SNR-curve analog eval, for E0 or any pruned E5 checkpoint
├── E7_eval_pruned_quantization.py       # Train-deploy separation quantization eval, on pruned backbone
│
├── E8_measure_latency.py                # Encoder inference latency across pruning ratios (CPU & GPU)
│
├── E9_train_bandwidth_ratio.py          # Train independent models at k/n = 1/6, 1/3, 1/2
├── E9_eval_bandwidth_ratio.py           # Evaluate all four k/n ratios (incl. reusing E0 for k/n=2/3)
│
├── E10_prune_multi_gamma.py             # Batch pruning + eval for γ = 0.2, 0.5, 0.9
│
├── E11_bpg_ldpc_qam.py                  # BPG+LDPC+4QAM full-curve baseline (pyldpc; NOT used in final paper)
├── E13_single_bpg_ldpc_example.py       # Single-image BPG+LDPC+QAM reconstruction (for Figure 5 only)
│
├── E12_generate_fig5_examples.py        # Generate the 4 reproduced columns of Figure 5's example images
│
├── plot_supplementary_e1_vs_e2.py       # γ=0 train-deploy-separation vs JCM supplementary plot
├── make_ablation_table.py               # Pruning-vs-quantization ablation summary table
├── make_summary_table.py                # E1-vs-E2 numeric summary table
│
└── results/                             # All checkpoints, JSON results, and generated figures (git-ignored)
```

---

## Setup

```bash
# Core dependencies
pip install torch torchvision numpy scikit-image matplotlib

# Only needed for E11 / E13 (BPG+LDPC baseline)
pip install pyldpc --no-build-isolation
```

Edit `common/config.py`:

```python
DATASET_PATH = r"./dataset/cityscapes"   # must contain train/ and val/ subfolders (flat .jpg/.png files)
```

For E11 / E13, you additionally need a locally built `bpgenc` / `bpgdec` ([bellard.org/bpg](https://bellard.org/bpg/)) and must set `BPGENC_CMD` / `BPGDEC_CMD` at the top of those scripts to the correct executable paths.

---

## Experiment index

| ID | Script(s) | What it does | Needs training? |
|---|---|---|---|
| E0 | `E0_pretrain_analog.py` | Base analog Deep JSCC, γ=0 | ✅ |
| E1 | `E1_eval_train_deploy_quantization.py` | Train-deploy separation quantization, γ=0, 4 modulation orders | ❌ (inference only) |
| E2 | `E2_jcm_model.py`, `E2_train_jcm.py` | JCM baseline, γ=0 | ✅ |
| E4 | `E4_capacity_bound.py` | Capacity-achieving separation bound | ❌ |
| E5 | `E5_prune_gamma.py` | Structured pruning for a single γ | ✅ |
| E6 | `E6_train_jcm_gamma07.py` | JCM baseline on pruned (γ=0.7) backbone | ✅ |
| E7 | `E7_eval_analog_full_curve.py`, `E7_eval_pruned_quantization.py` | Full-SNR-curve evaluation (analog / quantized) | ❌ |
| E8 | `E8_measure_latency.py` | Encoder inference latency, CPU & GPU | ❌ |
| E9 | `E9_train_bandwidth_ratio.py`, `E9_eval_bandwidth_ratio.py` | Systematic $k/n$ study | ✅ |
| E10 | `E10_prune_multi_gamma.py` | Batch pruning for the remaining γ values | ✅ |
| E11 | `E11_bpg_ldpc_qam.py` | Full BPG+LDPC+4QAM curves (pyldpc) | ❌ (not trained; LDPC decoding) |
| E12 | `E12_generate_fig5_examples.py` | Reconstructed-image examples (Original/analog/proposed/JCM) | ❌ |
| E13 | `E13_single_bpg_ldpc_example.py` | Single BPG+LDPC+QAM example image | ❌ |

Every script has a module docstring at the top with a more detailed explanation and its exact run command.

---

## Reproducing paper figures/tables

| Figure/Table | Requires |
|---|---|
| Table 2 (parameters / MACs / PSNR / SSIM per γ) | E0, E5 (γ=0.7), E10 (γ=0.2/0.5/0.9) |
| Table 3 (JCM parameter count) | Architecture-only, no training needed — see `E2_jcm_model.py` self-test |
| Table 4 (inference latency) | E8 |
| Figure 3 (pruning ratio comparison) | E0, E5, E10, E4 (capacity bound); BPG+LDPC curves are historical data, not reproduced |
| Figure 4 (γ=0.7 modulation comparison + JCM) | E7 (analog + quantization, γ=0.7), E6 (JCM, γ=0.7) |
| Figure 5 (reconstructed image examples) | E12 (4 columns) + E13 (BPG+LDPC+QAM column) |
| Figure 6 (pruning vs. direct bandwidth compression) | E10 (γ=0.5), E9 (k/n=1/3) |
| Figure 7 (pruning vs. quantization ablation) | E7 (analog + quantization, both γ=0 and γ=0.7) + `make_ablation_table.py` |
| Figure 8 (bandwidth compression ratio study) | E9 (all four k/n ratios) |

---

## Known limitations

We disclose these openly rather than presenting the reproduction as a perfect match to the original manuscript:

1. **Absolute reproduction gap.** Our independently retrained backbone converges to a PSNR ceiling a few dB below the values originally reported (e.g., γ=0 analog: 28.17 dB reproduced vs. 31.42 dB reported). We were unable to fully close this gap and attribute it to training configuration details not fully specified in the original text (e.g., a possible pretraining stage prior to Algorithm 1). All new experiments are internally self-consistent (trained and evaluated with the same code/channel model), so *relative* comparisons (e.g., JCM vs. proposed, pruned vs. unpruned) remain valid even though *absolute* values should not be directly compared against the original paper's published numbers.
2. **BPG+LDPC+QAM baseline is not fully reproduced.** `E11_bpg_ldpc_qam.py` implements a working BPG+LDPC+4QAM pipeline (verified to reproduce the classic cliff effect), but it is not used for the paper's main curves — the original historical data is retained instead. It is only used to generate a single qualitative example image for Figure 5 (`E13_single_bpg_ldpc_example.py`). 16-QAM is not implemented (would require multi-level soft-demodulation LLR computation beyond the simple I/Q-as-two-BPSK-streams approach used here).
3. **Original codebase's channel model was inconsistent.** We found three different Rayleigh fading variance conventions across the original code (`sqrt(1/2)`, `sqrt(1/8)`, and an unnormalized version), which we could not resolve. All code in this repository uses a single, unified convention matching the manuscript's stated specification ($h \sim \mathcal{CN}(0,1)$), defined once in `common/channel.py`.

---

## Citation

If you use this code, please cite the original paper (details to be added upon publication).

```bibtex
@article{TODO,
  title   = {Lightweight Low-SNR-Robust Semantic Communication System for Autonomous Driving},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO}
}
```

The JCM baseline reproduced here follows:
- Y. Bo, Y. Duan, S. Shao, M. Tao, "Learning based joint coding-modulation for digital semantic communication systems," WCSP 2022.
- Y. Bo, Y. Duan, et al., "Joint coding-modulation for digital semantic communications via variational autoencoder," IEEE Trans. Commun., 2024.
