# Ablation: Pruning vs. Quantization Contribution (near SNR=24dB, 64QAM)

| Condition | PSNR (dB) | SSIM |
|---|---|---|
| (a) Unpruned + analog | 28.13 | 0.8432 |
| (b) Unpruned + 64QAM | 28.25 | 0.8442 |
| (c) Pruned(γ=0.7) + analog | 28.16 | 0.8383 |
| (d) Pruned(γ=0.7) + 64QAM | 28.14 | 0.8379 |

**Pruning-only cost** = (a) − (c)  
**Quantization-only cost** = (a) − (b)  
**Combined cost** = (a) − (d), compared against (pruning-only + quantization-only) to check whether the two effects are roughly additive or interact.

- Pruning-only cost: -0.02 dB
- Quantization-only cost: -0.12 dB
- Combined (measured) cost: -0.01 dB
- Additive prediction (pruning + quantization): -0.14 dB
- Interaction term (combined − additive): +0.14 dB