| Arsitektur | Dim Laten | Params | Train Time/Epoch (s) | Latency (ms/batch64) | Latency mode | Test MSE | Test PSNR (dB) | Test SSIM | FID | Class entropy (coverage) | Loss CV tail (stabilitas) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Vanilla AE | 64 | 435536 | 1.37 | 0.2 | reconstruct | 0.0065 | 23 | 0.8662 | N/A | N/A | 0.0148 |
| Sparse AE | 64 (ReLU+L1) | 435536 | 1.45 | 0.22 | reconstruct | 0.00976 | 21.09 | 0.8085 | N/A | N/A | 0.0109 |
| VAE | 64 | 451984 | 1.79 | 0.15 | generate | 0.01322 | 19.64 | 0.7671 | 86.64 | 0.985 | 0.0008 |
| Vanilla GAN | 100 (prior noise) | 1486352 | 3.05 | 0.55 | generate | N/A | N/A | N/A | 120.3 | 0.99 | 0.0042 |
| DCGAN | 100 (prior noise) | 1781504 | 15.06 | 3.71 | generate | N/A | N/A | N/A | 20.71 | 0.995 | 0.0054 |
| DDPM | 1x28x28 (dimensi citra) | 6469889 | 79.72 | 3.298e+04 | generate | N/A | N/A | N/A | 9.22 | 0.99 | 0.0057 |
