# Studi Komparatif Empiris Model Generatif

**Deep Learning (D6421) - Sesi 07: Generative Models | Tugas Take-Home**

| | |
|---|---|
| Nama | Akira Agha Nugroho |
| NIM | 2802403555 |
| Kelas | Deep Learning LB01 |

Perbandingan enam model generatif, yaitu **Vanilla Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, dan DDPM**, pada dataset **Fashion-MNIST**. Repo ini juga berisi eksperimen yang membuktikan kenapa **MLP Classifier** tidak bisa dipakai untuk image generation (Tahap 1-5).

## Struktur Repositori

```
.
├── src/
│   ├── colab_01_autoencoders.ipynb     # Vanilla AE, Sparse AE, VAE + sweep λ, Figur 1, S1, S3
│   ├── colab_02_gans.ipynb             # Vanilla GAN (MLP), DCGAN
│   ├── colab_04_mlp_limitations.ipynb  # MLP Classifier Tahap 1-5, Figur 3 (versi Colab)
│   ├── colab_05_summary.ipynb          # Tabel komparasi (Bagian IV), Figur 2, 4, 5, S2
│   └── kaggle env/
│       ├── kaggle_01_autoencoders.ipynb
│       ├── kaggle_02_gans.ipynb
│       ├── kaggle_03_ddpm.ipynb        # DDPM (U-Net, T=1000), training + evaluasi resumable
│       ├── kaggle_04_mlp_limitations.ipynb
│       └── kaggle_05_summary.ipynb
├── final_results/                      # Hasil run final (checkpoint .pt tidak disertakan)
│   ├── figures/                        # Figur 1-5, S1-S3, progress/ (snapshot GAN per epoch)
│   ├── metrics/                        # JSON per model, tabel .md/.csv, sweep λ, interpolasi
│   ├── logs/                           # Riwayat loss training tiap model, info split data
│   └── README_SUMBER.txt               # Asal tiap file (notebook dan platform)
├── .gitignore
├── requirements.txt
└── README.md
```

Tiap notebook bisa dijalankan sendiri. Model yang dibutuhkan tapi checkpoint-nya belum ada akan dilatih di notebook itu. Definisi model, training loop, dan metrik ditulis sebagai fungsi/kelas terpisah di dalam tiap notebook.

## Platform Run Final

| Notebook | Platform | GPU |
|---|---|---|
| 01 Autoencoders | Google Colab | T4 |
| 02 GANs | Google Colab | T4 |
| 03 DDPM | Kaggle | T4 |
| 04 MLP Limitations | Kaggle | T4 |
| 05 Summary | Google Colab | T4 |

Versi Colab untuk notebook 03 tidak dibuat karena training DDPM (40 epoch, sekitar 80 detik per epoch) dan pembangkitan 7.000 sampel (sekitar 63 menit) lebih stabil dijalankan di Kaggle. Hasil 03 dan 04 dari Kaggle digabung ke `final_results/` sebelum notebook 05 dijalankan.

## Cara Menjalankan

### Kaggle (`src/kaggle env/`)

1. Import notebook ke Kaggle (*File → Import Notebook*).
2. Settings: **Accelerator → GPU T4**, **Internet → On** (untuk download Fashion-MNIST dan bobot InceptionV3 untuk FID).
3. Jalankan `kaggle_01` sampai `kaggle_04`. Semua hasil tersimpan di `/kaggle/working/final_results/` dan dikemas menjadi zip di akhir notebook.
4. *(Opsional, hemat waktu)* Tambahkan output notebook sebelumnya sebagai Input (*Add Input → Your Work*). Checkpoint dicari otomatis di `/kaggle/input/**/final_results/`, jadi model yang sudah ada dimuat, tidak dilatih ulang. Untuk memaksa training ulang, isi daftar `RETRAIN` di cell pengaturan.
5. `kaggle_03_ddpm` menyimpan checkpoint parsial tiap 2 epoch dan membangkitkan sampel evaluasi dalam 14 bagian × 500 citra (`samples/ddpm_parts/`). Kalau sesi terputus, jalankan ulang cell yang sama dan proses akan melanjutkan dari bagian terakhir.
6. Jalankan `kaggle_05_summary` dengan output `01`-`04` sebagai Input (cukup CPU).

### Google Colab (`src/colab_*`)

1. Buka notebook di Colab, pilih **Runtime → Change runtime type → T4 GPU**.
2. Hasil disimpan ke Google Drive di `MyDrive/Binus/Semester_5/DeepLearning/GSLC2/final_results/`. Ubah variabel path di cell pengaturan kalau struktur Drive berbeda.
3. Jalankan `colab_01`, `colab_02`, dan `colab_04`.
4. Untuk DDPM, salin `metrics/ddpm.json`, `metrics/ddpm_latency.json`, `logs/ddpm_history.json`, dan `samples/ddpm_trajectory.pt` dari output `kaggle_03` ke folder `final_results/` di Drive.
5. Jalankan `colab_05_summary`. Notebook ini mengecek apakah file DDPM sudah tersedia sebelum membuat tabel dan figur.

### Lokal

```bash
pip install -r requirements.txt
jupyter notebook src/
```

Versi Kaggle yang dijalankan di luar Kaggle menyimpan output ke `./kaggle_working/final_results/` dan mencari Input di `./kaggle_input/`.

## Setup Eksperimen

| Komponen | Nilai |
|---|---|
| Dataset | Fashion-MNIST, 70.000 citra (60k train + 10k test digabung), split **80/10/10** (56.000 / 7.000 / 7.000) |
| Random seed | 42 (split data, inisialisasi bobot, generator DataLoader di-reset sebelum tiap training) |
| Batch size | 128 |
| Backbone AE / Sparse AE / VAE | MLP `784 → 256 → 64 → 256 → 784`, identik dengan ekstraktor fitur MLP Classifier |
| Sparse AE | Laten ReLU + penalti L1, λ = 1e-4 (sweep λ ∈ {0, 1e-5, 1e-4, 1e-3}, 20 epoch) |
| VAE | BCE + KL, β = 1 |
| Vanilla GAN | G: `100 → 256 → 512 → 1024 → 784`, D: `784 → 512 → 256 → 1`, non-saturating loss |
| DCGAN | ConvTranspose/Conv berstride, BatchNorm, init N(0, 0.02) |
| DDPM | U-Net 3 level (64/128/128 kanal, self-attention di 7×7), T = 1000, β linear 1e-4 → 0.02, EMA 0.999, mixed precision |
| MLP Classifier | `784 → 256 → 64 → 10`, dropout 0.2, Cross-Entropy, target test accuracy ≥ 88% |
| Frozen-latent decoder | `64 → 256 → 784`, dilatih dari z_MLP dengan bobot classifier dibekukan |
| Optimizer | Adam (AE/VAE/MLP lr 1e-3; GAN lr 2e-4, β₁ = 0.5; DDPM lr 2e-4) |
| Epoch | AE/Sparse AE/VAE 30, Vanilla GAN 60, DCGAN 30, DDPM 40, MLP Classifier 25, decoder 30 |

## Metrik

- **Rekonstruksi** (keluarga AE, test set): MSE, PSNR, SSIM per citra, lalu dirata-rata.
- **Kualitas sampel**: FID (torchmetrics + InceptionV3; citra grayscale 28×28 diulang 3 channel, 7.000 sampel per model, diproses per batch 100 agar tidak out-of-memory). Nilainya hanya untuk perbandingan antarmodel di setup ini.
- **Mode coverage**: entropi ternormalisasi distribusi kelas sampel menurut MLP Classifier.
- **Efisiensi**: waktu training per epoch, latency inference per batch 64 (dengan `torch.cuda.synchronize`).
- **Analisis tambahan**: Hoyer sparsity dan jumlah unit aktif (Sparse AE), rasio variansi intra-kelas ρ (z_MLP vs z_AE), jarak titik interpolasi ke data terdekat.

## Hasil

### Komparasi enam model (test set, 7.000 citra, GPU T4)

| Model | Train time / epoch (s) | Latency (ms / batch 64) | MSE | PSNR (dB) | SSIM | FID | Coverage |
|---|---|---|---|---|---|---|---|
| Vanilla AE | 1.37 | 0.20ᵃ | 0.0065 | 23.00 | 0.866 | N/A | N/A |
| Sparse AE | 1.45 | 0.22ᵃ | 0.0098 | 21.09 | 0.809 | N/A | N/A |
| VAE | 1.79 | 0.15 | 0.0132 | 19.64 | 0.767 | 86.64 | 0.985 |
| Vanilla GAN | 3.05 | 0.55 | N/A | N/A | N/A | 120.26 | 0.990 |
| DCGAN | 15.06 | 3.71 | N/A | N/A | N/A | 20.71 | 0.995 |
| DDPM | 79.72 | 32,981 | N/A | N/A | N/A | **9.22** | 0.990 |

ᵃ Latency rekonstruksi (encode + decode). Model lain: latency generate dari noise.

### Batasan MLP Classifier (Tahap 1-5)

| Rekonstruksi dari | MSE | PSNR (dB) | SSIM |
|---|---|---|---|
| Proyeksi transpos Wᵀ (Tahap 3) | 0.2532 | 6.05 | 0.078 |
| Frozen z_MLP + decoder (Tahap 4) | 0.0200 | 17.82 | 0.692 |
| Vanilla AE, z_AE (pembanding) | 0.0065 | 23.00 | 0.866 |

Akurasi test MLP Classifier 89.0%. Rasio variansi intra-kelas ρ = 0.31 (z_MLP) vs 0.58 (z_AE). Sampling Gaussian di z_MLP menghasilkan 27.7% komponen negatif, padahal laten asli keluaran ReLU tidak pernah negatif.

### Temuan utama

- DDPM memberi kualitas terbaik (FID 9.22) dengan biaya latency sekitar 8.900× DCGAN, sesuai Generative Trilemma.
- Vanilla GAN stabil di dekat ekuilibrium (D(x) = 0.535) tanpa mode collapse level kelas, tetapi FID-nya paling buruk. Kestabilan loss tidak menjamin kualitas sampel.
- Titik tengah interpolasi Vanilla AE dan Sparse AE berupa dua objek yang ditumpuk, sedangkan VAE menghasilkan objek yang masuk akal.

Tabel lengkap: `final_results/metrics/metrics_table.md`. Sweep λ: `final_results/metrics/sparse_lambda_sweep.md`. Figur: `final_results/figures/`.

## Catatan Reproduksibilitas

- MLP Classifier dilatih dua kali: di Colab (akurasi 89.59%, dipakai untuk coverage VAE, Vanilla GAN, DCGAN) dan di Kaggle (89.03%, dipakai untuk coverage DDPM dan eksperimen Tahap 1-5). Riwayat keduanya ada di `logs/mlp_classifier_history.json` dan `logs/mlp_classifier_history_kaggle.json`.
- Hasil GPU tidak sepenuhnya deterministik antarplatform, jadi angka dapat sedikit berbeda saat dijalankan ulang meski seed sama.
- Checkpoint model (`.pt`) tidak disertakan karena ukurannya besar. Semua model dapat dilatih ulang dengan menjalankan notebook.
