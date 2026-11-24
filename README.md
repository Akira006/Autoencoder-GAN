# Studi Komparatif Empiris Model Generatif

**Deep Learning – Generative Models**

Perbandingan enam model generatif, yaitu **Vanilla Autoencoder, Sparse Autoencoder, VAE, Vanilla GAN, DCGAN, dan DDPM**, pada dataset **Fashion-MNIST**. Repo ini juga berisi eksperimen yang membuktikan kenapa **MLP Classifier** tidak bisa dipakai untuk image generation.

## Struktur Repositori

```
.
├── notebooks/
│   ├── 01_autoencoders.ipynb       # Vanilla AE, Sparse AE, VAE + sweep λ, Figur 1, S1, S3
│   ├── 02_gans.ipynb               # Vanilla GAN (MLP), DCGAN
│   ├── 03_ddpm.ipynb               # DDPM (U-Net, T=1000), trajektori reverse diffusion
│   ├── 04_mlp_limitations.ipynb    # MLP Classifier: Tahap 1–5, Figur 3
│   └── 05_summary.ipynb            # Tabel komparasi (Bagian IV), Figur 2, 4, 5, S2
├── outputs/                        # Hasil run penuh (figur, tabel, log). Checkpoint tidak disertakan.
│   ├── figures/
│   ├── results/
│   └── logs/
├── requirements.txt
└── README.md
```

Tiap notebook bisa dijalankan sendiri: model yang dibutuhkan tapi belum ada akan dilatih di notebook itu. Definisi model, training loop, dan metrik ditulis sebagai fungsi/kelas terpisah di dalam tiap notebook.

## Cara Menjalankan (Kaggle)

1. Import notebook ke Kaggle (*File → Import Notebook*).
2. Settings: **Accelerator → GPU T4**, **Internet → On** (untuk download Fashion-MNIST dan bobot InceptionV3 untuk FID).
3. Jalankan `01`–`04` lewat *Save Version → Save & Run All (Commit)*. `02` dan `03` bisa dijalankan paralel.
4. *(Opsional, hemat waktu)* Tambahkan output `01` sebagai Input di `02`–`04` (*Add Input → Your Work*). MLP Classifier dan Vanilla AE akan dimuat dari situ, tidak dilatih ulang.
5. Di `05_summary`, tambahkan output `01`–`04` sebagai Input, lalu jalankan (cukup CPU).

Semua hasil tersimpan di `/kaggle/working/outputs/`. `05_summary` juga membuat `outputs_bundle.zip`.

> `QUICK_TEST = True` di cell pengaturan hanya untuk cek cepat (epoch sangat kecil, output ke `trial/`). Hasil di laporan memakai `QUICK_TEST = False`.

### Menjalankan secara lokal

```bash
pip install -r requirements.txt
jupyter notebook notebooks/
```

Di luar Kaggle, output tersimpan ke `./kaggle_working/outputs/` dan Input dicari di `./kaggle_input/`.

## Setup Eksperimen

| Komponen | Nilai |
|---|---|
| Dataset | Fashion-MNIST, 70.000 citra (60k train + 10k test digabung), split **80/10/10** |
| Random seed | 42 (split data, inisialisasi bobot, urutan shuffle di-reset sebelum tiap training) |
| Batch size | 128 |
| Backbone AE / Sparse AE / VAE | MLP `784 → 256 → 64 → 256 → 784`, identik dengan ekstraktor fitur MLP Classifier |
| Sparse AE | Laten ReLU + penalti L1, λ = 1e-4 (sweep λ ∈ {0, 1e-5, 1e-4, 1e-3}) |
| VAE | BCE + KL, β = 1 |
| Vanilla GAN | G: `100 → 256 → 512 → 1024 → 784`, D: `784 → 512 → 256 → 1`, non-saturating loss |
| DCGAN | ConvTranspose/Conv berstride, BatchNorm, init N(0, 0.02) |
| DDPM | U-Net (28→14→7, self-attention di 7×7), T = 1000, β linear 1e-4 → 0.02, EMA 0.999 |
| MLP Classifier | `784 → 256 → 64 → 10`, Cross-Entropy, target test accuracy ≥ 88% |
| Optimizer | Adam (AE/VAE/MLP lr 1e-3; GAN lr 2e-4, β₁ = 0.5; DDPM lr 2e-4) |

## Metrik

- **Rekonstruksi** (AE family, test set): MSE, PSNR, SSIM per citra, lalu dirata-rata
- **Kualitas sampel**: FID (InceptionV3; citra grayscale 28×28 diulang 3 channel, jumlah sampel = ukuran test set). Nilainya hanya untuk perbandingan antarmodel di setup ini.
- **Mode coverage**: entropi distribusi kelas sampel menurut MLP Classifier
- **Efisiensi**: waktu training per epoch, latency inference per batch 64

## Hasil

_Diisi setelah run penuh: salin `outputs/results/metrics_table.md` ke sini._

## Keterangan

Tugas mata kuliah Deep Learning (D6421), Universitas Bina Nusantara.
