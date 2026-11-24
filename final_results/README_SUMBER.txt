Asal file di final_results/ (digabung untuk 05_summary)

- 01 Autoencoders  : Google Colab, T4 (AE dilatih ulang di T4)
- 02 GANs          : Google Colab, T4
- 03 DDPM          : Kaggle, T4 (training resume dari epoch 25, evaluasi resumable)
- 04 MLP limitations: Kaggle, T4 (versi Colab tidak dipakai)

Classifier:
- logs/mlp_classifier_history.json        : classifier Colab (test acc 0,8959), dipakai coverage VAE, Vanilla GAN, DCGAN, interpolasi GAN
- logs/mlp_classifier_history_kaggle.json : classifier Kaggle (test acc 0,8903), dipakai Bagian III (04) dan coverage DDPM

Vanilla AE:
- logs/vanilla_ae_history.json            : dari 01 (Colab)
- logs/vanilla_ae_history_kaggle04.json   : AE pembanding yang dilatih ulang di 04 (Kaggle); hasil identik dengan 01

samples/*.pt hanya berisi 256 sampel pertama (FID dihitung dari 7000 sampel saat run).
