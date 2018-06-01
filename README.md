# detektif-tomat
## Requirement
* OS >= Ubuntu 16.04.2 LTS
* Python >= 2.7.12
## Python Library
* Numpy >= 1.12.1
* OpenCV ~ 2.4.9.1
* PyYaml >= 3.11

## How to use
* Simpan semua kelas dalam dataset di dalam direktori `./dataset`
* Tahap praproses data jalankan `python preprocess.py` pada terminal. Hasilnya akan terlihat pada direktori `./dataset/preprocessed/*`
* Tahap ekstraksi ciri jalankan `python feature.py` pada terminal. Hasilnya akan terlihat pada direktori `./feature/feature.csv`
* Setting konfigurasi parameter Knn pada file system_config.json. Ubah settingan `classifier["parameters"]["k"]` sesuai keinginan. **(Note: (k%2 == 1) && (k >= 3))**.
* Tahap pelatihan dan test data jalankan `python classify.py` pada terminal. Hasilnya akan terlihat pada direktori `./model/classifier.p`
* Tahap identifikasi berdasarkan input dari user. Masukkan semua citra yang ingin di identifikasi pada direktori `./input.` Jalankan `python identify.py` pada terminal. Hasilnya akan terlihat pada terminal.