# Robot Marqeen micro:bit

Robot micro:bit yang dikendalikan gestur tangan memakai **transfer learning (MobileNetV2, ImageNet)**.

## Data

* Default: TFDS `rock\_paper\_scissors` (gambar tangan).
* Disarankan: foto gestur sendiri di `data/<kelas>/\*.jpg` (jalankan `python train.py --data\_dir data`).

## Cara pakai

```bash
pip install -r requirements.txt
python train.py                 # hasil: marqeen\_model.keras, labels.json
# flash microbit\_robot.py ke micro:bit
python control\_robot.py --port COM3
```

## Metode

1. Feature extraction: MobileNetV2 dibekukan, latih head Dense.
2. Fine-tuning: 30 layer terakhir dibuka, LR 1e-5.

## Hasil

Akurasi validasi: 94,4% (MobileNetV2, 5 epoch feature extraction + 5 epoch fine-tuning, dataset rock\_paper\_scissors)

