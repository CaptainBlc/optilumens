# Egitimli Pixel Iyilestirme: Egitim ve Kullanim

Pixel iyilestirme **egitimli bir yapay zeka (EnhanceNet)** ile yapilir. Guncel, performansli ve ileride farkli veri setleriyle yeniden egitilebilir.

---

## 1. Mimari

- **EnhanceNet**: PyTorch tabanli, U-Net benzeri encoder-decoder ag. Girdi ve cikti ayni boyutta (tam konvolusyonel).
- **Egitim**: Eslenmis veri (dusuk kalite -> yuksek kalite) uzerinde L1 loss ile ogrenir.
- **Inference**: Tek goruntu veya toplu; CPU/GPU destekli. Model dosyasi yoksa otomatik basit yedek (fallback) kullanilir.

---

## 2. Veri Yapisi (Egitim Icin)

Eslenmis ciftler gerekir: **girdi** (iyilestirilmesi gereken) ve **hedef** (iyi kalite).

```
data/
  train/
    input/   <- dusuk isik, bulanik veya gurultulu goruntuler
    target/  <- ayni sahnenin iyi kalite hali (ayni dosya isimleri)
```

- Ayni dosya adi: `input/001.png` ve `target/001.png` eslesmeli.
- Desteklenen formatlar: jpg, jpeg, png, bmp.
- Veri seti tek tip olmak zorunda degil: farkli kaynaklardan (telefon, CCTV, drone vb.) toplanan ciftler kullanilabilir; ag cesitlilige uyum saglar.

Ornek veri setleri (indirip input/target olarak kullanilabilir):
- **LOL** (low-light): https://github.com/flyywh/CVPR-2020-Semi-Low-Light
- **SID** (See in the Dark): https://github.com/cchen156/Learning-to-See-in-the-Dark
- Kendi cekimleriniz: ayni sahneyi dusuk ve normal isikta cekip eslestirin.

---

## 3. Egitim

```bash
cd src
python train.py --input_dir ../data/train/input --target_dir ../data/train/target --epochs 50 --batch 4 --size 256
```

- **--size 256**: Egitimde goruntuler 256x256 kullanilir (hizli, az bellek). Tam cozunurluk icin `--size 0` (dikkat: bellek).
- **--epochs**: 50 varsayilan; veri miktarina gore artirilabilir.
- **--checkpoints**: Model kayit klasoru (varsayilan: `../checkpoints`).
- Cikti: `checkpoints/enhance_net_latest.pt` (her 10 epoch + son epoch).

GPU yoksa CPU ile de egitim yapilir; daha yavas olur.

---

## 4. Inference (Calistirma)

Model egitildikten sonra:

```bash
cd src
python main.py
```

- `main.py`: `test_mix/` icindeki goruntuleri okur, `pixel_enhance()` cagirir, sonuclari `outputs/` a yazar.
- `pixel_enhance()`: Oncelikle `checkpoints/enhance_net_latest.pt` dosyasini yukler; varsa **egitimli model** ile iyilestirir. Dosya yoksa veya hata olursa **yedek** (hafif klasik iyilestirme) uygulanir.

Yani: Egitim yapmadan da `main.py` calisir (yedek ile); egitim yapildiktan sonra ayni kod AI ile calisir.

---

## 5. Guncel ve Gelecek Uyumluluk

- **PyTorch**: Guncel egitim ve model formatlari; ileride ONNX export veya baska framework’e aktarim mumkun.
- **Hafif ag**: Edge cihaz ve telefon tarafinda calistirilabilir; gerekirse model kucultme (pruning, quantize) veya ONNX ile hizlandirma eklenebilir.
- **Farkli veri**: Ag tek tip dataset’e bagli degil; yeni eslenmis veri ekleyip yeniden egiterek guncel ve gelecekteki kullanim senaryolarina uyarlanabilir.

Bu dokuman, "istatistik degil egitimli yapay zeka" ve "yenilikci, performans veren, guncel ve gelecege uyumlu" iddiasini kod tarafinda destekler.
