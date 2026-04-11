# Proje Tasarım Özeti

Bu dosya, **High Level Design**, **Project Specifications** ve **Analysis** raporlarından çıkarılan mimari ile mevcut kodun birebir eşleşmesini gösterir.

---

## 0. Güncel Kapsam ve Kurallar (Ekip Kararları)

- **Bu repoda sadece Pixel bölümü**: 3 kişilik ekip; pixel sorumluluğu bu tarafta.
- **Diğer modüller**: LabelBasedProcessor (Semantic) ve GeneralEnhancer (Global) ekip arkadaşlarından gelecek; entegrasyona hazır arayüzler mevcut.
- **Sistem vizyonu**: Neyin nerede, ne türde iyileştirildiğini kullanıcıya gösteren, cihazdan bağımsız bir sistem.
- **Explainable Processing**: Her adım loglanır, metriklerle ölçülür, heatmap ile görselleştirilir.

---

## 1. Doküman ↔ Kod Eşleşme Tablosu

| Doküman Referansı | Sınıf/Metot Adı | Dosya | Durum |
|---|---|---|---|
| Analysis 3.5.3 #3: ImageProcessor | `ImageProcessor` | `pixel_pipeline.py` | Aktif |
| Analysis 3.5.3 #3: analyzeImage() | `ImageProcessor.analyzeImage()` | `pixel_pipeline.py` | Aktif |
| Analysis 3.5.3 #3: enhanceImage() | `ImageProcessor.enhanceImage()` | `pixel_pipeline.py` | Aktif |
| Analysis 3.5.3 #5: PixelBasedProcessor | `PixelBasedProcessor` | `pixel_enhance.py` | Aktif |
| Analysis 3.5.3 #5: reduceNoise() | `PixelBasedProcessor.reduceNoise()` | `pixel_enhance.py` | Aktif |
| Analysis 3.5.3 #5: sharpenImage() | `PixelBasedProcessor.sharpenImage()` | `pixel_enhance.py` | Aktif |
| HLD 3.2.3: Edge-Preserving Filtering | `PixelBasedProcessor.edgePreservingFilter()` | `pixel_enhance.py` | Aktif |
| HLD 3.2.1: Image Profiling | `ImageProfiler` | `profiler.py` | Aktif |
| Analysis 3.5.3 #4: LabelBasedProcessor | — | — | Bekliyor (ekip arkadaşı) |
| Analysis 3.5.3 #6: GeneralEnhancer | — | — | Bekliyor (ekip arkadaşı) |

---

## 2. Mimari: İşlem Akışı

Raporlara göre işleme sırası (Analysis Report 3.5.4):

```
User → System → ImageProcessor
                    │
                    ├── 1. analyzeImage()        → ImageProfiler.profile()
                    │      brightness, contrast, blur, noise, edge_density, skin
                    │
                    ├── 2. LabelBasedProcessor   → [BEKLIYOR: ekip arkadaşı]
                    │      detectObjects(), enhanceLabeledRegions()
                    │      Şu an: otomatik merkez-ROI maskesi (test amaçlı)
                    │
                    ├── 3. PixelBasedProcessor    → [AKTİF: bu repo]
                    │      ├── reduceNoise()             (HLD 3.2.3)
                    │      ├── sharpenImage()             (HLD 3.2.3 + Specs 1.1)
                    │      │   ├── Unsharp Mask (Y kanal)
                    │      │   └── CLAHE (L kanal, detail restoration)
                    │      └── edgePreservingFilter()     (HLD 3.2.3)
                    │
                    ├── 4. GeneralEnhancer       → [BEKLIYOR: ekip arkadaşı]
                    │      adjustBrightness(), improveContrast(), optimizeColors()
                    │
                    └── 5. Quality Metrics + Difference Map
                           PSNR, SSIM, Entropy, Colorfulness, Heatmap
```

---

## 3. PixelBasedProcessor Detay

### reduceNoise() — "Adaptive Noise Suppression" (HLD 3.2.3)
- **Yöntem**: `cv2.fastNlMeansDenoisingColored`, profildeki `noise_level`'a göre güç ayarı
- **ROI davranışı**: ROI'de hafif (detay koruma), arka planda güçlü (gürültü temizleme)
- **Karar**: `noise_level < 5.0` ise atlanır (explainable: logda SKIPPED yazılır)

### sharpenImage() — "Controlled Sharpening" + "Detail Restoration" (HLD 3.2.3, Specs 1.1)
- **Alt adım 1**: Y kanalında Unsharp Mask; `blur_score`'a adaptif sigma ve güç
- **Alt adım 2**: L kanalında CLAHE; ROI'de yerel kontrast artırma (detayları öne çıkarma)
- **Her iki adım** sadece mask yüksek olan bölgelerde uygulanır

### edgePreservingFilter() — "Edge-Preserving Filtering" (HLD 3.2.3)
- **Yöntem**: `cv2.bilateralFilter(d=9, sigmaColor=75, sigmaSpace=75)`
- **ROI davranışı**: Sadece arka planda (mask düşük) uygulanır; ROI dokunulmaz
- **Amaç**: "Smooth out flat areas but lock down the edges" (HLD, birebir)

---

## 4. Kalite Metrikleri ve Açıklanabilirlik

Her işlenen görüntü için:

| Metrik | Açıklama |
|---|---|
| PSNR (dB) | Orijinal-enhanced arası bozulma ölçümü |
| SSIM | Yapısal benzerlik (0..1) |
| Entropy | Bilgi içeriği (orijinal vs enhanced) |
| Colorfulness | Renk canlılığı (Hasler & Süsstrunk) |
| Difference Map | Nerede ne değiştiğini gösteren renk kodlu heatmap |

Log çıktısı her adımı açıklar (APPLIED/SKIPPED + parametreler).

---

## 5. Dosya Yapısı

```
src/
├── pixel_pipeline.py     # ImageProcessor (orchestrator)
├── pixel_enhance.py      # PixelBasedProcessor (reduceNoise, sharpenImage, edgePreservingFilter)
├── profiler.py           # ImageProfiler (analyzeImage)
├── metrics.py            # MetricsCalculator (PSNR, SSIM, Entropy, Colorfulness, DiffMap)
├── main.py               # CLI entry point
├── mask_test_batch.py    # Batch test with visual comparison
├── train.py              # EnhanceNet training script
├── dataset.py            # Training datasets (paired / self-supervised)
└── models/
    ├── __init__.py
    └── enhance_net.py    # EnhanceNet CNN (future AI-based pixel enhancement)
```

---

## 6. Teknoloji (Raporlarla uyumlu)

- **Dil**: Python 3.x
- **Görüntü İşleme**: OpenCV (`cv2`)
- **Sayısal**: NumPy
- **AI (ileride)**: PyTorch (EnhanceNet)
- **Performans hedefi**: Görüntü başına ≤10 saniye (Analysis Report)
