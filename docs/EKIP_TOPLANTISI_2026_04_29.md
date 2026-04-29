# OptiLumen — Pixel Layer Statü Toplantısı

**Tarih:** 29 Nisan 2026 (akşam)
**Konuşan:** Batuhan Taşdemir
**Hedef:** Furkan ve Emir'e Pixel Layer'ın bugünkü durumunu, entegrasyon kontratlarını ve sıradaki işleri özetlemek.
**Süre hedefi:** ~10 dakika anlatım + 5 dakika demo + Q&A.

---

## 1. Tek cümlede sistem

> **GFPGAN v1.3 tabanlı yüz restorasyonu + 4 demo senaryosu (Live Cam, Batch, Chat, Live Filter) + ekip arası multi-PC AI handoff altyapısı.**

Repo: `https://github.com/CaptainBlc/Global-Enhancement`
Ana branch: `main` · Bizim branch: `Batuhan-Develop` · Birleşim: `Total-Develop`

---

## 2. Mimari hatırlatma

```
Input Image
     │
     ├─► Semantic / Edge Layer   ← Emir
     │       (segmentasyon, mask_map)
     │
     ├─► Pixel Layer              ← Batuhan (BEN)
     │       (GFPGAN + profiling + metrics)
     │
     ├─► Global Enhancement       ← Furkan
     │       (exposure, color, contrast)
     │
     └─► Fusion → Output
```

Anlatırken vurgu: **Her katman birbirinden bağımsız çalışabiliyor.** Pixel Layer'ı zaten bu repoda standalone gösterebiliyorum; arkadaşlar modüllerini `pipeline.py`'a bağlayınca `Total-Develop` üstünde tek borudan akacak.

---

## 3. Pixel Layer'da ne var?

| Modül | Dosya | Görev |
|---|---|---|
| Pipeline orchestrator | `src/pipeline.py` | `ImageEnhancementPipeline.restoreImage()` — tüm akışı koordine eder |
| Image profiler | `src/profiler.py` | brightness, contrast, blur, noise, skin oranı, scene flag'ler |
| Face restorer | `src/face_restorer.py` | RetinaFace ile yüz tespiti + GFPGAN inference + classical fallback |
| Metrics | `src/metrics.py` | PSNR, SSIM, Entropy, Colorfulness, diff heatmap |
| GFPGAN mimarisi | `src/models/gfpgan_arch.py` + `stylegan2_clean.py` | StyleGAN2 + SFT, **pure PyTorch** (basicsr bağımlılığı yok) |
| GUI | `src/gui/main_window.py` + `gui_main.py` | PyQt6 demo arayüzü |

**Önemli teknik karar (anlatımda söyle):**
GFPGAN'ın resmi paketi `basicsr` Python 3.13 ile uyumsuz çıktı. O yüzden GFPGAN v1.3 mimarisini **kendimiz pure PyTorch ile yeniden yazdık**, weight'leri orijinal release'ten indiriyoruz. Bu sayede:
- Tek bağımlılık çakışması yok
- 3.10 / 3.11 / 3.12 / 3.13 hepsinde çalışıyor
- "Karmaşık paket bağımlılığı" sorunu yaşamıyoruz

---

## 4. 4 Sprint — Analysis Report §3.5.1 use-case'leriyle eşleşmeli

| Sprint | Feature | Senaryo | Dosya |
|---|---|---|---|
| 1 | Real-time camera capture | §3.5.1 #4 | `src/camera_capture.py` |
| 2 | Background batch rendering | §3.5.1 #1, #2 | `src/batch_processor.py` |
| 3 | AI chat prompt | §3.5.1 #3 | `src/chat_commands.py` |
| 4 | Live video filter | §3.5.1 #4 ext. | `src/live_filters.py` |

Anlatım sırası önerisi: önce mimariyi söyle, sonra "her senaryo için ne çalışıyor"u kanıt olarak göster.

---

## 5. Entegrasyon kontratları — Furkan ve Emir bunu duyacak

### 5.1 Furkan — Global Enhancement Layer

**Çağrı noktası:** `src/pipeline.py` içinde `restoreImage()` adımları arasında **boş slot** hazır.

```python
# pipeline.py — gerçek çağrı sırası
profile  = self.analyzeImage(image)         # ImageProfiler
restored = self._restorer.restore(image)    # Pixel Layer (GFPGAN)
# ─── Furkan'ın modülü buraya girecek ─────────────
# enhanced = self._global.apply(restored, profile)
# ─────────────────────────────────────────────────
metrics  = MetricsCalculator.compute(...)
```

**Kontrat (Furkan'a):**
- Input: `np.ndarray` BGR uint8 (Pixel Layer'ın çıktısı) + `ProfileResult` (zaten elinde)
- Output: aynı tipte `np.ndarray` BGR uint8
- Hiçbir external state gerektirmesin (constructor'da config alabilir)
- Furkan'ın repo'su: `https://github.com/furkancabbar/Global-Enhancement`

### 5.2 Emir — Semantic / Edge Layer

**Pixel Layer'ın `mask_map` desteği var:** `FaceRestorer` ve `ImageProfiler` zaten optional bir `mask_map` parametresini düşünerek tasarlandı.

**Kontrat (Emir'e):**
- Format: `np.ndarray` shape `(H, W)` veya `(H, W, 1)`, dtype `float32`, range `[0.0, 1.0]`
- Yorum: 1.0 = ROI (yüz vb. korunacak yer), 0.0 = arka plan
- Çağrı noktası: pipeline'a girmeden önce, ya da `restoreImage(image, mask_map=...)` parametresi olarak

### 5.3 Birleştirme yeri

`Total-Develop` branch'i — herkes kendi `*-Develop` branch'inde geliştirir, hazır olunca `Total-Develop`'a merge ederiz.

---

## 6. Multi-PC AI Handoff Altyapısı (yeni)

Bu kısmı özellikle vurgula — bu sayede **ekipten herhangi biri** Cursor + Claude'la projeye katkı yapabilir, senkronize kalabilir.

| Dosya | Rol |
|---|---|
| `.cursor/rules/optilumen.mdc` | Cursor açıldığında AI'a otomatik yüklenen proje bağlamı: takım rolleri, mimari kuralları, branch akışı, dokunulmazlar. |
| `PROGRESS.md` | Her sprint sonrası güncellenen **paylaşılan AI hafızası**. Sprint durumu, changelog, açık kararlar, sıradaki aksiyonlar. |
| `README.md` → "Continuing Development on Another PC" | İnsan okunabilir kısa rehber. |

**Kullanım:** `git pull` → Cursor'da klasörü aç → AI hemen `optilumen.mdc` ve `PROGRESS.md`'yi okur → "nerede kaldık?" diye sormaya gerek kalmaz.

---

## 7. Branch yapısı

```
main              ← stable, ne anlattıysam orada çalışıyor
├── Batuhan-Develop    ← bizim aktif iş (Pixel Layer)
├── Furkan-Develop     ← Furkan'ın Global Layer'ı buraya
├── Emir-Develop       ← Emir'in Semantic Layer'ı buraya
└── Total-Develop      ← üçümüzün birleşik test branch'i
legacy                 ← eski classical-pixel sistemi (donduruldu, sunum öyküsü için kalsın)
```

Workflow:
1. Sen kendi `*-Develop` branch'inde çalış
2. Hazır olunca `Total-Develop`'a merge et
3. Her şey çalışıyorsa `main`'e PR aç
4. **`PROGRESS.md`'i güncellemeyi unutma** — paylaşılan AI hafızasıdır

---

## 8. Canlı Demo Akışı (5 dakika)

> **Anlatırken sırayla bunları yap:**

### Demo 1 — Tek görüntü iyileştirme (1 dk)
1. `py -3 src/gui_main.py` → GUI açılır
2. Toolbar **Open** → `test_mix/jurica-koletic-7YVZYZeITc8-unsplash.jpg` (yüksek kaliteli portre)
3. **Restore Face(s)** (kırmızı buton)
4. **Compare** sekmesine geç → orta sürgüyü sürükle: before/after canlı görsel
5. **Diff** sekmesi → değişikliklerin nerede olduğunu gösteren ısı haritası
6. Sağ panelde PSNR / SSIM / Entropy / Colorfulness metriklerini göster

### Demo 2 — Chat barı (45 sn)
En alttaki barda sırayla yaz:
- `more ai` → fidelity slider'ı kayar, mesaj log'a düşer
- `fidelity 25` → tam %25
- `compare` → otomatik view değişir
- `reset` → orijinale döner
- `help` → komut listesi

> "Şu an kural-tabanlı, offline çalışıyor — sunumda internet düşse bile demo çalışır. İleride LLM ile değiştirilebilecek şekilde tasarlandı."

### Demo 3 — Batch (45 sn)
- Toolbar **Batch…** → `test_mix` klasörünü seç
- Non-modal dialog açılır → progress bar + per-file tablo
- Her dosyayı sırayla işliyor, **Cancel** ile durdurulabiliyor
- Bittiğinde **Open Output Folder** → `outputs/batch_<timestamp>/`

### Demo 4 — Canlı kamera + filtre (1.5 dk)
- Toolbar **● Live** → webcam preview başlar
- **BEAUTY** filtresine bas → kendi yüzün üzerinde classical real-time filtre
- **AI** filtresine bas → GFPGAN ara ara çalışıp cache ediyor
- **Capture** → o anki filtreli görüntü "yüklü resim" gibi geçer
- Sonra **Restore Face(s)** ile tam pipeline'dan geçirebilirsin

### Demo sonu — kapanış cümlesi
> "Şu an itibariyle Pixel Layer demo-ready. Sizin modülleriniz hazır olunca `Total-Develop` üstünde aynı GUI'den entegre çalışır. Geri kalan tek iş `pipeline.py`'a iki çağrı eklemek."

---

## 9. Sıradaki Aksiyonlar (toplantı çıktısı)

Toplantıda netleştirmemiz gerekenler:

- [ ] **Furkan** — Global Layer'ın input/output kontratı kabul mu? Constructor parametreleri?
- [ ] **Emir** — Semantic Layer çıktısı `mask_map` formatı kesinleşti mi? Hangi class'larda mask üretilecek (face, skin, background, ...)?
- [ ] Hangi tarihte ilk birleşik **Total-Develop** denemesi yapacağız?
- [ ] Sunumda hangi senaryoları öne çıkaracağız (4 senaryomuz var, hepsini gösteremeyiz)?
- [ ] GPU'ya geçiş — labda CUDA destekli makine var mı? (AI live filter için kritik)

---

## 10. Sıkça Gelebilecek Sorulara Hazırlık

**S: GFPGAN'ı kendiniz mi yazdınız?**
- Mimariyi (`StyleGAN2GeneratorCSFT` + SFT'ler + U-Net encoder) pure PyTorch'ta yeniden implemente ettik. Eğitilmiş ağırlıkları (FFHQ üzerinde pre-train) TencentARC release'inden indiriyoruz. Eğitim yapmadık.

**S: Neden classical pixel sistemini bıraktınız?**
- Hocanın da önerisiyle GFPGAN'a geçtik. Eski sistem `legacy/` klasöründe duruyor, sunumda "evrim" hikayesi olarak kullanılabilir.

**S: Sistem CPU'da yavaş mı?**
- Tek görüntüde GFPGAN ~2-3 sn (CPU). Live AI filter bu yüzden hibrit yaklaşıyor: classical her frame, AI her N. frame + cache. GPU ile çok daha hızlı olur.

**S: 3 kişi 3 modül yapıyorsanız sunumda nasıl ayrılır?**
- Her birimiz kendi modülünü gösterir, sonunda `Total-Develop` üzerinde birleşik akışı çalıştırırız.

**S: Veri seti?**
- GFPGAN zaten pre-trained (FFHQ + degraded variants). `test_mix/` klasöründe 18 test görüntüsü var (Unsplash + sentetik bozulmalar). Eğitim verisi gerekmedi — bu projede transfer learning kullanıyoruz.

---

## 11. Kapanış Sözü

> "Pixel Layer 4 sprint'le birlikte demo-ready, repo herkes için clone-and-run. Multi-PC AI handoff sayesinde ikiniz de bilgisayarınızdan kendi modülünüzü Cursor'la geliştirebilirsiniz, AI bağlamı otomatik yüklenir. Toplantıdan sonra `PROGRESS.md`'yi okumanız 5 dakikanızı alır, neyin nerede olduğu net görünür."
