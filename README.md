# Ortalama kaymalı seriler için Chronos-2 gömmeleri

## İçerik

| Dosya | Açıklama |
|---|---|
| `veri/gomme_egitim.npz` | 4.000 seri: `gomme` (4.000 × 768, float32), `degerler` (ham seriler art arda), `uzunluklar` |
| `veri/gomme_test.npz` | 800 seri, aynı alanlar |
| `veri/meta_egitim.csv`, `veri/meta_test.csv` | Seri başına betise kayıtları; satır sırası npz ile aynı |
| `veri/gomme_egitim.csv`, `veri/gomme_test.csv` | Gömmelerin CSV hâli: her satır bir seri, `g0` … `g767` sütunları |
| `veri/seriler_egitim.csv`, `veri/seriler_test.csv` | Ham serilerin CSV hâli: `satir` (seri), `zaman`, `deger` |
| `chronos2_gomme.py` | Gömmeleri betise çıktısından üreten betik |

Bütün dosyalarda `satir` aynı seriyi gösteriyor: i'inci gömme, i'inci seri ve `meta` dosyasının i'inci satırı aynı seriye ait.

## Dosyaları açma

- **CSV dosyaları:** Excel ya da herhangi bir tablo programıyla açılır.
- **npz dosyaları:** Python ile açılır:

```python
import numpy as np
d = np.load("veri/gomme_egitim.npz")
gomme = d["gomme"]          # 4.000 x 768: her satır bir serinin gömmesi
uzunluklar = d["uzunluklar"]
degerler = d["degerler"]    # ham seriler art arda
bas = np.concatenate([[0], np.cumsum(uzunluklar)])
seri_0 = degerler[bas[0]:bas[1]]
```

## Seriler

- betise 0.6.0 ile üretilen eğitim ve test kümelerinden, tek anomalisi ortalama kayması olan seriler.
- Tezin dört grubu; her grupta eğitimde 1.000, testte 200 seri:
    - G06: durağan taban + ortalama kayması,
    - G11: deterministik trend + ortalama kayması,
    - G16: stokastik trend + ortalama kayması,
    - G21: volatilite + ortalama kayması.
- Uzunluklar 50–100 ya da 300–500 nokta.
- `meta` dosyalarında kaymanın konumu (`ortalama_kaymasi_konum`), büyüklüğü (`ortalama_kaymasi_buyukluk`), yönü ve konum kategorisi (başta / ortada / sonda) var.

## Gömme

- **Model:** Önceden eğitilmiş `amazon/chronos-2` (chronos-forecasting 2.2.2), dondurulmuş; ince ayar yok.
- **Gömme:** Serinin belirteç temsillerinin ortalaması, 768 boyut; ortalamaya Chronos-2'nin [REG] ve çıktı belirteçleri de dahil. Bu, Choi, Shook, Dubrawski (2026, Ek B.1) yöntemi ("mean pooling over token representations").
- **Dolgu:** Seriler farklı uzunlukta olduğu için her seri yalnız kendi verisiyle gömüldü; dolgu kullanılmadı. Aynı seri her zaman aynı gömmeyi veriyor.
- **Ön işlem:** Seri z-normalize edildi. Chronos-2 seriyi zaten kendi içinde standartlaştırdığı için bu adım sonucu değiştirmiyor.

## Betiği çalıştırma

Aşağıdaki komut, betise'nin eğitim ve test çıktılarından `veri/` klasöründeki dosyaların aynısını üretir:

```
python chronos2_gomme.py --kume egitim=<betise_cikti>/yenidata --kume test=<betise_cikti>/yenidata_test --cikti veri
```
