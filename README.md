# Ortalama kaymalı seriler için Chronos-2 gömmeleri

| Dosya | İçerik |
|---|---|
| `gommeler.csv` | Her satır bir seri (eğitim 4.000, test 800): seri bilgileri ve `g0` … `g767` gömme sütunları |
| `seriler.csv` | Ham seriler: `kume`, `seri`, `zaman`, `deger` |
| `chronos2_gomme.py` | İki dosyayı betise çıktısından üretir: `python chronos2_gomme.py --kume egitim=<yenidata> --kume test=<yenidata_test>` |

**Seriler:** betise 0.6.0 eğitim ve test kümelerinden, tek anomalisi ortalama kayması olanlar.

| Grup | Taban |
|---|---|
| G06 | durağan |
| G11 | deterministik trend |
| G16 | stokastik trend |
| G21 | volatilite |

- Uzunluk 50–100 ya da 300–500 nokta.
- Kaymanın konumu, büyüklüğü, yönü ve yeri (başta / ortada / sonda) `gommeler.csv`'de.
- Çoğu seride tek kayma var. 574 seride 2–3 kayma var; bunlarda yön rastgele.

**Gömme:**
- Model: önceden eğitilmiş, dondurulmuş `amazon/chronos-2`.
- Gömme: serinin belirteç temsillerinin ortalaması; [REG] ve çıktı belirteçleri dahil. Choi, Shook, Dubrawski (2026, Ek B.1) böyle yapıyor.
- Her seri yalnız kendi verisiyle gömüldü, dolgu yok.
- Girdi z-normalize edildi.
