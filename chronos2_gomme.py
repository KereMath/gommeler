"""Chronos-2 gömmeleri: betise 0.6.0 çıktısındaki, tek anomalisi ortalama kayması olan seriler.

Bağımsız betik: yalnız betise'nin parquet çıktısını okur.
Gereken paketler: numpy, pandas, pyarrow, torch, chronos-forecasting (2.2.2 ile denendi).

Model: önceden eğitilmiş amazon/chronos-2, dondurulmuş (ince ayar yok).
Gömme: serinin belirteç temsillerinin ortalaması (bağlam parçaları + [REG] + maskeli çıktı belirteci; 768 boyut),
  Choi, Shook, Dubrawski (2026) Ek B.1'deki gibi ("mean pooling over token representations").
  Makalede pencereler eşit uzunlukta olduğu için dolgu oluşmaz. Buradaki seriler farklı uzunlukta; Chronos2Pipeline.embed
  bir toplu işteki serileri en uzununa göre sola doldurduğu için seriler parça sayısına (ceil(n / 16)) göre gruplanarak
  gömülür. Böylece dolgu oluşmaz ve sonuç, her seriyi tek başına gömmekle aynıdır.
Ön işlem: seri z-normalize edilir. Chronos-2 seriyi kendi içinde standartlaştırıp arcsinh uyguladığı için bu adım
  sonucu değiştirmez (ham girdiyle en büyük fark ~3e-7).
Seçim: betise'nin kaydettiği bileşenlerden tek anomalisi "mean_shift" olan seriler (tezin G06, G11, G16, G21 grupları).
Sıra: dosya yolu sırası, dosya içinde series_id sırası.

Kullanım:
  python chronos2_gomme.py --kume egitim=PATH/yenidata --kume test=PATH/yenidata_test --cikti veri
Çıktı (her küme için):
  veri/gomme_<kume>.npz   gomme (N x 768, float32), degerler (ham seriler art arda), uzunluklar (N)
  veri/meta_<kume>.csv    seri başına betise kayıtları (satır sırası npz ile aynı)
  veri/gomme_<kume>.csv   gömmeler (satır başına bir seri, g0 ... g767)
  veri/seriler_<kume>.csv ham seriler (satir, zaman, deger)
"""
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ANOMALILER = {"point_anomaly", "collective_anomaly", "mean_shift", "variance_shift", "trend_shift"}
ALANLAR = ["base_components", "feature_components", "break_indices", "break_magnitudes", "location_mean_shift",
           "categorical_variant_ids"]


def liste(v):
    if v is None:
        return []
    return json.loads(v) if isinstance(v, str) else list(v)


def oku(kok):
    kok = Path(kok)
    seriler, meta = [], []
    for f in sorted(kok.glob("**/part-*.parquet")):
        df = pd.read_parquet(f)
        for sid, s in df.groupby("series_id", sort=True):
            ilk = s.iloc[0]
            if [c for c in liste(ilk["feature_components"]) if c in ANOMALILER] != ["mean_shift"]:
                continue
            x = s.sort_values("time")["data"].to_numpy(np.float64)
            satir = {"dosya": "/".join(f.relative_to(kok).parts), "series_id": int(sid), "uzunluk": len(x)}
            for a in ALANLAR:
                v = ilk[a] if a in s.columns else None
                satir[a] = v if (v is None or isinstance(v, str)) else json.dumps(v.tolist() if hasattr(v, "tolist") else v)
            bi = json.loads(satir["break_indices"]) if satir["break_indices"] else {}
            bm = json.loads(satir["break_magnitudes"]) if satir["break_magnitudes"] else {}
            satir["ortalama_kaymasi_konum"] = json.dumps(bi.get("mean_shift", []))
            satir["ortalama_kaymasi_buyukluk"] = json.dumps(bm.get("mean_shift", []))
            seriler.append(x)
            meta.append(satir)
    return seriler, pd.DataFrame(meta)


def znorm(x):
    x = np.asarray(x, np.float64)
    m, s = x.mean(), x.std()
    return (x - m if s < 1e-9 else (x - m) / s).astype(np.float32)


def gomme(pipe, ser, bs=256):
    grup = defaultdict(list)
    for i, s in enumerate(ser):
        grup[int(np.ceil(len(s) / 16))].append(i)
    out = np.zeros((len(ser), 768), np.float32)
    for npch, idx in grup.items():
        for b in range(0, len(idx), bs):
            ii = idx[b:b + bs]
            emb, _ = pipe.embed([torch.tensor(ser[i]) for i in ii], batch_size=bs)
            for i, e in zip(ii, emb):
                assert e.shape[1] == npch + 2, (e.shape, npch)
                out[i] = e[0].float().mean(0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", action="append", required=True, help="ad=betise_cikti_klasoru")
    ap.add_argument("--cikti", default="veri")
    a = ap.parse_args()
    from chronos import Chronos2Pipeline
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=dev, torch_dtype=torch.float32)
    cikti = Path(a.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    for k in a.kume:
        ad, kok = k.split("=", 1)
        t0 = time.time()
        ham, meta = oku(kok)
        G = gomme(pipe, [znorm(x) for x in ham])
        np.savez(cikti / f"gomme_{ad}.npz", gomme=G, degerler=np.concatenate(ham), uzunluklar=np.array([len(x) for x in ham]))
        meta.to_csv(cikti / f"meta_{ad}.csv", index_label="satir", encoding="utf-8")
        # Python'suz açmak için CSV: gömmeler (satır başına bir seri, 768 sütun) ve ham seriler (uzun biçim)
        pd.DataFrame(G, columns=[f"g{j}" for j in range(G.shape[1])]).to_csv(cikti / f"gomme_{ad}.csv", index_label="satir",
                                                                              float_format="%.7g")
        pd.DataFrame({"satir": np.repeat(np.arange(len(ham)), [len(x) for x in ham]),
                      "zaman": np.concatenate([np.arange(len(x)) for x in ham]),
                      "deger": np.concatenate(ham)}).to_csv(cikti / f"seriler_{ad}.csv", index=False, float_format="%.10g")
        print(ad, len(ham), "seri", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
