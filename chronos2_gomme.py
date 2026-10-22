"""Ortalama kaymalı betise 0.6.0 serileri için Chronos-2 gömmeleri (amazon/chronos-2, dondurulmuş).

Gömme: serinin belirteç temsillerinin ortalaması (Choi, Shook, Dubrawski 2026, Ek B.1); her seri dolgusuz, yalnız kendi
verisiyle gömülür (aynı uzunluktaki parça sayısına sahip seriler birlikte). Girdi z-normalize seri.
Kullanım: python chronos2_gomme.py --kume egitim=PATH/yenidata --kume test=PATH/yenidata_test
Çıktı: gommeler.csv (seri bilgileri + g0..g767), seriler.csv (kume, seri, zaman, deger)
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ANOM = {"point_anomaly", "collective_anomaly", "mean_shift", "variance_shift", "trend_shift"}
TABAN = {"G06": "durağan", "G11": "deterministik trend", "G16": "stokastik trend", "G21": "volatilite"}
YER = {"beginning": "başta", "middle": "ortada", "end": "sonda"}


def js(v):
    return json.loads(v) if isinstance(v, str) else (v if v is not None else {})


def oku(kok):
    kok, ser, bilgi = Path(kok), [], []
    for f in sorted(kok.glob("**/part-*.parquet")):
        for sid, s in pd.read_parquet(f).groupby("series_id", sort=True):
            r = s.iloc[0]
            if [c for c in js(r["feature_components"]) if c in ANOM] != ["mean_shift"]:
                continue
            x = s.sort_values("time")["data"].to_numpy(np.float64)
            grup = f.relative_to(kok).parts[0]
            tarif = js(r["categorical_variant_ids"]).get("mean_shift", "")
            yon = re.search(r"direction-(up|down)", tarif)
            yer = re.search(r"location-(beginning|middle|end)", tarif)
            konum = js(r["break_indices"]).get("mean_shift", [])
            ser.append(x)
            bilgi.append({"grup": grup, "taban": TABAN.get(grup, ""), "uzunluk": len(x), "kayma_sayisi": len(konum),
                          "kayma_konum": ";".join(str(k) for k in konum),
                          "kayma_buyukluk": ";".join(f"{m:.4g}" for m in js(r["break_magnitudes"]).get("mean_shift", [])),
                          "kayma_yon": {"up": "yukarı", "down": "aşağı"}[yon.group(1)] if yon else "rastgele",
                          "kayma_yeri": YER[yer.group(1)] if yer else ""})
    return ser, bilgi


def gomme(pipe, ser):
    grup = defaultdict(list)
    for i, s in enumerate(ser):
        grup[int(np.ceil(len(s) / 16))].append(i)
    out = np.zeros((len(ser), 768), np.float32)
    for idx in grup.values():
        for b in range(0, len(idx), 256):
            ii = idx[b:b + 256]
            emb, _ = pipe.embed([torch.tensor(((ser[i] - ser[i].mean()) / ser[i].std()).astype(np.float32)) for i in ii])
            for i, e in zip(ii, emb):
                out[i] = e[0].float().mean(0).cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kume", action="append", required=True, help="ad=betise_cikti_klasoru")
    a = ap.parse_args()
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="cuda" if torch.cuda.is_available() else "cpu")
    G, S = [], []
    for k in a.kume:
        ad, kok = k.split("=", 1)
        ser, bilgi = oku(kok)
        g = pd.DataFrame(bilgi)
        g.insert(0, "seri", range(len(ser)))
        g.insert(0, "kume", ad)
        G.append(pd.concat([g, pd.DataFrame(gomme(pipe, ser), columns=[f"g{j}" for j in range(768)])], axis=1))
        S.append(pd.DataFrame({"kume": ad, "seri": np.repeat(np.arange(len(ser)), [len(x) for x in ser]),
                               "zaman": np.concatenate([np.arange(len(x)) for x in ser]), "deger": np.concatenate(ser)}))
        print(ad, len(ser), "seri", flush=True)
    pd.concat(G).to_csv("gommeler.csv", index=False, float_format="%.7g", encoding="utf-8")
    pd.concat(S).to_csv("seriler.csv", index=False, float_format="%.10g", encoding="utf-8")


if __name__ == "__main__":
    main()
