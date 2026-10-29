"""
Inženjering karakteristika: data/processed/cars_clean.csv -> data/processed/cars_features.csv

Dodate karakteristike (svaka ima jasnu svrhu):

- car_age          = 2019 - year. Starost je ono što zaista određuje amortizaciju,
                     a ne kalendarska godina. Referentna godina je 2019 (najnovija
                     godina u skupu = godina kada su cene snimljene), a NE današnja
                     godina: računanje od 2026. bi svakom autu dodalo 7 lažnih godina.
- mileage_per_year = mileage_km / max(car_age, 1). Razdvaja "star auto sa malo
                     kilometara" od "mlad auto koji je bio taksi": 200.000 km na
                     5 godina i na 15 godina nisu ista priča. Delilac je bar 1 da
                     nova vozila (starost 0) ne dele nulom.
- brand_model      = make + "_" + model. Nazivi modela nisu jedinstveni između marki
                     ("3" je i mazda 3 i bmw 3-seriya), a par marka+model je ono što
                     nosi "osnovnu cenu" vozila. Ima 1.000+ vrednosti, pa se u
                     pretprocesiranju koduje target encoding-om, ne OneHot-om.

Predložene, a svesno NEuvedene karakteristike:

- engine_volume_liters: samo je volume_cm3 / 1000. Linearno skaliranje ne nosi
  nikakvu novu informaciju - stabla su na njega potpuno neosetljiva, a linearni
  model ga već dobija kroz StandardScaler.
- is_newer_car, is_high_mileage: fiksni pragovi na kolonama koje model već ima.
  Stabla sama nađu bolji prag, a za linearni model je prag proizvoljan. Dodali bi
  kolone, a ne informaciju.

Pokretanje:  python src/feature_engineering.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

NEW_FEATURES = ["car_age", "mileage_per_year", "brand_model"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Dodaje izvedene karakteristike. Radi i na jednom redu (predikcija)."""
    df = df.copy()
    df["car_age"] = (config.REFERENCE_YEAR - df["year"]).clip(lower=0)
    df["mileage_per_year"] = df["mileage_km"] / np.maximum(df["car_age"], 1)
    df["brand_model"] = df["make"].astype(str) + "_" + df["model"].astype(str)
    return df


def main() -> None:
    df = pd.read_csv(config.CLEAN_DATA_PATH)
    df = add_features(df)
    df.to_csv(config.FEATURES_DATA_PATH, index=False)

    print("=== Nove karakteristike ===")
    print(df[["year", "car_age", "mileage_km", "mileage_per_year", "brand_model"]].head(8).to_string())
    print()
    print(df[["car_age", "mileage_per_year"]].describe().round(1).to_string())
    print(f"\nbrand_model jedinstvenih: {df['brand_model'].nunique()}")
    print(f"Sačuvano: {config.FEATURES_DATA_PATH}  ({df.shape[0]} redova, {df.shape[1]} kolona)")


if __name__ == "__main__":
    main()
