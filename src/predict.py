"""
Predikcija cene za jedan automobil iz komandne linije.

Prolazi kroz iste korake kao trening (standardizacija kategorija, popravka
zapremine, "unknown" za nepoznat segment/pogon, izvedene karakteristike) pa
poziva sačuvani model. Ako postoje, prikazuje i interval "od-do" (10.-90.
percentil) i objašnjenje iz čitljivog formula-modela.

Primer (automobil iz teksta zadatka):
    python src/predict.py --make volkswagen --model golf --year 2014 \
        --condition "with mileage" --mileage 180000 --fuel diesel --volume 1600 \
        --color black --transmission mechanics --drive "front-wheel drive" --segment C

Ili iz JSON fajla sa istim ključevima kao kolone u cars.csv:
    python src/predict.py --json car.json
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data_cleaning import fix_volume, rename_columns, standardize_categories  # noqa: E402
from src.feature_engineering import add_features  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

RAW_COLUMNS = ["make", "model", "year", "condition", "mileage_km", "fuel_type",
               "volume_cm3", "color", "transmission", "drive_unit", "segment"]


def prepare_record(record: dict) -> pd.DataFrame:
    """Jedan oglas (dict) -> DataFrame spreman za model."""
    df = rename_columns(pd.DataFrame([record]))
    for col in RAW_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan
    df["year"] = df["year"].astype(int)
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df["volume_cm3"] = pd.to_numeric(df["volume_cm3"], errors="coerce")
    df = standardize_categories(df)
    df, _ = fix_volume(df)
    for col in config.GROUP_FILL_COLUMNS:
        df[col] = df[col].fillna(config.UNKNOWN_LABEL)
    return add_features(df)


def predict_price(record: dict) -> dict:
    """Vraća procenu, interval (ako postoji) i formulu (ako postoji)."""
    X = prepare_record(record)
    model = joblib.load(config.FINAL_MODEL_PATH)
    result = {"price_usd": float(model.predict(X)[0])}

    if config.INTERVAL_MODEL_LOW_PATH.exists() and config.INTERVAL_MODEL_HIGH_PATH.exists():
        lo = float(joblib.load(config.INTERVAL_MODEL_LOW_PATH).predict(X)[0])
        hi = float(joblib.load(config.INTERVAL_MODEL_HIGH_PATH).predict(X)[0])
        result["interval_10_90"] = [min(lo, hi), max(lo, hi)]

    formula_path = config.MODELS_DIR / "car_price_formula.joblib"
    if formula_path.exists():
        formula = joblib.load(formula_path)
        result["formula_price_usd"] = float(formula.predict(X)[0])
        result["formula"] = formula.formula_
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Procena cene polovnog automobila")
    parser.add_argument("--json", help="putanja do JSON fajla sa podacima o automobilu")
    parser.add_argument("--make"); parser.add_argument("--model"); parser.add_argument("--year", type=int)
    parser.add_argument("--condition", default="with mileage")
    parser.add_argument("--mileage", type=float, help="kilometraža u km")
    parser.add_argument("--fuel", default="petrol", help="petrol | diesel | electrocar")
    parser.add_argument("--volume", type=float, help="zapremina motora u cm3")
    parser.add_argument("--color", default="other")
    parser.add_argument("--transmission", default="mechanics", help="mechanics | auto")
    parser.add_argument("--drive", help="front-wheel drive | rear drive | all-wheel drive | part-time four-wheel drive")
    parser.add_argument("--segment", help="A | B | C | D | E | F | J | M | S")
    args = parser.parse_args()

    if args.json:
        with open(args.json, encoding="utf-8") as f:
            record = json.load(f)
    else:
        if not (args.make and args.model and args.year):
            parser.error("--make, --model i --year su obavezni (ili --json)")
        record = {"make": args.make, "model": args.model, "year": args.year, "condition": args.condition,
                  "mileage_km": args.mileage, "fuel_type": args.fuel, "volume_cm3": args.volume,
                  "color": args.color, "transmission": args.transmission,
                  "drive_unit": args.drive, "segment": args.segment}

    result = predict_price(record)
    print(f"Procenjena cena: {result['price_usd']:,.0f} $")
    if "interval_10_90" in result:
        lo, hi = result["interval_10_90"]
        print(f"Okvirno od-do (10.-90. percentil): {lo:,.0f} $ - {hi:,.0f} $")
    if "formula" in result:
        print(f"Formula-model kaže: {result['formula_price_usd']:,.0f} $")
        print(f"  {result['formula']}")


if __name__ == "__main__":
    main()
