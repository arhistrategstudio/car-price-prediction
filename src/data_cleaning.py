"""
Čišćenje podataka: data/cars.csv -> data/processed/cars_clean.csv

Svaka odluka je zasnovana na nalazima iz notebooks/01_eda.ipynb:

1.  Nazivi kolona: `mileage(kilometers)` -> `mileage_km`, `volume(cm3)` -> `volume_cm3`
    (zagrade i jedinice u nazivu smetaju u kodu i u formulama).
2.  Identični duplikati (87 redova) se brišu - isti oglas uvezen dva puta.
3.  Kategorijske vrednosti: mala slova + bez viška praznina (segment ostaje velikim
    slovom jer su to kodovi A-S). Podaci su već uredni, ali korak je obavezan da bi
    isti kod radio i na novim unosima.
4.  `condition == "for parts"` (454 redova, medijana 400 $) se izbacuje: to je cena
    delova/otpada, ne tržišna cena automobila koju platforma treba da predloži.
    `with damage` ostaje - oštećen auto je i dalje auto, a `condition` je ulazna
    karakteristika.
5.  Marka `eksklyuziv` (55 redova: retro, vojna tehnika, karting, samogradnja,
    kamp vozila) se izbacuje: to nije marka nego kategorija "sve ostalo".
6.  Godina < 1980 (258 redova) se izbacuje: oldtajmerima cena raste sa starošću,
    što je suprotna zakonitost od ostatka tržišta. Model je za moderna polovna vozila.
7.  Cena < 300 $ se izbacuje: na vozilu koje nije "for parts" to nije tržišna cena
    nego placeholder ("1", "100", "123") ili cena za dogovor.
8.  Kilometraža: vrednosti >= 1.000.000 km (1111111, 9999999, 3456789...) su
    očigledni placeholder-i; isto važi za ponovljene cifre i nizove ispod milion
    (111111 x195, 333333 x147, 123456 x61...), a < 1.000 km na vozilu starom 2+
    godine znači da kilometraža nije uneta (0, 1, 100, 172...). Svi ti slučajevi
    postaju NaN i kasnije ih popunjava SimpleImputer uz indikator nedostajanja.
    Redovi se NE brišu jer su ostale kolone ispravne.
9.  Zapremina motora: vrednosti > 7.000 cm³ su tipfeler sa jednom nulom viška
    (mazda 3 = 20000, renault 21 = 16000, peugeot 206 = 14000...) i popravljaju se
    deljenjem sa 10. Električni automobili dobijaju 0 (nemaju motor sa zapreminom).
10. Nedostajući `segment` (5.291) i `drive_unit` (1.905): to su osobine modela, pa se
    popunjavaju najčešćom vrednošću istog make+model iz skupa; gde ni to ne postoji
    upisuje se "unknown" kao posebna kategorija (nedostajanje je informativno).

Pokretanje:  python src/data_cleaning.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# Pojedinačni koraci (svaki vraća novi DataFrame, ne menja ulazni)
# ---------------------------------------------------------------------------
def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Učitava sirovi CSV."""
    return pd.read_csv(path)


def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Preimenuje kolone sa zagradama/jedinicama u nazive pogodne za kod."""
    return df.rename(columns=config.COLUMN_RENAMES)


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Mala slova i strip za sve kategorijske kolone; segment velikim slovom."""
    df = df.copy()
    for col in config.CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().str.lower()
    if "segment" in df.columns:
        df["segment"] = df["segment"].str.upper()
    return df


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Briše potpuno identične redove (isti oglas uvezen više puta)."""
    return df.drop_duplicates().reset_index(drop=True)


def drop_invalid_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Izbacuje redove koji ne pripadaju tržištu koje modelujemo ili sadrže
    nevalidne vrednosti ciljne promenljive / godine. Vraća i broj izbačenih
    redova po razlogu, za izveštaj.
    """
    removed = {}
    mask_parts = df["condition"].isin(config.DROP_CONDITIONS)
    removed["condition_for_parts"] = int(mask_parts.sum())
    df = df[~mask_parts]

    mask_make = df["make"].isin(config.DROP_MAKES)
    removed["make_eksklyuziv"] = int(mask_make.sum())
    df = df[~mask_make]

    mask_year = (df["year"] < config.MIN_YEAR) | (df["year"] > config.REFERENCE_YEAR)
    removed["year_out_of_range"] = int(mask_year.sum())
    df = df[~mask_year]

    mask_price = df[config.TARGET] < config.MIN_PRICE
    removed["price_below_min"] = int(mask_price.sum())
    df = df[~mask_price]

    return df.reset_index(drop=True), removed


def _is_placeholder_number(values: pd.Series) -> pd.Series:
    """
    Prepoznaje "kucane" vrednosti koje nisu merenje nego izgovor za obavezno polje:
    5+ istih cifara (111111, 333333, 99999...) ili niz cifara (12345, 123456...).
    """
    as_text = values.fillna(-1).astype("int64").astype(str)
    same_digit = as_text.apply(lambda t: len(t) >= 5 and len(set(t)) == 1)
    sequence = as_text.isin(config.PLACEHOLDER_SEQUENCES)
    return (same_digit | sequence) & values.notna()


def fix_mileage(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Placeholder kilometraže -> NaN (ne brišemo red, ostatak je validan).
    - >= 1.000.000 km: očigledno uneta glupost (1111111, 9999999, 3456789...)
    - ponovljene cifre / nizovi (111111, 333333, 123456...): "ne znam" kucano u polje
    - < 1.000 km na vozilu starom 2+ godine: kilometraža nije uneta (0, 1, 100...)
    """
    df = df.copy()
    age = config.REFERENCE_YEAR - df["year"]
    too_high = df["mileage_km"] >= config.MAX_MILEAGE_KM
    placeholder = _is_placeholder_number(df["mileage_km"]) & ~too_high
    too_low = (df["mileage_km"] < config.MIN_PLAUSIBLE_MILEAGE_KM) & (age >= 2)
    df.loc[too_high | placeholder | too_low, "mileage_km"] = np.nan
    return df, {"mileage_set_nan_too_high": int(too_high.sum()),
                "mileage_set_nan_placeholder": int(placeholder.sum()),
                "mileage_set_nan_too_low": int(too_low.sum())}


def fix_volume(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Zapremina > 7000 cm³ je tipfeler sa nulom viška -> delimo sa 10.
    Ako je i posle toga > 7000, vrednost je nepoznata (NaN).
    Električna vozila nemaju zapreminu motora -> 0.
    """
    df = df.copy()
    typo = df["volume_cm3"] > config.MAX_VOLUME_CM3
    df.loc[typo, "volume_cm3"] = df.loc[typo, "volume_cm3"] / 10
    still_bad = df["volume_cm3"] > config.MAX_VOLUME_CM3
    df.loc[still_bad, "volume_cm3"] = np.nan
    electric = df["fuel_type"] == "electrocar"
    df.loc[electric, "volume_cm3"] = 0.0
    return df, {"volume_typo_fixed": int(typo.sum()),
                "volume_set_nan": int(still_bad.sum()),
                "volume_electric_set_zero": int(electric.sum())}


def fill_missing_categories(df: pd.DataFrame,
                            lookup: dict[str, pd.Series] | None = None
                            ) -> tuple[pd.DataFrame, dict, dict[str, pd.Series]]:
    """
    Popunjava nedostajući segment/drive_unit najčešćom vrednošću istog make+model.
    Ostatak dobija "unknown". `lookup` (make|model -> vrednost) se može proslediti
    da bi se isti mapping primenio na nove podatke (predikcija).
    """
    df = df.copy()
    key = df["make"].astype(str) + "|" + df["model"].astype(str)
    report, lookup_out = {}, {}
    for col in config.GROUP_FILL_COLUMNS:
        if lookup is not None and col in lookup:
            modes = lookup[col]
        else:
            modes = (df.loc[df[col].notna()]
                       .groupby(key[df[col].notna()])[col]
                       .agg(lambda s: s.value_counts().index[0]))
        lookup_out[col] = modes
        missing = df[col].isna()
        filled = key[missing].map(modes)
        df.loc[missing, col] = filled
        report[f"{col}_filled_from_model"] = int(filled.notna().sum())
        still_missing = df[col].isna()
        report[f"{col}_set_unknown"] = int(still_missing.sum())
        df.loc[still_missing, col] = config.UNKNOWN_LABEL
    return df, report, lookup_out


# ---------------------------------------------------------------------------
# Ceo tok čišćenja
# ---------------------------------------------------------------------------
def clean_data(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Primenjuje sve korake redom i vraća očišćen skup + izveštaj."""
    report = {"rows_raw": int(len(df_raw))}

    df = rename_columns(df_raw)
    df = standardize_categories(df)

    n_before = len(df)
    df = drop_duplicates(df)
    report["duplicates_removed"] = int(n_before - len(df))

    df, removed = drop_invalid_rows(df)
    report.update(removed)

    df, rep = fix_mileage(df)
    report.update(rep)

    df, rep = fix_volume(df)
    report.update(rep)

    df, rep, _ = fill_missing_categories(df)
    report.update(rep)

    df = df.sort_values(["make", "model", "year"]).reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["rows_removed_total"] = int(report["rows_raw"] - report["rows_clean"])
    report["missing_after_cleaning"] = {c: int(v) for c, v in df.isna().sum().items() if v > 0}
    return df, report


def main() -> None:
    df_raw = load_raw()
    df_clean, report = clean_data(df_raw)

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(config.CLEAN_DATA_PATH, index=False)
    with open(config.PROCESSED_DIR / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print("=== Izveštaj čišćenja ===")
    for k, v in report.items():
        print(f"{k:32s} {v}")
    print(f"\nSačuvano: {config.CLEAN_DATA_PATH}  ({len(df_clean)} redova, {df_clean.shape[1]} kolona)")


if __name__ == "__main__":
    main()
