"""
Pretprocesiranje: definicija cilja, kolona i transformera + fiksna train/test podela.

data/processed/cars_features.csv -> data/processed/train.csv + data/processed/test.csv

Ciljna promenljiva: priceUSD. Modeli je uče u log-skali (log1p) kroz
TransformedTargetRegressor, jer je cena jako zakrivljena (medijana 5.500 $,
maksimum 235.000 $): u log-skali greška od 10 % vredi isto za auto od 2.000 $ i
od 50.000 $, a skupi outlieri ne diktiraju ceo model. Metrike se uvek računaju
u dolarima (posle inverzne transformacije).

Ulazne kolone:
- numeričke  : car_age, mileage_km, volume_cm3, mileage_per_year
               -> SimpleImputer(median, add_indicator) + StandardScaler
               (indikator pamti da je kilometraža bila nepoznata - to je samo po
               sebi signal; skaliranje treba linearnim modelima, stablima ne smeta)
- kategorijske sa malo vrednosti: make, condition, fuel_type, color, transmission,
               drive_unit, segment
               -> OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10)
               (nepoznata ili retka vrednost na predikciji ide u "infrequent" kolonu
               umesto da obori pipeline)
- kategorijska sa 1.000+ vrednosti: brand_model
               -> TargetEncoder (prosečna log-cena tog modela, sa cross-fitting-om
               na treningu da model ne bi "video" sopstvenu cenu). OneHot bi dao
               1.000+ retkih kolona.
- OrdinalEncoder nije potreban: segment (A, B, C, D, E, F, J, M, S) nije uređen
  (J = SUV, M = minivan, S = sportski), pa ide u OneHot.

Isključene kolone: year (kolinearna sa car_age, nosi istu informaciju),
model (zamenjen sa brand_model).

Podela: 80/20, random_state=42, sačuvana na disk - SVE skripte za treniranje,
evaluaciju i poređenje čitaju iste fajlove, pa je poređenje modela fer.

Pokretanje:  python src/data_preprocessing.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder
from sklearn.model_selection import KFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402

TARGET = config.TARGET
NUMERIC_FEATURES = ["car_age", "mileage_km", "volume_cm3", "mileage_per_year"]
ONEHOT_FEATURES = ["make", "condition", "fuel_type", "color", "transmission", "drive_unit", "segment"]
TARGET_ENCODED_FEATURES = ["brand_model"]
ALL_FEATURES = NUMERIC_FEATURES + ONEHOT_FEATURES + TARGET_ENCODED_FEATURES


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    """
    Vraća ColumnTransformer koji od sirovih kolona pravi numeričku matricu.
    `scale_numeric=False` je dozvoljeno za modele na stablima (skaliranje im nije
    potrebno), ali podrazumevano skaliramo da svi modeli dele isti preprocessor.
    """
    numeric_steps = [("imputer", SimpleImputer(strategy="median", add_indicator=True))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(numeric_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                  sparse_output=False), ONEHOT_FEATURES),
            ("target_enc", TargetEncoder(target_type="continuous", smooth="auto",
                                         cv=KFold(n_splits=5, shuffle=True,
                                                  random_state=config.RANDOM_STATE)),
             TARGET_ENCODED_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def target_transform(y):
    """Cena -> log-skala (za učenje)."""
    return np.log1p(y)


def target_inverse_transform(y_log):
    """Log-skala -> cena u dolarima (za predikciju i metrike)."""
    return np.expm1(y_log)


def get_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Razdvaja ulazne kolone i cilj. X zadržava sve ne-ciljne kolone (uključujući
    year i model) jer ih koristi baseline bez ML-a; ColumnTransformer iz
    build_preprocessor() sam bira samo kolone iz ALL_FEATURES (remainder="drop").
    """
    return df.drop(columns=[TARGET]).copy(), df[TARGET].copy()


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fiksna 80/20 podela sa random_state=42."""
    return train_test_split(df, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Učitava sačuvanu podelu (koriste je sve skripte za modele)."""
    return pd.read_csv(config.TRAIN_DATA_PATH), pd.read_csv(config.TEST_DATA_PATH)


def main() -> None:
    df = pd.read_csv(config.FEATURES_DATA_PATH)
    train_df, test_df = split_data(df)
    train_df.to_csv(config.TRAIN_DATA_PATH, index=False)
    test_df.to_csv(config.TEST_DATA_PATH, index=False)

    # Demonstracija: fit preprocessora na treningu i oblik rezultata
    X_train, y_train = get_X_y(train_df)
    pre = build_preprocessor()
    X_mat = pre.fit_transform(X_train, target_transform(y_train))

    print("=== Pretprocesiranje ===")
    print(f"Train: {train_df.shape[0]} redova | Test: {test_df.shape[0]} redova")
    print(f"Ciljna promenljiva: {TARGET} (uči se kao log1p, metrike u $)")
    print(f"Numeričke ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"OneHot ({len(ONEHOT_FEATURES)}): {ONEHOT_FEATURES}")
    print(f"TargetEncoder ({len(TARGET_ENCODED_FEATURES)}): {TARGET_ENCODED_FEATURES}")
    print(f"Matrica posle transformacije: {X_mat.shape[0]} x {X_mat.shape[1]} kolona")
    names = pre.get_feature_names_out()
    print(f"Primer kolona: {list(names[:6])} ... {list(names[-3:])}")
    print(f"\nSačuvano: {config.TRAIN_DATA_PATH}\n          {config.TEST_DATA_PATH}")


if __name__ == "__main__":
    main()
