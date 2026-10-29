"""
Treniranje modela i čuvanje u models/.

Podrazumevano trenira FINALNI model (HistGradientBoosting u log-skali - izbor
obrazložen u README na osnovu model_comparison.py) i čuva:
    models/car_price_model.joblib            - ceo pipeline (preprocessor + model)
    models/car_price_model_metadata.json     - šta je unutra, kada i na čemu trenirano
    models/car_price_interval_q10.joblib     - donja granica intervala (10. percentil)
    models/car_price_interval_q90.joblib     - gornja granica intervala (90. percentil)
    models/car_price_formula.joblib          - čitljivi "formula" model (za objašnjenje)

Model se trenira SAMO na train skupu, da bi model_evaluation.py na test skupu
merio pošteno. Za produkciju bi se isti pipeline ponovo istrenirao na svim podacima.

Pokretanje:  python src/model_training.py                 (finalni model + interval + formula)
             python src/model_training.py --model ridge   (jednostavan osnovni model)
             python src/model_training.py --model rf      (random forest)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.baselines import FormulaRegressor  # noqa: E402
from src.data_preprocessing import (ALL_FEATURES, NUMERIC_FEATURES, ONEHOT_FEATURES,  # noqa: E402
                                    TARGET_ENCODED_FEATURES, get_X_y, load_train_test)
from src.model_comparison import candidate_models, interval_models  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_CHOICES = {
    "hgb": "HistGradientBoosting",
    "rf": "RandomForest",
    "ridge": "Ridge",
    "linear": "LinearRegression",
    "tree": "DecisionTree",
}


def train_and_save(model_key: str = "hgb", with_interval: bool = True) -> dict:
    train_df, _ = load_train_test()
    X_train, y_train = get_X_y(train_df)
    name = MODEL_CHOICES[model_key]
    model, scale = candidate_models(include_gp=False)[name]

    print(f"Treniram: {name} (cilj u {'log' if scale == 'log' else '$'} skali) na {len(X_train)} redova...")
    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, config.FINAL_MODEL_PATH)
    metadata = {
        "model_name": name,
        "target": config.TARGET,
        "target_scale": scale,
        "features": {"numeric": NUMERIC_FEATURES, "onehot": ONEHOT_FEATURES,
                     "target_encoded": TARGET_ENCODED_FEATURES, "all": ALL_FEATURES},
        "reference_year": config.REFERENCE_YEAR,
        "train_rows": int(len(X_train)),
        "train_time_s": round(train_time, 1),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "random_state": config.RANDOM_STATE,
    }
    with open(config.MODEL_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  sačuvano: {config.FINAL_MODEL_PATH}  ({train_time:.0f}s)")

    if with_interval:
        low, high = interval_models()
        low.fit(X_train, y_train)
        high.fit(X_train, y_train)
        joblib.dump(low, config.INTERVAL_MODEL_LOW_PATH)
        joblib.dump(high, config.INTERVAL_MODEL_HIGH_PATH)
        print(f"  sačuvano: {config.INTERVAL_MODEL_LOW_PATH.name}, {config.INTERVAL_MODEL_HIGH_PATH.name}")

        formula = FormulaRegressor(max_terms=4).fit(X_train, y_train)
        joblib.dump(formula, config.MODELS_DIR / "car_price_formula.joblib")
        print(f"  sačuvano: car_price_formula.joblib\n  {formula.formula_}")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Treniranje i čuvanje modela")
    parser.add_argument("--model", choices=MODEL_CHOICES, default="hgb",
                        help="koji model trenirati kao finalni (podrazumevano hgb)")
    parser.add_argument("--no-interval", action="store_true",
                        help="ne trenirati intervalne i formula modele")
    args = parser.parse_args()
    train_and_save(args.model, with_interval=not args.no_interval)


if __name__ == "__main__":
    main()
