"""
Poređenje modela na ISTOM train/test skupu (data/processed/train.csv, test.csv).

Ortodoksni kandidati (sklearn, isti preprocessor iz data_preprocessing.py):
    LinearRegression, Ridge, DecisionTreeRegressor, RandomForestRegressor,
    HistGradientBoostingRegressor - svaki u log-skali cilja, a dva i u sirovoj
    skali da se vidi koliko log-transformacija vredi.

Neortodoksni kandidati (src/baselines.py):
    GroupMedian (lookup bez ML-a), Formula (osnovna cena + retka formula),
    Symbolic GP (gplearn genetsko programiranje).

Plus: 5-fold cross-validation na treningu za tri najbolja modela (stabilnost MAE)
i intervalna "od-do" predikcija (kvantilni gradient boosting, 10. i 90. percentil).

Izlazi (reports/):
    model_comparison.csv / .md  - tabela metrika
    cv_results.csv              - CV MAE za finaliste
    interval_metrics.json       - pokrivenost i širina intervala
    formulas.txt                - formule koje su modeli izveli
    figures/model_comparison_mae.png

Pokretanje:  python src/model_comparison.py            (sve, ~5 min na 2 jezgra)
             python src/model_comparison.py --fast     (bez GP i CV, ~1 min)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.baselines import FormulaRegressor, GroupMedianRegressor, SymbolicGPRegressor  # noqa: E402
from src.data_preprocessing import (build_preprocessor, get_X_y, load_train_test,  # noqa: E402
                                    target_inverse_transform, target_transform)

warnings.filterwarnings("ignore", category=FutureWarning)
N_JOBS = 2

# Hiperparametri finalnog boosting modela (izabrani 3-fold CV pretragom, vidi README)
HGB_PARAMS = dict(max_iter=800, learning_rate=0.05, max_leaf_nodes=127, min_samples_leaf=10,
                  l2_regularization=1.0, max_features=0.7, random_state=config.RANDOM_STATE)
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=1, max_features=0.5,
                 n_jobs=N_JOBS, random_state=config.RANDOM_STATE)
# Kvantilni modeli za interval su plići od tačkastog modela: duboki kvantilni boosting
# daje preuzak interval (pokrivenost 72 % umesto 80 %); ovo daje ~77 % uz 10 % užu
# traku nego još plići model (vidi README, odeljak "Interval")
INTERVAL_PARAMS = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
                       l2_regularization=1.0, random_state=config.RANDOM_STATE)


def sklearn_pipeline(estimator, log_target: bool):
    """Preprocessor + model; po potrebi uči log1p(cena) i vraća $ na predikciji."""
    pipe = Pipeline([("pre", build_preprocessor()), ("model", estimator)])
    if not log_target:
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=target_transform,
                                      inverse_func=target_inverse_transform)


def candidate_models(include_gp: bool = True) -> dict[str, tuple[object, str]]:
    """Naziv -> (model, skala cilja). Redosled = redosled u tabeli."""
    models = {
        "GroupMedian lookup (bez ML)": (GroupMedianRegressor(), "raw"),
        "LinearRegression": (sklearn_pipeline(LinearRegression(), False), "raw"),
        "Ridge": (sklearn_pipeline(Ridge(alpha=1.0), True), "log"),
        "DecisionTree": (sklearn_pipeline(DecisionTreeRegressor(min_samples_leaf=20,
                                                                random_state=config.RANDOM_STATE), True), "log"),
        "RandomForest": (sklearn_pipeline(RandomForestRegressor(**RF_PARAMS), True), "log"),
        "HistGradientBoosting (raw $)": (sklearn_pipeline(HistGradientBoostingRegressor(**HGB_PARAMS), False), "raw"),
        "HistGradientBoosting": (sklearn_pipeline(HistGradientBoostingRegressor(**HGB_PARAMS), True), "log"),
        "Formula: base + 4 člana": (FormulaRegressor(max_terms=4), "log"),
    }
    if include_gp:
        models["Symbolic GP (gplearn)"] = (SymbolicGPRegressor(n_jobs=N_JOBS), "log")
    return models


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    """MAE, MSE, RMSE, R² (u dolarima) + medijana relativne greške."""
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "R2": r2_score(y_true, y_pred),
        "MedAPE": float(np.median(np.abs(y_true - y_pred) / y_true)),
    }


def evaluate_candidates(models, X_train, y_train, X_test, y_test) -> tuple[pd.DataFrame, dict, dict]:
    """Trenira svaki model na treningu, meri na testu. Vraća tabelu, fitovane modele, formule."""
    rows, fitted, formulas = [], {}, {}
    for name, (model, scale) in models.items():
        t0 = time.time()
        try:
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
        except ImportError as exc:  # npr. gplearn nije instaliran
            print(f"  {name}: preskočeno ({exc})")
            continue
        metrics = regression_metrics(y_test, pred)
        rows.append({"model": name, "target": scale, **metrics, "train_time_s": time.time() - t0})
        fitted[name] = model
        if hasattr(model, "formula_"):
            formulas[name] = model.formula_
        print(f"  {name:32s} MAE={metrics['MAE']:7.0f}  RMSE={metrics['RMSE']:7.0f}  "
              f"R2={metrics['R2']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows), fitted, formulas


def cross_validate_finalists(table: pd.DataFrame, X_train, y_train, n_finalists: int = 3) -> pd.DataFrame:
    """5-fold CV na treningu: da li je razlika između najboljih veća od šuma?"""
    finalists = (table[~table["model"].str.contains("GP")]
                 .sort_values("MAE").head(n_finalists)["model"].tolist())
    models = candidate_models(include_gp=False)
    cv = KFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    rows = []
    for name in finalists:
        model, _ = models[name]
        t0 = time.time()
        scores = -cross_val_score(model, X_train, y_train, cv=cv,
                                  scoring="neg_mean_absolute_error", n_jobs=1)
        rows.append({"model": name, "cv_MAE_mean": scores.mean(), "cv_MAE_std": scores.std(),
                     "folds": " | ".join(f"{s:.0f}" for s in scores)})
        print(f"  {name:32s} CV MAE = {scores.mean():.0f} ± {scores.std():.0f}  ({time.time() - t0:.0f}s)", flush=True)
    return pd.DataFrame(rows)


def interval_models():
    """Kvantilni boosting za donju (10 %) i gornju (90 %) granicu cene."""
    low = sklearn_pipeline(HistGradientBoostingRegressor(loss="quantile", quantile=0.1, **INTERVAL_PARAMS), True)
    high = sklearn_pipeline(HistGradientBoostingRegressor(loss="quantile", quantile=0.9, **INTERVAL_PARAMS), True)
    return low, high


def evaluate_interval(X_train, y_train, X_test, y_test) -> tuple[dict, object, object]:
    """Pokrivenost (koliko % stvarnih cena upadne u interval) i širina intervala."""
    low, high = interval_models()
    low.fit(X_train, y_train)
    high.fit(X_train, y_train)
    lo, hi = low.predict(X_test), high.predict(X_test)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    y = np.asarray(y_test, dtype=float)
    inside = (y >= lo) & (y <= hi)
    metrics = {
        "coverage": float(inside.mean()),
        "target_coverage": 0.8,
        "mean_width_usd": float(np.mean(hi - lo)),
        "median_width_usd": float(np.median(hi - lo)),
        "median_relative_width": float(np.median((hi - lo) / np.maximum(lo, 1))),
    }
    print(f"  interval 10-90 %: pokrivenost {metrics['coverage']:.1%} (cilj 80 %), "
          f"medijana širine {metrics['median_width_usd']:.0f} $ "
          f"({metrics['median_relative_width']:.0%} donje granice)")
    return metrics, low, high


def plot_mae(table: pd.DataFrame, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = table.sort_values("MAE", ascending=False)
    colors = ["#c0392b" if "GP" in m or "bez ML" in m or "Formula" in m else "#2c3e50" for m in t["model"]]
    fig, ax = plt.subplots(figsize=(9, 0.5 * len(t) + 1.5))
    ax.barh(t["model"], t["MAE"], color=colors)
    for i, v in enumerate(t["MAE"]):
        ax.text(v + 30, i, f"{v:.0f} $", va="center", fontsize=9)
    ax.set_xlabel("MAE na test skupu [$]  (manje je bolje)")
    ax.set_title("Poređenje modela - crveno: neortodoksni, tamno: klasični sklearn")
    ax.set_xlim(0, t["MAE"].max() * 1.18)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Poređenje regresionih modela")
    parser.add_argument("--fast", action="store_true", help="bez genetskog programiranja i bez CV")
    args = parser.parse_args()

    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    train_df, test_df = load_train_test()
    X_train, y_train = get_X_y(train_df)
    X_test, y_test = get_X_y(test_df)
    print(f"Train {len(X_train)} | Test {len(X_test)} redova\n")

    print("=== Test skup ===")
    table, fitted, formulas = evaluate_candidates(candidate_models(include_gp=not args.fast),
                                                  X_train, y_train, X_test, y_test)
    table_sorted = table.sort_values("MAE").reset_index(drop=True)
    table_sorted.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    md = table_sorted.copy()
    for col in ["MAE", "RMSE"]:
        md[col] = md[col].map(lambda v: f"{v:,.0f}")
    md["MSE"] = md["MSE"].map(lambda v: f"{v:,.3e}")
    md["R2"] = md["R2"].map(lambda v: f"{v:.3f}")
    md["MedAPE"] = md["MedAPE"].map(lambda v: f"{v:.1%}")
    md["train_time_s"] = md["train_time_s"].map(lambda v: f"{v:.0f}")
    (config.REPORTS_DIR / "model_comparison.md").write_text(md.to_markdown(index=False), encoding="utf-8")
    plot_mae(table_sorted, config.FIGURES_DIR / "model_comparison_mae.png")

    if formulas:
        text = "\n\n".join(f"{name}:\n{formula}" for name, formula in formulas.items())
        (config.REPORTS_DIR / "formulas.txt").write_text(text, encoding="utf-8")
        print("\n=== Izvedene formule ===")
        print(text)

    if not args.fast:
        print("\n=== 5-fold CV na treningu (finalisti) ===")
        cv_table = cross_validate_finalists(table, X_train, y_train)
        cv_table.to_csv(config.REPORTS_DIR / "cv_results.csv", index=False)

    print("\n=== Intervalna predikcija (kvantilni boosting) ===")
    interval, _, _ = evaluate_interval(X_train, y_train, X_test, y_test)
    with open(config.REPORTS_DIR / "interval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(interval, f, indent=2)

    best = table_sorted.iloc[0]
    print(f"\nNajbolji po MAE na testu: {best['model']} (MAE {best['MAE']:.0f} $, R² {best['R2']:.3f})")
    print(f"Tabela: {config.REPORTS_DIR / 'model_comparison.md'}")


if __name__ == "__main__":
    main()
