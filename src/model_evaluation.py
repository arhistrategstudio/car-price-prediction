"""
Evaluacija sačuvanog modela na test skupu + tumačenje rezultata.

Računa MAE, MSE, RMSE, R² (u dolarima), pa ih stavlja u kontekst:
- tabela primera (stvarna cena, predviđena cena, greška),
- najveće greške (koje automobile model najviše promašuje i zašto),
- MAE po cenovnom razredu (greška od 1.000 $ nije ista za auto od 2.000 $ i 40.000 $),
- MAE po starosti,
- pokrivenost "od-do" intervala ako postoje kvantilni modeli,
- permutaciona važnost ulaza (šta model zaista koristi).

Izlazi (reports/):
    evaluation_metrics.json, evaluation_examples.csv, evaluation_worst.csv,
    evaluation_by_price_band.csv, feature_importance.csv,
    figures/pred_vs_actual.png, figures/residuals.png,
    figures/mae_by_price_band.png, figures/feature_importance.png

Pokretanje:  python src/model_evaluation.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402
from src.data_preprocessing import ALL_FEATURES, get_X_y, load_train_test  # noqa: E402
from src.model_comparison import regression_metrics  # noqa: E402

warnings.filterwarnings("ignore", category=FutureWarning)

PRICE_BANDS = [0, 3000, 7000, 15000, 30000, np.inf]
PRICE_LABELS = ["< 3.000 $", "3.000-7.000 $", "7.000-15.000 $", "15.000-30.000 $", "> 30.000 $"]
AGE_BANDS = [-1, 3, 7, 12, 20, 100]
AGE_LABELS = ["0-3 god", "4-7 god", "8-12 god", "13-20 god", "> 20 god"]


def interpret(metrics: dict, y_test: pd.Series) -> str:
    """Tekstualno tumačenje metrika u kontekstu cene automobila."""
    median_price = float(np.median(y_test))
    return (
        f"MAE = {metrics['MAE']:,.0f} $: model u proseku promaši cenu za toliko dolara.\n"
        f"  Medijana cene u testu je {median_price:,.0f} $, pa je prosečna greška "
        f"{metrics['MAE'] / median_price:.0%} medijane cene.\n"
        f"MedAPE = {metrics['MedAPE']:.1%}: za polovinu automobila relativna greška je ispod toga.\n"
        f"RMSE = {metrics['RMSE']:,.0f} $ je veći od MAE jer kvadrira greške, pa ga vuku retki\n"
        f"  veliki promašaji (skupi i neobični automobili).\n"
        f"R² = {metrics['R2']:.3f}: model objašnjava {metrics['R2']:.0%} varijanse cene\n"
        f"  (0 = ne bolje od proseka, 1 = savršeno)."
    )


def examples_table(X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray, n: int = 12) -> pd.DataFrame:
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(X_test), n, replace=False)
    table = pd.DataFrame({
        "make": X_test["make"].to_numpy()[idx], "model": X_test["model"].to_numpy()[idx],
        "year": X_test["year"].to_numpy()[idx], "mileage_km": X_test["mileage_km"].to_numpy()[idx],
        "stvarna_cena": y_test.to_numpy()[idx], "predvidjena_cena": np.round(pred[idx]),
    })
    table["greska"] = np.round(np.abs(table["stvarna_cena"] - table["predvidjena_cena"]))
    table["greska_%"] = np.round(100 * table["greska"] / table["stvarna_cena"], 1)
    return table


def worst_errors(X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray, n: int = 8) -> pd.DataFrame:
    err = np.abs(y_test.to_numpy() - pred)
    idx = np.argsort(err)[::-1][:n]
    table = X_test.iloc[idx][["make", "model", "year", "condition", "mileage_km", "volume_cm3"]].copy()
    table["stvarna_cena"] = y_test.to_numpy()[idx]
    table["predvidjena_cena"] = np.round(pred[idx])
    table["greska"] = np.round(err[idx])
    return table.reset_index(drop=True)


def error_by_band(values, y_test, pred, bands, labels, name) -> pd.DataFrame:
    df = pd.DataFrame({"band": pd.cut(values, bands, labels=labels), "y": y_test, "p": pred})
    df["abs_err"] = np.abs(df["y"] - df["p"])
    df["rel_err"] = df["abs_err"] / df["y"]
    out = df.groupby("band", observed=True).agg(
        n=("y", "size"), MAE=("abs_err", "mean"), MedAPE=("rel_err", "median")).reset_index()
    out = out.rename(columns={"band": name})
    return out


def plots(y_test, pred, by_price: pd.DataFrame, importance: pd.DataFrame) -> None:
    y = np.asarray(y_test, dtype=float)
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y, pred, s=4, alpha=0.25, color="#2c3e50")
    lim = [300, max(y.max(), pred.max()) * 1.05]
    ax.plot(lim, lim, "r--", lw=1, label="savršeno")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Stvarna cena [$]"); ax.set_ylabel("Predviđena cena [$]")
    ax.set_title("Predviđeno vs stvarno (test skup, log-log)"); ax.legend()
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "pred_vs_actual.png", dpi=130); plt.close(fig)

    resid = y - pred
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(np.clip(resid, -10000, 10000), bins=80, color="#2c3e50")
    axes[0].set_title("Raspodela greške (stvarna - predviđena), isečeno na ±10.000 $")
    axes[0].set_xlabel("greška [$]")
    axes[1].scatter(pred, resid / y, s=4, alpha=0.25, color="#2c3e50")
    axes[1].axhline(0, color="r", lw=1); axes[1].set_xscale("log")
    axes[1].set_ylim(-1.5, 1.5); axes[1].set_xlabel("predviđena cena [$]"); axes[1].set_ylabel("relativna greška")
    axes[1].set_title("Relativna greška po predviđenoj ceni")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "residuals.png", dpi=130); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(by_price.iloc[:, 0].astype(str), by_price["MAE"], color="#2c3e50")
    ax1.set_ylabel("MAE [$]"); ax1.set_title("Greška po cenovnom razredu: apsolutna raste sa cenom, relativna je najmanja u sredini tržišta")
    ax2 = ax1.twinx()
    ax2.plot(by_price.iloc[:, 0].astype(str), 100 * by_price["MedAPE"], "o-", color="#c0392b")
    ax2.set_ylabel("MedAPE [%]", color="#c0392b")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "mae_by_price_band.png", dpi=130); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    imp = importance.sort_values("importance_MAE")
    ax.barh(imp["feature"], imp["importance_MAE"], color="#2c3e50")
    ax.set_xlabel("porast MAE [$] kada se kolona izmeša (permutaciona važnost)")
    ax.set_title("Koje ulaze model zaista koristi")
    fig.tight_layout(); fig.savefig(config.FIGURES_DIR / "feature_importance.png", dpi=130); plt.close(fig)


def feature_importance(model, X_test, y_test, n_rows: int = 3000) -> pd.DataFrame:
    """Permutaciona važnost na podskupu testa, merena porastom MAE u dolarima."""
    rng = np.random.RandomState(config.RANDOM_STATE)
    idx = rng.choice(len(X_test), min(n_rows, len(X_test)), replace=False)
    Xs, ys = X_test.iloc[idx], y_test.iloc[idx]
    result = permutation_importance(model, Xs, ys, scoring="neg_mean_absolute_error",
                                    n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=1)
    # permutation_importance meša i kolone koje pipeline ne koristi; prikazujemo samo ulaze modela
    table = pd.DataFrame({"feature": Xs.columns, "importance_MAE": result.importances_mean,
                          "std": result.importances_std})
    return table[table["feature"].isin(ALL_FEATURES)].sort_values("importance_MAE", ascending=False)


def main() -> None:
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    model = joblib.load(config.FINAL_MODEL_PATH)
    with open(config.MODEL_METADATA_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    _, test_df = load_train_test()
    X_test, y_test = get_X_y(test_df)
    pred = model.predict(X_test)

    metrics = regression_metrics(y_test, pred)
    print(f"=== Evaluacija: {meta['model_name']} na test skupu ({len(X_test)} redova) ===")
    for k in ["MAE", "MSE", "RMSE", "R2", "MedAPE"]:
        print(f"{k:7s} {metrics[k]:,.4f}" if k in ("R2", "MedAPE") else f"{k:7s} {metrics[k]:,.0f}")
    print("\n--- Tumačenje ---")
    print(interpret(metrics, y_test))

    examples = examples_table(X_test, y_test, pred)
    print("\n--- Primeri (nasumičnih 12) ---")
    print(examples.to_string(index=False))
    examples.to_csv(config.REPORTS_DIR / "evaluation_examples.csv", index=False)

    worst = worst_errors(X_test, y_test, pred)
    print("\n--- Najveće greške ---")
    print(worst.to_string(index=False))
    worst.to_csv(config.REPORTS_DIR / "evaluation_worst.csv", index=False)

    by_price = error_by_band(y_test, y_test.to_numpy(), pred, PRICE_BANDS, PRICE_LABELS, "cenovni_razred")
    by_age = error_by_band(X_test["car_age"], y_test.to_numpy(), pred, AGE_BANDS, AGE_LABELS, "starost")
    print("\n--- Greška po cenovnom razredu ---")
    print(by_price.to_string(index=False))
    print("\n--- Greška po starosti ---")
    print(by_age.to_string(index=False))
    by_price.to_csv(config.REPORTS_DIR / "evaluation_by_price_band.csv", index=False)
    by_age.to_csv(config.REPORTS_DIR / "evaluation_by_age_band.csv", index=False)

    interval = None
    if config.INTERVAL_MODEL_LOW_PATH.exists() and config.INTERVAL_MODEL_HIGH_PATH.exists():
        lo = joblib.load(config.INTERVAL_MODEL_LOW_PATH).predict(X_test)
        hi = joblib.load(config.INTERVAL_MODEL_HIGH_PATH).predict(X_test)
        lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
        y = y_test.to_numpy(dtype=float)
        interval = {"coverage": float(((y >= lo) & (y <= hi)).mean()),
                    "median_width_usd": float(np.median(hi - lo)),
                    "median_relative_width": float(np.median((hi - lo) / np.maximum(lo, 1)))}
        print(f"\n--- Interval 10-90 % ---\npokrivenost {interval['coverage']:.1%} (cilj 80 %), "
              f"medijana širine {interval['median_width_usd']:,.0f} $ "
              f"({interval['median_relative_width']:.0%} donje granice)")

    print("\n--- Permutaciona važnost (podskup 3.000 redova) ---")
    importance = feature_importance(model, X_test, y_test)
    print(importance.to_string(index=False))
    importance.to_csv(config.REPORTS_DIR / "feature_importance.csv", index=False)

    plots(y_test, pred, by_price, importance)
    out = {"model": meta["model_name"], "test_rows": int(len(X_test)), "metrics": metrics,
           "interval": interval, "median_test_price": float(np.median(y_test))}
    with open(config.REPORTS_DIR / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\nSačuvano u {config.REPORTS_DIR} (metrike, tabele, grafici u figures/)")


if __name__ == "__main__":
    main()
