"""
Neortodoksni modeli koji ulaze u poređenje pored klasičnih sklearn regresora.
Svi prate sklearn API (fit/predict nad DataFrame-om iz get_X_y), pa prolaze
kroz isti kod za evaluaciju kao i ostali.

1. GroupMedianRegressor - "procenitelj bez mašinskog učenja".
   Predviđanje = medijana cene istih vozila iz TRENING skupa, hijerarhijski:
   (make, model, year) -> (make, model) -> (make) -> globalna medijana.
   To je ono što radi iskusan prodavac: pogleda šta koštaju isti takvi. Nema ni
   jedan parametar naučen optimizacijom. Svaki ML model mora ovo da pobedi,
   inače nema razloga da postoji.

2. FormulaRegressor - "cena = osnovna cena modela x formula amortizacije".
   log(cena) = base(marka_model) + f(starost, km, litri, oštećen)
   - base  : tabela od ~1.000 brojeva (TargetEncoder nad log-cenom očišćenom od
             globalnog trenda starosti, sa cross-fitting-om => bez curenja cilja)
   - f     : retka linearna kombinacija nad bibliotekom kandidat-članova
             (age, age², log(age+1), km, km², log(km+1), liters, log(liters+1),
             damaged, age*km, km/(age+1)...). Lasso bira članove, a onda se
             koeficijenti ponovo procene običnom regresijom (bez Lasso pristrasnosti).
             To je ideja "sparse identification" (SINDy): mašina BIRA koji članovi
             ulaze u formulu, čovek može da je pročita. Rezultat je jedna jednačina
             sa par članova, npr. -0.10*age => svaka godina starosti skida ~10 % cene.

3. SymbolicGPRegressor - ista dekompozicija (base + f), ali f traži genetsko
   programiranje (gplearn). Klasična "simbolička regresija": evolucija nad
   stablima izraza. Uključen je zato što je najneortodoksniji pristup, a rezultat
   (dobar ili loš) se iskreno prikazuje u tabeli poređenja. Podešavanja su
   rezultat probanja: sa log/sqrt u skupu funkcija ili jačom parsimony kaznom
   evolucija se "sruši" u konstantu; add/sub/mul/div + blaga kazna daje
   upotrebljivu formulu (vidi README). Traje ~3 min na 2 jezgra.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression, lasso_path
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler, TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import config  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Lookup bez ML-a
# ---------------------------------------------------------------------------
class GroupMedianRegressor(BaseEstimator, RegressorMixin):
    """Hijerarhijska medijana po (make, model, year) -> (make, model) -> make -> sve."""

    levels = [["make", "model", "year"], ["make", "model"], ["make"]]

    def fit(self, X: pd.DataFrame, y):
        df = X[["make", "model", "year"]].copy()
        df["_y"] = np.asarray(y, dtype=float)
        self.tables_ = [df.groupby(keys)["_y"].median() for keys in self.levels]
        self.global_median_ = float(np.median(df["_y"]))
        return self

    @staticmethod
    def _lookup(X: pd.DataFrame, keys: list[str], table: pd.Series) -> np.ndarray:
        idx = pd.MultiIndex.from_frame(X[keys]) if len(keys) > 1 else X[keys[0]]
        return table.reindex(idx).to_numpy(dtype=float)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        pred = np.full(len(X), np.nan)
        for keys, table in zip(self.levels, self.tables_):
            missing = np.isnan(pred)
            if not missing.any():
                break
            pred[missing] = self._lookup(X[missing], keys, table)
        pred[np.isnan(pred)] = self.global_median_
        return pred


# ---------------------------------------------------------------------------
# Zajednička osnova za oba "formula" modela: base(marka_model) + f(ulazi)
# ---------------------------------------------------------------------------
class _BasePlusFormula(BaseEstimator, RegressorMixin):
    """
    Uči: log1p(cena) = base(brand_model) + f(age, km_100k, liters, damaged).

    base se računa kao TargetEncoder nad reziduom log-cene posle globalnog trenda
    starosti (medijana log-cene po starosti, izglađena), tako da "osnovna cena"
    modela ne zavisi od toga koliko su stari primerci u skupu. f onda uči i sam
    trend starosti i sve korekcije. Potklase definišu kako se f uči.
    """

    input_names = ["age", "km_100k", "liters", "damaged"]

    def _inputs(self, X: pd.DataFrame) -> np.ndarray:
        km = X["mileage_km"].fillna(self.median_km_).to_numpy(dtype=float) / 100_000.0
        liters = X["volume_cm3"].fillna(self.median_vol_).to_numpy(dtype=float) / 1000.0
        damaged = (X["condition"] == "with damage").to_numpy().astype(float)
        age = X["car_age"].to_numpy(dtype=float)
        return np.column_stack([age, km, liters, damaged])

    def _age_trend(self, age: np.ndarray) -> np.ndarray:
        clipped = np.clip(age, self.trend_.index.min(), self.trend_.index.max())
        return self.trend_.reindex(clipped).to_numpy(dtype=float)

    def _fit_base(self, X: pd.DataFrame, y_log: np.ndarray) -> np.ndarray:
        self.median_km_ = float(X["mileage_km"].median())
        self.median_vol_ = float(X["volume_cm3"].median())
        age = X["car_age"].to_numpy(dtype=float)
        self.trend_ = (pd.Series(y_log).groupby(age).median()
                       .rolling(3, center=True, min_periods=1).mean())
        self.encoder_ = TargetEncoder(target_type="continuous", smooth="auto",
                                      cv=KFold(5, shuffle=True, random_state=config.RANDOM_STATE))
        return self.encoder_.fit_transform(X[["brand_model"]], y_log - self._age_trend(age)).ravel()

    def _base(self, X: pd.DataFrame) -> np.ndarray:
        return self.encoder_.transform(X[["brand_model"]]).ravel()

    def fit(self, X: pd.DataFrame, y):
        y_log = np.log1p(np.asarray(y, dtype=float))
        base = self._fit_base(X, y_log)
        self._fit_formula(self._inputs(X), y_log - base)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        y_log = self._base(X) + self._predict_formula(self._inputs(X))
        return np.expm1(np.clip(y_log, 0.0, 14.0))


# ---------------------------------------------------------------------------
# 2. Retka formula nad bibliotekom članova
# ---------------------------------------------------------------------------
class FormulaRegressor(_BasePlusFormula):
    """Lasso bira do `max_terms` članova iz biblioteke, OLS procenjuje koeficijente."""

    def __init__(self, max_terms: int = 4):
        self.max_terms = max_terms

    @staticmethod
    def _library(Z: np.ndarray) -> pd.DataFrame:
        age, km, lit, dmg = Z.T
        return pd.DataFrame({
            "age": age, "age^2": age ** 2, "sqrt(age)": np.sqrt(age), "log(age+1)": np.log1p(age),
            "km": km, "km^2": km ** 2, "log(km+1)": np.log1p(km),
            "liters": lit, "liters^2": lit ** 2, "log(liters+1)": np.log1p(lit),
            "damaged": dmg, "age*km": age * km, "age*liters": age * lit, "km/(age+1)": km / (age + 1),
        })

    def _fit_formula(self, Z: np.ndarray, target: np.ndarray) -> None:
        lib = self._library(Z)
        scaler = StandardScaler().fit(lib)
        alphas, coefs, _ = lasso_path(scaler.transform(lib), target, n_alphas=100)
        selected = None
        for coef_vec in coefs.T:                      # od najjače regularizacije ka slabijoj
            active = np.abs(coef_vec) > 1e-8
            if active.sum() <= self.max_terms:
                selected = active
        self.terms_ = list(lib.columns[selected])
        self.ols_ = LinearRegression().fit(lib[self.terms_], target)
        self.coefficients_ = dict(zip(self.terms_, self.ols_.coef_))
        self.intercept_ = float(self.ols_.intercept_)
        parts = [f"{self.intercept_:+.3f}"] + [f"{c:+.4f}*{t}" for t, c in self.coefficients_.items()]
        self.formula_ = "log(price) = base(brand_model) " + " ".join(parts)

    def _predict_formula(self, Z: np.ndarray) -> np.ndarray:
        return self.ols_.predict(self._library(Z)[self.terms_])


# ---------------------------------------------------------------------------
# 3. Genetsko programiranje (gplearn)
# ---------------------------------------------------------------------------
class SymbolicGPRegressor(_BasePlusFormula):
    """f traži gplearn.SymbolicRegressor nad 4 sirova ulaza; formula u `formula_`."""

    def __init__(self, population_size: int = 4000, generations: int = 50,
                 parsimony_coefficient: float = 0.0005, max_train_rows: int = 15000,
                 random_state: int = config.RANDOM_STATE, n_jobs: int = 1):
        self.population_size = population_size
        self.generations = generations
        self.parsimony_coefficient = parsimony_coefficient
        self.max_train_rows = max_train_rows
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _fit_formula(self, Z: np.ndarray, target: np.ndarray) -> None:
        from gplearn.genetic import SymbolicRegressor

        rng = np.random.RandomState(self.random_state)
        if len(Z) > self.max_train_rows:             # GP je spor; uzorak je dovoljan
            idx = rng.choice(len(Z), self.max_train_rows, replace=False)
            Z, target = Z[idx], target[idx]
        self.gp_ = SymbolicRegressor(
            population_size=self.population_size, generations=self.generations,
            function_set=("add", "sub", "mul", "div"),
            parsimony_coefficient=self.parsimony_coefficient, metric="mean absolute error",
            init_depth=(2, 4), const_range=(-2.0, 2.0), p_crossover=0.6,
            p_subtree_mutation=0.15, p_hoist_mutation=0.1, p_point_mutation=0.1,
            max_samples=0.8, feature_names=self.input_names,
            random_state=self.random_state, n_jobs=self.n_jobs, verbose=0,
        )
        self.gp_.fit(Z, target)
        self.formula_ = "log(price) = base(brand_model) + " + str(self.gp_._program)

    def _predict_formula(self, Z: np.ndarray) -> np.ndarray:
        return self.gp_.predict(Z)
