"""
Centralna konfiguracija projekta.

Sve putanje, nazivi kolona i pragovi za čišćenje su na jednom mestu,
da bi svaka skripta (čišćenje, inženjering, pretprocesiranje, treniranje,
evaluacija, poređenje, predikcija) koristila identične vrednosti.
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Putanje
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_PATH = DATA_DIR / "cars.csv"
CLEAN_DATA_PATH = PROCESSED_DIR / "cars_clean.csv"
FEATURES_DATA_PATH = PROCESSED_DIR / "cars_features.csv"
TRAIN_DATA_PATH = PROCESSED_DIR / "train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "test.csv"

FINAL_MODEL_PATH = MODELS_DIR / "car_price_model.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "car_price_model_metadata.json"
INTERVAL_MODEL_LOW_PATH = MODELS_DIR / "car_price_interval_q10.joblib"
INTERVAL_MODEL_HIGH_PATH = MODELS_DIR / "car_price_interval_q90.joblib"

# ---------------------------------------------------------------------------
# Kolone
# ---------------------------------------------------------------------------
TARGET = "priceUSD"

# Originalni nazivi -> nazivi pogodni za kod (bez zagrada i jedinica)
COLUMN_RENAMES = {
    "mileage(kilometers)": "mileage_km",
    "volume(cm3)": "volume_cm3",
}

# Kategorijske kolone koje standardizujemo (mala slova, bez praznina)
CATEGORICAL_COLUMNS = [
    "make", "model", "condition", "fuel_type", "color",
    "transmission", "drive_unit", "segment",
]

# Kolone čije nedostajuće vrednosti popunjavamo iz istog make+model
GROUP_FILL_COLUMNS = ["segment", "drive_unit"]
UNKNOWN_LABEL = "unknown"

# ---------------------------------------------------------------------------
# Pragovi za čišćenje (obrazloženi u README i u data_cleaning.py)
# ---------------------------------------------------------------------------
MIN_YEAR = 1980           # ispod ovoga su oldtajmeri; cena im raste sa starošću
MIN_PRICE = 300           # ispod ovoga nije tržišna cena vozila nego otpad/placeholder
MAX_MILEAGE_KM = 1_000_000  # >= 1 milion km su placeholder vrednosti (1111111, 9999999...)
MIN_PLAUSIBLE_MILEAGE_KM = 1_000  # <1000 km na automobilu starom 2+ godine = nije uneto
PLACEHOLDER_SEQUENCES = ["12345", "123456", "1234567", "123123", "654321"]  # "kucano", ne izmereno
MAX_VOLUME_CM3 = 7_000    # iznad ovoga je tipfeler (x10), npr. 20000 umesto 2000
DROP_CONDITIONS = ["for parts"]      # tržište delova, ne tržište automobila
DROP_MAKES = ["eksklyuziv"]          # "ekskluziv": retro, vojna tehnika, karting, samogradnja

# ---------------------------------------------------------------------------
# Inženjering karakteristika
# ---------------------------------------------------------------------------
# Godina snimka podataka = najnovija godina proizvodnje u skupu (2019).
# Starost se računa od nje, a ne od današnje godine, jer su cene iz 2019.
REFERENCE_YEAR = 2019

# ---------------------------------------------------------------------------
# Podela i reproduktivnost
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
