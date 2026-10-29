#!/usr/bin/env bash
# Pokreće ceo tok od nule: čišćenje -> karakteristike -> podela -> poređenje -> trening -> evaluacija
# Upotreba:  bash run_all.sh          (sve, ~6 min na 2 jezgra)
#            bash run_all.sh --fast   (bez genetskog programiranja i CV, ~2 min)
set -e
cd "$(dirname "$0")"
python src/data_cleaning.py
python src/feature_engineering.py
python src/data_preprocessing.py
python src/model_comparison.py "$@"
python src/model_training.py
python src/model_evaluation.py
python src/predict.py --make volkswagen --model golf --year 2014 --condition "with mileage" \
    --mileage 180000 --fuel diesel --volume 1600 --color black --transmission mechanics \
    --drive "front-wheel drive" --segment C
