"""Train a new random-forest valuation model from dataset.json.

Usage (from the project folder, next to app.py):

    python train_model.py
    python train_model.py --trees 500 --out random_forest_autovalue_model.pkl

The data is cleaned with the SAME code the web app uses (services/data_service.py),
so the values the dropdowns send at prediction time match what the model was
trained on. The saved file is a scikit-learn Pipeline (preprocessing + forest)
with exactly the input columns app.py sends, so it drops straight in.

Note: this is a NEW model. Its numbers will not match your original notebook model.
"""

import argparse
import shutil
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from services.data_service import NO_VARIANT, VehicleData
from services.model_service import FEATURE_COLUMNS

BASE_DIR = Path(__file__).resolve().parent

CATEGORICAL = ['Make / Brand', 'Body Type', 'Model', 'Variant / Trim', 'Transmission', 'Fuel Type']
NUMERIC = ['Year', 'Mileage (km)', 'Engine / Displacement']
TARGET = 'Listing / Bid Price'


def load_training_frame(dataset_path):
    """Cleaned, model-ready rows: the 9 model inputs plus the price."""
    df = VehicleData(dataset_path).df
    total = len(df)

    # Each model column maps to a cleaned dataset column (see FEATURE_COLUMNS).
    frame = pd.DataFrame({col: df[key] for col, key in FEATURE_COLUMNS.items()})
    frame['Variant / Trim'] = frame['Variant / Trim'].replace(NO_VARIANT, '').fillna('')  # blank trim = ''
    frame[TARGET] = df['price']

    # Rows with a missing input or price can't be used (e.g. mileage "N/A", EV engine size).
    frame = frame.dropna()
    frame = frame[frame[TARGET] > 0]
    after_missing = len(frame)

    # Identical listings (same car, same price) appear several times; keeping them
    # would leak copies of test rows into the training set and inflate the score.
    frame = frame.drop_duplicates().reset_index(drop=True)

    print(f'Rows in dataset:            {total}')
    print(f'Usable (no missing values): {after_missing}')
    print(f'After removing duplicates:  {len(frame)}')
    return frame


def build_pipeline(n_trees, seed):
    preprocessor = ColumnTransformer(
        [('categories', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL)],
        remainder='passthrough',  # Year, Mileage, Engine go through unchanged
    )
    forest = RandomForestRegressor(
        n_estimators=n_trees,
        min_samples_leaf=2,
        max_features=0.5,
        n_jobs=-1,
        random_state=seed,
    )
    return Pipeline([('preprocessor', preprocessor), ('model', forest)])


def report(label, y_true, y_pred):
    mape = np.median(np.abs(y_true - y_pred) / y_true) * 100
    print(
        f'{label:<22} R²={r2_score(y_true, y_pred):.3f}   '
        f'MAE=₱{mean_absolute_error(y_true, y_pred):,.0f}   '
        f'median error=₱{median_absolute_error(y_true, y_pred):,.0f} ({mape:.1f}%)'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=BASE_DIR / 'dataset.json', type=Path)
    parser.add_argument('--out', default=BASE_DIR / 'random_forest_autovalue_model.pkl', type=Path)
    parser.add_argument('--trees', default=300, type=int, help='number of trees (default 300)')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f'Dataset not found: {args.data}')

    frame = load_training_frame(args.data)
    X = frame[CATEGORICAL + NUMERIC]
    y = frame[TARGET]

    # --- Honest accuracy estimates -------------------------------------------------
    print('\nEvaluating (this model is NOT used for the final file)...')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.seed)
    check = build_pipeline(args.trees, args.seed).fit(X_train, y_train)
    report('Hold-out (20%)', y_test, check.predict(X_test))

    cv_pred = cross_val_predict(
        build_pipeline(args.trees, args.seed), X, y,
        cv=KFold(5, shuffle=True, random_state=args.seed),
    )
    report('5-fold cross-validation', y, cv_pred)

    # --- Final model: train on ALL rows --------------------------------------------
    print('\nTraining final model on all rows...')
    pipeline = build_pipeline(args.trees, args.seed).fit(X, y)

    if args.out.exists():
        backup = args.out.with_suffix(args.out.suffix + '.bak')
        shutil.move(args.out, backup)
        print(f'Existing file moved to {backup.name}')

    joblib.dump(pipeline, args.out, compress=3)
    size_mb = args.out.stat().st_size / 1_000_000

    # Reload + predict, exactly like app.py does, to prove the file works.
    reloaded = joblib.load(args.out)
    sample = X.iloc[[0]]
    price = float(reloaded.predict(sample)[0])
    print(f'\n✅ Saved {args.out.name} ({size_mb:.1f} MB), scikit-learn {sklearn.__version__}')
    print(f'   Sanity check: {sample.iloc[0]["Year"]:.0f} {sample.iloc[0]["Make / Brand"]} '
          f'{sample.iloc[0]["Model"]} -> predicted ₱{price:,.0f} (listed ₱{y.iloc[0]:,.0f})')
    print('\nNext: run `python app.py`. Keep the same scikit-learn version when deploying.')


if __name__ == '__main__':
    main()