"""Model performance & accuracy evaluation.

Produces, in the ./evaluation folder:
  table1_model_comparison.csv / .md   RMSE, MAE, R² for each model (your "Table 1")
  fig1_gradient_boosting.png          Actual vs. predicted + residual distribution
  fig2_random_forest.png
  fig3_ridge_regression.png
  feature_importance.png / .csv       Which inputs drive the price predictions
  metrics.json                        Everything above in machine-readable form

Usage (from the project folder):

    python evaluate_models.py
    python evaluate_models.py --max-price 3000000          # ignore extreme outliers
    python evaluate_models.py --old-model old_model.pkl    # also score an older .pkl
    python evaluate_models.py --seed 7 --trees 500

All models use the same cleaned data, the same input columns and the SAME
train/test split, so the comparison is fair. Metrics are computed on the 20% of
rows the models never saw during training.
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use('Agg')  # draw to files, no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from train_model import CATEGORICAL, NUMERIC, TARGET, load_training_frame

BASE_DIR = Path(__file__).resolve().parent

# Results reported in your paper (edit if they change). They are printed next to
# this run's numbers so you can see how the new run compares with the old one.
PAPER_REPORTED = {
    'Ridge Regression':        {'RMSE': 176_082.13, 'MAE': 106_093.34, 'R2': 0.8804},
    'Random Forest Regressor': {'RMSE': 175_865.18, 'MAE': 109_715.27, 'R2': 0.8807},
    'Gradient Boosting':       {'RMSE': 214_236.32, 'MAE': 140_092.29, 'R2': 0.8229},
}

# Which figure each model gets (matches the numbering in the paper).
FIGURE_FILES = {
    'Gradient Boosting':       'fig1_gradient_boosting.png',
    'Random Forest Regressor': 'fig2_random_forest.png',
    'Ridge Regression':        'fig3_ridge_regression.png',
}


def peso(value, decimals=2):
    return f'₱{value:,.{decimals}f}'


# ----------------------------------------------------------------------------- models

def build_models(n_trees, seed):
    """The three algorithms from the paper. Each is a full pipeline: raw columns in, price out."""
    one_hot = lambda: OneHotEncoder(handle_unknown='ignore')

    # Ridge is linear and scale-sensitive, so numeric columns are standardised.
    ridge_prep = ColumnTransformer(
        [('categories', one_hot(), CATEGORICAL), ('numbers', StandardScaler(), NUMERIC)]
    )
    # Tree models don't need scaling; numeric columns pass straight through.
    tree_prep = lambda: ColumnTransformer(
        [('categories', one_hot(), CATEGORICAL)], remainder='passthrough'
    )

    return {
        'Ridge Regression': Pipeline([
            ('preprocessor', ridge_prep), ('model', Ridge(alpha=1.0)),
        ]),
        'Random Forest Regressor': Pipeline([
            ('preprocessor', tree_prep()),
            ('model', RandomForestRegressor(
                n_estimators=n_trees, min_samples_leaf=2, max_features=0.5,
                n_jobs=-1, random_state=seed)),
        ]),
        'Gradient Boosting': Pipeline([
            ('preprocessor', tree_prep()),
            ('model', GradientBoostingRegressor(random_state=seed)),
        ]),
    }


def score(y_true, y_pred):
    return {
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAE': float(mean_absolute_error(y_true, y_pred)),
        'R2': float(r2_score(y_true, y_pred)),
    }


# ----------------------------------------------------------------------------- plots

def plot_actual_vs_predicted(name, y_true, y_pred, path):
    """Figure style of the paper: parity plot on the left, residual distribution on the right."""
    millions = FuncFormatter(lambda v, _: f'{v / 1e6:.1f}M')
    residuals = y_true - y_pred

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    ax1.scatter(y_true, y_pred, s=18, alpha=0.55, color='#C2785C', edgecolor='none')
    lim = [0, max(y_true.max(), y_pred.max()) * 1.03]
    ax1.plot(lim, lim, '--', color='#0F3040', linewidth=1.5, label='Perfect prediction (y = x)')
    ax1.set(xlim=lim, ylim=lim, xlabel='Actual price (PHP)', ylabel='Predicted price (PHP)',
            title='Actual vs. Predicted')
    ax1.xaxis.set_major_formatter(millions)
    ax1.yaxis.set_major_formatter(millions)
    ax1.legend(loc='upper left')
    ax1.grid(alpha=0.25)

    ax2.hist(residuals, bins=40, color='#0F3040', alpha=0.85, edgecolor='white')
    ax2.axvline(0, color='#C2785C', linestyle='--', linewidth=1.5)
    ax2.set(xlabel='Residual = actual − predicted (PHP)', ylabel='Number of cars',
            title='Residual Distribution')
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v / 1e3:,.0f}k'))
    ax2.grid(alpha=0.25)

    fig.suptitle(f'{name}: Actual vs. Predicted Car Prices and Residual Distribution',
                 fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feature_importance(importance, path):
    ordered = importance.sort_values('importance')
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(ordered['feature'], ordered['importance'], xerr=ordered['std'],
            color='#C2785C', ecolor='#0F3040', capsize=3)
    ax.set(xlabel='Permutation importance (drop in R² when the feature is shuffled)',
           title='Feature Importance – Random Forest')
    ax.grid(axis='x', alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', default=BASE_DIR / 'dataset.json', type=Path)
    parser.add_argument('--out-dir', default=BASE_DIR / 'evaluation', type=Path)
    parser.add_argument('--old-model', type=Path,
                        help='an older .pkl pipeline to score on the same test rows')
    parser.add_argument('--trees', default=300, type=int)
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--test-size', default=0.2, type=float)
    parser.add_argument('--max-price', type=float,
                        help='drop rows priced above this (e.g. 3000000) to remove extreme outliers')
    args = parser.parse_args()

    if not args.data.exists():
        sys.exit(f'Dataset not found: {args.data}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_training_frame(args.data)
    if args.max_price:
        before = len(frame)
        frame = frame[frame[TARGET] <= args.max_price]
        print(f'Dropped {before - len(frame)} rows priced above {peso(args.max_price, 0)}')

    X, y = frame[CATEGORICAL + NUMERIC], frame[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed)
    print(f'\nTrain rows: {len(X_train)}   Test rows: {len(X_test)}   (seed {args.seed})\n')

    results, predictions, fitted = {}, {}, {}
    for name, pipeline in build_models(args.trees, args.seed).items():
        print(f'Training {name}...')
        pipeline.fit(X_train, y_train)
        predictions[name] = pipeline.predict(X_test)
        results[name] = score(y_test, predictions[name])
        fitted[name] = pipeline

    # Optional: an older saved model, scored on the exact same test rows.
    if args.old_model:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                old = joblib.load(args.old_model)
            old_pred = old.predict(X_test)
            results['Previous model (old .pkl)'] = score(y_test, old_pred)
            predictions['Previous model (old .pkl)'] = old_pred
            print('Scored the previous model. NOTE: it may have seen these test rows while '
                  'training, which would flatter its numbers.')
        except Exception as exc:
            print(f'⚠️  Could not load/score {args.old_model.name}: {type(exc).__name__}: {exc}')

    # ---- Table 1 -------------------------------------------------------------------
    table = pd.DataFrame(results).T[['RMSE', 'MAE', 'R2']]
    table.index.name = 'Model Algorithm'
    table.round(4).to_csv(args.out_dir / 'table1_model_comparison.csv')

    lines = ['| Model Algorithm | RMSE (PHP) | MAE (PHP) | R2 Score |', '|---|---|---|---|']
    for name, row in table.iterrows():
        lines.append(f'| {name} | {peso(row.RMSE)} | {peso(row.MAE)} | {row.R2:.4f} |')
    (args.out_dir / 'table1_model_comparison.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('\nTable 1: Performance Comparison of Regression Models (hold-out test set)')
    print('\n'.join(lines))

    # ---- New run vs. numbers reported in the paper ----------------------------------
    print('\nThis run vs. results reported in the paper')
    print(f'{"Model":<26}{"Metric":<7}{"Paper":>16}{"This run":>16}{"Change":>14}')
    for name, paper in PAPER_REPORTED.items():
        for metric in ('RMSE', 'MAE', 'R2'):
            old_v, new_v = paper[metric], results[name][metric]
            fmt = (lambda v: f'{v:.4f}') if metric == 'R2' else (lambda v: peso(v, 0))
            delta = f'{new_v - old_v:+.4f}' if metric == 'R2' else f'{new_v - old_v:+,.0f}'
            print(f'{name if metric == "RMSE" else "":<26}{metric:<7}{fmt(old_v):>16}{fmt(new_v):>16}{delta:>14}')

    # ---- Figures 1-3 ----------------------------------------------------------------
    for name, filename in FIGURE_FILES.items():
        plot_actual_vs_predicted(name, y_test.to_numpy(), predictions[name], args.out_dir / filename)

    # ---- Feature importance (permutation, on the test set) -------------------------------
    # Shuffling one ORIGINAL input column at a time and measuring how much R² drops
    # gives one clear number per feature (no one-hot fragments) and works for any model.
    print('\nComputing feature importance (Random Forest)...')
    perm = permutation_importance(
        fitted['Random Forest Regressor'], X_test, y_test,
        n_repeats=20, random_state=args.seed, scoring='r2', n_jobs=-1)
    importance = pd.DataFrame({
        'feature': X_test.columns, 'importance': perm.importances_mean, 'std': perm.importances_std,
    }).sort_values('importance', ascending=False)
    importance.round(5).to_csv(args.out_dir / 'feature_importance.csv', index=False)
    plot_feature_importance(importance, args.out_dir / 'feature_importance.png')
    print(importance.round(4).to_string(index=False))

    # ---- metrics.json ---------------------------------------------------------------
    (args.out_dir / 'metrics.json').write_text(json.dumps({
        'settings': {'seed': args.seed, 'trees': args.trees, 'test_size': args.test_size,
                     'max_price': args.max_price, 'train_rows': len(X_train), 'test_rows': len(X_test)},
        'metrics': results,
        'paper_reported': PAPER_REPORTED,
        'feature_importance': importance.round(5).to_dict(orient='records'),
    }, indent=2, ensure_ascii=False), encoding='utf-8')

    print(f'\n✅ Saved tables, figures and metrics.json to {args.out_dir}')


if __name__ == '__main__':
    main()
