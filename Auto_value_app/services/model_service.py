"""Wraps the trained random-forest pipeline (random_forest_autovalue_model.pkl).

The app starts even if the model cannot be loaded; /api/v1/predict then returns
a clear 503 explaining why, instead of the whole server crashing at import time.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# Column names the fitted ColumnTransformer expects (unchanged from the original app.py).
FEATURE_COLUMNS = {
    'Year': 'year',
    'Mileage (km)': 'mileage',
    'Engine / Displacement': 'engine_displacement',
    'Make / Brand': 'make',
    'Body Type': 'body_type',
    'Model': 'model',
    'Variant / Trim': 'variant',
    'Transmission': 'transmission',
    'Fuel Type': 'fuel_type',
}


class ModelService:
    def __init__(self, model_path):
        self.path = Path(model_path)
        self.pipeline = None
        self.error = None
        self._load()

    def _load(self):
        if not self.path.exists():
            self.error = f'Model file not found: {self.path.name}'
            log.warning('⚠️  %s', self.error)
            return
        try:
            import joblib

            with warnings.catch_warnings():
                warnings.simplefilter('default')  # surface sklearn version warnings
                self.pipeline = joblib.load(self.path)
            log.info('✅ Loaded ML model: %s', self.path.name)
        except Exception as exc:  # corrupt file, version mismatch, missing dependency...
            self.error = (
                f'Model file {self.path.name} could not be loaded '
                f'({type(exc).__name__}: {exc}). The file may be corrupted or '
                'saved with a different scikit-learn version.'
            )
            log.error('❌ %s', self.error)

    @property
    def loaded(self):
        return self.pipeline is not None

    @staticmethod
    def _frame(vehicle):
        return pd.DataFrame([{
            column: vehicle[key] for column, key in FEATURE_COLUMNS.items()
        }])

    def predict(self, vehicle):
        """Return (predicted_price, (low, high) | None) for one vehicle dict."""
        frame = self._frame(vehicle)
        price = max(0.0, round(float(self.pipeline.predict(frame)[0]), 2))
        return price, self._range(frame, price)

    def _range(self, frame, point_estimate):
        """10th-90th percentile of the individual trees, when it is meaningful.

        Skipped (None) if the final step is not a forest, or if the tree average
        does not agree with the pipeline's own prediction (e.g. the target was
        log-transformed), so we never show a misleading range.
        """
        try:
            forest = self.pipeline[-1]
            if not hasattr(forest, 'estimators_'):
                return None
            features = self.pipeline[:-1].transform(frame)
            per_tree = np.array([t.predict(features)[0] for t in forest.estimators_])
            if point_estimate <= 0 or abs(per_tree.mean() - point_estimate) / point_estimate > 0.05:
                return None
            low, high = np.percentile(per_tree, [10, 90])
            return round(float(low), 2), round(float(high), 2)
        except Exception:
            log.exception('Could not compute prediction range')
            return None
