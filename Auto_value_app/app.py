import logging
import os
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from services.data_service import LEVELS, VehicleData
from services.model_service import ModelService
from services.valuation import build_valuation

logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')

BASE_DIR = Path(__file__).resolve().parent
APP_NAME = os.getenv('APP_NAME', 'AutoValue PH')
DATASET_PATH = Path(os.getenv('DATASET_PATH', BASE_DIR / 'dataset.json'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', BASE_DIR / 'random_forest_autovalue_model.pkl'))

# Templates live in ./templates and assets in ./static (the old setup served the
# whole project folder as static files, which exposed app.py and the .pkl).
app = Flask(__name__)

vehicles = VehicleData(DATASET_PATH)
model = ModelService(MODEL_PATH)


def _error(message, status=400):
    return jsonify({'status': 'error', 'message': message}), status


# ---------------------------------------------------------------------- pages


@app.route('/')
def home():
    return render_template(
        'index.html',
        app_name=APP_NAME,
        years=vehicles.options('year', {}),
        regions=vehicles.regions(),
    )


# ------------------------------------------------------------------------ API


@app.get('/api/v1/options')
def options():
    """Next dropdown's values, given the choices made so far.

    GET /api/v1/options?level=make&year=2021
    """
    level = request.args.get('level', '')
    if level not in LEVELS:
        return _error(f'level must be one of: {", ".join(LEVELS)}')
    try:
        values = vehicles.options(level, request.args)
    except ValueError as exc:
        return _error(str(exc))
    return jsonify({'status': 'success', 'level': level, 'options': values})


def _parse_vehicle(data):
    """Validate the request body and return (vehicle_dict, error_message)."""
    vehicle, problems = {}, []

    for key in ('make', 'model', 'body_type', 'transmission', 'fuel_type'):
        vehicle[key] = str(data.get(key) or '').strip()
        if not vehicle[key]:
            problems.append(key.replace('_', ' '))
    vehicle['variant'] = str(data.get('variant') or '').strip()  # optional

    for key, cast, check in (
        ('year', int, lambda v: 1980 <= v <= 2100),
        ('mileage', float, lambda v: 0 <= v <= 2_000_000),
        ('engine_displacement', float, lambda v: 0 < v <= 20),
    ):
        try:
            vehicle[key] = cast(data.get(key))
            if not check(vehicle[key]):
                raise ValueError
        except (TypeError, ValueError):
            problems.append(key.replace('_', ' '))

    # Optional: the price the user is being asked / bidding. Used only for the verdict.
    vehicle['asking_price'] = None
    raw_price = data.get('asking_price')
    if raw_price not in (None, ''):
        try:
            vehicle['asking_price'] = float(raw_price)
            if not 0 < vehicle['asking_price'] <= 100_000_000:
                raise ValueError
        except (TypeError, ValueError):
            problems.append('asking price')

    if problems:
        return None, 'Missing or invalid: ' + ', '.join(problems) + '.'
    return vehicle, None


@app.post('/api/v1/predict')
def predict():
    if not model.loaded:
        return _error(f'Valuation model is unavailable. {model.error}', 503)

    data = request.get_json(silent=True)
    if not data:
        return _error('No JSON payload provided in request.')

    vehicle, problem = _parse_vehicle(data)
    if problem:
        return _error(problem)

    try:
        predicted, price_range = model.predict(vehicle)
    except Exception as exc:
        app.logger.exception('Prediction failed')
        return _error(f'Prediction failed: {exc}', 500)

    valuation = build_valuation(
        vehicles,
        vehicle,
        predicted,
        price_range,
        region=str(data.get('region') or '').strip() or None,
        asking_price=vehicle['asking_price'],
    )
    return jsonify({'status': 'success', 'vehicle': vehicle, **valuation})


if __name__ == '__main__':
    app.run(
        host='127.0.0.1',
        port=int(os.getenv('PORT', 5000)),
        debug=os.getenv('FLASK_DEBUG', '1') == '1',
    )