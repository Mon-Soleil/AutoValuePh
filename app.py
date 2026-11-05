import os
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

# Initialize Flask app to serve index.html directly from the root directory
app = Flask(__name__, template_folder='.', static_folder='.')

MODEL_PATH = 'random_forest_autovalue_model.pkl'

# Load fitted model pipeline
if os.path.exists(MODEL_PATH):
    model_pipeline = joblib.load(MODEL_PATH)
    print(f'✅ Successfully loaded ML model: {MODEL_PATH}')
else:
    model_pipeline = None
    print(
        f'⚠️ Warning: {MODEL_PATH} not found in root directory! Please export'
        ' it from your notebook.'
    )


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if model_pipeline is None:
        return (
            jsonify({
                'status': 'error',
                'message': (
                    f'Model file ({MODEL_PATH}) not loaded on server. Make'
                    ' sure it exists in the project root.'
                ),
            }),
            500,
        )

    try:
        data = request.get_json(force=True)

        if not data:
            return (
                jsonify({
                    'status': 'error',
                    'message': 'No JSON payload provided in request.',
                }),
                400,
            )

        # Construct input DataFrame matching exact feature names expected by ColumnTransformer
        input_df = pd.DataFrame([{
            'Year': int(data.get('year', 0)),
            'Mileage (km)': float(data.get('mileage', 0.0)),
            'Engine / Displacement': float(
                data.get('engine_displacement', 0.0)
            ),
            'Make / Brand': str(data.get('make', '')),
            'Body Type': str(data.get('body_type', '')),
            'Model': str(data.get('model', '')),
            'Variant / Trim': str(data.get('variant', '')),
            'Transmission': str(data.get('transmission', '')),
            'Fuel Type': str(data.get('fuel_type', '')),
        }])

        # Execute prediction
        raw_prediction = model_pipeline.predict(input_df)[0]
        predicted_price = max(0.0, round(float(raw_prediction), 2))
        formatted_price = f'₱{predicted_price:,.2f}'

        return (
            jsonify({
                'status': 'success',
                'predicted_price': predicted_price,
                'formatted_price': formatted_price,
            }),
            200,
        )

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)