import os
import joblib
import pandas as pd
from flask import Blueprint, render_template, jsonify, request
from app.utils import load_dataset, get_dropdown_options

main_bp = Blueprint('main', __name__)

# Resolve absolute paths relative to the project root directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
MODEL_PATH = os.path.join(BASE_DIR, 'random_forest_autovalue_model.pkl')
DATASET_PATH = os.path.join(BASE_DIR, 'dataset_2.json')

# Load the trained Random Forest model once when the server boots up
model = joblib.load(MODEL_PATH)

@main_bp.route('/')
def index():
    """Renders the frontend user interface."""
    return render_template('index.html')

@main_bp.route('/api/v1/options', methods=['GET'])
def api_options():
    """API endpoint to send unique dropdown options to the frontend JS."""
    try:
        df = load_dataset(DATASET_PATH)
        options = get_dropdown_options(df)
        return jsonify({"success": True, "options": options})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@main_bp.route('/api/v1/predict', methods=['POST'])
def api_predict():
    """API endpoint to handle car valuation predictions via the ML model."""
    try:
        # Grab JSON payload sent from frontend JavaScript fetch()
        incoming_data = request.get_json()
        
        # Convert the dictionary into a DataFrame format expected by your scikit-learn pipeline
        input_df = pd.DataFrame([incoming_data])
        
        # Run inference using the trained Random Forest model
        prediction = model.predict(input_df)
        
        # Return predicted price back to the frontend
        return jsonify({
            "success": True,
            "predicted_valuation": float(prediction[0])
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400