AutoValuePh
System Setup, Architecture & Execution Guide (README)

AutoValuePh is an interactive three-tier web application delivering instant Philippine market valuations and bank auction estimates for repossessed vehicles. Powered by a serialized Random Forest Regressor backend and a lightweight Python Flask REST API, the platform enables buyers, sellers, and evaluators to input vehicle specifications via reactive cascading dropdowns and obtain real-time resale predictions in PHP.
1. Step-by-Step Installation & Setup
Step 1: Clone the Repository
Download the source code and navigate into the root project directory:
git clone https://github.com/Mon-Soleil/AutoValuePh.git
cd AutoValuePh


Step 2: Create & Activate a Virtual Environment
Isolate project dependencies using Python's built-in venv module:
# Windows (PowerShell):
python -m venv venv
venv\Scripts\Activate

# macOS / Linux:
python3 -m venv venv
source venv/bin/activate


Step 3: Install Required Dependencies
Install Flask, Scikit-Learn, Pandas, and supporting libraries from requirements.txt:
pip install -r requirements.txt


Step 4: Launch the Flask Application
Navigate into the auto_value_app subfolder where app.py resides and start the server:
cd auto_value_app
python app.py


Step 5: Access the Web Interface — Once the Werkzeug development server initializes, open any modern web browser and navigate to http://127.0.0.1:5000.
2. Core API Endpoints & ML Pipeline Scripts
The application exposes two primary REST endpoints and two standalone ML scripts in the root directory:
GET /api/v1/options: Dynamically filters valid vehicle specifications (make, model, variant, body type, engine displacement, transmission, fuel type) based on upstream dropdown selections.
POST /api/v1/predict: Accepts selected vehicle attributes and mileage, executes Random Forest inference, and returns the predicted PHP market value, price range, and deal verdict.
python train_model.py: Re-trains the Random Forest Regressor pipeline on dataset.json and exports updated serialized model artifacts.
python evaluate_models.py: Evaluates and compares R² and Mean Absolute Error (MAE) metrics across Ridge Regression, Gradient Boosting, and Random Forest models.
