import json
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

# 1. Load dataset
with open('dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
df = pd.DataFrame(data)

# 2. Clean numeric columns (strips out commas/symbols)
df['Listing / Bid Price ()'] = pd.to_numeric(df['Listing / Bid Price ()'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
df['Mileage (km)'] = pd.to_numeric(df['Mileage (km)'].astype(str).str.replace(r'[^\d.]', '', regex=True), errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df = df.dropna(subset=['Listing / Bid Price ()', 'Year', 'Mileage (km)'])

# 3. Define features (X) and target (y) and force categorical columns to string type
categorical_features = ['Make / Brand', 'Model', 'Transmission', 'Fuel Type']
numerical_features = ['Year', 'Mileage (km)']

for col in categorical_features:
    if col in df.columns:
        df[col] = df[col].astype(str)

X = df[categorical_features + numerical_features]
y = df['Listing / Bid Price ()']

# 4. Build pipeline for categorical and numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', 'passthrough', numerical_features)
    ]
)

model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 5. Train and save model pipeline
model_pipeline.fit(X, y)
joblib.dump(model_pipeline, 'random_forest_autovalue_model.pkl')
print("Fresh model pipeline trained and saved successfully!")