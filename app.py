'''
This file is for the backend (Flask)
XGBoost House Price Prediction
Dataset : data.csv (King County, WA)
'''

import os
import pickle
import numpy as np
from flask import Flask, render_template, request

# ── Load Model ────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "House_Price_Model.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")


@app.route("/predict", methods=['GET', 'POST'])
def predict():
    try:
        f = request.form

        # ── Collect raw inputs ────────────────────────────────────
        bedrooms     = float(f['bedrooms'])
        bathrooms    = float(f['bathrooms'])
        sqft_living  = float(f['sqft_living'])
        sqft_lot     = float(f['sqft_lot'])
        floors       = float(f['floors'])
        waterfront   = float(f['waterfront'])
        view         = float(f['view'])
        condition    = float(f['condition'])
        sqft_above   = float(f['sqft_above'])
        sqft_basement= float(f['sqft_basement'])
        city         = float(f['city'])
        yr_built     = float(f['yr_built'])
        yr_renovated = float(f['yr_renovated'])

        # ── Feature Engineering (must match main.py exactly) ──────
        house_age             = 2024 - yr_built
        renovated             = 1.0 if yr_renovated > 0 else 0.0
        years_since_renovated = (2024 - yr_renovated) if yr_renovated > 0 else house_age
        total_sqft            = sqft_living + sqft_basement
        bed_bath_ratio        = bedrooms / (bathrooms + 1)
        living_to_lot_ratio   = sqft_living / (sqft_lot + 1)
        luxury_score          = waterfront * 5 + view

        # ── Build Feature Array (same order as training) ──────────
        # bedrooms, bathrooms, sqft_living, sqft_lot, floors,
        # waterfront, view, condition, sqft_above, sqft_basement,
        # city, house_age, renovated, years_since_renovated,
        # total_sqft, bed_bath_ratio, living_to_lot_ratio, luxury_score
        input_array = np.array([[
            bedrooms, bathrooms, sqft_living, sqft_lot, floors,
            waterfront, view, condition, sqft_above, sqft_basement,
            city, house_age, renovated, years_since_renovated,
            total_sqft, bed_bath_ratio, living_to_lot_ratio, luxury_score
        ]])

        prediction = model.predict(input_array)[0]
        formatted  = f"${prediction:,.0f}"

        return render_template("index.html", prediction_text=formatted)

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")


if __name__ == "__main__":
    app.run(debug=True)