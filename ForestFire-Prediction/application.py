import pickle
from flask import Flask, request, render_template
import numpy as np

# Initialize Flask app
application = Flask(__name__)
app = application

# Load trained model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# ==========================
# ROUTES
# ==========================

# Index (Landing Page)
@app.route('/')
def index():
    return render_template('index.html')

# Home Page (Prediction Form)
@app.route('/home')
def home():
    return render_template('home.html')

# Prediction Route
@app.route('/predictdata', methods=['POST'])
def predict_datapoint():

    # Read inputs from form
    Temperature = float(request.form['Temperature'])
    RH = float(request.form['RH'])
    Ws = float(request.form['Ws'])
    Rain = float(request.form['Rain'])
    FFMC = float(request.form['FFMC'])
    DMC = float(request.form['DMC'])
    ISI = float(request.form['ISI'])
    Classes = float(request.form['Classes'])   # 0 = not fire, 1 = fire
    Region = float(request.form['Region'])     # 0 or 1

    # Scale input
    new_data_scaled = standard_scaler.transform(
        [[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]]
    )

    # Predict FWI
    result = ridge_model.predict(new_data_scaled)
    fwi_value = result[0]

    # Risk classification
    if fwi_value < 1:
        risk = "Not Fire 🔵"
    elif fwi_value < 5:
        risk = "Moderate Fire Risk 🟡"
    else:
        risk = "High Fire Risk 🔴"

    # Render result
    return render_template(
        'home.html',
        result=round(fwi_value, 2),
        risk=risk
    )

# Run app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
