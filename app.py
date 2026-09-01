# app.py
import pickle
import numpy as np
from flask import Flask, request, render_template

app = Flask(__name__)

# Load transformation templates upfront when backend spins up
with open("saved_models/scalerDT.pkl", 'rb') as file:
    scaler = pickle.load(file)

with open("saved_models/finalized_model_SFP_DecisionTree.sav", 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # 1. Gather structural parameters from frontend index.html form names
    rul_days = float(request.form['Remaining_Useful_Life_days'])
    
    features = [
        float(request.form['Operational_Hours']),
        float(request.form['Temperature_C']),
        float(request.form['Vibration_mms']),
        float(request.form['Oil_Level_pct']),
        rul_days
    ]
    
    # 2. Rule-Based Threshold Check (Override condition)
    # If remaining useful life is less than 5 days, force a faulty state instantly
    if rul_days < 5:
        prediction = 1
        status_message = "Embedded Machine operates with Sensor Faulty (Rule Override: RUL < 5 Days)"
        alert_class = "faulty-alert"
        
    else:
        # 3. Model Inference Pipeline (Runs only if RUL >= 5 days)
        scaled_input = scaler.transform([features])
        prediction = int(loaded_model.predict(scaled_input)[0])
        
        # Condition matching block based on model output
        if prediction == 0:
            status_message = "Embedded Machine operates with Sensor Healthy"
            alert_class = "healthy-alert"
        else:
            status_message = "Embedded Machine operates with Sensor Faulty"
            alert_class = "faulty-alert"
        
    return render_template('result.html', prediction_text=status_message, alert_type=alert_class)

if __name__ == '__main__':
    app.run(debug=True)
