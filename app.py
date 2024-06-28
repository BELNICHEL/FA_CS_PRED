from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained ML models and scaler
XGB_model = joblib.load("models/XGB_model1.pkl")
RF_model = joblib.load("models/RF_model1.pkl")
scaler = joblib.load("models/scaler1.pkl")

# Define parameter names
parameter_names = [
    "W/b ratio",
    "Cement Content",
    "Fine Aggregate",
    "Coarse Aggregate",
    "SiO2",
    "CaO",
    "Fe2O3",
    "Al2O3",
    "Loss on Ignition",
    "Superplastisizer",
    "Curing Days",
    "Replacement Percentage"
]

@app.route('/')
def home():
    return render_template('index.html', parameter_names=parameter_names)

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values for all parameters
    params = [float(request.form[param]) for param in parameter_names]
    
    # Scale the input data
    scaled_params = scaler.transform([params])
    
    # Perform prediction based on selected model
    selected_model = request.form['model']
    if selected_model == "Random Forest":
        model = RF_model
    elif selected_model == "XGBoost":
        model = XGB_model
    
    # Perform prediction using the selected model
    prediction = model.predict(scaled_params)[0]
    
    # Display prediction
    return render_template('index.html', parameter_names=parameter_names, prediction_text=f'The predicted value is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
