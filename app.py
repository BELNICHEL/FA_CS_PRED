from flask import Flask, render_template, request
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load your trained ML models and scaler
DT_model = joblib.load("DT_model1.pkl")
XGB_model = joblib.load("XGB_model1.pkl")
RF_model = joblib.load("RF_model1.pkl")
SVR_model = joblib.load("SVM_model1.pkl")
scaler = joblib.load("scaler1.pkl")

# Define parameter names
parameter_names = [
    "W/b ratio[0.36-0.55]",
    "Cement Content (kg/m3)[140-290]",
    "Fine Aggregate (kg/m3)[684-895]",
    "Coarse Aggregate (kg/m3)[975-1180]",
    "SiO2 (%)[49.7-60]",
    "CaO (%)[2.35-6]",
    "Fe2O3 (%)[5.27-8]",
    "Al2O3 (%)[21.63-28]",
    "Loss on Ignition (%)[1-3.6]",
    "Superplastisizer (kg/m3)[0-4.25]",
    "Curing Days[7-90]",
    "Replacement Percentage[30-60]"
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
    elif selected_model == "Extreme Gradient Boosting":
        model = XGB_model
    elif selected_model == "Decision Tree":
        model = DT_model
    elif selected_model == "Support Vector Regression":
        model = SVR_model
    
    # Perform prediction using the selected model
    prediction = model.predict(scaled_params)[0]
    
    # Display prediction
    return render_template('index.html', parameter_names=parameter_names, prediction_text=f'The predicted value is: {prediction}')

if __name__ == "__main__":
    app.run(debug=True)
