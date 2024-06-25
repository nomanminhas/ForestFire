import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)

scalar_model = pickle.load(open('Models/scaler_project.pkl', 'rb'))
ridge_model = pickle.load(open('Models/ridge_model.pkl', 'rb'))

# open home page

@app.route('/')
def home_page():
    return render_template('index.html', Result=None)

@app.route('/predict', methods=['POST', 'GET'])
def predict_datapoint():
    if request.method=='POST':
        Temperature = float(request.form.get('Temperature'))
        Relative_Humidity = float(request.form.get('Relative_Humidity'))
        Wind_speed = float(request.form.get('Wind_speed'))
        Rain = float(request.form.get('Rain'))
        Fine_Fuel_Moisture_Code = float(request.form.get('Fine_Fuel_Moisture_Code'))
        Duff_Moisture_Code = float(request.form.get('Duff_Moisture_Code'))
        Initial_Spread_IDX = float(request.form.get('Initial_Spread_IDX'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))


        new_data = scalar_model.transform([[Temperature, Relative_Humidity, Wind_speed, Rain, Fine_Fuel_Moisture_Code,
                                 Duff_Moisture_Code, Initial_Spread_IDX, Classes, Region]])
        
        result = ridge_model.predict(new_data)
        return render_template('index.html', Result="{:.2f}".format(result[0]))
    else:
        return render_template('index.html', Result=None)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
