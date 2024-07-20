import os
import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request
from sklearn.ensemble import GradientBoostingRegressor

app = Flask(__name__, static_url_path='/static')

# Get the directory of the current file (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to the model and scaler files
model_path = os.path.join(current_dir, 'cement.pkl')
scaler_path = os.path.join(current_dir, 'scaler.pkl')

# Load the model and scaler
model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/Prediction', methods=['GET', 'POST'])
def prediction():
    if request.method == 'POST':
        try:
            input_features = [float(x) for x in request.form.values()]
            features_name = ['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']
            x = pd.DataFrame([input_features], columns=features_name)

            # Scaling the input features
            x_scaled = scaler.transform(x)

            # Predicting
            prediction = model.predict(x_scaled)

            # Render prj3.html with prediction result
            return render_template('result2.html', prediction_text=prediction[0])
        except Exception as e:
            error_message = str(e)
            return f"An error occurred: {error_message}"

    # Render prj2.html for GET requests
    return render_template('index1.html')

@app.route('/Home')
def my_home():
    return render_template('result2.html')

if __name__ == "__main__":
    app.run(debug=True)



