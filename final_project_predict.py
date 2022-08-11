#import os
import pickle
import numpy as np
import pandas as pd

import mlflow
from flask import Flask, request, jsonify

#logged_model = f'C:/Users/victo/PycharmProjects/mlops/mlruns/1/4526dbe8bab74998a23b76acf0e9d296/artifacts/models_mlflow'
#model = mlflow.pyfunc.load_model(logged_model)

cat_cols = ['size_class', 'fire_origin', 'det_agent_type', 'initial_action_by', 'fire_type', 'fire_position_on_slope',
            'weather_conditions_over_fire', 'fuel_type', 'day_period', 'seasons', 'forest_protection_area']

logged_model = './artifacts/12/0362c9f10c404e53b5b149cdec2d5a69/artifacts/log_rfc_model'
model = mlflow.pyfunc.load_model(logged_model)

# Load label encoder
dict_encoders = {}

for col in cat_cols:
    encoder_path = './artifacts/12/0362c9f10c404e53b5b149cdec2d5a69/artifacts/encoder/' + col + '.pkl'
    pkl_file = open(encoder_path, 'rb')
    dict_encoders[col] = pickle.load(pkl_file) 
    pkl_file.close() 

def prepare_features(register): # Transform json to dataframe
    features = pd.json_normalize(register) # Normalize semi-structured JSON data into a flat table
                                           # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.json_normalize.html
    for col in cat_cols:
        features[col] = dict_encoders[col].transform(features[col])

    return np.array(features)

def predict(features):
    preds = model.predict(features)
    if preds[0] == 0:
        return 'human'
    elif preds[0] == 1:
        return 'natural'

app = Flask('classification-alberta')

@app.route('/predict', methods=['POST'])

def predict_endpoint():
    wildfire = request.get_json()

    features = prepare_features(wildfire)

    pred = predict(features)

    result = {
        'fire_cause': pred
    }
    
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
