import json

import pandas as pd
import joblib

from flask import Flask, request, abort, jsonify

app = Flask('churn')
@app.route('/', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        # READ MODEL:
        model = joblib.load('model.joblib')

        # READ DATA:
        station_status = request.get_json()
        station_status_pd = pd.json_normalize(station_status)

        # PREPROCESS DATA:
        stations = ['259', '303', '3069', '3113', '3357', '3564', '3690', '3726', '3771', '385', '466']
        for station in stations:
            station_status_pd['station_id_'+station] = station_status_pd.station_id == station
        
        station_status_pd['year_2020'] = station_status_pd.year == 2020
        station_status_pd['year_2021'] = station_status_pd.year == 2021

        station_status_pd = station_status_pd.drop(['station_id', 'year'], axis=1)

        # MODEL PREDICTION:
        y_pred = model.predict(station_status_pd)[0]

        result = {
            'bike_percentage_availability': float(y_pred),
            'dock_percentage_availability': float(1-y_pred),
        }
        return jsonify(result)
    else:
        return "OK"

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
