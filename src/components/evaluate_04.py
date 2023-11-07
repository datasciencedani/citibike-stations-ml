# Imports
from kfp.dsl import (
    Input, 
    Model,
    Dataset, 
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install = ["scikit-learn==1.3.2"],
)
def evaluate(
    in_df_test: Input[Dataset], 
    label: str,
    in_model: Input[Model], 
    mult_stations: bool = True,
    stations: list = [],
    pick_features: list = None,
)-> str:

    import pandas as pd
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    # 1. Read artifacts:
    df_test = pd.read_pickle(in_df_test.uri + ".pkl") 
    model = joblib.load(in_model.path + '.joblib')

    # 2. Prepare dataset:
    y_test = df_test[label].copy()
    x_test = df_test.drop([label], axis=1)

    # 2. Preprocess categorical variables:
    # Staion:
    if mult_stations:
        for station in stations:
            x_test['station_id_'+station] = x_test.station_id == station
    # Year:    
    x_test['year_2020'] = x_test.year == 2020
    x_test['year_2021'] = x_test.year == 2021

    # Drop previous:
    x_test = x_test.drop(['station_id', 'true_bike_capacity', 'year'], axis=1)

    # Pick labels:
    if pick_features:
       x_test = x_test[pick_features]
        
    # 3. Make prediction:
    y_pred = model.predict(x_test)
    
    # Evaluate:
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    dict = {'rmse':rmse, 'mae':mae, 'r2':r2}
    return str(dict)

evaluate_func = evaluate.python_func