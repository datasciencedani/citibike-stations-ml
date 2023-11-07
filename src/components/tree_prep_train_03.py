# Imports
from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    Model, 
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install = ["scikit-learn==1.3.2"],
)
def tree_prep_train(
    in_df_train: Input[Dataset],
    label: str,
    out_dt: Output[Model],
    max_depth: int = None,
    min_samples_leaf: int = 1,
    mult_stations: bool = True,
    stations: list = [],
    pick_features: list = None,
):
    import pandas as pd 
    import numpy as np

    import joblib
    import pickle

    from sklearn.tree import DecisionTreeRegressor

    # 1. Read data:
    df_train = pd.read_pickle(in_df_train.uri + ".pkl") 

    y_train = df_train[label].copy()
    x_train = df_train.drop([label], axis=1)

    # 2. Preprocess categorical variables:
    # Staion:
    if mult_stations:
        for station in stations:
            x_train['station_id_'+station] = x_train.station_id == station
    # Year:    
    x_train['year_2020'] = x_train.year == 2020
    x_train['year_2021'] = x_train.year == 2021

    # Drop previous:
    x_train = x_train.drop(['station_id', 'true_bike_capacity', 'year'], axis=1)

    # Pick labels:
    if pick_features:
       x_train = x_train[pick_features] 

    # 3. Training:
    dt = DecisionTreeRegressor(
        max_depth = max_depth, 
        min_samples_leaf = min_samples_leaf,
    )
    dt.fit(x_train, y_train)

    with open(out_dt.path+'.joblib', 'wb') as f:
        pickle.dump(dt, f)

tree_prep_train_func = tree_prep_train.python_func