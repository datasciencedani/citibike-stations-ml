from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
    packages_to_install = ['holidays==0.36'],
)
def feature_eng(
    in_df_ml: Input[Dataset], 
    in_stations: Input[Dataset], 
    out_df_prep: Output[Dataset],
):
    """
    Funtion that:
    1. Reformats time by station on 5-min intervals
    2. Obtains time features from timestamp
    """
    import pandas as pd
    import datetime
    import holidays 

    import numpy as np

    df_ml = pd.read_pickle(in_df_ml.uri + ".pkl") 
    stations = pd.read_pickle(in_stations.uri + ".pkl") 

    # Step 1: Obtain time features
    def reformat_time(df, stations):
        # Create an empty list to store DataFrames
        dfs = []
        
        for station_id in stations.station_id.unique():
            df_station = df.query(f"station_id=='{station_id}'").copy()
            df_station = df_station.set_index("station_status_last_reported").sort_index()
            df_station = df_station.resample("5T").last()
            df_station = df_station.fillna(method="ffill")
            dfs.append(df_station)
        
        # Concatenate the DataFrames into one DataFrame
        df_prep = pd.concat(dfs)
        
        return df_prep
    
    df_prep = reformat_time(df_ml, stations)

    # Step 2: Obtain time features
    def extract_time_features(df):
        df['dayofweek'] = df.index.dayofweek
        df['weekend'] = np.where(df.index.weekday > 5, 1, 0)
        df['dayofmonth'] = df.index.day
        df['dayofyear'] = df.index.dayofyear
        df['month'] = df.index.month
        df['year'] = df.index.year
        df['hour'] = df.index.hour
        df['minute'] = df.index.minute

        # Holidays (can also help!)
        us_holidays = holidays.US(years=range(2010, 2030)) 
        df["is_holiday"] = df.index.to_series().apply(lambda x: 1 if x in us_holidays else 0)
        return df

    df_prep = extract_time_features(df_prep)   
    df_prep.to_pickle(out_df_prep.uri+".pkl") 