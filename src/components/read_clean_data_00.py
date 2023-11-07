from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest",
)
def read_clean_data(
    data_path: str,
    out_df_ml: Output[Dataset], 
    out_stations: Output[Dataset], 
):
    """
    Funtion that:
    1. Reads the data from a path (either local or Google Cloud Storage).
    2. Perform recommended preprocessing: https://www.kaggle.com/datasets/rosenthal/citi-bike-stations/
    3. Additionally, filter the rows to avoid COVID as the behavior was different to what we may see know a days.
    4. Finally, remove rows with missing station information (name, lat, lon, etc). We remove it as
    there is no known reason why this data is missing and we've checked.
    """
    import pandas as pd

    # 1. Read data:
    df = pd.read_csv(
        data_path,
        sep=",",
        na_values="\\N",
        dtype={
            "station_id": str,
            # Use Pandas Int16 dtype to allow for nullable integers
            "num_bikes_available": "Int16",
            "num_ebikes_available": "Int16",
            "num_bikes_disabled": "Int16",
            "num_docks_available": "Int16",
            "num_docks_disabled": "Int16",
            "is_installed": "Int16",
            "is_renting": "Int16",
            "is_returning": "Int16",
            "station_status_last_reported": "Int64",
            "station_name": str,
            "lat": float,
            "lon": float,
            "region_id": str,
            "capacity": "Int16",
            # Use pandas boolean dtype to allow for nullable booleans
            "has_kiosk": "boolean",
            "station_information_last_updated": "Int64",
            "missing_station_information": "boolean"
        },
    )
    # 2. Preprocess:
    # Read in timestamps as UNIX/POSIX epochs but then convert to the local
    # bike share timezone.
    df["station_status_last_reported"] = pd.to_datetime(
        df["station_status_last_reported"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("US/Eastern")

    df["station_information_last_updated"] = pd.to_datetime(
        df["station_information_last_updated"], unit="s", origin="unix", utc=True
    ).dt.tz_convert("US/Eastern")

    # 3. Filter only POST Covid dates:
    # Define the start and end date:
    start_date = pd.to_datetime("2020-12-01", utc=True).tz_convert("US/Eastern")
    end_date = pd.to_datetime("2021-12-01", utc=True).tz_convert("US/Eastern")
    # Filter the DataFrame for dates within the specified range
    df = df[(df["station_status_last_reported"] >= start_date)&(df["station_status_last_reported"] <= end_date)]

    # 4. Drop rows without station information:
    df = df.dropna()

    # 5. Select only the region of interest:
    df_nyc = df.query("region_id == '71'")
    station_variables = ['station_id', 'station_name', 'lat', 'lon', 'region_id', 'capacity']
    stations = df_nyc.groupby(station_variables).size().reset_index(name='count').copy()
    stations = stations.sort_values(by='count', ascending=False).set_index('station_id')
    # 6. Obtain True Capacity:
    df_prep = df_nyc.copy()
    df_prep.loc[df_prep["station_id"] == "3726", "capacity"] = 23
    df_prep['true_bike_capacity'] = df_prep['capacity']-(df_prep['num_bikes_disabled']+df_prep['num_docks_disabled'])
    df_prep['percentage_bikes_available'] = df_prep['num_bikes_available']*1.0/df_prep['true_bike_capacity'] 
    df_prep['percentage_docks_available'] = df_prep['num_docks_available']*1.0/df_prep['true_bike_capacity']
    df_prep['sum_percentage_validation'] = df_prep['percentage_bikes_available'] + df_prep['percentage_docks_available'] # Needs to be 1
    # 7. Removed capture errors:
    df_prep = df_prep[df_prep['sum_percentage_validation'] == 1]
    # 8. Export data:
    df_ml = df_prep[['station_id', 'station_status_last_reported', 'lat', 'lon', 'true_bike_capacity', 'percentage_bikes_available']].copy()
    stations = df_prep.groupby(station_variables).size().reset_index(name='count')

    pd.to_pickle(df_ml, out_df_ml.uri + ".pkl")
    pd.to_pickle(stations, out_stations.uri + ".pkl")