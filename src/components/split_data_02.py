# Imports
from kfp.dsl import (
    Input, 
    Output, 
    Dataset,
    component,
)
@component(
    base_image="gcr.io/deeplearning-platform-release/base-cpu.py310:latest", # The docker image this could would need to run as a container.
)
def split_data(
    in_df: Input[Dataset],
    out_df_test: Output[Dataset],
    out_df_full_train: Output[Dataset], 
    out_df_val: Output[Dataset],
    out_df_train: Output[Dataset], 
):
    import pandas as pd

    df = pd.read_pickle(in_df.uri + ".pkl") 
    
    # We have 12 months, we'll take 8 months training,
    # 2 months for validation and 2 months for testing:
    split_date_test = pd.to_datetime("2021-10-01", utc=True).tz_convert("US/Eastern")
    split_date_val = pd.to_datetime("2021-08-01", utc=True).tz_convert("US/Eastern")
    # Filter the DataFrame for dates within the specified range
    df_test = df[df.index >= split_date_test]
    df_full_train = df[df.index < split_date_test]
    df_val = df[df.index >= split_date_val]
    df_train = df[df.index < split_date_val]

    pd.to_pickle(df_test, out_df_test.uri + ".pkl")
    pd.to_pickle(df_full_train, out_df_full_train.uri + ".pkl")
    pd.to_pickle(df_val, out_df_val.uri + ".pkl")
    pd.to_pickle(df_train, out_df_train.uri + ".pkl")

split_data_func = split_data.python_func