# Imports
from kfp import dsl

from components.read_clean_data_00 import read_clean_data
from components.feature_eng_01 import feature_eng
from components.split_data_02 import split_data
from components.tree_prep_train_03 import tree_prep_train
from components.evaluate_04 import evaluate

# Pipeline
@dsl.pipeline(
    name="pipeline-citibike",
)
def pipeline(
    data_gcs_uri: str,
    label: str,
):    
    o1 = read_clean_data(
        data_path=data_gcs_uri,
    )
    o2 = feature_eng(
        in_df_ml=o1.outputs["out_df_ml"], 
        in_stations=o1.outputs["out_stations"], 
    )
    o3 = split_data(
        in_df = o2.outputs["out_df_prep"],
    )
    o4 = tree_prep_train(
        in_df_train = o3.outputs["out_df_full_train"],
        label = label,
        stations = ['259', '303', '3069', '3113', '3357', '3564', '3690', '3726', '3771', '385', '466'], 
        max_depth = 5, # Chosen with parameter tuning
        min_samples_leaf=500, # Chosen with parameter tuning
    )
    o5 = evaluate(
        in_df_test = o3.outputs["out_df_test"], 
        label = label,
        in_model = o4.outputs["out_dt"], 
        stations = ['259', '303', '3069', '3113', '3357', '3564', '3690', '3726', '3771', '385', '466'], 
    )