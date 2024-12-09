import pandas as pd

from typing import List
from copy import deepcopy

import EDA.eda_addon as eda_addon


# Machine learning EDA wrapper
def ml_eda_step(
    farm_data_df: pd.DataFrame,
    part1_target_col: str,
    part1_enc_col_list: List,
    part1_corr_thresh: float,
    part2_target_list: List,
    part2_target_comb: str,
    part2_corr_thresh: float,
    low_case_col_list: List,
    nutrient_col_list: List,
    drop_col_list: List,
    model_test_size: float,
    model_random_state: int,
):
    # Change 'Plant Type' and 'Plant Stage' columns' data to lowercase characters
    for col_name in low_case_col_list:
        farm_data_df[col_name] = farm_data_df[col_name].str.lower()

    # Remove 'ppm' tag from cells in 'Nutrient * Sensor' column and convert data to numeric type
    for col_name in nutrient_col_list:
        farm_data_df[col_name] = farm_data_df[col_name].str.replace(
            "ppm", "", regex=False
        )
        farm_data_df[col_name] = pd.to_numeric(farm_data_df[col_name], errors="coerce")

    ## Drop 'Humidity Sensor' column due to huge bias when replacing missing values with median -> Can be modified in user-config if needed
    farm_data_df = farm_data_df.drop(columns=drop_col_list)

    # Remove negative sign from cells in 'Temperature Sensor' column
    ## Get all column names and find 'Temperature Sensor'
    farm_data_df_col_list = farm_data_df.columns
    part1_target_col = [
        col_name for col_name in farm_data_df_col_list if part1_target_col in col_name
    ][0]
    farm_data_df[part1_target_col] = farm_data_df[part1_target_col].abs()

    # Part 1 - Regression
    part1_farm_data_df = deepcopy(farm_data_df)
    (
        part1_feat_farm_data_df,
        part1_X_train,
        part1_X_test,
        part1_Y_train,
        part1_Y_test,
    ) = part1_eda(
        part1_farm_data_df,
        part1_target_col,
        farm_data_df_col_list,
        part1_enc_col_list,
        part1_corr_thresh,
        model_test_size,
        model_random_state,
    )

    # Part 2 - Classification
    part2_farm_data_df = deepcopy(farm_data_df)
    (
        part2_feat_farm_data_df,
        part2_X_train,
        part2_X_test,
        part2_Y_train,
        part2_Y_test,
    ) = part2_eda(
        part2_farm_data_df,
        part2_target_comb,
        part2_target_list,
        farm_data_df_col_list,
        part2_corr_thresh,
        model_test_size,
        model_random_state,
    )

    return (
        part1_feat_farm_data_df,
        part1_X_train,
        part1_X_test,
        part1_Y_train,
        part1_Y_test,
        part2_feat_farm_data_df,
        part2_X_train,
        part2_X_test,
        part2_Y_train,
        part2_Y_test,
    )


# Part 1 (Regression) EDA wrapper
def part1_eda(
    data_df,
    target_col,
    data_df_col_list,
    enc_col_list,
    corr_thresh,
    model_test_size,
    model_random_state,
):
    ## Drop all rows with missing values
    data_df = data_df.dropna().reset_index()

    ## Perform feature engineering (Label encoding and one-hot encoding)
    feat_farm_data_df = eda_addon.feat_eng(
        data_df,
        data_df_col_list,
        enc_col_list,
    )

    # Drop features that are below correlation threshold value
    feat_farm_data_df = eda_addon.corr_eval(feat_farm_data_df, target_col, corr_thresh)

    (
        data_df,
        X_train,
        X_test,
        Y_train,
        Y_test,
    ) = eda_addon.model_prep(
        feat_farm_data_df,
        target_col,
        model_test_size,
        model_random_state,
    )

    return feat_farm_data_df, X_train, X_test, Y_train, Y_test


# Part 2 (Classification) EDA wrapper
def part2_eda(
    data_df: pd.DataFrame,
    target_col: str,
    target_list: List,
    data_df_col_list: List,
    corr_thresh: float,
    model_test_size: float,
    model_random_state: int,
):
    ## Do further processing to combine 'Plant Type' and 'Plant Stage' to form 'Plant Type-Stage'
    data_df[target_col] = data_df[target_list[0]] + "-" + data_df[target_list[1]]

    ## Drop 'Plant Type' and 'Plant Stage' columns as their information are already combined to form a new column
    data_df = data_df.drop(columns=target_list)
    data_df_col_list = [
        col_name for col_name in data_df_col_list if col_name not in target_list
    ]

    ## Drop all rows with missing values
    data_df = data_df.dropna().reset_index()

    ## Perform feature engineering (Label encoding and one-hot encoding)
    feat_farm_data_df = eda_addon.feat_eng(
        data_df,
        data_df_col_list,
        [target_col],
    )

    feat_farm_data_df = eda_addon.corr_eval(feat_farm_data_df, target_col, corr_thresh)

    (
        feat_farm_data_df,
        X_train,
        X_test,
        Y_train,
        Y_test,
    ) = eda_addon.model_prep(
        feat_farm_data_df,
        target_col,
        model_test_size,
        model_random_state,
    )

    return feat_farm_data_df, X_train, X_test, Y_train, Y_test
