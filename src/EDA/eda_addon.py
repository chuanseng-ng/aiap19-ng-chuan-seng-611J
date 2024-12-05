import pandas as pd

from typing import List
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


def feat_eng(
    data_df: pd.DataFrame, full_col_list: List, part_num: int, lab_enc_list=[]
):
    # Perform label encoding
    lab_enc = LabelEncoder()
    for col_name in lab_enc_list:
        data_df[col_name] = lab_enc.fit_transform(data_df[col_name])

    # Perform one-hot encoding
    cat_col_list = data_df.select_dtypes(include=["object"]).columns.tolist()
    if part_num == 2:
        cat_col_list = cat_col_list + lab_enc_list
    data_df = pd.get_dummies(data_df, columns=cat_col_list, drop_first=True)
    bool_col = data_df.select_dtypes(include=["bool"]).columns
    data_df[bool_col] = data_df[bool_col].astype(int)

    # Perform standard scaling
    num_col_list = [
        col_name
        for col_name in full_col_list
        if (col_name not in cat_col_list and col_name not in lab_enc_list)
    ]
    scaler = StandardScaler()
    data_df[num_col_list] = scaler.fit_transform(data_df[num_col_list])

    return data_df


def corr_eval(data_df: pd.DataFrame, target_col: str, corr_thresh: float):
    # Calculate correlation matrix
    corr_matrix = data_df.corr()

    # Set correlation matrix to target variable
    target_corr = corr_matrix[target_col]

    # Sort correlations by absolute value
    sorted_target_corr = target_corr.abs().sort_values(ascending=False)

    # Get list of column names that have correlation values lower than threshold
    target_drop_col_list = sorted_target_corr[sorted_target_corr < corr_thresh].index
    data_df = data_df.drop(columns=target_drop_col_list, axis=1)

    return data_df


def model_prep(
    data_df: pd.DataFrame,
    target_col: str,
    model_test_size: float,
    model_random_state: int,
    part_num: int,
):
    # Split features (X) and target variable (Y)
    X = data_df.drop(columns=target_col)
    Y = data_df[target_col]

    # Split data into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=model_test_size, random_state=model_random_state
    )

    if part_num == 2:
        smote = SMOTE(sampling_strategy="auto", random_state=model_random_state)
        resample_X_train, resample_Y_train = smote.fit_resample(X_train, Y_train)

        return data_df, resample_X_train, X_test, resample_Y_train, Y_test
    else:
        return data_df, X_train, X_test, Y_train, Y_test
