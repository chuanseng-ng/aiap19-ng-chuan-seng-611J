import sqlite3
import pandas as pd
import time

import setup.setup as setup
import setup.duration_cal as duration_cal
import EDA.eda_step as eda_step

start_time = time.time()
step_cnt = 1  # Initialize step count

(
    db_path,
    part1_target_col,
    part1_enc_col_list,
    part1_corr_thresh,
    part2_target_list,
    part2_target_comb,
    part2_corr_thresh,
    low_case_col_list,
    nutrient_col_list,
    drop_col_list,
    model_test_size,
    model_random_state,
    model_search_method,
    model_cv_num,
    model_scoring,
    model_num_iter,
    model_num_jobs,
    part1_model_name_list,
    part2_model_name_list,
    part1_model_param_dict,
    part2_model_param_dict,
) = setup.setup_stage()

# Create connection to SQL database
print(f"{step_cnt}. Connecting to SQL database....")
conn = sqlite3.connect(db_path)
print("Connection done!")

part1_time, step_cnt = duration_cal.duration_print(start_time, step_cnt)

# Get data from SQL database table
print(f"{step_cnt}. Extract SQL database table as DataFrame...")
farm_data_table_query = "SELECT name FROM sqlite_master WHERE type='table';"
farm_data_table_name = pd.read_sql(farm_data_table_query, conn).iloc[0, 0]
farm_data_query = f"SELECT * FROM {farm_data_table_name};"
farm_data_df = pd.read_sql_query(farm_data_query, conn)
print("Extraction done!")

part2_time, step_cnt = duration_cal.duration_print(part1_time, step_cnt)

# Using analysis from task_1 EDA, perform data pre-processing and feature engineering
print(f"{step_cnt}. Performing EDA on DataFrame...")
(
    part1_feat_farm_data_df,
    part1_X_train,
    part1_X_test,
    part1_Y_train,
    part1_Y_test,
    part2_feat_farm_data_df_list,
    part2_X_train_list,
    part2_X_test_list,
    part2_Y_train_list,
    part2_Y_test_list,
) = eda_step.ml_eda_step(
    farm_data_df,
    part1_target_col,
    part1_enc_col_list,
    part1_corr_thresh,
    part2_target_list,
    part2_target_comb,
    part2_corr_thresh,
    low_case_col_list,
    nutrient_col_list,
    drop_col_list,
    model_test_size,
    model_random_state,
)
print("EDA done!")

# Best model decision -
## If model variance is priority, look for highest R^2
## If predictive accuracy is priority, look for lowerst MSE (0 == Perfect model)

end_time = time.time()
final_time = end_time - start_time
final_duration, final_tag = duration_cal.duration_cal(final_time)

print("Script has reached end of line - It will terminate now!")
print(f"Script has run for {final_duration:.3f} {final_tag}!")
