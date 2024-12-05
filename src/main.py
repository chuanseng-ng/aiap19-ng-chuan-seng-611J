import sqlite3
import pandas as pd
import time

import setup.setup as setup
import setup.duration_cal as duration_cal

start_time = time.time()
step_cnt = 1  # Initialize step count

(
    db_path,
    part1_target_col,
    part2_target_list,
    part2_target_comb,
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

part1_time = duration_cal.duration_print(start_time, step_cnt)
step_cnt = step_cnt + 1

# Best model decision -
## If model variance is priority, look for highest R^2
## If predictive accuracy is priority, look for lowerst MSE (0 == Perfect model)

end_time = time.time()
final_time = end_time - start_time
final_duration, final_tag = duration_cal.duration_cal(final_time)

print("Script has reached end of line - It will terminate now!")
print(f"Script has run for {final_duration:.3f} {final_tag}!")
