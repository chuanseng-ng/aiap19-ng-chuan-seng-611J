import pandas as pd
import time

import setup.duration_cal as duration_cal

from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    confusion_matrix,
    classification_report,
)
from typing import Dict


def model_evaluation(
    X_test: pd.DataFrame,
    Y_test: pd.DataFrame,
    best_estimator_dict: Dict,
    task_type: str,
    col_name: str,
):
    ## Initialize dictionary to store results
    eval_result_dict = {}

    for model_name, model in best_estimator_dict.items():
        model_start_time = time.time()
        print(f"Evaluating {model_name} {col_name} now...")
        ### Predict on test set
        Y_predict = model.predict(X_test)
        ### Calculate evaluation metrics
        if task_type == "Regression":
            eval_mse = mean_squared_error(Y_test, Y_predict)
            eval_r2 = r2_score(Y_test, Y_predict)
            ### Store results
            eval_result_dict[model_name] = {
                "Mean Squared Error": eval_mse,
                "R2 Score": eval_r2,
            }
        elif task_type == "Classification":
            eval_confuse_matrix = confusion_matrix(Y_test, Y_predict)
            eval_class_rpt = classification_report(Y_test, Y_predict)
            ### Store results
            eval_result_dict[model_name] = {
                "Confusion Matrix": eval_confuse_matrix,
                "Classification Report": eval_class_rpt,
            }

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(
            f"{model_name} {col_name} has run evaluation for {model_duration:.3f} {model_tag}!"
        )
        print()

    ## Info -
    ### Accuracy - Measures overall percentage of correct predictions
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### Precision - Out of instances predicted as positive, how many were actually positive
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### Recall - Out of actual positives, how many were correctly predicted
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ### F1-Score - A combined metric that balances precision and recall
    ### Optimal value = 1.0 (100%), Lowest value = 0.0

    ## Display results
    results_df = pd.DataFrame(eval_result_dict).T
    print(results_df)
