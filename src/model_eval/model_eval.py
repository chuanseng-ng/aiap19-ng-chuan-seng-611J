import pandas as pd
import joblib
import time
import os

import setup.duration_cal as duration_cal
from log.log_setup import logger

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
    model_save: bool,
    task_type: str,
):
    ## Initialize dictionary to store results
    eval_result_dict = {}

    for model_name, model in best_estimator_dict.items():
        model_start_time = time.time()
        print(f"Evaluating {model_name} now...")
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
            logger.info(f"{model_name} evaluation - ")
            logger.info(f"Mean Squared Error: {eval_mse}")
            logger.info(f"R2 Score: {eval_r2}")
        elif task_type == "Classification":
            eval_confuse_matrix = confusion_matrix(Y_test, Y_predict)
            eval_class_rpt = classification_report(Y_test, Y_predict)
            ### Store results
            eval_result_dict[model_name] = {
                "Confusion Matrix": eval_confuse_matrix,
                "Classification Report": eval_class_rpt,
            }
            logger.info(f"{model_name} evaluation - ")
            logger.info(f"Confusion Matrix: \n{eval_confuse_matrix}")
            logger.info(f"Classification Report: \n{eval_class_rpt}")

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(f"{model_name} has run evaluation for {model_duration:.3f} {model_tag}!")

        if model_save:
            model_save_path = "./model_saved"

            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)

            model_save_name = f"{model_save_path}/{model_name}.pkl"
            joblib.dump(model, model_save_name)
            print(f"Model has been saved as {model_save_name}!")
            print()
        else:
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
