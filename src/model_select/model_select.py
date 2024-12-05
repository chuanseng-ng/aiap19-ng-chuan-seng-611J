import pandas as pd
import time

import setup.duration_cal as duration_cal
from log.log_setup import logger

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, LogisticRegression, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.naive_bayes import BernoulliNB

from typing import Dict


def model_selection(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    model_random_state: int,
    model_num_jobs: int,
    model_cv_num: int,
    model_scoring: str,
    model_num_iter: int,
    model_param_dict: Dict,
    model_search_method: str,
    task_type: str,
    col_name: str,
):
    # Define models and hyper-parameters
    model_dict = {}

    if task_type == "Regression":
        model_dict = regression_model_param_det(
            model_param_dict, model_dict, model_random_state
        )
    elif task_type == "Classification":
        model_dict = classification_model_param_det(
            model_param_dict, model_dict, model_random_state
        )
    else:
        ValueError(
            f"Invalid value for task_type: {task_type}! - Only valid values are 'Regression'/'Classification'"
        )

    # Initialize empty dictionary to store best models
    best_estimators_dict = {}

    ## Loop through each model
    for model_name, mp in model_dict.items():
        model_start_time = time.time()
        print(f"Processing {model_name} now for {col_name}...")
        ### Create pipeline with preprocessing and model
        pipeline = Pipeline(steps=[("model", mp["model"])])

        if model_search_method == "grid":
            #### Use GridSearchCV for hyper-parameter tuning
            search = GridSearchCV(
                pipeline,
                param_grid=mp["params"],
                cv=model_cv_num,
                scoring=model_scoring,
                n_jobs=model_num_jobs,
            )
        elif model_search_method == "random":
            #### Use RandomizedSearchCV for hyper-parameter tuning
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=mp["params"],
                n_iter=model_num_iter,
                cv=model_cv_num,
                scoring=model_scoring,
                random_state=model_random_state,
                n_jobs=model_num_jobs,
            )

        search.fit(X_train, Y_train)

        ### Save best model and use parameters for model evaluation
        best_estimators_dict[model_name] = search.best_estimator_
        print(f"Best parameters for {model_name} - {col_name}: {search.best_params_}")
        logger.info(
            f"Best parameters for {model_name} - {col_name}: {search.best_params_}"
        )
        logger.info("--------------------------------------------")

        model_end_time = time.time()
        model_total_time = model_end_time - model_start_time
        model_duration, model_tag = duration_cal.duration_cal(model_total_time)
        print(
            f"{model_name} - {col_name} has run tuning for {model_duration:.3f} {model_tag}!"
        )
        print()

    return best_estimators_dict


def regression_model_param_det(model_param_dict, model_dict, model_random_state):
    if "Linear Regression" in model_param_dict:
        model_dict["Linear Regression"] = {
            "model": LinearRegression(),
            "params": model_param_dict["Linear Regression"],
        }
    if "Random Forest" in model_param_dict:
        model_dict["Random Forest"] = {
            "model": RandomForestRegressor(random_state=model_random_state),
            "params": model_param_dict["Random Forest"],
        }
    if "SVR" in model_param_dict:  # Support Vector Regressor
        model_dict["SVR"] = {"model": SVR(), "params": model_param_dict["SVR"]}
    if "MLP" in model_param_dict:
        model_dict["MLP"] = {
            "model": MLPRegressor(random_state=model_random_state),
            "params": model_param_dict["MLP Regression"],
        }
    if "Bayesian Ridge" in model_param_dict:
        model_dict["Bayesian Ridge"] = {
            "model": BayesianRidge(),
            "params": model_param_dict["Bayesian Ridge"],
        }
    if "XGBoost" in model_param_dict:
        model_dict["XGBoost"] = {
            "model": XGBRegressor(
                objective="reg:squarederror", random_state=model_random_state
            ),
            "params": model_param_dict["XGBoost"],
        }

    return model_dict


def classification_model_param_det(model_param_dict, model_dict, model_random_state):
    if "Logistic Regression" in model_param_dict:
        model_dict["Logistic Regression"] = {
            "model": LogisticRegression(random_state=model_random_state),
            "params": model_param_dict["Logistic Regression"],
        }
    if "Random Forest" in model_param_dict:
        model_dict["Random Forest"] = {
            "model": RandomForestClassifier(random_state=model_random_state),
            "params": model_param_dict["Random Forest"],
        }
    if "SVC" in model_param_dict:  # Support Vector Classifier
        model_dict["SVC"] = {"model": SVC(), "params": model_param_dict["SVC"]}
    if "MLP" in model_param_dict:
        model_dict["MLP"] = {
            "model": MLPClassifier(random_state=model_random_state),
            "params": model_param_dict["MLP"],
        }
    if "Naive Bayes" in model_param_dict:
        model_dict["Naive Bayes"] = {
            "model": BernoulliNB(),
            "params": model_param_dict["Naive Bayes"],
        }
    if "XGBoost" in model_param_dict:
        model_dict["XGBoost"] = {
            "model": XGBClassifier(
                objective="reg:squarederror", random_state=model_random_state
            ),
            "params": model_param_dict["XGBoost"],
        }

    return model_dict
