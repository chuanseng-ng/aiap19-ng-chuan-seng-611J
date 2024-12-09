import yaml


# Get all user-configs from config.yaml and load them into variables for use in latter steps
def setup_stage():
    # Load configuration file
    with open("cfg/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Accessing configuration settings
    db_path = config["database"]["path"]
    part1_target_col = config["features"]["part1"]["target"]
    part1_enc_col_list = config["features"]["part1"]["enc_list"]
    part1_corr_thresh = config["features"]["part1"]["corr_threshold"]
    part2_target_list = config["features"]["part2"]["target_list"]
    part2_target_comb = config["features"]["part2"]["target_comb"]
    part2_corr_thresh = config["features"]["part2"]["corr_threshold"]
    low_case_col_list = config["features"]["low_case_list"]
    nutrient_col_list = config["features"]["nutrient_list"]
    drop_col_list = config["features"]["drop_list"]

    model_test_size = config["model"]["test_size"]
    model_random_state = config["model"]["random_state"]
    model_search_method = config["model"]["search_method"]
    model_cv_num = config["model"]["cv_num"]
    model_num_iter = config["model"]["num_iter"]
    model_num_jobs = config["model"]["num_jobs"]
    part1_model_name_list = config["model"]["part1"]["model_name_list"]
    part2_model_name_list = config["model"]["part2"]["model_name_list"]
    part1_model_task_type = config["model"]["part1"]["task_type"]
    part2_model_task_type = config["model"]["part2"]["task_type"]
    part1_model_scoring = config["model"]["part1"]["scoring"]
    part2_model_scoring = config["model"]["part2"]["scoring"]
    part1_model_save_model = config["model"]["part1"]["save_model"]
    part2_model_save_model = config["model"]["part2"]["save_model"]

    part1_model_param_dict = {}
    part2_model_param_dict = {}

    if not (part1_model_name_list):
        part1_model_enable = False
    else:
        part1_model_enable = True

    if not (part2_model_name_list):
        part2_model_enable = False
    else:
        part2_model_enable = True

    # Handle part1 model parameters (Regression)
    if part1_model_enable:
        if "Linear Regression" in part1_model_name_list:
            part1_model_param_dict["Linear Regression"] = {}

        if "Random Forest" in part1_model_name_list:
            part1_model_param_dict["Random Forest"] = {
                "model__n_estimators": config["part1"]["rand_forest"]["est_list"],
                "model__max_depth": config["part1"]["rand_forest"]["depth_list"],
            }

        if "SVR" in part1_model_name_list:
            part1_model_param_dict["SVR"] = {
                "model__C": config["part1"]["svr"]["c_list"],
                "model__kernel": config["part1"]["svr"]["kernel_list"],
            }

        if "MLP" in part1_model_name_list:
            part1_model_param_dict["MLP"] = {
                "model__hidden_layer_sizes": config["part1"]["mlp"][
                    "hidden_layer_sizes_list"
                ],
                "model__activation": config["part1"]["mlp"]["activation_list"],
                "model__solver": config["part1"]["mlp"]["solver_list"],
                "model__learning_rate": config["part1"]["mlp"]["learning_rate_list"],
                "model__max_iter": config["part1"]["mlp"]["max_iter_list"],
            }

        if "Bayesian Ridge" in part1_model_name_list:
            part1_model_param_dict["Bayesian Ridge"] = {
                "model__max_iter": config["part1"]["bayes"]["max_iter_list"],
                "model__alpha_1": config["part1"]["bayes"]["alpha_1_list"],
                "model__alpha_2": config["part1"]["bayes"]["alpha_2_list"],
                "model__lambda_1": config["part1"]["bayes"]["lambda_1_list"],
                "model__lambda_2": config["part1"]["bayes"]["lambda_2_list"],
            }

        if "XGBoost" in part1_model_name_list:
            part1_model_param_dict["XGBoost"] = {
                "model__n_est": config["part1"]["xgb"]["n_est_list"],
                "model__learning_rate": config["part1"]["xgb"]["learning_rate_list"],
                "model__max_depth": config["part1"]["xgb"]["max_depth_list"],
                "model__subsample": config["part1"]["xgb"]["subsample_list"],
            }

    # Handle part2 model parameters (Classifier)
    if part2_model_enable:
        if "Logistic Regression" in part2_model_name_list:
            part2_model_param_dict["Logistic Regression"] = {
                "model__solver": config["part2"]["logistic"]["solver_list"],
                "model__max_iter": config["part2"]["logistic"]["max_iter_list"],
                "model__C": config["part2"]["logistic"]["c_list"],
                "model__class_weight": config["part2"]["logistic"]["class_weight_list"],
            }

        if "Random Forest" in part2_model_name_list:
            part2_model_param_dict["Random Forest"] = {
                "model__n_estimators": config["part2"]["rand_forest"]["est_list"],
                "model__max_depth": config["part2"]["rand_forest"]["depth_list"],
                "model__class_weight": config["part2"]["logistic"]["class_weight_list"],
            }

        if "SVC" in part2_model_name_list:
            part2_model_param_dict["SVC"] = {
                "model__C": config["part2"]["svc"]["c_list"],
                "model__kernel": config["part2"]["svc"]["kernel_list"],
                "model__class_weight": config["part2"]["logistic"]["class_weight_list"],
            }

        if "MLP" in part2_model_name_list:
            part2_model_param_dict["MLP"] = {
                "model__hidden_layer_sizes": config["part2"]["mlp"][
                    "hidden_layer_sizes_list"
                ],
                "model__activation": config["part2"]["mlp"]["activation_list"],
                "model__solver": config["part2"]["mlp"]["solver_list"],
                "model__learning_rate": config["part2"]["mlp"]["learning_rate_list"],
                "model__max_iter": config["part2"]["mlp"]["max_iter_list"],
            }

        if (
            "Naive Bayes" in part2_model_name_list
        ):  # Check on parameter list for Naive Bayes
            part2_model_param_dict["Naive Bayes"] = {
                "model__alpha": config["part2"]["bayes"]["alpha_list"],
            }

        if "XGBoost" in part2_model_name_list:
            part2_model_param_dict["XGBoost"] = {
                "model__learning_rate": config["part2"]["xgb"]["learning_rate_list"],
                "model__max_depth": config["part2"]["xgb"]["max_depth_list"],
                "model__subsample": config["part2"]["xgb"]["subsample_list"],
            }

    return (
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
        model_num_iter,
        model_num_jobs,
        part1_model_name_list,
        part2_model_name_list,
        part1_model_task_type,
        part2_model_task_type,
        part1_model_scoring,
        part2_model_scoring,
        part1_model_save_model,
        part2_model_save_model,
        part1_model_param_dict,
        part2_model_param_dict,
    )
