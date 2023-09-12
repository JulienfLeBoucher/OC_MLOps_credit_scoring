import mlflow
from mlflow.models import infer_signature
import lightgbm
from lightgbm import LGBMClassifier

import sys
# Append path to find my_scorers and config
sys.path.append('./lightgbm')
import my_scorers
import config

from project_tools import utils
from project_tools.scorer import Scorer

########################################################################
# MAIN PARAMETER ZONE
########################################################################
# MLflow experiment name
experiment_name = 'test_best_lightgbm'

# utils.load_split_clip_scale_and_impute() parameters.
pre_processing_params = dict(
        predictors=None, 
        n_sample=2000,
        ohe=False,
        clip=False,
        scaling_method=None,
        imputation_method=None,
)

# Tags
mlflow_tags= {
    'pre_processing': str(pre_processing_params)
}

# Load the data
print('>>>>>> Load and pre-process raw features <<<<<<\n')
(
    X_train_pp, X_test_pp, y_train, y_test, pre_processors,
    
) = utils.load_split_clip_scale_and_impute_data(
    config.DATA_PATH,
    **pre_processing_params
)

# Load all scorers instantiated in my_scorers.py
scorers = Scorer.all 

# List of categorical features
categorical_features = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START'
]

# Set the best params found with hyperopt
lgbm_best_params = dict(
    objective='binary',
    nthread=-1,
    n_estimators=10_000,
    boosting_type='gbdt',
    data_sample_strategy='goss',
    random_state= 103,
    verbosity=1,
    min_child_samples=230,
    min_split_gain=0.0402532222541853,
    learning_rate=0.01752285342870759,
    reg_lambda=0.5957468159776326,
    colsample_bytree=0.7886281999030231,
    max_depth=14,
    reg_alpha=0.8631398118506094,
    is_unbalance=False,
    subsample_for_bin=140000,
    subsample=0.8679759773737233,
    num_leaves=54,
)

lgbm =LGBMClassifier(
    **lgbm_best_params
)    

########################################################################
# WARNING: this section only works when the file is run by the python
# interpreter, otherwise, the mlflow run command does not take that into 
# account because it has already created an experiment by default. 
#
# Create the MLflow experiment if needed.
# Possible to customize the experiment with the create_experiment
# method. Also set and get the exp id.
if mlflow.get_experiment_by_name(experiment_name) is None:
    mlflow.create_experiment(
        experiment_name,
        #artifact_location=...,
        tags={
            'pre_processing': str(pre_processing_params),
        }
    )
# mlflow.set_tracking_uri(f"{config.TRACKING_URI}")    
exp = mlflow.set_experiment(experiment_name)
exp_id = exp.experiment_id
########################################################################

with mlflow.start_run(experiment_id=exp_id) as run:
    # Fit the model
    lgbm.fit(
        X_train_pp,
        y_train,
        eval_set=[(X_train_pp, y_train), (X_test_pp, y_test)],
        eval_names=['training', 'validation'],
        eval_metric='auc',
        categorical_feature=categorical_features,
        callbacks=[lightgbm.early_stopping(100)],
    )
    
    # Predict
    y_pred_train = lgbm.predict_proba(X_train_pp) 
    y_pred_test = lgbm.predict_proba(X_test_pp)
    
    # Define the model signature
    signature = infer_signature(X_train_pp, y_pred_train)
    
    # Evaluate
    # Search best threshold and score for each scorer on the training set:
    metrics_= utils.compute_scorers_best_threshold_and_score(
        scorers,
        y_train,
        y_pred_train[:, 1]
    )
    metrics_train = {
        "train_"+k: v for (k, v) in metrics_.items()
    }
    
    # Compute scores on the test set using best threshold.
    metrics_test = {}
    for scorer in scorers:
        # Get the threshold that optimized the scorer on the train set.
        threshold = metrics_train[f'train_threshold_{scorer.name}']
        # Compute the metric accordingly
        metrics_test[f'test_{scorer.name}'] = (
            scorer.evaluate_predictions(
                y_test,
                utils.threshold_class_from_proba(
                    y_pred_test[:, 1],
                    threshold=threshold
                )
            )
        )

    # MLflow tracking
    mlflow.log_params(lgbm_best_params)
    metrics = {**metrics_test, **metrics_train}
    mlflow.log_metrics(metrics)
    if mlflow_tags is not None:
        mlflow.set_tags(mlflow_tags) 
    # Model logging    
    artifact_path='lgbm_model'    
    mlflow.lightgbm.log_model(lgbm, artifact_path, signature=signature)