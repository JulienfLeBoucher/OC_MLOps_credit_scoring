# Script objective:
# 
# Find the best lightgbm model without pre-processing.
#
#
# The script can be launched with python or with the `mlflow run` 
# command from the parent folder. 

# In the latter case, some bugs occurs when passing arguments to
# mlflow.start_run(). To avoid that:
# - pass redundantly the --experiment_name "..." to the run command.
# https://github.com/mlflow/mlflow/issues/2735
# - a workaround has been implemented directly in the code as suggested 
# here: 
# https://github.com/mlflow/mlflow/issues/2804#issuecomment-640056129


import mlflow
import hyperopt
import sys
from hyperopt import hp
from hyperopt.pyll import scope
# Append path to find my_scorers and config
sys.path.append('./lightgbm')
import my_scorers
import config
from project_tools import utils
from project_tools.hyperopt_estimators import HyperoptEstimator
from project_tools.scorer import Scorer
from numpy import log
from lightgbm import LGBMClassifier
########################################################################
# MAIN PARAMETER ZONE
########################################################################
# MLflow experiment name
experiment_name = 'lightgbm'

# utils.load_split_clip_scale_and_impute() parameters.
pre_processing_params = [
    dict(
        predictors=None, 
        n_sample=50_000,
        ohe=False,
        clip=False,
        scaling_method=None,
        imputation_method=None,
    ),
]
# CV folds
stratified_folds = True

# Get Scorers and choose the optimization metric :
optimization_scorer_name = 'loss_of_income'

########################################################################
# Chose the folds iterator type
folds_iterator = utils.make_folds(stratified=stratified_folds)

# - load all scorers instantiated in my_scorers.py
# - focus on the optimization scorer.
scorers = Scorer.all 
optimization_scorer = [
    sc for sc in scorers if sc.name == optimization_scorer_name
][0]
########################################################################
lightgbm_eval_metric = utils.convert_scorer_to_lightgbm_eval_metric(
    optimization_scorer
)

categorical_features = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START'
]

hyperopt_estimators = [
    HyperoptEstimator(
        name="LightGBM Classifier",
        estimator=LGBMClassifier(
            objective='binary',
            metric="None", 
            # first_metric_only=True,
            nthread=-1,
            n_estimators=10_000,
            random_state= 103,
            verbosity=0
        ),
        space={
                'boosting_type': hp.choice('boosting_type', ['gbdt', 'goss']),
                'num_leaves': hp.uniformint('num_leaves', 10, 100),
                'max_depth': hp.uniformint('max_depth', 3, 15),
                'learning_rate': hp.loguniform('learning_rate', log(0.004), log(0.2)),
                # 'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 120000, 300000, 20000)),
                'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 300, 10)),
                'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
                'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
                'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
                'is_unbalance': hp.choice('is_unbalance', [True, False]),
                'min_split_gain': hp.uniform('min_split_gain', 0.01, 0.05),
                'subsample': hp.uniform('subsample', 0.6, 1.0)
            },
        max_evals=10,
        early_stopping_rounds=100,
        eval_metric=lightgbm_eval_metric,
        categorical_features=categorical_features,
    )
]

print(
    f"\n>>> Find best hyperparameters for lightGBMClassifier"
    "with no preprocessing.\n"
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
            'stratified_cv': str(stratified_folds),
            'optimizer': optimization_scorer_name,
        }
    )
# mlflow.set_tracking_uri(f"{config.TRACKING_URI}")    
exp = mlflow.set_experiment(experiment_name)
exp_id = exp.experiment_id
########################################################################
# Load and pre-process data with some variations.
for pp_params in pre_processing_params:
    print('>>>>>> Load and pre-process raw features <<<<<<\n')
    (
        X_train_pp, X_test_pp, y_train, y_test, pre_processors,
        
    ) = utils.load_split_clip_scale_and_impute_data(
        config.DATA_PATH,
        **pp_params
    )

    # Derived a set of parameters to create the hyperopt objective function
    # that does not depend on the model type but rather on input datasets
    # and scorers.
    fixed_params = dict(
        X_train=X_train_pp,
        X_test=X_test_pp,
        y_train=y_train,
        y_test=y_test,
        folds_iterator=folds_iterator,
        scorers=scorers,
        optimization_scorer=optimization_scorer,
        exp_id=exp_id,
        mlflow_tags= {
            'pre_processing': str(pp_params),
            'stratified_cv': str(stratified_folds),
            'optimizer': optimization_scorer_name,
        },
    )

    # Loop on estimators to create the hyperopt objective 
    # and tune hyperparameters while tracking performance.
    for h_estim in hyperopt_estimators:
        
        parent_run_name = h_estim.name
        fmin_params = h_estim.get_fmin_params()
        
        print(f'\n>>>>>> Entering hyperparameter tuning '
            f'for {parent_run_name} <<<<<<')
        
        objective = utils.objective_adjusted_to_data_and_model(
            **fixed_params,
            h_estimator=h_estim,
        )
        trials = hyperopt.Trials()
        # Start the parent run
        with mlflow.start_run(
            experiment_id=exp_id,
            #run_name=parent_run_name
        ) as run:
            # Workaround line as specified in the incipit
            mlflow.set_tag("mlflow.runName", f"{parent_run_name}")
            # Model hyperparameter tuning.
            # space and max_evals are passed through fmin_params.
            best_result = hyperopt.fmin(
                fn=objective,
                algo=hyperopt.tpe.suggest,
                trials=trials,
                **fmin_params,
            )