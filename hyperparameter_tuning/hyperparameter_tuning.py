# Script objective:
# 
# Track (MLflow) performance of different classifiers while being tuned with
# hyperopt(with regard to an optimization Scorer) using cross-validation.
# 
# GOAL: Compare model and choose the one to fine-tune.
#
# The script can be launched with python or with the `mlflow run` command. 

# In the latter case, some bugs occurs naturally when passing arguments to
# mlflow.start_run(). To avoid that:
# - pass redundantly the --experiment_name "..." to the run command.
# https://github.com/mlflow/mlflow/issues/2735
# - a workaround has been implemented directly in the code as suggested here: 
# https://github.com/mlflow/mlflow/issues/2804#issuecomment-640056129

import mlflow
import hyperopt
import my_hyperopt_estimators 
import config
import sys
# Append path to the parent folder to find the project_tools package.
sys.path.append('../')
from project_tools import utils
from project_tools.hyperopt_estimators import HyperoptEstimator

########################################################################
# MAIN PARAMETER ZONE
# MLflow experiment name
experiment_name = 'test_GBM_compatibility'
# utils.load_split_clip_scale_and_impute() parameters.
pre_processing_params = dict(
    predictors=None, 
    n_sample=2_000,
    ohe=True,
    clip=True,
    scaling_method='minmax',
    imputation_method='knn',
)
# CV folds
stratified_folds = True
folds_iterator = utils.make_folds(stratified=stratified_folds)
# Get Scorers and choose the optimization metric :
scorers = utils.my_Scorers
optimization_scorer_name = 'loss_of_income'
# Get HyperoptEstimators defined in my_hyperopt_estimators
hyperopt_estimators = HyperoptEstimator.all
print(f"\n>>> Models to be tuned :\n")
for estimator in hyperopt_estimators:
    print(f"- {estimator.name}")
print('')    

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
# Load and pre-process data
print('>>>>>> Load and pre-process raw features <<<<<<\n')
(
    X_train_pp, X_test_pp, y_train, y_test, pre_processors,
    
) = utils.load_split_clip_scale_and_impute_data(
    config.DATA_PATH,
    **pre_processing_params
)
# Derived a set of parameters to create the objective function that does 
# not depend on the model type but rather on input datasets and scorers.
fixed_params = dict(
    X_train=X_train_pp,
    X_test=X_test_pp,
    y_train=y_train,
    y_test=y_test,
    folds_iterator=folds_iterator,
    scorers=scorers,
    optimization_scorer=scorers[optimization_scorer_name],
    exp_id=exp_id,
    mlflow_tags= {
        'pre_processing': str(pre_processing_params),
        'stratified_cv': str(stratified_folds),
        'optimizer': optimization_scorer_name,
    },
)
# Loop on estimators to create the hyperopt objective 
# and tune hyperparameters.
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