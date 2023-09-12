# Script objective:
# 
# Find the best final random forest with minmax scaling and zero 
# imputation.
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
sys.path.append('./random_forest')
import my_scorers
import config
from project_tools import utils
from project_tools.hyperopt_estimators import HyperoptEstimator
from project_tools.scorer import Scorer

from sklearn.ensemble import RandomForestClassifier
########################################################################
# MAIN PARAMETER ZONE
########################################################################
# MLflow experiment name
experiment_name = 'random_forest_final_model'

# utils.load_split_clip_scale_and_impute() parameters.
pre_processing_params = [
    dict(
        predictors=None, 
        n_sample=None,
        ohe=True,
        clip=True,
        scaling_method='minmax',
        imputation_method='zero',
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
rf = RandomForestClassifier(criterion='log_loss')

hyperopt_estimators = [
    HyperoptEstimator(
    name="Random Forest",
    estimator=rf,
    space={
        'n_estimators': hp.uniformint('n_estimators', 125, 250),
        'max_depth':hp.uniformint('max_depth', 14, 18),
        'min_samples_leaf':hp.uniformint('min_samples_leaf', 1, 5),
        'min_samples_split':hp.uniformint('min_samples_split', 4, 6),
        'class_weight': hp.choice('class_weight', ['balanced', None]),
        'max_samples': scope.int(
            hp.quniform('max_samples', 1_000, 10_000, 1_000)
        ),
        
    },
    max_evals=100,
    ),
]

print(
    f"\n>>> Find best hyperparameters for the random forest"
    "classifier\nwith minmax scaler and zero imputation.")
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