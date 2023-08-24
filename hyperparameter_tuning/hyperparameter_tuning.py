# Script objective:
# 
# Track (MLflow) performance of different classifiers while being tuned with
# hyperopt(with regard to an optimization Scorer) using cross-validation.
# 
# GOAL : compare model and choose the one to fine-tune.
#
# You can launch the script with python or with the `mlflow run` command. 

# In the latter case, some bugs occurs naturally when passing arguments to
# mlflow.start_run().

# To avoid that, pass the --experiment_name "..." as provided above in
# that file. https://github.com/mlflow/mlflow/issues/2735
# 
# To avoid another problem with the parent_run_name, a workaround
# suggested here is used : 
# https://github.com/mlflow/mlflow/issues/2804#issuecomment-640056129

import mlflow
import hyperopt
import utils
import models_config
import config

########################################################################
# MAIN PARAMETER ZONE
# MLflow experiment name
experiment_name = 'all_features_minmax_knn'
# load_split_clip_scale_and_impute() parameters.
pre_processing_params = dict(
    predictors=None, 
    n_sample=2_000,
    ohe=True,
    clip=True,
    scaling_method='minmax',
    imputation_method='knn',
)
# CV folds type
stratified_folds = True
# Load Scorers and choose the optimization metric :
scorers = utils.my_Scorers
optimization_scorer_name = 'loss_of_income'
# Load models and associated search space + max_evals
models_config = models_config.models_config
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
print('>>>>>> Load and pre-process <<<<<<\n')
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
    cv=utils.make_folds(stratified=stratified_folds),
    scorers=scorers,
    optimization_scorer=scorers[optimization_scorer_name],
    exp_id=exp_id,
    mlflow_tags= {
        'pre_processing': str(pre_processing_params),
        'stratified_cv': str(stratified_folds),
        'optimizer': optimization_scorer_name,
    }
)
# Loop on models to create the hyperopt objective 
# and tune hyperparameters.
for parent_run_name, model_dict in models_config.items():
    model = model_dict['model']
    fmin_params = model_dict['fmin_params']
    
    print(f'>>>>>> Entering hyperparameter tuning'
          f' for {parent_run_name} <<<<<<')
    
    objective = utils.objective_adjusted_to_data_and_model(
        **fixed_params,
        model=model,
    )
    
    trials = hyperopt.Trials()

    # Start the parent run
    with mlflow.start_run(
        experiment_id=exp_id,
        #run_name=parent_run_name
    ) as run:
        # Workaround line as specified in the incipit
        mlflow.set_tag("mlflow.runName", f"{parent_run_name}")
        # model hyperparameter tuning
        # space and max_evals are passed through fmin_params
        best_result = hyperopt.fmin(
            fn=objective,
            algo=hyperopt.tpe.suggest,
            trials=trials,
            **fmin_params,
        )