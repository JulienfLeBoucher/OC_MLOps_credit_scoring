# Script objective:
# 
# Track (MLflow) performance of different classifiers while being tuned with
# hyperopt(with regard to an optimization Scorer) using cross-validation.
# 
# GOAL : compare model and choose the one to fine-tune.

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
    n_sample=1_000,
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
# Create the MLflow experiment if needed.
# Possible to customize the experiment with the create_experiment
# method. Also set and get the exp id.
existing_exp = mlflow.get_experiment_by_name(experiment_name)
if not existing_exp:
    mlflow.create_experiment(
        experiment_name,
        # artifact_location="...", can be chosen here
    )
exp = mlflow.set_experiment(experiment_name)
exp_id = exp.experiment_id
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
    X_train=X_train_pp.sparse.to_dense(), #sparse.to_dense() optional.
    X_test=X_test_pp.sparse.to_dense(),
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
          f'for {parent_run_name} <<<<<<')
    
    objective = utils.objective_adjusted_to_data_and_model(
        **fixed_params,
        model=model,
    )
    
    trials = hyperopt.Trials()

    # Start the parent run
    with mlflow.start_run(
        experiment_id=exp_id,
        run_name=parent_run_name
    ) as run:
        # model hyperparameter tuning
        # space and max_evals are passed through fmin_params
        best_result = hyperopt.fmin(
            fn=objective,
            algo=hyperopt.tpe.suggest,
            trials=trials,
            **fmin_params,
        )