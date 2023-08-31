# Script objective:
# 
# Track (MLflow) pipeline performance with optional SMOTENC, 
# undersampling, and one-hot encoder before a random forest classifier. 
# while being tuned with hyperopt(with regard to an optimization Scorer) 
# using cross-validation. 
# 
# GOAL: Observe SMOTE impact and select most important features.
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
# Append path to find my_scorers and config
sys.path.append('./random_forest')
import my_scorers
import config
from project_tools import utils
from project_tools.hyperopt_estimators import HyperoptEstimator
from project_tools.scorer import Scorer

from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTENC
from imblearn.under_sampling import RandomUnderSampler
########################################################################
# MAIN PARAMETER ZONE
########################################################################
# MLflow experiment name
experiment_name = 'random_forest'

# utils.load_split_clip_scale_and_impute() parameters.
# Here, ohe is set to False because it will be introduced in the 
# imblearn pipeline or not.
# Of course, more combinations exist, but for time motivations, I have
# chosen to explore only those.

pre_processing_params = [
    dict(
        predictors=None, 
        n_sample=1_500,
        ohe=False,
        clip=True,
        scaling_method='minmax',
        imputation_method='knn',
    ),
    # dict(
    #     predictors=None, 
    #     n_sample=3_000,
    #     ohe=False,
    #     clip=True,
    #     scaling_method='minmax',
    #     imputation_method='median',
    # ),
    # dict(
    #     predictors=None, 
    #     n_sample=3_000,
    #     ohe=False,
    #     clip=True,
    #     scaling_method='standard',
    #     imputation_method='median',
    # ),
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
# TODO:
# - build lightgbm user defined eval_metric.
# - build xgboost eval_metric
scorers = Scorer.all 
optimization_scorer = [
    sc for sc in scorers if sc.name == optimization_scorer_name
][0]
########################################################################
# Define 2 HypertoptEstimator pipelines
categorical_features = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
    'WEEKDAY_APPR_PROCESS_START'
]

ohe = OneHotEncoder(sparse_output=False, dtype='int16')
# create some minority synthetic individuals to yield a rate indicated in sampling_strategy
over = SMOTENC(
    categorical_features=categorical_features,
    categorical_encoder = ohe,
    k_neighbors=5, sampling_strategy=0.4
)
# downsample the majority class to something more balanced
under = RandomUnderSampler(sampling_strategy=0.7)
rf = RandomForestClassifier()
# Prepare a one hot encoder only for categorical features
ohe_cat = ColumnTransformer(
    transformers=[('ohe', ohe, categorical_features)],
    remainder="passthrough",
)

pipeline_smote = Pipeline(
    steps = [('o', over), ('u', under), ('rfc', rf)]
)
pipeline_smote_ohe = Pipeline(
    steps=[('o', over), ('u', under), ('ohe', ohe_cat), ('rfc', rf)]
)
pipeline_ohe = Pipeline(
    steps=[('ohe', ohe_cat), ('rfc', rf)]
)



# Hyperparameters spaces are similar, only the one-hot encoder step 
# differs from the 2 pipelines.
hyperopt_estimators = [
    HyperoptEstimator(
        name="SMOTE pipeline with Random Forest",
        estimator=pipeline_smote,
        space={
            'o__k_neighbors': hp.uniformint('o__k_neighbors', 2, 5),
            'o__random_state': 25,
            'o__sampling_strategy': hp.uniform('o__sampling_strategy', 0.1, 0.6),
            'u__random_state': 25,
            'u__sampling_strategy': hp.uniform('u__sampling_strategy', 0.61, 0.99),
            'rfc__n_estimators': hp.uniformint('n_estimators', 125, 250),
            'rfc__max_depth':hp.uniformint('max_depth',12,18),
            'rfc__min_samples_leaf':hp.uniformint('min_samples_leaf',1,7),
            'rfc__min_samples_split':hp.uniformint('min_samples_split',2,6),
            'rfc__criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
            'rfc__class_weight': hp.choice('class_weight', ['balanced', None]),
        },
        max_evals=60,
    ),
    HyperoptEstimator(
    name="SMOTE pipeline, ohe categorical, with Random Forest",
    estimator=pipeline_smote_ohe,
    space={
        'o__k_neighbors': hp.uniformint('o__k_neighbors', 2, 5),
        'o__random_state': 25,
        'o__sampling_strategy': hp.uniform('o__sampling_strategy', 0.1, 0.99),
        'u__random_state': 25,
        'u__sampling_strategy': hp.uniform('u__sampling_strategy', 0.5, 1),
        'rfc__n_estimators': hp.uniformint('n_estimators', 125, 250),
        'rfc__max_depth':hp.uniformint('max_depth',12,18),
        'rfc__min_samples_leaf':hp.uniformint('min_samples_leaf',1,7),
        'rfc__min_samples_split':hp.uniformint('min_samples_split',2,6),
        'rfc__criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
        'rfc__class_weight': hp.choice('class_weight', ['balanced', None]),
    },
    max_evals=60,
    ),
    HyperoptEstimator(
    name="ohe categorical with Random Forest",
    estimator=pipeline_ohe,
    space={
        'rfc__n_estimators': hp.uniformint('n_estimators', 125, 250),
        'rfc__max_depth':hp.uniformint('max_depth',12,18),
        'rfc__min_samples_leaf':hp.uniformint('min_samples_leaf',1,7),
        'rfc__min_samples_split':hp.uniformint('min_samples_split',2,6),
        'rfc__criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
        'rfc__class_weight': hp.choice('class_weight', ['balanced', None]),
    },
    max_evals=60,
    ),
]

print(
    f"\n>>> Evaluation of SMOTENC, and OHE impact "
    "on the random forest classifier")

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