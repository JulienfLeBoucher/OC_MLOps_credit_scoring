from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from hyperopt import hp
import numpy as np 
from typing import Callable, Dict, Any

DEFAULT_MAX_EVALS = 2

class HyperoptEstimator:
    
    def __init__(
        self,
        name: str,
        estimator: Estimator,
        space: Dict[str, Any]=None,
        max_evals: int=DEFAULT_MAX_EVALS,
        make_base_estimator_fit_params: Callable[[]]=None,  
    ):
        """
        - name: serve as the model name and the mlflow parent run.
        - estimator: part of the model with its fixed hyperparameters.
        - space: hyperopt fmin space argument. Describe where
        hyperparameters can take values.
        - max_evals: number of search iterations before stopping tuning.
        - make_fit_params: a callable that receives training and
        validation sets and builds extra fit parameters for the models
        if we, for example, want to use early stopping technique.
        
        #TODO : think if it is possible to augment this class with an 
        # imb_pipeline, so it is the finishing step of a resampling 
        # process. Should be possible as the pipeline take fit_params
        prefixed with the step.
        """
        self.name = name
        self.base_estimator = base_estimator
        self.space = space
        self.max_evals = max_evals
        
    def set_estimator_params(params):
        return base_estimator.set_params(**params)

    def get_estimator_fit_params(X_train, X_valid, y_train, y_valid):
        if self.make_fit_params is not None:
            return self.make_fit_params(X_train, X_valid, y_train, y_valid)
        else:
            return None
        
    def get_fmin_params():
        return {'space': self.space, 'max_evals': self.max_evals}


# LogisticRegression with L1 regulation
Lasso = HyperoptEstimator(
    name="Lasso-type Logistic Regression",
    estimator=LogisticRegression(penalty='l1', max_iter=5_000),
    space={
            'C': hp.lognormal('C', 0, 1.0),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice('solver', ['liblinear']),
        }, 
    #max_evals= ,
)

# LogisticRegression with L2 regulation
Ridge = HyperoptEstimator(
    name="Rigde-type Logistic Regression",
    estimator=LogisticRegression(penalty='l2', max_iter=5_000),
    space={
            'C': hp.lognormal('C', 0, 1.0),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
        }, 
    #max_evals= ,
)

# Random Forest
RandomForest = HyperoptEstimator(
    name="Random Forest",
    estimator=RandomForestClassifier(),
    space={
        'n_estimators': hp.uniformint('n_estimators', 20, 300),
        'max_depth':hp.uniformint('max_depth',5,20),
        'min_samples_leaf':hp.uniformint('min_samples_leaf',1,5),
        'min_samples_split':hp.uniformint('min_samples_split',2,6),
        'criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
        'class_weight': hp.choice('class_weight', ['balanced', None]),
    },
    #max_evals= ,
)

# SVCs
SVC_rbf = HyperoptEstimator(
    name="SVC_rbf",
    estimator=SVC(kernel='rbf', probability=True),
    space={
        'C': hp.loguniform('C', -10, 2),
        'gamma': hp.loguniform('gamma', -10, 2),
    },
    #max_evals= ,
)

SVC_poly = HyperoptEstimator(
    name="SVC_poly",
    estimator=SVC(kerner='poly', probability=True),
    space={
        'C': hp.lognormal('C', 0, 1),
        'degree': hp.choice('degree', [2, 3, 4, 5]),
        'gamma': hp.loguniform('gamma', -10, 2),
        'coef0': hp.uniform('coef0', -5, 5),
    },
    #max_evals= ,
)

# lightGBM
LightGBM = HyperoptEstimator(
    name="LightGBM",
    base_estimator=LGBMClassifier(
        objective='binary', 
        nthread=-1,
        n_estimators=10_000,
        random_state= 103,
        make_fit_params=lightGBM_make_fit_param,
    ),
    space={
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': hp.quniform('num_leaves', 10, 100, 1),
            'max_depth': hp.quniform('max_depth', 3, 15, 1),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.004), np.log(0.2)),
            'subsample_for_bin': hp.quniform('subsample_for_bin', 120000, 300000, 20000),
            'min_child_samples': hp.quniform('min_child_samples', 80, 400, 10),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
            'is_unbalance': hp.choice('is_unbalance', [True, False]),
            'min_split_gain': hp.uniform('min_split_gain', 0.01, 0.05),
            #'min_child_weight': hp.quniform('min_child_weight', 10, 50, 1),
            #'subsample': hp.uniform('subsample', 0.6, 1.0)
        },
    max_evals=5,
    make_base_estimator_fit_params=lightGBM_make_fit_param
)

def lightGBM_make_fit_param(X_train, X_valid, y_train, y_valid):
    fit_params = dict(
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        eval_names=['training', 'validation'],
        eval_metric=['auc'], #TODO: possibly to be changed with my metric
        # categorical_feature=categorical_feature,
        callbacks=[lgb.early_stopping(100, first_metric_only=True)],
    )
    return fit_params

# # xgboost
# XGBoost = model_configs["XGBOOST"] = dict(
#     model=XGBClassifier(
#         nthread= 4,
#         n_estimators=10000,
#         random_state= 103,
#         silent=-1,
#         verbose=-1
#     ),
#     fmin_params = dict(
        
#         space={
#             'boosting_type': hp.choice('boosting_type',
#                                        [{'boosting_type': 'gbdt',
#                                          'subsample': hp.uniform('gdbt_subsample', 0.6, 1)},
#                                         {'boosting_type': 'goss', 'subsample': 1.0}]),
#             'num_leaves': hp.quniform('num_leaves', 36, 86, 1),
#             'max_depth': hp.quniform('max_depth', 4, 14, 1),
#             'learning_rate': hp.loguniform('learning_rate', np.log(0.004), np.log(0.2)),
#             'subsample_for_bin': hp.quniform('subsample_for_bin', 120000, 300000, 20000),
#             'min_child_samples': hp.quniform('min_child_samples', 80, 400, 10),
#             'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
#             'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
#             'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
#             'is_unbalance': hp.choice('is_unbalance', [True, False]),
#             'min_split_gain': hp.uniform('min_split_gain', 0.01, 0.05),
#             #'min_child_weight': hp.quniform('min_child_weight', 10, 50, 1),
#             #'subsample': hp.uniform('subsample', 0.6, 1.0)
#         },
#         max_evals=200,
        
#     )
# )



    
hyperopt_estimators = [
    Lasso,
    Ridge,
    RandomForest,
    SVC_rbf,
    SVC_poly,
    LightGBM,
]