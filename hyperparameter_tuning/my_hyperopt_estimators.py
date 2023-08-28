from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from hyperopt import hp
from hyperopt.pyll import scope

from numpy import log 
from typing import Callable, Dict, Any

import hyperparameter_tuning

import sys
sys.path.append('..')
from project_tools.hyperopt_estimators import HyperoptEstimator

#TODO: may be change eval metric which defaults to 'auc' 
# for xbg and lgmb.


imbalance_ratio = 10 #(N_neg / N_positive)

# ########################################################################
# # Define all HyperoptEstimators here :
# # LogisticRegression with L1 regulation
# Lasso = HyperoptEstimator(
#     name="Lasso-type Logistic Regression",
#     estimator=LogisticRegression(penalty='l1', max_iter=5_000),
#     space={
#         'C': hp.lognormal('C', 0, 1.0),
#         'class_weight': hp.choice('class_weight', ['balanced', None]),
#         'solver': hp.choice('solver', ['liblinear']),
#     }, 
#     max_evals=50,
# )

# # LogisticRegression with L2 regulation
# Ridge = HyperoptEstimator(
#     name="Rigde-type Logistic Regression",
#     estimator=LogisticRegression(penalty='l2', max_iter=5_000),
#     space={
#         'C': hp.lognormal('C', 0, 1.0),
#         'class_weight': hp.choice('class_weight', ['balanced', None]),
#         'solver': hp.choice('solver', ['liblinear', 'lbfgs']),
#     }, 
#     max_evals=50,
# )

# # Random Forest
# RandomForest = HyperoptEstimator(
#     name="Random Forest",
#     estimator=RandomForestClassifier(),
#     space={
#         'n_estimators': hp.uniformint('n_estimators', 20, 300),
#         'max_depth':hp.uniformint('max_depth',5,20),
#         'min_samples_leaf':hp.uniformint('min_samples_leaf',1,5),
#         'min_samples_split':hp.uniformint('min_samples_split',2,6),
#         'criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
#         'class_weight': hp.choice('class_weight', ['balanced', None]),
#     },
#     max_evals=50,
# )

# # SVCs
# SVC_rbf = HyperoptEstimator(
#     name="SVC_rbf",
#     estimator=SVC(kernel='rbf', probability=True),
#     space={
#         'C': hp.loguniform('C', -10, 2),
#         'gamma': hp.loguniform('gamma', -10, 2),
#     },
#     max_evals=50,
# )

# SVC_poly = HyperoptEstimator(
#     name="SVC_poly",
#     estimator=SVC(kernel='poly', probability=True),
#     space={
#         'C': hp.lognormal('C', 0, 1),
#         'degree': hp.choice('degree', [2, 3, 4, 5]),
#         'gamma': hp.loguniform('gamma', -10, 2),
#         'coef0': hp.uniform('coef0', -5, 5),
#     },
#     max_evals=50 ,
# )

# lightGBM
lightGBM = HyperoptEstimator(
    name="LightGBM Classifier",
    estimator=LGBMClassifier(
        objective='binary',  nthread=-1, n_estimators=10_000, random_state= 103
    ),
    space={
            'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
            'num_leaves': hp.uniformint('num_leaves', 10, 100),
            'max_depth': hp.uniformint('max_depth', 3, 15),
            'learning_rate': hp.loguniform('learning_rate', log(0.004), log(0.2)),
            'subsample_for_bin': scope.int(hp.quniform('subsample_for_bin', 120000, 300000, 20000)),
            'min_child_samples': scope.int(hp.quniform('min_child_samples', 10, 300, 10)),
            'reg_alpha': hp.uniform('reg_alpha', 0.0, 1.0),
            'reg_lambda': hp.uniform('reg_lambda', 0.0, 1.0),
            'colsample_bytree': hp.uniform('colsample_by_tree', 0.5, 1.0),
            'is_unbalance': hp.choice('is_unbalance', [True, False]),
            'min_split_gain': hp.uniform('min_split_gain', 0.01, 0.05),
            'subsample': hp.uniform('subsample', 0.6, 1.0)
        },
    max_evals=2,
    early_stopping_rounds=50,
    eval_metric=hyperparameter_tuning.lightgbm_eval_metric,
)

# For xgboost early stopping round and eval_metric are part of the estimator
# instance. No need to specify them a second time.
xgboost = HyperoptEstimator(
    name="XGBoost Classifier",
    estimator=XGBClassifier(
        objective='binary:logistic',
        n_estimators=10_000,
        early_stopping_rounds=50,
        random_state= 103,
        njobs=-1,
        eval_metric=hyperparameter_tuning.xgboost_eval_metric,
        verbosity=1,
    ),
    space={
        'learning_rate': hp.loguniform('learning_rate', log(0.004), log(0.2)),
        'booster': hp.choice('booster', ['gbtree', 'dart', 'gblinear']),
        'grow_policy': hp.choice('grow_policy', ['depthwise', 'lossguide']),
        'max_depth':  hp.uniformint('max_depth', 1, 10),
        'min_child_weight': hp.uniformint('min_child_weight', 1, 8),
        'subsample': hp.quniform('subsample', 0.5, 1, 0.01),
        'gamma': hp.loguniform('gamma', log(1e-5), log(100)),
        'colsample_bytree': hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        'scale_pos_weight': hp.choice('scale_pos_weight', [1, imbalance_ratio])
    },
    max_evals=2,
)





    
