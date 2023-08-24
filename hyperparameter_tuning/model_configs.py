from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from hyperopt import hp
import numpy as np 

# Create a dict to gather the model and its fmin parameters.
# Key: used as the mlflow parent name run,
# Value: a dict with two fields:
#   - 'model' : the model with its fixed params (used to define 
#   the hyperopt objective function).
#   - 'fmin_params': A dict with `space` and `max_evals` adapted to the 
#   space size. To be unpacked directly in the hyperopt fmin function.
model_configs = {}

# LogisticRegression with L1 regulation
model_configs["Lasso-type Logistic Regression"] = dict(
    model=LogisticRegression(penalty='l1', max_iter=5_000),
    fmin_params = dict(
        
        space={
            'C': hp.lognormal('C', 0, 1.0),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice('solver', ['liblinear']),
        },
        max_evals=50,
        
    )
)

# LogisticRegression with L2 regulation
model_configs["Rigde-type Logistic Regression"] = dict(
    model=LogisticRegression(penalty='l2', max_iter=5_000),
    fmin_params = dict(
        
        space={
            'C': hp.lognormal('C', 0, 1.0),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice(
                'solver',
                [
                    'lbfgs',
                    'liblinear',
                ]
            ),
        },
        max_evals=50,
        
    )
)

# # LogisticRegression no regulation
# model_configs["Logistic Regression with no regulation"] = dict(
#     model=LogisticRegression(penalty=None, max_iter=5_000),
#     fmin_params = dict(
#         space={
#             'class_weight': hp.choice('class_weight', ['balanced', None]),
#             'solver': hp.choice(
#                 'solver',
#                 [
#                     'lbfgs',
#                     'newton-cg', 
#                     'newton-cholesky',
#                     'sag',
#                     'saga',
#                 ]
#             ),
#         },
#         max_evals=,
#     )
# )

# Random Forest
model_configs["Random Forest"] = dict(
    model=RandomForestClassifier(),
    fmin_params = dict(
        
        space={
            'n_estimators': hp.uniformint('n_estimators', 20, 300),
            'max_depth':hp.uniformint('max_depth',5,20),
            'min_samples_leaf':hp.uniformint('min_samples_leaf',1,5),
            'min_samples_split':hp.uniformint('min_samples_split',2,6),
            'criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
        },
        max_evals=50,
        
    )
)

# SVC
model_configs["SVC"] = dict(
    model=SVC(probability=True),
    fmin_params = dict(
        
        space={
            'C': hp.lognormal('C', 0, 1),
            'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf']),
            'degree': hp.choice('degree', [2, 3]),
            'gamma': hp.choice('gamma', ['auto', 'scale']),
            'coef0': hp.uniform('coef0', -1, 1),
        },
        max_evals=100,
        
    )
)

# lightGBM
model_configs["LightGBM"] = dict(
    model=LGBMClassifier(
        nthread= 4,
        n_estimators=10000,
        random_state= 103,
        silent=-1,
        verbose=-1
    ),
    fmin_params = dict(
        
        space={
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
                                         'subsample': hp.uniform('gdbt_subsample', 0.6, 1)},
                                        {'boosting_type': 'goss', 'subsample': 1.0}]),
            'num_leaves': hp.quniform('num_leaves', 36, 86, 1),
            'max_depth': hp.quniform('max_depth', 4, 14, 1),
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
        max_evals=200,
        
    )
)