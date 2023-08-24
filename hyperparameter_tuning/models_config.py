from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from lightgbm import LGBMClassifier
from hyperopt import hp

# Create a dict to gather the model and its fmin parameters.
# Key: used as the mlflow parent name run,
# Value: a dict with two fields:
#   - 'model' : the model with its fixed params (used to define 
#   the hyperopt objective function).
#   - 'fmin_params': A dict with `space` and `max_evals` adapted to the 
#   space size. To be unpacked directly in the hyperopt fmin function.
models_config = {}

# LogisticRegression with L1 regulation
models_config["Lasso-type Logistic Regression"] = dict(
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

# # LogisticRegression with L2 regulation
# models_config["Rigde-type Logistic Regression"] = dict(
#     model=LogisticRegression(penalty='l2', max_iter=5_000),
#     fmin_params = dict(
#         space={
#             'C': hp.lognormal('C', 0, 1.0),
#             'class_weight': hp.choice('class_weight', ['balanced', None]),
#             'solver': hp.choice(
#                 'solver',
#                 [
#                     'lbfgs',
#                     'liblinear',
#                 ]
#             ),
#         },
#         max_evals=50,
#     )
# )

# # # LogisticRegression no regulation
# # models_config["Logistic Regression with no regulation"] = dict(
# #     model=LogisticRegression(penalty=None, max_iter=5_000),
# #     fmin_params = dict(
# #         space={
# #             'class_weight': hp.choice('class_weight', ['balanced', None]),
# #             'solver': hp.choice(
# #                 'solver',
# #                 [
# #                     'lbfgs',
# #                     'newton-cg', 
# #                     'newton-cholesky',
# #                     'sag',
# #                     'saga',
# #                 ]
# #             ),
# #         },
# #         max_evals=,
# #     )
# # )

# # Random Forest
# models_config["Random Forest"] = dict(
#     model=RandomForestClassifier(),
#     fmin_params = dict(
#         space={
#             'n_estimators': hp.uniformint('n_estimators', 20, 300),
#             'max_depth':hp.uniformint('max_depth',5,20),
#             'min_samples_leaf':hp.uniformint('min_samples_leaf',1,5),
#             'min_samples_split':hp.uniformint('min_samples_split',2,6),
#             'criterion': hp.choice('criterion', ['gini', 'log_loss', 'entropy']),
#             'class_weight': hp.choice('class_weight', ['balanced', None]),
#         },
#         max_evals=50,
#     )
# )

# lightGBM
models_config["LightGBM"] = dict(
    model=LGBMClassifier(
        nthread= self.fixed_params['nthread'],
        n_estimators=10000,
        boosting_type= params['boosting_type']['boosting_type'],
        learning_rate= params['learning_rate'],
        num_leaves= params['num_leaves'],
        max_depth= params['max_depth'],
        subsample_for_bin= params['subsample_for_bin'],
        min_child_samples= params['min_child_samples'],
        reg_alpha= params['reg_alpha'],
        reg_lambda= params['reg_lambda'],
        colsample_bytree= params['colsample_bytree'],
        is_unbalance= params['is_unbalance'],
        min_split_gain= params['min_split_gain'],
        subsample= params['boosting_type']['subsample'],
        objective= params['objective'],
        random_state= self.seed,
        silent=-1,
        verbose=-1
    ),
    fmin_params = dict(
        space={
            'boosting_type': hp.choice('boosting_type',
                                       [{'boosting_type': 'gbdt',
                                         'subsample': hp.uniform('gdbt_subsample', 0.6, 1)},
                                        {'boosting_type': 'goss', 'subsample': 1.0}]),

            #'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
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
        }
    )
)