from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
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
            'solver': hp.choice('solver', ['liblinear', 'saga']),
        },
        max_evals=3,
    )
)

# LogisticRegression with L2 regulation
models_config["Rigde-type Logistic Regression"] = dict(
    model=LogisticRegression(penalty='l2', max_iter=1_000),
    fmin_params = dict(
        space={
            'C': hp.lognormal('C', 0, 1.0),
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice(
                'solver',
                [
                    'lbfgs',
                    'liblinear',
                    'newton-cg', 
                    'newton-cholesky',
                    'sag',
                    'saga',
                ]
            ),
        },
        max_evals=3,
    )
)

# LogisticRegression no regulation
models_config["Logistic Regression with no regulation"] = dict(
    model=LogisticRegression(penalty=None, max_iter=1_000),
    fmin_params = dict(
        space={
            'class_weight': hp.choice('class_weight', ['balanced', None]),
            'solver': hp.choice(
                'solver',
                [
                    'lbfgs',
                    'newton-cg', 
                    'newton-cholesky',
                    'sag',
                    'saga',
                ]
            ),
        },
        max_evals=5,
    )
)

# Random Forest
models_config["Random Forest"] = dict(
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
        max_evals=3,
    )
)