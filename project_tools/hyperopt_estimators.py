from typing import Dict, Any, Callable
import xgboost
import lightgbm

class HyperoptEstimator:
    """
    Union of an estimator and its search space for hyperparameter 
    optimization via hyperopt.
    
    
    TODO : 
    - Think if it is possible to augment this class with an 
    mb_pipeline, so it is the finishing step of a resampling 
    process. Should be possible as the pipeline take fit_params
    prefixed with the step.
    
    - See if passing the pre-processors can help revert one-hot encoding
    and leverage categorical features as label encoded for lightGBM and XGBoost? 
    
    """
    # to store all instances
    all = []
    
    def __init__(
        self,
        name: str,
        estimator,
        space: Dict[str, Any]=None,
        max_evals: int=10,
        early_stopping_rounds: int=None,  
        eval_metric=None,
    ):
        """
        - name: serve as the model name and the mlflow parent run.
        - estimator: part of the model with its fixed hyperparameters.
        - space: hyperopt fmin space argument. Describe where
        hyperparameters can take values.
        - max_evals: number of search iterations before stopping tuning.
        - early_stopping_rounds: stop training if the eval score has not
        improved for that many rounds.
        - eval_metric: The evaluation metric to track performance on the
        train and valid sets.
        
        """
        assert estimator is not None, f"No estimator passed."
        # Assign to self object
        self.name = name
        self.estimator = estimator
        self.space = space
        self.max_evals = max_evals
        self.early_stopping_rounds = early_stopping_rounds  
        self.eval_metric = eval_metric,

        # Actions
        
        # Append to the class list
        HyperoptEstimator.all.append(self)
        # Get early_stopping_rounds from the model definition if the
        # estimator is XGBClassifier 
        if self.is_xgboost_classifier():
            self.early_stopping_rounds = (
                self.estimator
                .get_params()["early_stopping_rounds"]
            )
            self.eval_metric =  (
                self.estimator
                .get_params()["eval_metric"]
            )
        
    def set_params(self, params: Dict[str, Any]):
        """ Set estimator params."""
        return self.estimator.set_params(**params)
    
    def is_xgboost_classifier(self):
        return isinstance(self.estimator, xgboost.sklearn.XGBClassifier)
    
    def is_lightGBM_classifier(self):
        return isinstance(self.estimator, lightgbm.sklearn.LGBMClassifier)
    
    def fit(self, X_train, y_train, X_valid, y_valid):
        """ fit the estimator with or without early stopping technique."""
        # Without early stopping
        if self.early_stopping_rounds is None:
            self.estimator.fit(X_train, y_train)
        # With
        else:    
            if self.is_lightGBM_classifier():
                self.fit_lightgbm(
                    X_train, y_train, X_valid, y_valid,
                )
            if self.is_xgboost_classifier():
                # the eval_metric is already part of the model definition.
                self.fit_xgboost(
                    X_train, y_train, X_valid, y_valid,
                )
                
    def fit_xgboost(self, X_train, y_train, X_valid, y_valid):
        self.estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            # verbose=25
        )
   
    
    def fit_lightgbm(self, X_train, y_train, X_valid, y_valid):
        self.estimator.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_valid, y_valid)],
            eval_names=['training', 'validation'],
            #TODO: test and validate with custom metric?
            eval_metric=[self.eval_metric], 
            callbacks=[
                lightgbm.early_stopping(
                    self.early_stopping_rounds,
                    first_metric_only=True
                )
            ],
        )
   
    def predict_proba(self, X):
        """ Predict proba, and get best model config to do so if
        early stopping was used."""
        if self.early_stopping_rounds is None:
            return self.estimator.predict_proba(X)
        else:    
            if self.is_lightGBM_classifier():
                return self.estimator.predict_proba(
                    X=X,
                    num_iteration=self.estimator.best_iteration_
                )
            if self.is_xgboost_classifier():
                return self.estimator.predict_proba(
                    X=X,
                    iteration_range=(0, self.estimator.best_iteration + 1)
                )
                
    def get_fmin_params(self):
        return {'space': self.space, 'max_evals': self.max_evals}
    
    def print(self):
        print(
            f"name : {self.name},\nearly :{self.early_stopping_rounds}"
        )
        