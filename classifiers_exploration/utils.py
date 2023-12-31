import re
import gc
import numpy as np
import pandas as pd
import config
import mlflow
import hyperopt
import time

from typing import (
    Any,
    Dict,
    Union,
    List,
)

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
    cross_val_predict
)
from sklearn.impute import (
    KNNImputer,
    SimpleImputer,
)
from sklearn.metrics import (
    confusion_matrix,
    make_scorer,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    fbeta_score,
)
from sklearn.preprocessing import(
    OneHotEncoder,
    StandardScaler,
    PowerTransformer,
    MinMaxScaler,
)

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

########################################################################
### Load and prepare data
########################################################################
def prepare_data(
    data_path: str,
    drop_NAN: bool=False,
    drop_INF: bool=False,
    n_sample: int=None,
) -> tuple[pd.DataFrame, pd.Series]:
    """ 
    - Read the pickle file.
    - Retain individuals with a known target value.
    - Determine useful columns for prediction.
    - Return the feature dataframe and the target series.
    """
    try:
        data = pd.read_pickle(data_path)
    except Exception as e:
        logger.exception(
                """Unable to load features. 
                Check the config FEATURE_PATH.
                Error: %s", e"""
        )
    
    data = data[data['TARGET'].notnull()]
    
    # Eventually replace inf with NaN
    if drop_INF:
        data = data.replace(np.inf, np.NaN)
        data = data.replace(-np.inf, np.NaN)
    # Eventually drop columns with nulls 
    if drop_NAN:

        data = data.dropna(axis=1)
        
    
    not_predictors = [
        'TARGET',
        'SK_ID_CURR',
        'SK_ID_BUREAU',
        'SK_ID_PREV',
        'index',
        'level_0',
    ]
    predictors = [feat for feat in data.columns if feat not in not_predictors]
    
    if n_sample:
        print(
            f">>> Sampling with a fixed random state : {n_sample}"
            "random samples."
        )
        data = data.sample(n_sample, random_state=2)

    print(
        "Data shape: {}, target shape: {}"
        .format(data[predictors].shape, data['TARGET'].shape)
    )
    return data[predictors], data['TARGET'].astype('int8')


sel_fts_intrinsic = [
    'EXT_SOURCES_MEAN',
    'ORGANIZATION_TYPE',
    'EXT_SOURCES_NANMEDIAN',
    'EXT_SOURCES_MIN',
    'EXT_SOURCES_MAX',
    'CREDIT_TO_GOODS_RATIO',
    'OCCUPATION_TYPE',
    'EXT_SOURCE_3',
    'EXT_SOURCES_PROD',
    'GROUP2_EXT_SOURCES_MEDIAN',
    'EXT_SOURCE_2',
    'DAYS_EMPLOYED',
    'BUREAU_CREDIT_DEBT_CREDIT_DIFF_MEAN',
    'DAYS_LAST_PHONE_CHANGE',
    'PREV_Consumer_AMT_ANNUITY_MAX',
    'BUREAU_CLOSED_DAYS_CREDIT_VAR',
    'INCOME_TO_EMPLOYED_RATIO',
    'AMT_ANNUITY',
    'EXT_SOURCE_1',
    'CURRENT_TO_APPROVED_ANNUITY_MEAN_RATIO',
    'EMPLOYED_TO_BIRTH_RATIO',
    'PAYMENT_MEAN_TO_ANNUITY_RATIO',
    'DAYS_BIRTH',
    'BUREAU_ACTIVE_DEBT_CREDIT_DIFF_MEAN',
    'PREV_DAYS_DECISION_MEAN',
    'CREDIT_TO_ANNUITY_RATIO',
    'PREV_DAYS_TERMINATION_MAX'
]


def prepare_features_from_lightgbm(
    data_path: str,
    drop_NAN: bool=False,
    drop_INF: bool=False,
    n_sample: int=None,
) -> tuple[pd.DataFrame, pd.Series]:
    # Get the features and target of individuals with a target.
    X, y = prepare_data(
        data_path=data_path,
        drop_INF=drop_NAN,
        drop_NAN=drop_INF,
        n_sample=n_sample,        
    )
    # Keep selected features.
    X = X.loc[:, sel_fts_intrinsic]
    print(f'Kept only the feature selection from lightgbm: {X.shape}')
    # Remove the 2 individuals with inf values.
    inf_mask = ((X == np.inf) | (X ==-np.inf)).any(axis=1)
    X = X.loc[~inf_mask, :]
    y = y[~inf_mask]
    return X, y


def prepare_and_impute_features_from_lightgbm(
    data_path: str,
    drop_NAN: bool=False,
    drop_INF: bool=False,
    n_sample: int=None,
) -> tuple[pd.DataFrame, pd.Series]:
    # Get the features and target of individuals with a target.
    X, y = prepare_features_from_lightgbm(
        data_path=data_path,
        drop_INF=drop_NAN,
        drop_NAN=drop_INF,
        n_sample=n_sample,        
    )
    #### TODO:
    print(">>> Imputation with 0's")
    X = X.fillna(0)
    
    return X, y


def load_split_clip_scale_and_impute_data(
    data_path: str,
    predictors: List[str]=None, 
    n_sample: int=None,
    ohe: bool=False,
    clip: bool=False,
    scaling_method: str=None,
    imputation_method: str='zero',
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Dict[str, Any]]:
    """ 
    - Read the pickle file.
    - Retain individuals with a known target value.
    - Filter individuals with little information.
    - Eventually select a set of predictors and ensure predictors are valid.
    - Eventually sample.
    - Split into train and test sets. 
    For categorical features
    - Impute with the closest neighbor the label encoded value provided.
    - One hot encode categorical features or not.
    For numerical features :
    - Filter inf and outliers values by clipping (based on the train set).
    - Scale (based on the train set)
    - Impute (based on the train set).
    
    
    Args:
    
    Return X_train, X_test, y_train, y_test, A dictionary with pre-processors
    
    TODO : If I want to reuse that in another project, it could be nice
    to implement a choice between a removal of the outliers and a clipping.
    
    TODO : A transformer to clip the value should be implemented if I want it to
    be part of the pre-processing pipeline of a future individual. 
    """
    # Read the pickle file (previously merged features from Aguiar's script)
    try:
        t0 = time.time()
        print(">>> Read the pickle file")
        data = pd.read_pickle(data_path)
        print(f"It took {time.time()-t0:.3f}s")
    except Exception as e:
        logger.exception(
                """Unable to load features. 
                Check the config FEATURE_PATH.
                Error: %s", e"""
        )
    # Retain individuals with target
    data = data[data['TARGET'].notnull()]    
    # Keep individuals with decent information
    data = keep_individuals_with_info(data, threshold=400)
    # Build predictors list if not passed as parameter
    # and filter non-valid predictors
    if predictors is None:
        predictors = list(data.columns)
    else:
        pass
    
    not_predictors = [
        'TARGET',
        'SK_ID_CURR',
        'SK_ID_BUREAU',
        'SK_ID_PREV',
        'index',
        'level_0',
    ]
    predictors = [feat for feat in predictors if feat not in not_predictors]
    # Extract selected data
    data = data[[*predictors, 'TARGET']]
    # Get categorical and numerical feature names among selected predictors
    initial_categorical_features = [
        'CODE_GENDER', 'FLAG_OWN_CAR', 'NAME_CONTRACT_TYPE',
        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
        'NAME_INCOME_TYPE', 'OCCUPATION_TYPE', 'ORGANIZATION_TYPE',
        'WEEKDAY_APPR_PROCESS_START', 
    ]
    categorical_features = [
        ft for ft in predictors if ft in initial_categorical_features
    ]
    numerical_features = [
        ft for ft in predictors if ft not in initial_categorical_features
    ]
    # Print data shape
    print(
        ">>> Feature selection :"
        "   Data shape: {}, target shape: {}"
        .format(data[predictors].shape, data['TARGET'].shape)
    )
    # It is a trick but I am fitting the ohe_enc before sampling 
    # to ensure it knows about all categories (some could be missing
    # if n_samples is low).
    ohe_enc = OneHotEncoder(sparse_output=False, dtype='int32')
    ohe_enc.fit(data[categorical_features])
    # Eventually sample
    if n_sample:
        print(
            f">>> Sampling with a fixed random state : {n_sample}"
            " random samples."
        )
        data = data.sample(n_sample, random_state=2)
    # Split into train and test sets
    X_train, X_test, y_train, y_test = split_data(
        X=data[predictors],
        y=data['TARGET']
    )
    
    ################################################################
    # Process categorical features based on the X_train information.
    ################################################################
    # Imputation (Use 1NN imputer on LE categorical features)
    cat_imputer = KNNImputer(n_neighbors=1)
    cat_imputer.fit(X_train[categorical_features])
    X_train.loc[:, categorical_features] = (
        cat_imputer.transform(X_train[categorical_features])
    )
    X_test.loc[:, categorical_features] = (
        cat_imputer.transform(X_test[categorical_features])
    )
    if ohe:
        print(">>> One-hot encoding categorical features")
        # One-hot encode categorical features
        ohe_train = pd.DataFrame(
            ohe_enc.transform(X_train[categorical_features]),
            index=X_train.index,
            columns=ohe_enc.get_feature_names_out()
        )
        ohe_test = pd.DataFrame(
            ohe_enc.transform(X_test[categorical_features]),
            index=X_test.index,
            columns=ohe_enc.get_feature_names_out()
        )
        # Replace raw categorical by joining numerical and ohe
        # categorical features
        X_train = pd.concat(
            [X_train[numerical_features], ohe_train],
            axis=1,
        )
        X_test = pd.concat(
            [X_test[numerical_features], ohe_test],
            axis=1,
        )
        # Update feature names
        categorical_features = ohe_enc.get_feature_names_out()
        predictors = [*numerical_features, *categorical_features]
    ##############################################################
    # Process numerical features based on the X_train information.
    ##############################################################
    if clip:
        print(">>> Clipping inf and outliers values")
        # Find boundaries to clip values observing only the train set
        boundaries = define_boundaries_for_numerical_features(
            X_train[numerical_features]
        )
        # Clip values in the train set
        X_train.loc[:, numerical_features] = clip_with_boundaries(
            X_train[numerical_features], 
            boundaries
        )
        # Clip values in the test set with a slight tolerance (10%) if values are 
        # out of the boundaries.
        X_test.loc[:, numerical_features] = clip_with_boundaries(
            X_test[numerical_features], 
            boundaries * 1.1
        )
    # If a scaling method is specified, fit the scaler on the train set and
    # transform both train and test set.
    if scaling_method is not None:
        print(f">>> Start Scaling : {scaling_method}")
        t0 = time.time()
        match scaling_method:
            case 'minmax':
                scaler = MinMaxScaler()
                X_train.loc[:, numerical_features] = (
                    scaler
                    .fit_transform(X_train[numerical_features])
                )
                X_test.loc[:, numerical_features] = (
                    scaler
                    .transform(X_test[numerical_features])
                )
            case 'standard':
                scaler = StandardScaler()
                X_train.loc[:, numerical_features] = (
                    scaler
                    .fit_transform(X_train[numerical_features])
                )
                X_test.loc[:, numerical_features] = (
                    scaler
                    .transform(X_test[numerical_features])
                )
            case 'power_transform':
                scaler = PowerTransformer()
                X_train.loc[:, numerical_features] = (
                    scaler
                    .fit_transform(X_train[numerical_features])
                )
                X_test.loc[:, numerical_features] = (
                    scaler
                    .transform(X_test[numerical_features])
                )
            case _:
                print('No scaling was done because the specified method is not'
                      ' implemented/recognized.\nPlease specify a method among'
                      ' "minmax", "standard", "power_transform".')
        print(f'Scaling completed in {time.time()-t0:.3f}s')
    else:
        print('>>> No Scaling')
    # Imputation
    if imputation_method is not None:
        (
            X_train.loc[:, numerical_features],
            X_test.loc[:, numerical_features],
            num_imputer
        ) = impute(
            X_train.loc[:, numerical_features],
            X_test.loc[:, numerical_features],
            method=imputation_method
        )
    return (
        X_train,
        X_test,
        y_train,
        y_test,
        {
            'categorical_imputer': cat_imputer,
            'one_hot_encoder': ohe_enc,   
            'boundaries': boundaries,
            'numerical_scaler': scaler,
            'numerical_imputer': num_imputer,
        }
    )
    


def keep_individuals_with_info(data, threshold=400):
    """ Remove individuals with more than `threshold` null values.
    
    The default value of 400 is based on the histogram of null values per
    individual."""
    print(f">>> Filtering individual with more than {threshold} null values")
    print(f"before : {len(data)}")
    mask = (data.isnull().sum(axis=1) > 400)
    df = data.copy().loc[~mask, :]
    print(f"after : {len(df)}")
    del data
    gc.collect()
    return df


def define_boundaries_for_numerical_features(
    num_df: pd.DataFrame,
    q_lower: float=0.1,
    q_upper: float=99.9,
)-> pd.DataFrame:
    
    """
    Search for the q_lower and the q_upper percentile in num_df
    (without considering inf values nor nan values).
    
    Note : This must not be applied on categorical features but only on
    a dataframe with numerical features.
    
    Return : A dataframe which indexes correspond to numerical features and
    columns are the associated low and high percentiles q_lower and q_upper."""    
    # Made a Series of a high percentile for each numerical feature
    high_percentiles = (
        num_df
        .replace(np.inf, np.NaN)
        .apply(np.nanpercentile, q=q_upper, axis=0)
    )
    high_percentiles.name = "high_percentiles"
    # Made a Series of a low percentile for each numerical feature
    low_percentiles = (
        num_df
        .replace(-np.inf, np.NaN)
        .apply(np.nanpercentile, q=q_lower, axis=0)
    )
    low_percentiles.name = "low_percentiles"
    # Concatenate boundaries and clip values with both boundaries
    boundaries = pd.concat([low_percentiles, high_percentiles], axis=1)
    return boundaries


def clip_with_boundaries(
    df: pd.DataFrame,
    boundaries: pd.DataFrame
)-> pd.DataFrame:
    """In order to filter inf and -inf values and, by the way, filter outliers
    Values are clipped between boundaries. df must be numerical."""
    clipped_df = (
        df
        .loc[:, list(boundaries.index)]
        .clip(
            lower=boundaries.low_percentiles,
            upper=boundaries.high_percentiles,
            axis=1,
        )
    )
    del df
    gc.collect()
    return clipped_df


def impute(X_train, X_test, method='median'):
    """Fit imputer on X_train and transform X_train and X_test
    
    Return (X_train_imp, X_test_imp, imp) where imp is the fit imputer.
    
    
    --> Problem : This is ok if used on the training set to produce the
    final model and extract the imputer to pre-process individual to be inferred.
    But if done previously to the cross-validation, there will be
    some data leakage.
    
    Note : https://arxiv.org/abs/2010.00718
    
    This article shows that practically, it is often NOT very important to
    proceed imputation on each CV run, and that imputation, which has a very
    high computational cost can be done before cross-validation.
    
    That's why, I will impute the full training set before CV, and use
    the fit imputer on the testing set to keep a measurement of that leakage
    wrt scores on the CV.
    
    """
    t0 = time.time()
    match method:
        case 'median':
            print(f'>>> Imputation with {method}')
            imp = SimpleImputer(strategy='median')
            imp.fit(X_train)
            t1 = time.time()
            fit_duration = t1 - t0
            print(f"Imputer fit duration: {fit_duration:.3f}s")
            X_train_imp = imp.transform(X_train)
            X_test_imp = imp.transform(X_test)
            t2 = time.time()
            imputation_duration = t2 - t1
            print(f"Imputation duration: {imputation_duration:.3f}s")
        case 'knn':
            knn_n_neighbors=2
            print(f'>>> Imputation with {method} -- k={knn_n_neighbors}')
            imp = KNNImputer(n_neighbors=knn_n_neighbors)
            imp.fit(X_train)
            t1 = time.time()
            fit_duration = t1 - t0
            print(f"Imputer fit duration: {fit_duration:.3f}s")
            X_train_imp = imp.transform(X_train)
            X_test_imp = imp.transform(X_test)
            t2 = time.time()
            imputation_duration = t2 - t1
            print(f"Imputation duration: {imputation_duration:.3f}s")
        case 'zero':
            print(f'>>> Imputation with {method}')
            imp = SimpleImputer(strategy='constant', fill_value=0)
            imp.fit(X_train)
            t1 = time.time()
            fit_duration = t1 - t0
            print(f"Imputer fit duration: {fit_duration:.3f}s")
            X_train_imp = imp.transform(X_train)
            X_test_imp = imp.transform(X_test)
            t2 = time.time()
            imputation_duration = t2 - t1
            print(f"Imputation duration: {imputation_duration:.3f}s")
        case _:
            print("No imputation was made.")
    return (X_train_imp, X_test_imp, imp)
            

# def impute_train_and_test(X_train, X_test, method='median'):
#     imputer, X_train_imp = impute(X_train, method=method)
#     X_test_imp = imputer.transform(X_test)
#     return imputer, X_train_imp, X_test_imp
    
    
    
def rename_col_for_lgbm_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    # Change columns names ([LightGBM] Do not support special JSON characters in feature name.)
    new_names = {col: re.sub(r'[^A-Za-z0-9_]+', '', col) for col in df.columns}
    new_n_list = list(new_names.values())
    # [LightGBM] Feature appears more than one time.
    new_names = {col: f'{new_col}_{i}' if new_col in new_n_list[:i] else new_col for i, (col, new_col) in enumerate(new_names.items())}
    return df.rename(columns=new_names)


def split_data(X, y, test_size_=0.2):
    """
    - Merge features X and target y
    - Split in train/test sets
    - Re-extract targets.
    
    Return (train_df, test_df, y_train, y_test )
    """ 
    data = X.merge(y, on='SK_ID_CURR')
    train_df, test_df = train_test_split(
        data,
        test_size=test_size_,
        random_state=108,
        stratify=data.TARGET
    )
    y_train = train_df.pop('TARGET')
    y_test = test_df.pop('TARGET')
    print(
        f"Train shapes : {train_df.shape}, {y_train.shape}\n"
        + f"Test shapes : {test_df.shape}, {y_test.shape}"
    )
    return train_df, test_df, y_train, y_test 
    

def class_percentages(target):
    """
    Print percentages of each element in target.
    """
    return target.value_counts() * 100 / len(target)


def make_folds(stratified=True):
    """ Return an sklearn Kfold(possibly stratified) cross-validator 
    with a fixed random state to enable comparisons between models.
    """
    if not stratified:
        folds = KFold(
            n_splits=config.NUM_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_SEED
        )
    else:
        folds = StratifiedKFold(
            n_splits=config.NUM_FOLDS,
            shuffle=True,
            random_state=config.RANDOM_SEED
        )
    return folds


########################################################################
### Define some metrics and Scorers
########################################################################
class Scorer:
    
    def __init__(
        self,
        name: str,
        score_func,
        score_kwargs: dict()=None, 
        greater_is_better: bool=True,
    ):
        self.name = name
        self.score_func = score_func
        self.score_kwargs = score_kwargs
        self.greater_is_better = greater_is_better

    def evaluate_predictions(self, y_true, y_pred):
        """ Compute the score."""
        if self.score_kwargs:
            return self.score_func(y_true, y_pred, **self.score_kwargs)
        else:
            return self.score_func(y_true, y_pred)
            
    def find_best_threshold_and_score(
        self,
        y_true,
        proba_pred,
        threshold_step=0.01
    ):
        eps = threshold_step * 0.01
        thresholds = np.arange(0, 1+eps, threshold_step)
        scores = np.zeros_like(thresholds)
        for idx, t in enumerate(thresholds):
            scores[idx] = self.evaluate_predictions(
                y_true,
                threshold_class_from_proba(proba_pred, threshold=t)    
            )
        if self.greater_is_better:
            best_score = np.max(scores)
            best_threshold = thresholds[np.argmax(scores)]  
        else:
            best_score = np.min(scores)
            best_threshold = thresholds[np.argmin(scores)]  
        return best_threshold, best_score
    
    def print(self):
        print(f"name : {self.name}")
        print(f"score_func : {self.score_func}")
        print(f"score_kwargs : {self.score_kwargs}")
        print(f"greater_is_better : {self.greater_is_better}")
# End Scorer


def threshold_class_from_proba(probabilities, threshold=0.5):
    """ Return an array of probabilities.shape
    where the values are 0 or 1, depending on the threshold value.
    """
    return (probabilities > threshold).astype('int8')


def specificity_score(y_true, y_pred):
    """ Specificity is the recall of the negative class"""
    return recall_score(y_true, y_pred, pos_label=0)


def weighted_geometric_mean_score(y_true, y_pred, recall_weight=5):
    """
    Compute the weighted geometric mean of recall and specificity.
    
    It is the (recall_weight + 1)th root of the product
    (recall^recall_weight * specificity)
    """
    recall = recall_score(y_true, y_pred)
    specificity = specificity_score(y_true, y_pred)
    return np.power(
        np.power(recall, recall_weight) * specificity,
        1 / (recall_weight + 1)   
    )


def loss_of_income_score(y_true, y_pred, fn_weight=5):
    """ Compute the loss of income due to false negatives
    and false positives over-penalizing false negative with a weight. 
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return fn_weight*fn + fp
   
            
# Define a dict of Scorers       
# I commented some scorers for time economy. 
my_Scorers = {
    # Scorer('Accuracy', accuracy_score),
    'AUC': Scorer('AUC', roc_auc_score),
    # Scorer('f1', fbeta_score, score_kwargs={'beta': 1}),
    'f2': Scorer('f2', fbeta_score, score_kwargs={'beta': 2}),
    'recall_specificity_G_mean': Scorer(
        'recall_specificity_G_mean',
        weighted_geometric_mean_score,
    ),
    'loss_of_income' : Scorer(
        'loss_of_income',
        loss_of_income_score,
        greater_is_better=False
    ),
}
    
    
def compute_scorers_best_threshold_and_score(
    scorers: Dict[str, Scorer],
    y_true,
    proba
):
    """ From the probability of being in the positive class and for each
    Scorer in scorers:
    
    - Search the best threshold to optimize the Scorer metric.
    - Add the best threshold and the best score to the returned
    dictionary.
    
    ARGS:
    scorers : A dict of Scorers.
    y_true : True labels
    proba : same shape as y_true (probability predictions of being
    in the positive class.)
    """
    metrics = {}
    for scorer_key, scorer in scorers.items():
        (
            metrics[f'threshold_{scorer.name}'],
            metrics[f'{scorer.name}']
        ) = scorer.find_best_threshold_and_score(y_true, proba)
    return metrics


########################################################################
# Inspired from https://www.phdata.io/blog/bayesian-hyperparameter-optimization-with-mlflow/
# functions for hyperparameter tuning with hyperopt using mlflow tracking and Cross-validation.
# How fmin works: https://github.com/hyperopt/hyperopt/wiki/FMin
#
########################################################################
# CROSS-VALIDATION
########################################################################
# def predict_proba(model, X, cv):
#     return cross_val_predict(model, X, cv, method='predict_proba')

########################################################################
# fmin needs an objective function which :
#   - takes a set of hyperparameters
#   - return a value or dictionary with a value which is to be minimized.
# Below, I chose to return only the relevant value.
########################################################################
def objective_adjusted_to_data_and_model(
    X_train: Union[pd.DataFrame, np.array],
    y_train: Union[pd.Series, np.array],
    X_test: Union[pd.DataFrame, np.array],
    y_test: Union[pd.Series, np.array],
    cv,
    model,
    scorers: Dict[str, Scorer],
    optimization_scorer: Scorer,
    exp_id: str='0',
    mlflow_tags: Dict[str, str]=None,
):
    """ This is a wrapper. Build the Hyperopt objective function 
    for :
    - a given dataset,
    - a given model,
    - a given evaluation metric to be optimized among those computed.    

    Args:
      X_train: feature matrix for training/CV data
      y_train: label array for training/CV data
      X_test: feature matrix for test data
      y_test: label array for test data
      model: Estimator to be set with the hyperparams returned by hyperopt
      scorers : A list of Scorers (class created for this project)
      optimization_scorer: the scorer which computes the metric to be optimized
      exp_id: An mlflow experiment id to ensure logs of nested runs are in
      the right place.
      mlflow_tags: to be added to all nested runs.

    Returns:
        Objective function set up to take hyperparams dict from Hyperopt.
    """

    def objective(hyperparams):
        """Extract the loss from a model fit multiple times on CV-folds
        with the hyperparams configuration.
        
        While building the objective function, 
        the make_cv_predictions_evaluate_and_log() function takes care
        to track the model with MLflow nested runs.
        """
        metrics = make_cv_predictions_evaluate_and_log(
            X_train,
            y_train,
            X_test,
            y_test,
            cv,
            model,
            model_params=hyperparams,
            scorers=scorers,
            mlflow_tags=mlflow_tags,
            exp_id=exp_id,
            nested=True,
        )
        
        # Extract the optimization metric and modify it (sign + or -)
        # to ensure the value is to be minimized.
        opt_metric_name = f"CV_{optimization_scorer.name}"
        if optimization_scorer.greater_is_better:
            evaluation_metric = -metrics[opt_metric_name]
        else:
            evaluation_metric = metrics[opt_metric_name]
            
        return {'status': hyperopt.STATUS_OK, 'loss': evaluation_metric}
    return objective


def make_cv_predictions_evaluate_and_log(
    x_train: Union[pd.DataFrame, np.array],
    y_train: Union[pd.Series, np.array],
    x_test: Union[pd.DataFrame, np.array],
    y_test: Union[pd.Series, np.array],
    cv,
    model,
    model_params: Dict[str, Any],
    scorers: Dict[str, Scorer],
    mlflow_tags: Dict[str, str],
    exp_id: str,
    nested: bool = False
) -> Dict[str, Any]:
    """ 
    1) Instantiate a model with model_params.
    2) Make prediction (default to predict_proba) on the CV-folds 
    using cross_val_predict (fitting several models on folds combinations)
    3) Compute best threshold and score for each scorer on the out-of-fold prediction.
    4) Fit on the training set and compute metrics on the testing set.
    5) Log to exp_id folder in MLflow
    6) Return all metrics.
    
    Args:
        x_train: feature matrix for training/CV data
        y_train: label array for training/CV data
        x_test: feature matrix for test data
        y_test: label array for test data
        cv: a sklearn cross-validation iterator on folds.
        model: a classifier
        model_params: the non-default parameters of the model
        scorers: a list of Scorer used for evaluation and finding the best 
        threshold from the probability prediction.
        nested: if true, mlflow run will be started as child
        of existing parent
    """
    with mlflow.start_run(experiment_id=exp_id, nested=nested) as run:
        # Instantiate the model
        model = model.set_params(**model_params)
        
        # Work on cross-validation folds.
        # Fit models and extract the probability of being in the
        # positive class (out-of-fold prediction is used underneath).
        proba_pos_cv = cross_val_predict(
            model, x_train, y_train,
            cv=cv,
            method='predict_proba',
        )[:, 1]
        # Search best threshold and score for each scorer
        metrics_= compute_scorers_best_threshold_and_score(
            scorers,
            y_train,
            proba_pos_cv
        )
        metrics_cv = {
           "CV_"+k: v for (k, v) in metrics_.items()
        }
        
        # Fit on the whole training set and evaluate on the testing set 
        # in order to assess under/over-fitting.
        model.fit(x_train, y_train)
        proba_pos_test = model.predict_proba(x_test)[:, 1]
        metrics_= compute_scorers_best_threshold_and_score(
            scorers,
            y_test,
            proba_pos_test
        )
        metrics_test = {
           "test_"+k: v 
           for (k, v) in metrics_.items()
        }
        
        metrics = {**metrics_test, **metrics_cv}
        # MLflow tracking
        mlflow.log_params(model_params)
        mlflow.log_metrics(metrics)
        if mlflow_tags is not None:
            mlflow.set_tags(mlflow_tags)
        return metrics


def log_best(run: mlflow.entities.Run, metric: str) -> None:
    """Log the best parameters from optimization to the parent experiment.

    Args:
        run: current run to log metrics
        metric: name of metric to select best and log
    """
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        [run.info.experiment_id],
        "tags.mlflow.parentRunId = '{run_id}' ".format(run_id=run.info.run_id)
    )

    best_run = min(runs, key=lambda run: run.data.metrics[metric])

    mlflow.set_tag("best_run", best_run.info.run_id)
    mlflow.log_metric(f"best_{metric}", best_run.data.metrics[metric])
    
    
    
# def display_confusion_matrix(y_true, y_pred):
    
#     cm = confusion_matrix(y_true, y_pred)
#     plt.figure(figsize=(5,5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title('Confusion matrix @{:.2f}'.format(threshold))
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')

#     print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
#     print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
#     print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
#     print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
#     print('Total Fraudulent Transactions: ', np.sum(cm[1]))

    
def experiment_id_from_experiment_name(experiment_name):
    experiment=dict(mlflow.get_experiment_by_name(experiment_name))
    return experiment['experiment_id'] 





