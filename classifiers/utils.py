import re
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    train_test_split,
)

from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    fbeta_score,
)

from imblearn.metrics import geometric_mean_score

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

### load and prepare the data
def prepare_data(
    data_path: str,
    drop_NAN: bool=True,
    drop_INF: bool=True,
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
        print(f"took only {n_sample} random samples.")
        data = data.sample(n_sample)

    print(
        "Data shape: {}, target shape: {}"
        .format(data[predictors].shape, data['TARGET'].shape)
    )
    return data[predictors], data['TARGET'].astype('int8')


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
    display(target.value_counts() * 100 / len(target))
    return None    


def make_folds(stratified=True):
    """ Return an sklearn Kfold(possibly stratified) cross-validator with a 
    fixed random state to enable comparisons between models.
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


### Define some metrics and scorers
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


def loss_of_income_func(y_true, y_pred, fn_weight=5):
    """ Compute the loss of income due to false negatives
    and false positives over-penalizing false negative with a weight. 
    """
    tn, fp, fn, tp = confusion_matrix(**vectors).ravel()
    return fn_weight*fn + fp
   
   
scorers = {
    'accuracy_score': make_scorer(accuracy_score),
    'auc_score': make_scorer(roc_auc_score),
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'f1_score': make_scorer(fbeta_score, beta=1),
    'f2_score': make_scorer(fbeta_score, beta=2),
    'f3_score': make_scorer(fbeta_score, beta=3),
    'geometric_mean_score': make_scorer(geometric_mean_score, average='binary'),
    'weighted_geometric_mean_score': make_scorer(weighted_geometric_mean_score),
    'loss_of_income': make_scorer(loss_of_income_func, greater_is_better=False)
}

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

def prepare_and_impute_intrinsic_selected_features():
    # Get the features and target of individuals with a target.
    X, y = utils.prepare_data(
        config.FEATURE_PATH,
        drop_INF=False,
        drop_NAN=False,
    )
    # Keep selected features.
    X = X[sel_fts_intrinsic]
    # Remove 2 individuals with inf values.
    inf_mask = ((X == np.inf) | (X ==-np.inf)).any(axis=1)
    X = X.loc[~inf_mask, :]
    y = y[~inf_mask]
    
    
    return X, y