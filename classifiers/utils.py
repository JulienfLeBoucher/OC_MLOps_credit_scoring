import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    make_scorer,
    fbeta_score,
)

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

### SCORING
ftwo_scorer = make_scorer(fbeta_score, beta=2)

def prepare_data(
    data_path: str,
    drop_NAN: bool=True,
    drop_INF: bool=True,
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
    
    # Eventually drop columns with nulls 
    # (eventually with inf being first replace with NaNs)
    if drop_NAN:
        if drop_INF:
            data = data.replace(np.inf, np.NaN)
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
    
    print(
        "Data shape: {}, target shape: {}"
        .format(data[predictors].shape, data['TARGET'].shape)
    )
    return data[predictors], data['TARGET']


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


### DISPLAY ###