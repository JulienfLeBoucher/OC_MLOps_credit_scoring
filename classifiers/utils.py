#%% 
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
)
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
)

#%%
FEATURE_PATH = "./pickle_files/features.pkl.gz"
STRATIFIED_KFOLD = True
NUM_FOLDS = 5
RANDOM_SEED = 784


def prepare_data(
    data_path: str
) -> tuple[pd.DataFrame, pd.Series]:
    """ 
    - Read the pickle file.
    - Retain individuals with a known target value.
    - Determine useful columns for prediction.
    - Return the feature dataframe and the target series.
    """
    data = pd.read_pickle(data_path)
    
    data = data[data['TARGET'].notnull()]
    
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
# %%

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
    
df, target = prepare_data(FEATURE_PATH)
