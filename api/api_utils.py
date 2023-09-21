import mlflow
import pandas as pd
from mlflow import MlflowClient
from typing import Dict


# MLFLOW
def set_mlflow_tracking_URI(path):
    return mlflow.set_tracking_uri(path)


def get_model_run_id_from_name_stage_version(
    name: str,
    stage: str,
    version: int,
) -> str:
    """ return the model rnu_id from some information of the model registered
    in the model registry.
    
    TODO: add check of name, stage and version and return error accordling
    if no model is found."""
    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{name}'"):
        if (mv.current_stage == stage) and (mv.version == version):
            return mv.run_id
    return None     


def get_model_metrics(run_id: str) -> Dict[str, float]:
    run = mlflow.get_run(run_id=run_id)
    return run.data.metrics


def get_model_threshold(run_id: str) -> float:
    """ From the model run_id, return the model threshold saved in 
    metrics."""
    return get_model_metrics(run_id)['train_threshold_loss_of_income']  


def make_model_uri(run_id):
    return f"runs:/{run_id}/model"


def load_model(model_uri):
    """ Load the model from the artifacts in the MLflow model registry """
    return mlflow.sklearn.load_model(model_uri=model_uri)


def get_sorted_features_by_importance(model, features):
    """ Use the feature_importance attributes of
    the lightgbm model to return features names sorted by importance
    from the most to the least important."""
    return list(features.columns[model.feature_importances_.argsort()[::-1]])


# Load data
def load_data(DATA_PATH):
    """ Load both the features and the target associated to customers """
    df = pd.read_pickle(DATA_PATH).astype("float64")
    target = df.pop('TARGET')
    return df, target


# Prediction
def format_customer_data(features, customer_id):
    """ Passing a pd.Series to the model fails. It must be converted
    first as such. """
    return pd.DataFrame(features.loc[customer_id,:]).T 


def get_customer_proba(model, features, customer_id):
    """ Get the customer probability for being in the risky class. 
    
    - model is a loaded mlflow model.
    - features is a pd.DataFrame with index being customer_ids and
    columns, all the features necessary to predict.
    - customer_id is used to select the row of the data."""
    return model.predict_proba(
        format_customer_data(features, customer_id)
    )[0][1]


def get_index(customer_id, features):
    """ return the line index of a customer in features as a number
    ranging from 0 to n-1, n being the shape[0] of features."""
    return (
        features
        .reset_index()
        .query('SK_ID_CURR == @customer_id')
        .index
        .values[0]
    )