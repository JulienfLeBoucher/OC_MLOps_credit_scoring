from flask import Flask#, render_template, jsonify
import json
import pandas as pd

import mlflow.pyfunc
import pickle
import json
import requests

DEBUG = True

app = Flask(__name__)


def load_model(model_name: str, stage: str):
    """ Load the model from the artifacts in the MLflow model registry """
    return mlflow.pyfunc.load_model(model_uri=f"models:/{model_name}/{stage}")


def load_data(DATA_PATH):
    """ Load both the features and the target associated to customers """
    df = pd.read_pickle(DATA_PATH).astype("float64")
    target = df.pop('TARGET')
    return df, target


def shutdown_server():
    # https://stackoverflow.com/questions/15562446/how-to-stop-flask-application-without-using-ctrl-c
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    

# Load model and data
model_name = "lgbm_test"
stage = "Staging"
APP_DATA_PATH = "/home/louberehc/OCR/projets/7_scoring_model/pickle_files/reduced_data.pkl"

model = load_model(model_name, stage)
features, target = load_data(APP_DATA_PATH)
valid_customer_ids = features.index


@app.route("/")
def hello():
    return "Hello World!"

@app.route('/prediction/')
def print_id_list():
    return f'The list of valid client ids :\n\n{list(features.index)}'

# Note : the following should automatically call jsonify on the dict.
@app.route('/prediction/<int:id_client>')
def prediction(id_client):
    if id_client in valid_customer_ids:
        customer_pred = {
            'customer_id': id_client,
            'customer_score': 0.48,
        }
        #return f'valid client: {id_client}'
        return customer_pred
    else:
        return 'no valid client number'
    

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == "__main__":
    app.run("localhost", port=8435, debug=DEBUG)