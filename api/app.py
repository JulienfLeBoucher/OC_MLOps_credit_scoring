from flask import Flask, jsonify
import pandas as pd
import api_utils

########################################################################
# Variables
########################################################################
DEBUG = True
# Path to find model information
MLFLOW_TRACKING_URI = "/home/louberehc/OCR/projets/7_scoring_model/mlruns"
# Choose the model in the MLflow registry
model_name = "lgbm_test_2"
stage = "Staging"
version = 2
# Choose the data to consider and load in the app
APP_DATA_PATH = "/home/louberehc/OCR/projets/7_scoring_model/pickle_files/reduced_data.pkl"
########################################################################
def shutdown_server():
    # https://stackoverflow.com/questions/15562446/how-to-stop-flask-application-without-using-ctrl-c
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    

# Set the Mlflow tracking URI
api_utils.set_mlflow_tracking_URI(MLFLOW_TRACKING_URI)

# Instantiate the flask object
app = Flask(__name__)
    
    
# Load the model 
model_run_id = (
    api_utils
    .get_model_run_id_from_name_stage_version(model_name, stage, version)
)
model_uri = api_utils.make_model_uri(model_run_id)
model = api_utils.load_model(model_uri)
# Get model threshold
model_threshold = api_utils.get_model_threshold(model_run_id)

# Load data
features, target = api_utils.load_data(APP_DATA_PATH)
valid_customer_ids = features.index


@app.route("/")
def welcome():
    return (
        "Welcome! Go to the '/prediction/<customer_id>' "
        "to get a model inference."
    )


@app.route('/prediction/')
def print_id_list():
    return f'The list of valid client ids :\n\n{list(valid_customer_ids)}'


@app.route('/prediction/<int:customer_id>')
def prediction(customer_id):
    if customer_id in valid_customer_ids:
        #return f'valid client: {id_client}'
        proba = api_utils.get_customer_proba(model, features, customer_id)
        customer_score_info = {
            'customer_id': customer_id,
            'proba_risk_class': proba,
            'model_threshold': model_threshold,
            'customer_class': 'no_risk' if proba <= model_threshold else 'risk' 
        }
        return jsonify(customer_score_info)
    else:
        return 'Customer_id is not valid.'


@app.get('/data')
def send_data():
    return features, target
    

@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'


if __name__ == "__main__":
    app.run("localhost", port=8435, debug=DEBUG)