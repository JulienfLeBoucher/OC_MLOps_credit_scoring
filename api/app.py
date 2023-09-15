from flask import Flask, jsonify, render_template

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg') # https://stackoverflow.com/questions/69924881/userwarning-starting-a-matplotlib-gui-outside-of-the-main-thread-will-likely-fa
import base64
import pandas as pd
import shap
import api_utils

########################################################################
"""
This API can find and load a model from the MLflow registry if
the MLflow tracking folder is accessible. (It works locally.)

Nevertheless, for deployment ease, I've also implemented a way
to load a serialized pickle model which is provided directly in the 
api directory in order to avoid deploying the mlflow tracking folder.
"""

########################################################################
# Variables
########################################################################
DEBUG = True
# Choose the data to consider and load in the app
APP_DATA_PATH = "/home/louberehc/OCR/projets/7_scoring_model/pickle_files/reduced_data.pkl"

# Above for local use.
MLFLOW_BACKEND_AVAILABLE = True 
MLFLOW_TRACKING_URI = "/home/louberehc/OCR/projets/7_scoring_model/mlruns"
# model attributes in the MLflow registry
model_name = "lgbm_test_2"
stage = "Staging"
version = 2
########################################################################    
### Load the model 
if MLFLOW_BACKEND_AVAILABLE:
    # if the model registry is accessible from the API
    # Set the Mlflow tracking URI
    api_utils.set_mlflow_tracking_URI(MLFLOW_TRACKING_URI)
    # get the model run id
    model_run_id = (
        api_utils
        .get_model_run_id_from_name_stage_version(model_name, stage, version)
    )
    # Build a valid model uri from it
    model_uri = api_utils.make_model_uri(model_run_id)
    # Load the model from the model registry
    model = api_utils.load_model(model_uri)
    # Get model threshold
    model_threshold = api_utils.get_model_threshold(model_run_id)
else:
    # Load the model attached as a pkl object.
    model_name = "A_model_not_from_the_MLflow_registry"
    stage = ""
    version = 0
    #TODO: load the pickle model.
    # model_threshold = 

### Load data
features, target = api_utils.load_data(APP_DATA_PATH)
valid_customer_ids = features.index

# Get model Shap interpretability
explainer = shap.TreeExplainer(model)
# shap values explainer with 3 fiels (values, base_values, data)
sv = explainer(features)
# Retain explanations only for being in the positive class.
exp = shap.Explanation(
    sv.values[:,:,1], 
    sv.base_values[:,1], 
    data=features.values, 
    feature_names=features.columns
)

# Instantiate the flask object
app = Flask(__name__)

@app.route("/")
def welcome():
    return ("Welcome!")


@app.route('/prediction/')
def print_id_list():
    return f'The list of valid client ids :\n\n{list(valid_customer_ids)}'


@app.get('/model_info')
def model_info():
    return jsonify(
        {
            'name': model_name,
            'stage': stage,
            'version': version,
            'type': type(model).__name__,
            'decision_threshold': model_threshold
        }
    )


@app.route('/prediction/<int:customer_id>')
def prediction(customer_id):
    if customer_id in valid_customer_ids:
        #return f'valid client: {id_client}'
        proba = api_utils.get_customer_proba(model, features, customer_id)
        customer_info = {
            'id': customer_id,
            'proba_risk_class': proba,
            'class': 'no_risk' if proba <= model_threshold else 'risk' 
        }
        return jsonify(customer_info)
    else:
        return 'Customer_id is not valid.'


@app.get('/data')
def send_data():
    return features.to_dict()
    

@app.get('/target')
def send_target():
    return pd.DataFrame(target).to_dict()


@app.get('/global_shap')
def send_global_shap():
    img_path = './shap_output/global_shap.png'
    # Get the pyplot object without showing it
    fig = plt.figure()
    shap.summary_plot(
        exp,
        features,
        max_display=30,
        plot_size=(10,13),
        show=False
    )
    # Save the image locally
    fig.tight_layout()
    fig.savefig(img_path)
    # read the image file and encode it adding the adapted prefix
    with open(img_path, 'rb') as img:
        img_binary_file_content = img.read()
        encoded = base64.b64encode(img_binary_file_content)
        return (b'data:image/png;base64,' + encoded)
    
@app.get('/local_shap/<customer_id>')
def send_global_shap():
    img_path = './shap_output/global_shap.png'
    # Get the pyplot object without showing it
    fig = plt.figure()
    shap.summary_plot(
        exp,
        features,
        max_display=30,
        plot_size=(10,13),
        show=False
    )
    # Save the image locally
    fig.tight_layout()
    fig.savefig(img_path)
    # read the image file and encode it adding the adapted prefix
    with open(img_path, 'rb') as img:
        img_binary_file_content = img.read()
        encoded = base64.b64encode(img_binary_file_content)
        return (b'data:image/png;base64,' + encoded)    
    


if __name__ == "__main__":
    app.run("localhost", port=8435, debug=DEBUG)